import asyncio
import io
import json

import httpx
import pytest
from PIL import Image

from vizwiz.serve.config import ServeConfig
from vizwiz.serve.gateway import create_app
from vizwiz.serve.mock_model import create_mock_app
from vizwiz.serve.model_client import ModelClient
from vizwiz.serve.registry import (
    InMemoryRegistry,
    RegistryError,
    download_cached,
    metadata_for,
    promote,
    rollback,
)

BASE = "Qwen/Qwen3-VL-4B-Instruct"
NAME = "e5-multi-lora"


def adapter_dir(tmp_path, tag, rank=16, base=BASE):
    d = tmp_path / f"adapter-{tag}"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "r": rank, "lora_alpha": 2 * rank})
    )
    (d / "adapter_model.safetensors").write_bytes(tag.encode())
    return d


def registry_with(tmp_path, specs):
    """specs: [(rank, base), ...] -> registry with versions v0, v1, ...; v0 is production."""
    reg = InMemoryRegistry()
    for i, (rank, base) in enumerate(specs):
        d = adapter_dir(tmp_path, f"v{i}", rank, base)
        reg.register(NAME, d, metadata_for(d, "multi", f"run{i}", None))
    promote(reg, NAME, "v0", BASE, 16)
    return reg


def test_promote_and_rollback_move_aliases(tmp_path):
    reg = registry_with(tmp_path, [(16, BASE), (16, BASE), (32, BASE), (16, "other/model")])
    assert reg.get(NAME, "production").version == "v0"
    new, old = promote(reg, NAME, "v1", BASE, 16)
    assert (new.version, old.version) == ("v1", "v0")
    assert reg.get(NAME, "previous").version == "v0"
    assert [v.aliases for v in reg.versions(NAME)][:2] == [("previous",), ("production",)]
    assert promote(reg, NAME, "v1", BASE, 16)[1] is None  # already live: no change
    new, old = rollback(reg, NAME)
    assert (new.version, old.version) == ("v0", "v1")
    with pytest.raises(RegistryError, match="rank 32"):
        promote(reg, NAME, "v2", BASE, 16)
    with pytest.raises(RegistryError, match="trained on"):
        promote(reg, NAME, "v3", BASE, 16)
    assert reg.get(NAME, "production").version == "v0"  # refused promotions change nothing
    fresh = (
        registry_with(tmp_path / "x", [(16, BASE)]) if (tmp_path / "x").mkdir() is None else None
    )
    with pytest.raises(RegistryError, match="no previous"):
        rollback(fresh, NAME)


def test_metadata_and_cached_download(tmp_path):
    d = adapter_dir(tmp_path, "m")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "git_commit": "abc",
                "tasks": {
                    "answer": {"metrics": {"vqa_accuracy": 62.56}},
                    "quality": {"metrics": {"macro_f1": 65.59}},
                },
            }
        )
    )
    md = metadata_for(d, "multi", "e5", metrics)
    assert md == {
        "task": "multi",
        "base_model": BASE,
        "rank": 16,
        "alpha": 32,
        "run_id": "e5",
        "git_commit": "abc",
        "metrics_test": {"answer": 62.56, "quality": 65.59},
    }
    reg = InMemoryRegistry()
    v = reg.register(NAME, d, md)
    path = download_cached(reg, v, tmp_path / "cache")
    assert (path / "adapter_model.safetensors").read_bytes() == b"m"
    assert download_cached(reg, v, tmp_path / "cache") == path  # second call: no download


# ---------------------------------------------------------------- gateway with registry


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 80).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def make_app(tmp_path, reg, token="s3cret", enabled=True):
    cfg = ServeConfig(
        model_url="http://mock",
        timeouts={"backoff_s": 0},
        registry={"enabled": enabled, "cache_dir": str(tmp_path / "cache")},
    )
    mock = create_mock_app(adapters=[])  # nothing loaded until the gateway loads it
    client = ModelClient("http://mock", backoff_s=0, transport=httpx.ASGITransport(app=mock))
    return create_app(cfg, client, registry=reg, admin_token=token)


async def session(app, calls):
    """Run requests inside the app's real startup/shutdown (startup loads adapters)."""
    out = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            for fn in calls:
                out.append(await fn(c))
    return out


def test_gateway_promote_rollback_and_refusals(tmp_path):
    reg = registry_with(tmp_path, [(16, BASE), (16, BASE), (16, BASE), (32, BASE)])
    app = make_app(tmp_path, reg)
    admin = {"X-Admin-Token": "s3cret"}

    def assist(c):
        return c.post(
            "/v1/assist",
            files={"image": ("p.jpg", jpeg(), "image/jpeg")},
            data={"question": "What is this?"},
        )

    stages = lambda c: c.get("/adapters")  # noqa: E731
    r = asyncio.run(
        session(
            app,
            [
                stages,
                assist,
                lambda c: c.post(f"/adapters/{NAME}/promote", json={"version": "v1"}),  # no token
                lambda c: c.post(
                    f"/adapters/{NAME}/promote",
                    json={"version": "v1"},
                    headers={"X-Admin-Token": "wrong"},
                ),
                lambda c: c.post(
                    f"/adapters/{NAME}/promote", json={"version": "v1"}, headers=admin
                ),
                assist,
                lambda c: c.post(
                    f"/adapters/{NAME}/promote", json={"version": "v2"}, headers=admin
                ),
                stages,
                lambda c: c.post(f"/adapters/{NAME}/rollback", headers=admin),
                lambda c: c.post(
                    f"/adapters/{NAME}/promote", json={"version": "v3"}, headers=admin
                ),
                lambda c: c.post(
                    "/adapters/unknown/promote", json={"version": "v0"}, headers=admin
                ),
                stages,
                lambda c: c.get("/metrics"),
                lambda c: c.post(f"/adapters/{NAME}/load", json={"version": "v0"}, headers=admin),
            ],
        )
    )
    start, a1, no_tok, bad_tok, p1, a2, p2, after_p2, rb, p3, unknown, end, metrics, load = r
    assert set(start.json()["stages"].values()) == {f"{NAME}-v0"}
    assert start.json()["registry"]["live"][NAME]["version"] == "v0"
    assert a1.json()["trace"][0]["adapter"] == f"{NAME}-v0"
    assert no_tok.status_code == 401 and bad_tok.status_code == 401
    assert p1.json()["live"] == "v1" and p1.json()["previous"] == "v0"
    assert a2.json()["status"] == "answered" and a2.json()["trace"][0]["adapter"] == f"{NAME}-v1"
    assert p2.json()["live"] == "v2"
    assert after_p2.json()["registry"]["loaded"] == [f"{NAME}-v1", f"{NAME}-v2"]  # v0 unloaded
    assert f"{NAME}-v0" not in after_p2.json()["loaded"]
    assert rb.json() == {
        "live": "v1",
        "previous": "v2",
        "stages": {s: f"{NAME}-v1" for s in ("quality", "answerability", "answer")},
    }
    assert p3.status_code == 409 and "rank 32" in p3.json()["detail"]
    assert unknown.status_code == 409
    assert set(end.json()["stages"].values()) == {f"{NAME}-v1"}  # refusals changed nothing
    assert reg.get(NAME, "production").version == "v1"
    assert f'vizwiz_adapter_live{{adapter="{NAME}-v1",stage="answer"}} 1.0' in metrics.text
    assert f'adapter="{NAME}-v0"' not in metrics.text
    assert load.status_code == 200 and load.json()["loaded"] == f"{NAME}-v0"


def test_admin_disabled_without_token_or_registry(tmp_path):
    reg = registry_with(tmp_path, [(16, BASE)])
    no_token = make_app(tmp_path, reg, token=None)
    (r,) = asyncio.run(
        session(
            no_token,
            [lambda c: c.post(f"/adapters/{NAME}/rollback", headers={"X-Admin-Token": "x"})],
        )
    )
    assert r.status_code == 403
    plain = create_app(
        ServeConfig(model_url="http://mock"),
        ModelClient("http://mock", transport=httpx.ASGITransport(app=create_mock_app())),
        admin_token="x",
    )
    (r,) = asyncio.run(
        session(
            plain, [lambda c: c.post(f"/adapters/{NAME}/rollback", headers={"X-Admin-Token": "x"})]
        )
    )
    assert r.status_code == 404
