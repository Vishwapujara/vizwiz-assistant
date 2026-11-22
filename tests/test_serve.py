import asyncio
import io
import math
import random

import httpx
import pytest
from PIL import Image

from vizwiz.data.config import load_data_config
from vizwiz.serve import guidance
from vizwiz.serve.cascade import Cascade, unanswerable_score
from vizwiz.serve.config import ServeConfig, load_serve_config
from vizwiz.serve.gateway import create_app
from vizwiz.serve.mock_model import create_mock_app
from vizwiz.serve.model_client import ModelClient, ModelError
from vizwiz.serve.preprocess import to_data_url


def photo(kind: str, size=(96, 64)) -> Image.Image:
    rng = random.Random(0)
    lo, hi = {"normal": (0, 255), "dark": (0, 60), "flat": (128, 130)}[kind]
    data = bytes(rng.randint(lo, hi) for _ in range(size[0] * size[1] * 3))
    return Image.frombytes("RGB", size, data)


def jpeg(kind: str) -> bytes:
    buf = io.BytesIO()
    photo(kind).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def cfg(**kw) -> ServeConfig:
    return ServeConfig(model_url="http://mock", timeouts={"backoff_s": 0}, **kw)


GATE = {"unusable_labels": ["unrecognizable"]}  # the optional hard retake gate (D-064)


def mock_client(**kw) -> ModelClient:
    transport = httpx.ASGITransport(app=create_mock_app(**kw))
    return ModelClient("http://mock", backoff_s=0, transport=transport)


# ---------------------------------------------------------------- guidance and config


def test_guidance_covers_every_quality_issue():
    labels = set(load_data_config().quality_labels) - {"none"}
    assert set(guidance.TIPS) == labels


def test_retake_and_decline_messages():
    msg, tips = guidance.retake_message(["unrecognizable", "blur", "framing", "dark"], 2)
    assert msg.startswith(guidance.UNUSABLE)
    assert tips == [guidance.TIPS["dark"], guidance.TIPS["blur"]]  # priority order, capped
    assert guidance.retake_message(["unrecognizable"], 2)[1] == [guidance.GENERIC_RETAKE]
    assert guidance.decline_message([], 2) == (
        f"{guidance.CANNOT_ANSWER} {guidance.NOT_VISIBLE}", [guidance.NOT_VISIBLE])  # fmt: skip


def test_serve_config_env_overrides_and_no_key_from_yaml(tmp_path, monkeypatch):
    path = tmp_path / "serve.yaml"
    path.write_text("model_url: http://a\nmodel_api_key: leaked\nmode: sequential\n")
    monkeypatch.setenv("VIZWIZ_MODEL_URL", "http://b")
    c = load_serve_config(path)
    assert c.model_url == "http://b" and c.model_api_key is None and c.mode == "sequential"
    monkeypatch.setenv("VIZWIZ_MODEL_API_KEY", "from-env")
    assert load_serve_config(path).model_api_key == "from-env"
    assert load_serve_config().adapters["answer"] == "e5-multi-lora"  # the committed file
    with pytest.raises(ValueError):
        ServeConfig(adapters={"answer": "x"})


def test_unanswerable_score_from_token_strings():
    top = {"un": math.log(0.6), "Un": math.log(0.2), "answer": math.log(0.2), "The": -1.0}
    score, from_lp = unanswerable_score(top, "unanswerable")
    assert from_lp and score == pytest.approx(0.8)
    assert unanswerable_score({"The": -0.1}, "Unanswerable.") == (1.0, False)
    assert unanswerable_score({}, "answerable") == (0.0, False)


# ---------------------------------------------------------------- cascade


@pytest.mark.parametrize("mode", ["sequential", "parallel_checks", "speculative"])
@pytest.mark.parametrize(
    "kind,status", [("normal", "answered"), ("flat", "retake"), ("dark", "unanswerable")]
)
def test_cascade_branches(mode, kind, status):
    async def go():
        client = mock_client()
        try:
            c = cfg(mode=mode, rules=GATE)
            return await Cascade(client, c).run(to_data_url(photo(kind)), "What?")
        finally:
            await client.aclose()

    r = asyncio.run(go())
    assert r.status == status
    used = [t.stage for t in r.trace if t.used]
    if status == "answered":
        assert r.answer == "mock answer" and r.message == "mock answer"
        assert used == ["quality", "answerability", "answer"] and r.unanswerable_score < 0.1
    elif status == "retake":
        assert r.answer is None and r.message.startswith(guidance.UNUSABLE)
        assert used == ["quality"] and r.issues == ["unrecognizable"]
        if mode == "sequential":
            assert [t.stage for t in r.trace] == ["quality"]  # nothing else ran
    else:
        assert r.answer is None and r.message.startswith(guidance.CANNOT_ANSWER)
        assert r.issues == ["dark"] and r.tips == [guidance.TIPS["dark"]]
        assert used == ["quality", "answerability"] and r.unanswerable_score >= 0.5


def test_default_is_tips_only():
    async def go(kind):
        client = mock_client()
        try:
            return await Cascade(client, cfg()).run(to_data_url(photo(kind)), "What?")
        finally:
            await client.aclose()

    flat = asyncio.run(go("flat"))  # unrecognizable: declined by answerability, not forced
    assert flat.status == "unanswerable"
    assert flat.tips == [guidance.TIPS["unrecognizable"], guidance.GENERIC_RETAKE]
    assert [t.stage for t in flat.trace if t.used] == ["quality", "answerability"]
    assert asyncio.run(go("normal")).status == "answered"
    assert (
        cfg().rules.answerability_threshold == 0.10
        and load_serve_config().rules.unusable_labels == []
    )


def test_threshold_moves_the_decision():
    async def go(threshold):
        client = mock_client()
        try:
            c = cfg(rules={"answerability_threshold": threshold})
            return (await Cascade(client, c).run(to_data_url(photo("dark")), "What?")).status
        finally:
            await client.aclose()

    assert asyncio.run(go(0.5)) == "unanswerable"
    assert asyncio.run(go(0.9)) == "answered"  # mock P(unanswerable) for dark = 0.8


# ---------------------------------------------------------------- model client


def test_client_retries_transient_errors_then_gives_up():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    async def go(handler, retries=2):
        c = ModelClient("http://x", retries=retries, backoff_s=0,
                        transport=httpx.MockTransport(handler))  # fmt: skip
        try:
            return await c.models()
        finally:
            await c.aclose()

    assert asyncio.run(go(flaky)) == ["m"] and calls["n"] == 2
    with pytest.raises(ModelError, match="after 3 attempts"):
        asyncio.run(go(lambda r: httpx.Response(502)))
    with pytest.raises(ModelError, match="returned 400"):  # not retried
        asyncio.run(go(lambda r: httpx.Response(400)))


# ---------------------------------------------------------------- gateway


def call(app, method, path, **kw):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            return await c.request(method, path, **kw)

    return asyncio.run(go())


def assist(app, image: bytes, question="What is this?"):
    files = {"image": ("p.jpg", image, "image/jpeg")}
    return call(app, "POST", "/v1/assist", files=files, data={"question": question})


def test_gateway_assist_branches_and_metrics():
    app = create_app(cfg(rules=GATE), mock_client())
    r = assist(app, jpeg("normal"))
    assert r.status_code == 200 and r.headers["X-Request-ID"] == r.json()["request_id"]
    body = r.json()
    assert body["status"] == "answered" and body["answer"] == "mock answer"
    assert [t["stage"] for t in body["trace"]] == ["quality", "answerability", "answer"]
    assert assist(app, jpeg("flat")).json()["status"] == "retake"
    assert assist(app, jpeg("dark")).json()["status"] == "unanswerable"
    text = call(app, "GET", "/metrics").text
    for status in ("answered", "retake", "unanswerable"):
        assert f'vizwiz_requests_total{{status="{status}"}} 1.0' in text
    assert 'vizwiz_stage_seconds_count{stage="answer"} 1.0' in text


def test_gateway_rejects_bad_input():
    app = create_app(cfg(limits={"max_upload_mb": 0.001}), mock_client())
    assert assist(app, b"not a photo").status_code == 415
    assert assist(app, jpeg("normal"), question="   ").status_code == 422
    assert assist(app, jpeg("normal"), question="x" * 501).status_code == 422
    big = create_app(cfg(limits={"max_upload_mb": 0.0001}), mock_client())
    assert assist(big, jpeg("normal")).status_code == 413


def test_gateway_model_down_and_health():
    def down(request):
        raise httpx.ConnectError("refused")

    dead = ModelClient("http://x", retries=1, backoff_s=0, transport=httpx.MockTransport(down))
    app = create_app(cfg(), dead)
    r = assist(app, jpeg("normal"))
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    assert call(app, "GET", "/health").status_code == 503

    ok = create_app(cfg(), mock_client())
    h = call(ok, "GET", "/health")
    assert h.status_code == 200 and h.json()["missing_adapters"] == []
    assert "e5-multi-lora" in call(ok, "GET", "/adapters").json()["loaded"]
    missing = create_app(cfg(adapters={"quality": "q", "answerability": "a", "answer": "x"}),
                         mock_client())  # fmt: skip
    assert call(missing, "GET", "/health").json()["missing_adapters"] == ["a", "q", "x"]


def test_gateway_api_key_guards_assist_only():
    app = create_app(cfg(), mock_client(), api_key="k3y")
    assert assist(app, jpeg("normal")).status_code == 401
    files = {"image": ("p.jpg", jpeg("normal"), "image/jpeg")}
    wrong = call(app, "POST", "/v1/assist", files=files, data={"question": "What?"},
                 headers={"X-API-Key": "nope"})  # fmt: skip
    assert wrong.status_code == 401
    ok = call(app, "POST", "/v1/assist", files=files, data={"question": "What?"},
              headers={"X-API-Key": "k3y"})  # fmt: skip
    assert ok.status_code == 200 and ok.json()["status"] == "answered"
    assert call(app, "GET", "/health").status_code == 200  # health stays open for probes


def test_demo_collect_records_every_sample(tmp_path):
    from vizwiz.serve.demo import collect

    (tmp_path / "a.jpg").write_bytes(jpeg("normal"))
    (tmp_path / "b.jpg").write_bytes(jpeg("dark"))
    samples = [{"id": "a", "file": "a.jpg", "question": "What is this?"},
               {"id": "b", "file": "b.jpg", "question": "What does it say?"}]  # fmt: skip
    app = create_app(cfg(rules=GATE), mock_client())

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            return await collect(c, samples, tmp_path)

    report = asyncio.run(go())
    assert [a["id"] for a in report["answers"]] == ["a", "b"]
    assert [a["status"] for a in report["answers"]] == ["answered", "unanswerable"]
    assert report["mode"] and report["adapters"]
