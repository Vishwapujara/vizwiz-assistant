"""Adapter registry (Step 5.3, D-026, D-061).

Each trained adapter is a versioned artifact (``<name>:v0``, ``v1``, ...) with
metadata: task, base model, LoRA rank, training run, headline metrics. The
``production`` alias marks the live version of each collection, ``previous`` the
one before it. Promotion moves ``production`` (and ``previous`` to the old live
version); rollback swaps them back.

Code depends on the ``Registry`` protocol. ``InMemoryRegistry`` serves tests and
CI (no network, no key); ``WandbRegistry`` uses W&B artifacts and reads
``WANDB_API_KEY`` from the environment.

    python -m vizwiz.serve.registry register --name e5-multi-lora --path <adapter dir> \\
        --task multi --run-id e5-multi-lora --metrics results/eval/e5-multi-lora/metrics.json \\
        [--alias production]
    python -m vizwiz.serve.registry list --name e5-multi-lora
    python -m vizwiz.serve.registry promote --name e5-multi-lora --version v1
    python -m vizwiz.serve.registry rollback --name e5-multi-lora
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PRODUCTION, PREVIOUS = "production", "previous"
HEADLINE = {"answer": "vqa_accuracy", "answerability": "average_precision", "quality": "macro_f1"}


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterVersion:
    name: str  # collection, e.g. "e5-multi-lora"
    version: str  # "v0", "v1", ...
    aliases: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def served_name(self) -> str:
        """Name on the model server: one per version, so two versions can be loaded."""
        return f"{self.name}-{self.version}"


class Registry(Protocol):
    def versions(self, name: str) -> list[AdapterVersion]: ...
    def get(self, name: str, ref: str) -> AdapterVersion: ...  # ref: alias or "vN"
    def add_alias(self, name: str, version: str, alias: str) -> None: ...  # moves it
    def remove_alias(self, name: str, version: str, alias: str) -> None: ...
    def download(self, v: AdapterVersion, dest: Path) -> Path: ...
    def register(
        self, name: str, path: Path, metadata: dict, aliases: tuple[str, ...] = ()
    ) -> AdapterVersion: ...


# ---------------------------------------------------------------- rules


def check_compatible(v: AdapterVersion, base_model: str, max_rank: int) -> None:
    """Refuse an adapter the model server cannot serve correctly."""
    md = v.metadata
    if md.get("base_model") != base_model:
        raise RegistryError(f"{v.name}:{v.version} was trained on {md.get('base_model')!r}, "
                            f"the server runs {base_model!r}")  # fmt: skip
    if not isinstance(md.get("rank"), int) or md["rank"] > max_rank:
        raise RegistryError(f"{v.name}:{v.version} rank {md.get('rank')} > server max {max_rank}")


def promote(reg: Registry, name: str, version: str, base_model: str, max_rank: int):
    """Make ``version`` live. Returns (new production, old production or None)."""
    new = reg.get(name, version)
    check_compatible(new, base_model, max_rank)
    try:
        old = reg.get(name, PRODUCTION)
    except RegistryError:
        old = None
    if old is not None and old.version == new.version:
        return new, None
    if old is not None:
        reg.add_alias(name, old.version, PREVIOUS)
    reg.add_alias(name, new.version, PRODUCTION)
    return reg.get(name, new.version), old


def rollback(reg: Registry, name: str):
    """Swap ``production`` and ``previous``. Returns (new production, old production)."""
    try:
        prev = reg.get(name, PREVIOUS)
    except RegistryError as e:
        raise RegistryError(f"{name} has no previous version to roll back to") from e
    cur = reg.get(name, PRODUCTION)
    reg.add_alias(name, prev.version, PRODUCTION)
    reg.add_alias(name, cur.version, PREVIOUS)
    return reg.get(name, prev.version), cur


def metadata_for(path: Path, task: str, run_id: str, metrics_file: Path | None) -> dict:
    """Registry metadata from the adapter's own config and the committed metrics."""
    cfg = json.loads((Path(path) / "adapter_config.json").read_text())
    md = {"task": task, "base_model": cfg.get("base_model_name_or_path"), "rank": cfg.get("r"),
          "alpha": cfg.get("lora_alpha"), "run_id": run_id}  # fmt: skip
    if metrics_file:
        m = json.loads(Path(metrics_file).read_text())
        md["git_commit"] = m.get("git_commit")
        md["metrics_test"] = {t: v["metrics"][HEADLINE[t]] for t, v in m.get("tasks", {}).items()}
    return md


# ---------------------------------------------------------------- in-memory (tests, CI)


class InMemoryRegistry:
    def __init__(self):
        self._versions: dict[str, list[dict]] = {}  # name -> [{"metadata", "aliases", "files"}]

    def _entry(self, name: str, ref: str) -> tuple[int, dict]:
        items = self._versions.get(name, [])
        for i, e in enumerate(items):
            if ref == f"v{i}" or ref in e["aliases"]:
                return i, e
        raise RegistryError(f"{name}:{ref} not found")

    def _view(self, name: str, i: int, e: dict) -> AdapterVersion:
        return AdapterVersion(name, f"v{i}", tuple(sorted(e["aliases"])), dict(e["metadata"]))

    def versions(self, name):
        return [self._view(name, i, e) for i, e in enumerate(self._versions.get(name, []))]

    def get(self, name, ref):
        return self._view(name, *self._entry(name, ref))

    def add_alias(self, name, version, alias):
        _, target = self._entry(name, version)
        for e in self._versions[name]:
            e["aliases"].discard(alias)  # an alias points at one version only
        target["aliases"].add(alias)

    def remove_alias(self, name, version, alias):
        self._entry(name, version)[1]["aliases"].discard(alias)

    def download(self, v, dest):
        dest = Path(dest)
        for rel, data in self._entry(v.name, v.version)[1]["files"].items():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes(data)
        return dest

    def register(self, name, path, metadata, aliases=()):
        path = Path(path)
        files = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}
        self._versions.setdefault(name, []).append(
            {"metadata": dict(metadata), "aliases": set(), "files": files}
        )
        version = f"v{len(self._versions[name]) - 1}"
        for a in aliases:
            self.add_alias(name, version, a)
        return self.get(name, version)


# ---------------------------------------------------------------- W&B


class WandbRegistry:
    """W&B artifacts of type ``lora-adapter`` in one project. W&B keeps each alias on
    one version of a collection, so adding an alias moves it."""

    TYPE = "lora-adapter"

    def __init__(self, project: str, entity: str | None = None):
        import wandb

        self._wandb = wandb
        self.api = wandb.Api()
        self.project = project
        self.entity = entity or self.api.default_entity
        self.path = f"{self.entity}/{project}"

    def _artifact(self, name, ref):
        try:
            return self.api.artifact(f"{self.path}/{name}:{ref}", type=self.TYPE)
        except Exception as e:  # wandb raises CommError / ValueError for unknown refs
            raise RegistryError(f"{name}:{ref} not found in {self.path}") from e

    def _view(self, a) -> AdapterVersion:
        aliases = tuple(sorted(x for x in a.aliases if x != "latest"))
        return AdapterVersion(a.name.split(":")[0], a.version, aliases, dict(a.metadata))

    def versions(self, name):
        arts = self.api.artifacts(self.TYPE, f"{self.path}/{name}")
        return sorted((self._view(a) for a in arts), key=lambda v: int(v.version[1:]))

    def get(self, name, ref):
        return self._view(self._artifact(name, ref))

    def add_alias(self, name, version, alias):
        a = self._artifact(name, version)
        if alias not in a.aliases:
            a.aliases.append(alias)
            a.save()

    def remove_alias(self, name, version, alias):
        a = self._artifact(name, version)
        if alias in a.aliases:
            a.aliases.remove(alias)
            a.save()

    def download(self, v, dest):
        return Path(self._artifact(v.name, v.version).download(root=str(dest)))

    def register(self, name, path, metadata, aliases=()):
        run = self._wandb.init(project=self.project, entity=self.entity,
                               job_type="register-adapter", name=f"register-{name}")  # fmt: skip
        art = self._wandb.Artifact(name, type=self.TYPE, metadata=metadata)
        art.add_dir(str(path))
        run.log_artifact(art, aliases=list(aliases))
        art.wait()
        run.finish()
        return self.get(name, art.version)


def download_cached(reg: Registry, v: AdapterVersion, cache_dir: Path) -> Path:
    """Download once per version; a half-finished download is never reused."""
    dest = Path(cache_dir) / v.name / v.version
    if (dest / "adapter_config.json").exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    reg.download(v, tmp)
    if not (tmp / "adapter_config.json").exists():
        raise RegistryError(f"{v.name}:{v.version} has no adapter_config.json")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from vizwiz.serve.config import load_serve_config

    p = argparse.ArgumentParser(description="Adapter registry (W&B)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--path", type=Path, required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--metrics", type=Path)
    r.add_argument("--alias", action="append", default=[])
    for cmd in ("list", "promote", "rollback"):
        s = sub.add_parser(cmd)
        s.add_argument("--name", required=True)
        if cmd == "promote":
            s.add_argument("--version", required=True)
    args = p.parse_args(argv)
    cfg = load_serve_config().registry
    reg = WandbRegistry(cfg.project, cfg.entity)
    if args.cmd == "register":
        md = metadata_for(args.path, args.task, args.run_id, args.metrics)
        v = reg.register(args.name, args.path, md)
        if PRODUCTION in args.alias:  # through promote, so the compatibility check runs
            v, _ = promote(reg, args.name, v.version, cfg.base_model, cfg.max_lora_rank)
        for a in set(args.alias) - {PRODUCTION}:
            reg.add_alias(args.name, v.version, a)
        print("REGISTERED:", v.name, v.version, reg.get(args.name, v.version).aliases, md)
    elif args.cmd == "list":
        for v in reg.versions(args.name):
            print(v.version, list(v.aliases), v.metadata)
    elif args.cmd == "promote":
        new, old = promote(reg, args.name, args.version, cfg.base_model, cfg.max_lora_rank)
        print("PROMOTED:", new.version, "previous:", old.version if old else None)
    else:
        new, old = rollback(reg, args.name)
        print("ROLLED BACK to", new.version, "from", old.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
