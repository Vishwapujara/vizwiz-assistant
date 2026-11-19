"""Live adapter management for the gateway (Step 5.3, D-061).

At startup each registry collection named in ``adapters`` is resolved to its
``production`` version, downloaded once, loaded into the model server under a
per-version name (``e5-multi-lora-v1``) and switched in for its cascade stages.

Promotion order: download -> load on the server -> move the alias -> switch. A
failure in the first two steps changes nothing that serves traffic. The previous
version stays loaded, so a rollback is only an alias swap and a switch; older
versions are unloaded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from vizwiz.serve import registry as reg_mod
from vizwiz.serve.cascade import Cascade
from vizwiz.serve.config import ServeConfig
from vizwiz.serve.model_client import ModelClient
from vizwiz.serve.registry import AdapterVersion, Registry


class AdapterManager:
    def __init__(self, cascade: Cascade, client: ModelClient, registry: Registry, cfg: ServeConfig):
        self.cascade, self.client, self.registry = cascade, client, registry
        self.rcfg = cfg.registry
        self.collection_of = dict(cfg.adapters)  # stage -> registry collection
        self.loaded: dict[str, AdapterVersion] = {}  # served name -> version
        self.live: dict[str, AdapterVersion] = {}  # collection -> live version
        self._lock = asyncio.Lock()

    async def _call(self, fn, *args):
        return await asyncio.to_thread(fn, *args)  # W&B calls block

    async def _ensure_loaded(self, v: AdapterVersion) -> None:
        reg_mod.check_compatible(v, self.rcfg.base_model, self.rcfg.max_lora_rank)
        if v.served_name in self.loaded:
            return
        path = await self._call(
            reg_mod.download_cached, self.registry, v, Path(self.rcfg.cache_dir)
        )
        await self.client.load_adapter(v.served_name, str(path))
        self.loaded[v.served_name] = v

    def _switch(self, v: AdapterVersion) -> None:
        for stage, collection in self.collection_of.items():
            if collection == v.name:
                self.cascade.adapters[stage] = v.served_name
        self.live[v.name] = v

    async def _prune(self, name: str, keep: set[str]) -> None:
        for served, v in list(self.loaded.items()):
            if v.name == name and served not in keep:
                await self.client.unload_adapter(served)
                del self.loaded[served]

    async def sync(self) -> None:
        """Load and switch to the production version of every collection."""
        async with self._lock:
            for name in sorted(set(self.collection_of.values())):
                v = await self._call(self.registry.get, name, self.rcfg.alias)
                await self._ensure_loaded(v)
                self._switch(v)

    async def load(self, name: str, version: str) -> AdapterVersion:
        """Load a version on the server without serving it (warm-up before promotion)."""
        async with self._lock:
            v = await self._call(self.registry.get, name, version)
            await self._ensure_loaded(v)
            return v

    async def promote(self, name: str, version: str) -> dict:
        self._check_known(name)
        async with self._lock:
            v = await self._call(self.registry.get, name, version)
            await self._ensure_loaded(v)
            new, old = await self._call(reg_mod.promote, self.registry, name, version,
                                        self.rcfg.base_model, self.rcfg.max_lora_rank)  # fmt: skip
            before = self.live.get(name)
            self._switch(new)
            keep = {new.served_name} | ({before.served_name} if before else set())
            await self._prune(name, keep)
            return {"live": new.version, "previous": before.version if before else None}

    async def rollback(self, name: str) -> dict:
        self._check_known(name)
        async with self._lock:
            prev = await self._call(self.registry.get, name, reg_mod.PREVIOUS)
            await self._ensure_loaded(prev)
            new, old = await self._call(reg_mod.rollback, self.registry, name)
            self._switch(new)
            await self._prune(name, {new.served_name, old.served_name})
            return {"live": new.version, "previous": old.version}

    def _check_known(self, name: str) -> None:
        if name not in self.collection_of.values():
            raise reg_mod.RegistryError(f"{name} is not used by any cascade stage")

    def status(self) -> dict:
        return {
            "live": {
                n: {"version": v.version, "metadata": v.metadata} for n, v in self.live.items()
            },
            "loaded": sorted(self.loaded),
        }
