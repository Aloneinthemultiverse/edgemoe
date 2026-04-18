"""Local-SSD backend: mmap + pread for zero-copy random reads."""

from __future__ import annotations

import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from edgemoe.storage.base import StorageBackend


class LocalSSDBackend(StorageBackend):
    """Reads per-expert .bin files from disk, mmap'd for zero-copy access.

    The OS page cache acts as the natural LRU — frequently used experts stay
    hot in RAM automatically. For manifest lookup we keep a single
    per-layer shard when available (faster than millions of tiny files).
    """

    def __init__(self, model_path: str | Path, prefetch_workers: int = 4):
        super().__init__(model_path)
        self.experts_dir = self.model_path / "experts"
        self._mmaps: dict[str, mmap.mmap] = {}
        self._mmap_lock = threading.Lock()
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="ssd-prefetch"
        )

    def _expert_file(self, layer_id: int, expert_id: int) -> Path:
        return self.experts_dir / f"L{layer_id:02d}_E{expert_id:03d}.bin"

    def _open_mmap(self, path: Path) -> mmap.mmap:
        key = str(path)
        with self._mmap_lock:
            mm = self._mmaps.get(key)
            if mm is not None:
                return mm
            fd = os.open(key, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            self._mmaps[key] = mm
            return mm

    def load_expert(self, layer_id: int, expert_id: int) -> bytes:
        path = self._expert_file(layer_id, expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert not on disk: {path}")
        mm = self._open_mmap(path)
        return bytes(mm)

    def prefetch_layer(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        for eid in expert_ids:
            self._prefetch_pool.submit(self._warm_page_cache, layer_id, eid)

    def _warm_page_cache(self, layer_id: int, expert_id: int) -> None:
        path = self._expert_file(layer_id, expert_id)
        if not path.exists():
            return
        try:
            mm = self._open_mmap(path)
            if hasattr(mm, "madvise"):
                mm.madvise(mmap.MADV_WILLNEED)
            else:
                _ = mm[0:1]
        except Exception:
            pass

    def load_backbone(self) -> bytes:
        path = self.model_path / "backbone.bin"
        return path.read_bytes()

    def _read_manifest(self) -> dict:
        path = self.experts_dir / "manifest.json"
        return self._parse_manifest_bytes(path.read_bytes())

    def __del__(self):
        for mm in self._mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        self._prefetch_pool.shutdown(wait=False)
