"""HuggingFace Hub backend: zero-setup streaming via hf-xet.

The Hub's HfFileSystem gives us POSIX-style reads with HTTP range requests.
Parallel streams hit 100-500 MB/s on Kaggle; hf-xet deduplicates chunks
across revisions so subsequent model downloads reuse shared bytes.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterable

from edgemoe.storage.base import StorageBackend

try:
    from huggingface_hub import HfFileSystem, hf_hub_download
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False


class HuggingFaceBackend(StorageBackend):
    """Stream experts directly from huggingface.co without full download.

    model_path is the repo ID (e.g. "Qwen/Qwen3-235B-A22B-edgemoe").
    The repo must be laid out identically to the LocalSSDBackend layout.
    """

    def __init__(
        self,
        model_path: str,
        revision: str = "main",
        cache_dir: str | None = None,
        prefetch_workers: int = 16,
        token: str | None = None,
    ):
        if not _HAVE_HF:
            raise ImportError(
                "huggingface_hub missing. pip install huggingface_hub hf_xet"
            )
        super().__init__(model_path)
        self.repo_id = str(model_path)
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token
        self.fs = HfFileSystem(token=token)
        self._bytes_cache: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self._in_flight: dict[str, Future] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="hf"
        )

    def _hf_path(self, relative: str) -> str:
        return f"{self.repo_id}@{self.revision}/{relative}"

    def _fetch_sync(self, relative: str) -> bytes:
        try:
            with self.fs.open(self._hf_path(relative), "rb") as f:
                data = f.read()
            with self._cache_lock:
                self._bytes_cache[relative] = data
                self._in_flight.pop(relative, None)
            return data
        except Exception:
            with self._cache_lock:
                self._in_flight.pop(relative, None)
            raise

    def _fetch(self, relative: str) -> bytes:
        with self._cache_lock:
            cached = self._bytes_cache.get(relative)
            if cached is not None:
                return cached
            fut = self._in_flight.get(relative)
            if fut is None:
                fut = self._pool.submit(self._fetch_sync, relative)
                self._in_flight[relative] = fut
        return fut.result()

    def load_expert(self, layer_id: int, expert_id: int) -> bytes:
        return self._fetch(f"experts/L{layer_id:02d}_E{expert_id:03d}.bin")

    def prefetch_layer(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        for eid in expert_ids:
            rel = f"experts/L{layer_id:02d}_E{eid:03d}.bin"
            with self._cache_lock:
                if rel in self._bytes_cache or rel in self._in_flight:
                    continue
                self._in_flight[rel] = self._pool.submit(self._fetch_sync, rel)

    def load_backbone(self) -> bytes:
        """Prefer hf_hub_download for backbone — it's large, hf-xet dedupes."""
        local = hf_hub_download(
            repo_id=self.repo_id,
            filename="backbone.bin",
            revision=self.revision,
            cache_dir=self.cache_dir,
            token=self.token,
        )
        return Path(local).read_bytes()

    def _read_manifest(self) -> dict:
        return self._parse_manifest_bytes(self._fetch("experts/manifest.json"))
