"""Google Drive backend: stream experts from the user's own 2 TB Drive.

Authenticates via OAuth (credentials.json from Google Cloud Console).
Supports both name-based lookup (file path inside Drive) and direct
file-ID lookup (recommended, ~5x faster).

Requires: google-api-python-client, google-auth, google-auth-oauthlib.
"""

from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterable

from edgemoe.storage.base import StorageBackend

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    _HAVE_GDRIVE = True
except ImportError:
    _HAVE_GDRIVE = False


_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveBackend(StorageBackend):
    """Streaming backend backed by the user's Google Drive.

    The Drive folder layout mirrors LocalSSDBackend:
        <drive_root>/
            backbone.bin
            tokenizer.bin
            experts/manifest.json
            experts/L00_E000.bin
            ...
    """

    def __init__(
        self,
        model_path: str | Path,
        credentials_path: str | Path = "credentials.json",
        token_path: str | Path = "token.json",
        prefetch_workers: int = 8,
    ):
        if not _HAVE_GDRIVE:
            raise ImportError(
                "google-api-python-client missing. "
                "pip install google-api-python-client google-auth-oauthlib"
            )
        super().__init__(model_path)
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.service = self._authenticate()
        self._id_cache: dict[str, str] = {}
        self._id_cache_lock = threading.Lock()
        self._bytes_cache: dict[str, bytes] = {}
        self._bytes_cache_lock = threading.Lock()
        self._in_flight: dict[str, Future] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="gdrive"
        )
        self._root_id = self._resolve_path_id(str(self.model_path).replace("\\", "/"))

    def _authenticate(self):
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), _SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), _SCOPES
                )
                creds = flow.run_local_server(port=0)
            self.token_path.write_text(creds.to_json())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _resolve_path_id(self, path: str) -> str:
        """Walk the slash-separated path, returning the final folder/file ID."""
        parent = "root"
        for part in [p for p in path.split("/") if p]:
            q = (
                f"name = '{part}' and '{parent}' in parents "
                f"and trashed = false"
            )
            resp = (
                self.service.files()
                .list(q=q, spaces="drive", fields="files(id, mimeType)", pageSize=1)
                .execute()
            )
            files = resp.get("files", [])
            if not files:
                raise FileNotFoundError(f"Drive path missing: {path} (at '{part}')")
            parent = files[0]["id"]
        return parent

    def _lookup_file_id(self, relative_path: str) -> str:
        with self._id_cache_lock:
            cached = self._id_cache.get(relative_path)
        if cached:
            return cached
        parent = self._root_id
        for part in [p for p in relative_path.split("/") if p]:
            q = f"name = '{part}' and '{parent}' in parents and trashed = false"
            resp = (
                self.service.files()
                .list(q=q, spaces="drive", fields="files(id)", pageSize=1)
                .execute()
            )
            files = resp.get("files", [])
            if not files:
                raise FileNotFoundError(
                    f"Drive file missing: {relative_path} (at '{part}')"
                )
            parent = files[0]["id"]
        with self._id_cache_lock:
            self._id_cache[relative_path] = parent
        return parent

    def _download_bytes(self, file_id: str) -> bytes:
        req = self.service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=4 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    def _fetch_path(self, relative_path: str) -> bytes:
        with self._bytes_cache_lock:
            cached = self._bytes_cache.get(relative_path)
            if cached is not None:
                return cached
            fut = self._in_flight.get(relative_path)
        if fut is not None:
            return fut.result()
        with self._bytes_cache_lock:
            fut = self._in_flight.get(relative_path)
            if fut is None:
                fut = self._pool.submit(self._download_path_sync, relative_path)
                self._in_flight[relative_path] = fut
        return fut.result()

    def _download_path_sync(self, relative_path: str) -> bytes:
        try:
            file_id = self._lookup_file_id(relative_path)
            data = self._download_bytes(file_id)
            with self._bytes_cache_lock:
                self._bytes_cache[relative_path] = data
                self._in_flight.pop(relative_path, None)
            return data
        except Exception:
            with self._bytes_cache_lock:
                self._in_flight.pop(relative_path, None)
            raise

    def load_expert(self, layer_id: int, expert_id: int) -> bytes:
        return self._fetch_path(f"experts/L{layer_id:02d}_E{expert_id:03d}.bin")

    def prefetch_layer(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        for eid in expert_ids:
            rel = f"experts/L{layer_id:02d}_E{eid:03d}.bin"
            with self._bytes_cache_lock:
                if rel in self._bytes_cache or rel in self._in_flight:
                    continue
                self._in_flight[rel] = self._pool.submit(
                    self._download_path_sync, rel
                )

    def load_backbone(self) -> bytes:
        return self._fetch_path("backbone.bin")

    def _read_manifest(self) -> dict:
        return self._parse_manifest_bytes(self._fetch_path("experts/manifest.json"))
