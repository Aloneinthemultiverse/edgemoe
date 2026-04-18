"""Upload a prepared EdgeMoE model directory to Google Drive.

Creates a nested folder hierarchy that mirrors the local layout. Skips
files that already exist with the same size (cheap resume). OAuth flow
is the same as GoogleDriveBackend.
"""

from __future__ import annotations

from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _service(credentials_path: str, token_path: str):
    creds = None
    tok = Path(token_path)
    if tok.exists():
        creds = Credentials.from_authorized_user_file(str(tok), _SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, _SCOPES)
            creds = flow.run_local_server(port=0)
        tok.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _ensure_folder(service, name: str, parent_id: str) -> str:
    q = (
        f"name = '{name}' and '{parent_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    resp = service.files().list(q=q, spaces="drive", fields="files(id)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(body=meta, fields="id").execute()["id"]


def _find_file(service, name: str, parent_id: str) -> dict | None:
    q = f"name = '{name}' and '{parent_id}' in parents and trashed = false"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, size)").execute()
    files = resp.get("files", [])
    return files[0] if files else None


def upload_dir(
    local_dir: str,
    remote_parent: str = "root",
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
    mirror_root_name: str | None = None,
) -> None:
    svc = _service(credentials_path, token_path)
    root = Path(local_dir)
    if mirror_root_name is None:
        mirror_root_name = root.name
    top_id = _ensure_folder(svc, mirror_root_name, remote_parent)

    total_files = sum(1 for _ in root.rglob("*") if _.is_file())
    done = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parent_id = top_id
        for part in rel.parts[:-1]:
            parent_id = _ensure_folder(svc, part, parent_id)
        existing = _find_file(svc, rel.name, parent_id)
        if existing and int(existing.get("size", -1)) == path.stat().st_size:
            done += 1
            print(f"[up] skip {rel} (exists, size match)")
            continue
        media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024 * 1024)
        body = {"name": rel.name, "parents": [parent_id]}
        if existing:
            svc.files().update(fileId=existing["id"], media_body=media).execute()
        else:
            svc.files().create(body=body, media_body=media, fields="id").execute()
        done += 1
        print(f"[up] {done}/{total_files}  {rel}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("local_dir")
    p.add_argument("--remote-parent", default="root")
    p.add_argument("--credentials", default="credentials.json")
    p.add_argument("--token", default="token.json")
    args = p.parse_args()
    upload_dir(
        args.local_dir,
        remote_parent=args.remote_parent,
        credentials_path=args.credentials,
        token_path=args.token,
    )
