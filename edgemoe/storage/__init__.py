from edgemoe.storage.base import StorageBackend
from edgemoe.storage.local_ssd import LocalSSDBackend
from edgemoe.storage.gdrive import GoogleDriveBackend
from edgemoe.storage.huggingface import HuggingFaceBackend


def get_backend(name: str, **kwargs) -> StorageBackend:
    """Factory: returns the right backend for a name."""
    name = name.lower()
    if name in ("local", "local_ssd", "ssd"):
        return LocalSSDBackend(**kwargs)
    if name in ("gdrive", "google_drive", "googledrive"):
        return GoogleDriveBackend(**kwargs)
    if name in ("hf", "huggingface", "hub"):
        return HuggingFaceBackend(**kwargs)
    raise ValueError(f"Unknown storage backend: {name}")


__all__ = [
    "StorageBackend",
    "LocalSSDBackend",
    "GoogleDriveBackend",
    "HuggingFaceBackend",
    "get_backend",
]
