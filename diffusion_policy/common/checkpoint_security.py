from __future__ import annotations

import os
from pathlib import Path
from typing import Union


PathLike = Union[str, os.PathLike]


def require_trusted_pickle_artifact(
    path: PathLike,
    *,
    trusted: bool,
    artifact_name: str = "checkpoint",
) -> Path:
    """Gate executable pickle/dill deserialization behind explicit caller trust."""
    artifact_path = Path(path).expanduser()
    if not trusted:
        raise PermissionError(
            f"Refusing to load untrusted {artifact_name} {str(artifact_path)!r}. "
            "PyTorch dill/pickle artifacts can execute arbitrary code while loading. "
            "Only load an artifact from a trusted source and explicitly set "
            "trust_checkpoint=True (or pass --trust-checkpoint)."
        )
    if not artifact_path.exists():
        raise FileNotFoundError(f"{artifact_name.capitalize()} does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise ValueError(f"{artifact_name.capitalize()} must be a regular file: {artifact_path}")
    return artifact_path
