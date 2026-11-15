"""Access to ``registry.yaml`` and the on-disk dataset layout.

Layout under the data root (default ``./datasets``, or ``$UWBENCH_DATA``)::

    <root>/<key>/archives/   downloaded archives, unchanged
    <root>/<key>/extracted/  one sub-folder per archive (or manual deposit)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")
MANIFEST_DIR = Path(__file__).with_name("manifests")


def default_root() -> Path:
    return Path(os.environ.get("UWBENCH_DATA", "datasets"))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_entry(key: str, registry: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    registry = load_registry() if registry is None else registry
    if key not in registry:
        raise KeyError(f"unknown dataset {key!r}; available: {', '.join(sorted(registry))}")
    return registry[key]
