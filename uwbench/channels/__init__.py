"""Reference channels: AWGN, modelled acoustic (OFDM), Watermark replay (OFDM), optical
(IM/DD).

Scenarios are YAML files (``configs/scenarios/``) with ``channel``, ``params`` and an
``snr_db`` grid; :func:`load_scenario` builds the channel and the grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uwbench.channels.awgn import AWGN
from uwbench.channels.base import Channel
from uwbench.channels.uac import UACOFDM
from uwbench.channels.uwoc import UWOC

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def build_channel(kind: str, params: dict[str, Any] | None = None) -> Channel:
    params = dict(params or {})
    if kind == "awgn":
        return AWGN()
    if kind == "uac_ofdm":
        return UACOFDM(**params)
    if kind == "watermark_ofdm":
        from uwbench.channels.watermark import WatermarkOFDM

        return WatermarkOFDM(**params)
    if kind == "uwoc":
        water = yaml.safe_load((CONFIG_DIR / "uwoc_water.yaml").read_text(encoding="utf-8"))
        c = water["water_types"][params.pop("water_type")]["c"]
        return UWOC(attenuation_c=c, **params)
    raise ValueError(f"unknown channel {kind!r}")


def snr_grid(spec: dict[str, float]) -> np.ndarray:
    return np.arange(spec["start"], spec["stop"] + spec["step"] / 2, spec["step"])


def load_scenario(path: Path | str) -> tuple[Channel, np.ndarray, dict[str, Any]]:
    """Return ``(channel, snr_grid_db, config)`` for a scenario file."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return build_channel(cfg["channel"], cfg.get("params")), snr_grid(cfg["snr_db"]), cfg
