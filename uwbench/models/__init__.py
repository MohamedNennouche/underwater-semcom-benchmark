"""Learned analog schemes: the model interface (:mod:`uwbench.models.base`) and the
reference models DeepJSCC, ADJSCC and WITT.

A model is described in YAML by ``model: "package.module:ClassName"`` (any importable
class implementing :class:`~uwbench.models.base.JSCCModel`, including your own) and
``params`` passed to its constructor; :func:`load_model` builds it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from uwbench.models.base import JSCCModel


def load_model(cfg: dict[str, Any], symbols: str, checkpoint: Path | str | None = None,
               device: str = "cpu") -> JSCCModel:
    """Instantiate ``cfg["model"]`` with ``cfg["params"]`` and ``symbols``; load weights."""
    module, _, name = cfg["model"].partition(":")
    cls = getattr(importlib.import_module(module), name)
    model = cls(**{**cfg.get("params", {}), "symbols": symbols})
    if not isinstance(model, JSCCModel):
        raise TypeError(f"{cfg['model']} does not subclass uwbench.models.base.JSCCModel")
    if checkpoint is not None:
        # weights_only: a checkpoint from elsewhere cannot execute code when loaded
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    return model.to(device).eval()
