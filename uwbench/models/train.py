"""``uwbench train``: train a learned model on one scenario (one command per pair).

Protocol (work plan, phase 8; parameters in ``configs/models/*.yaml``):

* data: the ``train`` splits of the listed datasets (default LSUI, EUVP, UIEB), which by
  construction share no duplicate with any test split; model selection on their ``val``
  splits;
* random ``crop x crop`` crops (reflection padding of smaller images) with horizontal
  flips; validation on fixed centre crops;
* MSE loss, Adam (optional ``weight_decay``; optional ``scheduler: {type: plateau,
  factor, patience, min_lr}`` on the validation PSNR); the SNR of every training image is
  drawn uniformly in the scenario's
  range (``snr_db.start`` to ``snr_db.stop``, or ``train.snr_db``), and the image goes
  through the scenario's channel exactly as in evaluation
  (:func:`uwbench.models.base.transmit`);
* validation PSNR (mean over a fixed subset of ``val_images`` validation crops, 512 by
  default, and the scenario's SNR grid, fixed seed); the best epoch is kept;
* resumable: re-running the same command continues from ``last.pt`` (weights,
  optimiser, scheduler and random generators); only ``epochs`` may change, so a
  finished training can also be extended.

Outputs in ``<out>/<model>_<scenario>/``: ``best.pt`` and ``last.pt`` (weights, both
configurations, epoch, validation PSNR, provenance) and ``log.csv``.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from uwbench.channels import build_channel
from uwbench.metrics.fidelity import psnr
from uwbench.models import load_model
from uwbench.models.base import transmit
from uwbench.utils.seeding import seed_everything


class Crops(Dataset):
    """Crops of images given as paths or ``(H, W, 3)`` uint8 arrays."""

    def __init__(self, images: Sequence[Path | np.ndarray], crop: int, train: bool, seed: int):
        self.images, self.crop, self.train = list(images), crop, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.images[i]
        if not isinstance(img, np.ndarray):
            img = np.asarray(Image.open(img).convert("RGB"))
        x = torch.tensor(img).permute(2, 0, 1).float() / 255
        c = self.crop
        ph, pw = max(0, c - x.shape[1]), max(0, c - x.shape[2])
        if ph or pw:
            x = F.pad(x[None], (0, pw, 0, ph), mode="reflect")[0]
        h, w = x.shape[1:]
        if self.train:
            top, left = int(self.rng.integers(h - c + 1)), int(self.rng.integers(w - c + 1))
            x = x[:, top:top + c, left:left + c]
            return x.flip(-1) if self.rng.random() < 0.5 else x
        top, left = (h - c) // 2, (w - c) // 2
        return x[:, top:top + c, left:left + c]


def _seed_worker(_: int) -> None:
    """Own crop randomness per data-loading process: ``torch.initial_seed()`` differs per
    worker and per epoch and derives from the loader's seeded generator."""
    torch.utils.data.get_worker_info().dataset.rng = np.random.default_rng(torch.initial_seed())


def _resumable(cfg: dict[str, Any]) -> dict[str, Any]:
    """The configuration a resumed run must match (the epoch count may grow)."""
    return {**cfg, "train": {k: v for k, v in cfg["train"].items() if k != "epochs"}}


def symbols_for(scenario: dict[str, Any]) -> str:
    return "real" if scenario["channel"] == "uwoc" else "complex"


def train(model_cfg: dict[str, Any], scenario: dict[str, Any],
          train_images: Sequence[Path | np.ndarray], val_images: Sequence[Path | np.ndarray],
          out_dir: Path, device: str = "cpu", epochs: int | None = None) -> Path:
    from uwbench.eval.runner import provenance

    tc = model_cfg["train"]
    seed = tc.get("seed", 0)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    channel = build_channel(scenario["channel"], scenario.get("params"))
    model = load_model(model_cfg, symbols_for(scenario), device=device).train()
    lo, hi = tc.get("snr_db", [scenario["snr_db"]["start"], scenario["snr_db"]["stop"]])
    grid = np.arange(scenario["snr_db"]["start"], scenario["snr_db"]["stop"] + 1e-9,
                     scenario["snr_db"]["step"])
    gen = torch.Generator().manual_seed(seed)
    crops = Crops(train_images, tc["crop"], True, seed)
    loader = DataLoader(crops, batch_size=tc["batch_size"], shuffle=True,
                        num_workers=tc.get("workers", 0), drop_last=True, generator=gen,
                        worker_init_fn=_seed_worker)
    n_val = tc.get("val_images", 512)  # fixed subset spread over the datasets: bounded cost
    val_images = list(val_images)[::max(1, len(val_images) // n_val)][:n_val]
    val = torch.stack([v for v in Crops(val_images, tc["crop"], False, seed)])
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0))
    sched = None
    if (sc := tc.get("scheduler")) and sc["type"] == "plateau":  # on validation PSNR
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=sc["factor"], patience=sc["patience"], min_lr=sc["min_lr"])
    m_index = model_cfg.get("modulation_index", 1.0)
    run = out_dir / f"{model_cfg['name']}_{scenario['name']}"
    run.mkdir(parents=True, exist_ok=True)
    meta = {"model_cfg": model_cfg, "scenario": scenario, "provenance": provenance()}
    start, best = 1, -np.inf
    if (run / "last.pt").exists():  # resume an interrupted or extended training
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if _resumable(state["model_cfg"]) != _resumable(model_cfg) \
                or state["scenario"] != scenario:
            raise ValueError(f"{run} holds a training with another configuration; "
                             "delete it to start over")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if sched is not None:
            sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state, crops.rng.bit_generator.state = state["rng"], state["crop_rng"]
        gen.set_state(state["loader_rng"])
        start, best = state["epoch"] + 1, state["best_val_psnr_db"]
        print(f"resuming {run.name} after epoch {state['epoch']}", flush=True)
    with open(run / "log.csv", "a" if start > 1 else "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f, lineterminator="\n")
        if start == 1:
            log.writerow(["epoch", "train_mse", "val_psnr_db", "seconds"])
        for epoch in range(start, (epochs or tc["epochs"]) + 1):
            t0, losses = time.time(), []
            model.train()
            for x in loader:
                x = x.to(device)
                snr = torch.as_tensor(rng.uniform(lo, hi, x.shape[0]), dtype=torch.float32,
                                      device=device)
                loss = F.mse_loss(transmit(model, x, channel, snr, rng, m_index), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            model.eval()
            vrng, scores = np.random.default_rng(seed + 1), []
            with torch.no_grad():
                for s in grid:
                    for b in range(0, len(val), tc["batch_size"]):
                        x = val[b:b + tc["batch_size"]].to(device)
                        snr = torch.full((x.shape[0],), float(s), device=device)
                        scores += psnr(transmit(model, x, channel, snr, vrng, m_index).cpu(),
                                       x.cpu()).tolist()
            val_psnr = float(np.mean(scores))
            if sched is not None:
                sched.step(val_psnr)
            best = max(best, val_psnr)
            state = {"model": model.state_dict(), "epoch": epoch, "val_psnr_db": val_psnr,
                     "best_val_psnr_db": best, "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict() if sched is not None else None,
                     "rng": rng.bit_generator.state, "crop_rng": crops.rng.bit_generator.state,
                     "loader_rng": gen.get_state(), **meta}
            if val_psnr == best:  # before last.pt: an interruption never loses a new best
                torch.save(state, run / "best.pt")
            torch.save(state, run / "last.pt")
            log.writerow([epoch, float(np.mean(losses)), val_psnr, round(time.time() - t0, 1)])
            f.flush()
            print(f"epoch {epoch}: train MSE {np.mean(losses):.5f}, val PSNR {val_psnr:.2f} dB",
                  flush=True)
    return run / "best.pt"
