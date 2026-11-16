"""WITT (Yang, Wang, Dai, Tan, Niu, Zhang, "WITT: A wireless image transmission
transformer for semantic communications", ICASSP 2023, arXiv:2211.00937; survey ref.
[137]).

The architecture follows the authors' official implementation,
https://github.com/KeYang8/WITT (``net/encoder.py``, ``net/decoder.py``,
``net/modules.py`` and the high-resolution configuration in ``train.py``), which was read
for its hyperparameters and structure. That repository has no license, so no code is
copied: this module is written independently (Swin blocks in
:mod:`uwbench.models.layers`, from Liu et al., 2021).

Architecture (official high-resolution configuration):

* encoder: 2x2 patch embedding (stride-2 convolution, no normalisation); four stages with
  ``[2, 2, 6, 2]`` Swin blocks, ``[128, 192, 256, 320]`` channels, ``[4, 6, 8, 10]``
  heads, window 8, MLP ratio 4, patch merging between stages (downsampling 16); final
  LayerNorm; Channel ModNet; linear head to ``C`` latent channels;
* Channel ModNet (official form; the paper's equations differ): hidden width
  ``M = 1.5 x dim``; ``t = FC_0(stop_grad(x))``; for ``i = 0..6``:
  ``t = FC_i(t) * SM_i(snr)`` (``FC_0`` already applied for ``i = 0``), each SM module
  ``Linear(1, M), ReLU, Linear(M, M), ReLU, Linear(M, M), Sigmoid``; output
  ``x * sigmoid(FC_7(t))``;
* decoder: linear head ``C -> 320``; Channel ModNet; four stages with ``[2, 6, 2, 2]``
  blocks, ``[320, 256, 192, 128]`` channels, ``[10, 8, 6, 4]`` heads, each followed by a
  patch reverse merging (LayerNorm, bias-free linear, pixel shuffle), the last one to 3
  channels; no output activation, the reconstruction is clamped to ``[0, 1]`` (as in the
  official loss and metrics);
* initialisation: truncated normal (std 0.02) for linear layers, zero biases, unit
  LayerNorm.

Differences that remain: the power normalisation is per image (benchmark convention)
instead of per batch; feature maps whose side is not a multiple of the window are padded
(the official code requires multiples); single-stage training.
"""

from __future__ import annotations

import torch
from torch import nn

from uwbench.models.base import JSCCModel, latent_channels
from uwbench.models.layers import PatchDivision, PatchMerging, SwinBlock


class AdaptiveModulator(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, m), nn.ReLU(), nn.Linear(m, m), nn.ReLU(),
                                nn.Linear(m, m), nn.Sigmoid())

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        return self.fc(snr_db.reshape(-1, 1))


class ChannelModNet(nn.Module):
    """SNR-adaptive gating of ``(B, H, W, dim)`` features (see the module docstring)."""

    def __init__(self, dim: int, layers: int = 7):
        super().__init__()
        m = int(dim * 1.5)
        widths = [dim] + [m] * layers + [dim]
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:],
                                                                  strict=True))
        self.sms = nn.ModuleList(AdaptiveModulator(m) for _ in range(layers))

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        t = x.detach()
        for fc, sm in zip(self.fcs[:-1], self.sms, strict=True):
            t = fc(t) * sm(snr_db.to(x))[:, None, None, :]
        return x * torch.sigmoid(self.fcs[-1](t))


def _stage(dim: int, depth: int, heads: int, window: int) -> nn.ModuleList:
    return nn.ModuleList(SwinBlock(dim, heads, window, shift=i % 2 == 1) for i in range(depth))


class WITT(JSCCModel):
    downsample = 16

    def __init__(self, bandwidth_ratio: float = 1 / 6, symbols: str = "complex",
                 dims: tuple[int, ...] = (128, 192, 256, 320),
                 depths: tuple[int, ...] = (2, 2, 6, 2),
                 heads: tuple[int, ...] = (4, 6, 8, 10), window: int = 8):
        super().__init__()
        if 2 ** len(dims) != self.downsample:
            raise ValueError("WITT here has four stages (downsampling 16)")
        self.symbols = symbols
        c = latent_channels(bandwidth_ratio, self.downsample, symbols)
        self.embed = nn.Conv2d(3, dims[0], 2, 2)
        self.enc_stages = nn.ModuleList(_stage(d, n, h, window)
                                        for d, n, h in zip(dims, depths, heads, strict=True))
        self.merges = nn.ModuleList(PatchMerging(a, b) for a, b in zip(dims[:-1], dims[1:],
                                                                       strict=True))
        self.enc_norm = nn.LayerNorm(dims[-1])
        self.enc_mod, self.to_latent = ChannelModNet(dims[-1]), nn.Linear(dims[-1], c)
        rdims, rdepths, rheads = dims[::-1], depths[::-1], heads[::-1]
        self.from_latent, self.dec_mod = nn.Linear(c, rdims[0]), ChannelModNet(rdims[0])
        self.dec_stages = nn.ModuleList(_stage(d, n, h, window)
                                        for d, n, h in zip(rdims, rdepths, rheads, strict=True))
        self.reverses = nn.ModuleList(PatchDivision(a, b) for a, b in zip(rdims, (*rdims[1:], 3),
                                                                          strict=True))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        y = self.embed(x).permute(0, 2, 3, 1)  # (B, H/2, W/2, C1)
        for i, stage in enumerate(self.enc_stages):
            if i:
                y = self.merges[i - 1](y)
            for block in stage:
                y = block(y)
        y = self.to_latent(self.enc_mod(self.enc_norm(y), snr_db))
        return y.permute(0, 3, 1, 2)

    def decode(self, z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        y = self.dec_mod(self.from_latent(z.permute(0, 2, 3, 1)), snr_db)
        for stage, reverse in zip(self.dec_stages, self.reverses, strict=True):
            for block in stage:
                y = block(y)
            y = reverse(y)
        return y.permute(0, 3, 1, 2).clamp(0, 1)
