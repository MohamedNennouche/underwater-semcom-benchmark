"""Building blocks of the reference models, written from their papers.

* :class:`GDN`: generalized divisive normalization, Balle, Laparra, Simoncelli, "Density
  modeling of images using a generalized normalization transformation", ICLR 2016:
  ``y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2)``; ``inverse=True`` gives IGDN,
  ``y_i = x_i * sqrt(...)``. Non-negativity by squaring the parameters, with the
  initialisation of ``tensorflow_compression.GDN`` (``beta = 1``, ``gamma = 0.1 I``).
* :class:`SwinBlock`: Swin Transformer block, Liu et al., "Swin Transformer:
  hierarchical vision transformer using shifted windows", ICCV 2021: (shifted) window
  multi-head self-attention with relative position bias, then an MLP, both pre-norm with
  residual connections. Feature maps of any size are padded to a multiple of the window.
* :class:`PatchMerging` (2x2 neighbours concatenated, LayerNorm, bias-free linear: halves
  the resolution) and :class:`PatchDivision` (LayerNorm, bias-free linear, pixel shuffle:
  doubles it), as in the Swin and WITT implementations. Shifted-window masks use -100
  for blocked pairs, as in those implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GDN(nn.Module):
    def __init__(self, channels: int, inverse: bool = False, eps: float = 1e-6):
        super().__init__()
        self.inverse, self.eps = inverse, eps
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.sqrt(0.1 * torch.eye(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        beta = self.beta.square() + self.eps
        gamma = self.gamma.square().reshape(c, c, 1, 1)
        norm = torch.sqrt(F.conv2d(x.square(), gamma, beta))
        return x * norm if self.inverse else x / norm


def _windows(x: torch.Tensor, w: int) -> torch.Tensor:
    b, h, wd, c = x.shape
    x = x.reshape(b, h // w, w, wd // w, w, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, c)


def _merge(win: torch.Tensor, w: int, b: int, h: int, wd: int) -> torch.Tensor:
    x = win.reshape(b, h // w, wd // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, wd, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.heads, self.window, self.scale = heads, window, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window),
                                            indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer("index", rel[..., 0] * (2 * window - 1) + rel[..., 1],
                             persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        bw, n, c = x.shape
        q, k, v = self.qkv(x).reshape(bw, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.bias[self.index].permute(2, 0, 1)
        if mask is not None:  # (num_windows, n, n), 0 or -inf
            nw = mask.shape[0]
            attn = attn.reshape(bw // nw, nw, self.heads, n, n) + mask[None, :, None]
            attn = attn.reshape(bw, self.heads, n, n)
        out = attn.softmax(dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(bw, n, c))


class SwinBlock(nn.Module):
    """One Swin block on ``(B, H, W, C)`` features; ``shift`` for odd blocks of a stage."""

    def __init__(self, dim: int, heads: int, window: int, shift: bool, mlp_ratio: float = 4.0):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        win, s = self.window, self.window // 2
        label = torch.zeros(1, h, w, 1, device=device)
        cnt = 0
        for hs in (slice(0, -win), slice(-win, -s), slice(-s, None)):
            for ws in (slice(0, -win), slice(-win, -s), slice(-s, None)):
                label[:, hs, ws, :] = cnt
                cnt += 1
        lw = _windows(label, win).squeeze(-1)
        diff = lw[:, None, :] - lw[:, :, None]
        return torch.zeros_like(diff).masked_fill(diff != 0, -100.0)  # as in Swin / WITT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        win = self.window
        y = F.pad(self.norm1(x), (0, 0, 0, -w % win, 0, -h % win))
        hp, wp = y.shape[1:3]
        shift = self.shift and min(hp, wp) > win
        if shift:
            y = torch.roll(y, (-(win // 2), -(win // 2)), dims=(1, 2))
        mask = self._mask(hp, wp, x.device) if shift else None
        y = _merge(self.attn(_windows(y, win), mask), win, b, hp, wp)
        if shift:
            y = torch.roll(y, (win // 2, win // 2), dims=(1, 2))
        x = x + y[:, :h, :w]
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm, self.reduce = nn.LayerNorm(4 * dim), nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C), H and W even
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduce(self.norm(x))


class PatchDivision(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm, self.expand = nn.LayerNorm(dim), nn.Linear(dim, 4 * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C) -> (B, 2H, 2W, out)
        y = self.expand(self.norm(x)).permute(0, 3, 1, 2)
        return F.pixel_shuffle(y, 2).permute(0, 2, 3, 1)
