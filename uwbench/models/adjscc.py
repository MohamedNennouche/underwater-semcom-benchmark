"""ADJSCC (Xu, Ai, Chen, Shi, Gong, "Wireless image transmission using deep source
channel coding with attention modules", IEEE TCSVT 32(4), 2022; survey ref. [253]).

PyTorch port of the authors' TensorFlow code, https://github.com/alexxu1988/ADJSCC
(``util_module.py``: ``Attention_Encoder``, ``Attention_Decoder``, ``AF_Module``),
MIT License, Copyright (c) 2020 alexxu1988. The MIT notice applies to this file.

Encoder: ``GFR`` modules = convolution + GDN (+ PReLU): ``9x9x256/2, 5x5x256/2,
5x5x256/1, 5x5x256/1`` with PReLU, then ``5x5xc/1`` with GDN only; an attention feature
(AF) module after each of the first four. Decoder: transposed convolutions + IGDN:
``5x5x256/1`` x3 and ``5x5x256/2`` with PReLU and AF modules, then ``9x9x3/2`` with a
sigmoid. AF module: global average pooling, concatenation with the SNR in dB (not
normalised, as in the original), dense ``C/16`` + ReLU, dense ``C`` + sigmoid, channel-wise
product. Differences from the original: ``tensorflow_compression``'s GDN is replaced by
:class:`uwbench.models.layers.GDN` (same formula and initialisation); "same" zero padding
is written explicitly. Downsampling factor 4.
"""

from __future__ import annotations

import torch
from torch import nn

from uwbench.models.base import JSCCModel, latent_channels
from uwbench.models.layers import GDN


class AFModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels + 1, channels // 16), nn.ReLU(),
                                nn.Linear(channels // 16, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        m = torch.cat([x.mean(dim=(2, 3)), snr_db.reshape(-1, 1).to(x)], dim=1)
        return x * self.fc(m)[:, :, None, None]


def _gfr_enc(cin: int, cout: int, k: int, stride: int, prelu: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2), GDN(cout)]
    return nn.Sequential(*layers, *([nn.PReLU(cout)] if prelu else []))


def _gfr_dec(cin: int, cout: int, k: int, stride: int, last: bool) -> nn.Sequential:
    conv = nn.ConvTranspose2d(cin, cout, k, stride, k // 2, output_padding=stride - 1)
    return nn.Sequential(conv, GDN(cout, inverse=True),
                         nn.Sigmoid() if last else nn.PReLU(cout))


class ADJSCC(JSCCModel):
    downsample = 4

    def __init__(self, bandwidth_ratio: float = 1 / 6, symbols: str = "complex",
                 width: int = 256):
        super().__init__()
        self.symbols = symbols
        c = latent_channels(bandwidth_ratio, self.downsample, symbols)
        w = width
        enc = [(3, w, 9, 2, True), (w, w, 5, 2, True), (w, w, 5, 1, True), (w, w, 5, 1, True),
               (w, c, 5, 1, False)]  # (in, out, kernel, stride, PReLU)
        dec = [(c, w, 5, 1, False), (w, w, 5, 1, False), (w, w, 5, 1, False),
               (w, w, 5, 2, False), (w, 3, 9, 2, True)]  # (in, out, kernel, stride, last)
        self.enc = nn.ModuleList(_gfr_enc(*spec) for spec in enc)
        self.enc_af = nn.ModuleList([AFModule(width) for _ in range(4)])
        self.dec = nn.ModuleList(_gfr_dec(*spec) for spec in dec)
        self.dec_af = nn.ModuleList([AFModule(width) for _ in range(4)])

    def encode(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        for layer, af in zip(self.enc[:4], self.enc_af, strict=True):
            x = af(layer(x), snr_db)
        return self.enc[4](x)

    def decode(self, z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        for layer, af in zip(self.dec[:4], self.dec_af, strict=True):
            z = af(layer(z), snr_db)
        return self.dec[4](z)
