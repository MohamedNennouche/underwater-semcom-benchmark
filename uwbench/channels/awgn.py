"""Complex AWGN channel (control reference)."""

from __future__ import annotations

import numpy as np
import torch

from uwbench.channels.base import SNR, Channel, gaussian, noise_variance, normalize_power


class AWGN(Channel):
    """``y = x + n`` with ``x`` normalised to ``E|x|^2 = 1`` and ``n ~ CN(0, sigma^2)``.

    SNR definition: ``SNR = E|x|^2 / sigma^2 = 1 / sigma^2``, i.e.
    ``sigma^2 = 10^(-snr_db / 10)`` (per complex symbol).
    """

    def forward(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator) -> torch.Tensor:
        return self.forward_csi(x, snr_db, rng)[0]

    def forward_csi(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator):
        if not x.is_complex():
            raise TypeError("AWGN expects complex baseband symbols")
        x = normalize_power(x)
        if snr_db is None:
            return x, torch.zeros_like(x.real)
        var = noise_variance(snr_db, x)
        return x + var.sqrt() * gaussian(tuple(x.shape), rng, x), var.expand(x.shape)
