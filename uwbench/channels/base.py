"""Common channel interface.

Every channel implements ``forward(x, snr_db, rng) -> y`` once, in PyTorch, so it is
differentiable for training. Calling a channel (``channel(x, snr_db, rng)``) also accepts
NumPy arrays: they are converted, processed without gradient and returned as NumPy
(evaluation mode). Both modes draw all randomness from the same
``numpy.random.Generator``, so a seed gives identical realisations in both modes.

Conventions
-----------
* Symbols are ``(B, n)``: one row per transmitted image. Acoustic and AWGN channels take
  complex baseband symbols; optical (IM/DD) channels take real non-negative intensities.
* Power normalisation happens inside the channel, per row: unit mean power
  ``E|x|^2 = 1`` (complex channels) or unit mean optical power ``E[x] = 1`` (optical).
* ``snr_db`` is a float, or a ``(B,)`` tensor (one SNR per row, e.g. drawn uniformly
  during training), or ``None`` for a noiseless channel. Each channel's docstring defines
  its SNR exactly.
* Channels return symbols in the transmit domain, after perfect-CSI compensation (see
  each channel), so a decoder can be trained on any of them.
"""

from __future__ import annotations

import numpy as np
import torch

SNR = float | torch.Tensor | None


def normalize_power(x: torch.Tensor) -> torch.Tensor:
    """Scale each row to unit mean power ``mean(|x|^2) = 1``."""
    p = x.abs().square().mean(dim=-1, keepdim=True)
    return x / p.clamp_min(1e-12).sqrt()


def noise_variance(snr_db: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``10^(-snr_db/10)`` as a tensor broadcastable against the ``(B, ...)`` rows of ``like``."""
    snr = torch.as_tensor(snr_db, dtype=torch.float64, device=like.device)
    var = (10.0 ** (-snr / 10)).to(like.real.dtype)
    return var.reshape(-1, *([1] * (like.ndim - 1))) if var.ndim == 1 else var


def gaussian(shape: tuple[int, ...], rng: np.random.Generator, like: torch.Tensor) -> torch.Tensor:
    """Unit-variance Gaussian noise drawn from ``rng``: real, or circular complex
    (``E|n|^2 = 1``) when ``like`` is complex."""
    z = rng.standard_normal(shape)
    if like.is_complex():
        z = (z + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    return torch.as_tensor(z, dtype=like.dtype, device=like.device)


def equalize(y: torch.Tensor, h: torch.Tensor, var: torch.Tensor | None, csi: bool):
    """One-tap equalisation of ``y = h x + n`` with ``Var[n] = var`` (``None``: noiseless).

    ``csi=False``: MMSE estimate ``h* y / (|h|^2 + var)`` (zero forcing without noise).
    ``csi=True``: unbiased zero-forcing estimate ``y / h`` and its per-symbol noise
    variance ``var / |h|^2``, as needed by a soft demapper.
    """
    if csi:
        nv = torch.zeros_like(y.real) if var is None else var / h.abs().square()
        return y / h, nv.expand(y.shape)
    return y / h if var is None else h.conj() * y / (h.abs().square() + var)


class Channel:
    """Base class; subclasses implement :meth:`forward` and :meth:`forward_csi`."""

    def forward(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator) -> torch.Tensor:
        raise NotImplementedError

    def forward_csi(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y, noise_var)``: unbiased estimate of the normalised transmitted symbols and
        the noise variance of each (perfect CSI), for digital receivers."""
        raise NotImplementedError

    def __call__(self, x, snr_db: SNR, rng: np.random.Generator | None = None):
        rng = np.random.default_rng() if rng is None else rng
        if isinstance(x, np.ndarray):
            with torch.no_grad():
                return self.forward(torch.from_numpy(x), snr_db, rng).numpy()
        return self.forward(x, snr_db, rng)
