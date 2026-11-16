"""Interface for learned (analog) schemes and their transmission through a channel.

**To evaluate your own model**, subclass :class:`JSCCModel` (an ``nn.Module``):

* class attributes ``downsample`` (the factor between image and latent sides) and
  ``symbols`` (``"complex"`` for the AWGN/acoustic channels, ``"real"`` for the optical
  IM/DD channel);
* ``encode(x, snr_db) -> z``: images ``(B, 3, H, W)`` in ``[0, 1]`` to a **real** latent
  ``(B, C, H / downsample, W / downsample)``; ``snr_db`` is a ``(B,)`` tensor that the
  encoder may ignore;
* ``decode(z_hat, snr_db) -> x_hat``: the noisy latent back to images in ``[0, 1]``.

Everything else is done by :func:`transmit`, identically for every model: padding of
the image to a multiple of ``downsample`` (reflection, cropped back after decoding),
mapping of the latent to channel symbols, power normalisation and the channel itself.

* complex symbols: the flattened latent ``(f_1 .. f_2k)`` gives ``k`` complex symbols
  ``f_i + j f_{k+i}`` (as in ADJSCC's TensorFlow code); the channel normalises them to
  unit mean power (``E|z|^2 = 1``, eq. (1) of Bourtsoulatze et al., TCCN 2019, P = 1);
* real symbols (IM/DD): DC bias and clipping, ``s = max(0, 1 + m f / rms(f))``
  (:func:`uwbench.channels.uwoc.bias_and_clip`, modulation index ``m``), inverted after
  the channel.

Bandwidth ratio (:mod:`uwbench.metrics.rate`): ``k / n`` with ``n = 3 H W`` and ``k``
the number of channel uses, i.e. ``C h w / 2`` (complex) or ``C h w`` (real).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from uwbench.channels.base import Channel
from uwbench.channels.uwoc import bias_and_clip, remove_bias


class JSCCModel(torch.nn.Module):
    """Base class of the benchmark's model interface (see the module docstring)."""

    downsample: int = 1
    symbols: str = "complex"

    def encode(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def latent_channels(bandwidth_ratio: float, downsample: int, symbols: str) -> int:
    """Number of latent channels ``C`` giving ``bandwidth_ratio`` (must be an integer)."""
    c = bandwidth_ratio * 3 * downsample**2 * (2 if symbols == "complex" else 1)
    if abs(c - round(c)) > 1e-6:
        raise ValueError(f"k/n = {bandwidth_ratio} needs {c} latent channels (not an integer)")
    return round(c)


def bandwidth_ratio(model: JSCCModel, latent_channels_: int) -> float:
    return latent_channels_ / (3 * model.downsample**2 * (2 if model.symbols == "complex" else 1))


def pad_to(x: torch.Tensor, multiple: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = -h % multiple, -w % multiple
    return F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x


def transmit(model: JSCCModel, x: torch.Tensor, channel: Channel, snr_db: torch.Tensor,
             rng: np.random.Generator, modulation_index: float = 1.0) -> torch.Tensor:
    """Encode, send through ``channel`` at ``snr_db`` (``(B,)`` tensor) and decode.

    Differentiable in the model parameters (used for training and evaluation).
    """
    h, w = x.shape[-2:]
    z = model.encode(pad_to(x, model.downsample), snr_db)
    flat = z.flatten(1)
    snr = snr_db.cpu()  # channels draw their randomness on the CPU
    if model.symbols == "complex":
        if flat.shape[1] % 2:
            raise ValueError("complex symbols need an even latent size")
        k = flat.shape[1] // 2
        sym = torch.complex(flat[:, :k].float(), flat[:, k:].float()).cpu()
        out = channel(sym, snr, rng).to(flat.device)
        flat_hat = torch.cat([out.real, out.imag], dim=1)
    else:
        s = bias_and_clip(flat.double().cpu(), modulation_index)
        flat_hat = remove_bias(channel(s, snr, rng), modulation_index).to(flat)
    x_hat = model.decode(flat_hat.reshape(z.shape).to(z.dtype), snr_db)
    return x_hat[..., :h, :w]
