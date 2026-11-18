"""Data conventions shared by every module.

Images
    ``torch.float32`` tensors of shape ``(B, C, H, W)`` with values in ``[0, 1]``.
    Checked by :func:`check_image_batch`.

Channel symbols
    * Acoustic (UAC) and AWGN control channel: complex baseband symbols
      (``torch.complex64`` / ``numpy.complex128``).
    * Optical (UWOC, IM/DD): real, non-negative intensity symbols.
    These are enforced by the channel interface (``uwbench.channels.base``).

Randomness
    See :mod:`uwbench.utils.seeding`.
"""

from __future__ import annotations

import torch

IMAGE_DTYPE = torch.float32


def check_image_batch(x: torch.Tensor) -> None:
    """Raise ``ValueError`` unless ``x`` follows the image convention.

    Parameters
    ----------
    x : torch.Tensor
        Candidate image batch.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
    if x.dtype != IMAGE_DTYPE:
        raise ValueError(f"expected dtype {IMAGE_DTYPE}, got {x.dtype}")
    if x.numel() and (x.min() < 0 or x.max() > 1):
        raise ValueError("pixel values must lie in [0, 1]")
