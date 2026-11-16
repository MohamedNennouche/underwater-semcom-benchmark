"""Fidelity metrics, one value per image (aggregation over a dataset is the caller's).

All metrics take ``(B, C, H, W)`` float tensors; inputs are clamped to ``[0, 1]`` first
(reconstructions of analog schemes can leave the range), and values are computed in
float, without 8-bit quantisation.

* PSNR: ``10 log10(1 / MSE)`` per image, MSE over all pixels and channels, data range 1.
  Identical images (``MSE = 0``) give the documented ceiling ``PSNR_MAX_DB = 100`` instead
  of infinity, so that dataset means stay finite; no lossy scheme at benchmark rates gets
  near it.
* MS-SSIM: Wang, Simoncelli, Bovik, "Multiscale structural similarity for image quality
  assessment", Asilomar 2003, via ``pytorch_msssim.ms_ssim`` with its defaults (Gaussian
  window 11, sigma 1.5, the five scale weights of the paper). The five scales require a
  smallest image side above 160 pixels; smaller images are rejected rather than scored
  with a modified metric.
* LPIPS: Zhang et al., "The unreasonable effectiveness of deep features as a perceptual
  metric", CVPR 2018, ``lpips`` package v0.1 with the AlexNet backbone (as in the
  literature); inputs mapped to ``[-1, 1]``. Lower is better. The first call downloads
  the torchvision AlexNet weights.
"""

from __future__ import annotations

import functools

import torch
from pytorch_msssim import ms_ssim as _ms_ssim

from uwbench.utils.conventions import check_image_batch

PSNR_MAX_DB = 100.0
MS_SSIM_MIN_SIDE = 161  # (win_size - 1) * 2**4 + 1 for win_size 11 and five scales


def _prepare(x: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != ref.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(ref.shape)}")
    ref = ref.float()
    check_image_batch(ref)
    return x.float().clamp(0, 1), ref


def psnr(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB, capped at ``PSNR_MAX_DB``."""
    x, ref = _prepare(x, ref)
    mse = (x.double() - ref.double()).square().flatten(1).mean(dim=1)  # float64: stable sums
    return (10 * torch.log10(1 / mse)).clamp(max=PSNR_MAX_DB).float()


def ms_ssim(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-image MS-SSIM in ``[0, 1]`` (1 for identical images)."""
    x, ref = _prepare(x, ref)
    if min(ref.shape[-2:]) < MS_SSIM_MIN_SIDE:
        raise ValueError(f"MS-SSIM needs both sides >= {MS_SSIM_MIN_SIDE} pixels, "
                         f"got {tuple(ref.shape[-2:])}")
    return _ms_ssim(x, ref, data_range=1.0, size_average=False)


@functools.lru_cache(maxsize=2)
def _lpips_model(device: str):
    import warnings

    import lpips

    with warnings.catch_warnings():  # lpips 0.1.4 uses torchvision's deprecated `pretrained`
        warnings.simplefilter("ignore", UserWarning)
        return lpips.LPIPS(net="alex", verbose=False).eval().to(device)


def lpips(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-image LPIPS (AlexNet); 0 for identical images, lower is better."""
    x, ref = _prepare(x, ref)
    model = _lpips_model(str(ref.device))
    with torch.no_grad():
        return model(x, ref, normalize=True).flatten()  # normalize: [0, 1] -> [-1, 1]
