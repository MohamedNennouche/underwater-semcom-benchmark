import math

import pytest
import torch

from uwbench.metrics.complexity import count_flops, count_parameters, measure_latency, platform_info
from uwbench.metrics.fidelity import PSNR_MAX_DB, lpips, ms_ssim, psnr
from uwbench.metrics.rate import (
    bandwidth_ratio,
    bandwidth_ratio_to_bpp,
    bits_per_pixel,
    bpp_to_bandwidth_ratio,
)


def images(b=2, h=192, w=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, h, w, generator=g)


# ---------------------------------------------------------------- PSNR

def test_psnr_identical_images_hit_documented_ceiling():
    x = images()
    assert torch.equal(psnr(x, x), torch.full((2,), PSNR_MAX_DB))


def test_psnr_known_error_per_image():
    ref = torch.full((2, 3, 8, 8), 0.5)
    x = ref.clone()
    x[0] += 0.1  # MSE 0.01 -> 20 dB
    x[1] -= 0.01  # MSE 1e-4 -> 40 dB
    torch.testing.assert_close(psnr(x, ref), torch.tensor([20.0, 40.0]), rtol=1e-4, atol=1e-4)


def test_psnr_gaussian_noise_matches_variance():
    ref = torch.full((4, 3, 128, 128), 0.5)
    sigma = 0.02
    x = ref + sigma * torch.randn(ref.shape, generator=torch.Generator().manual_seed(0))
    assert psnr(x, ref).mean().item() == pytest.approx(-10 * math.log10(sigma**2), abs=0.05)


def test_psnr_clamps_reconstruction_to_unit_range():
    ref = torch.ones(1, 3, 8, 8)
    assert psnr(ref + 5, ref).item() == PSNR_MAX_DB  # clamped back to 1


# ---------------------------------------------------------------- MS-SSIM / LPIPS

def test_ms_ssim_of_image_with_itself_is_one():
    x = images()
    torch.testing.assert_close(ms_ssim(x, x), torch.ones(2), atol=1e-5, rtol=0)
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    assert (ms_ssim(noisy, x) < 0.99).all()


def test_ms_ssim_rejects_small_images():
    x = images(h=128, w=128)
    with pytest.raises(ValueError, match="161"):
        ms_ssim(x, x)


def test_lpips_zero_for_identical_and_grows_with_distortion():
    x = images(h=64, w=64)
    same = lpips(x, x)
    assert same.shape == (2,) and (same.abs() < 1e-6).all()
    mild = lpips((x + 0.05 * torch.randn_like(x)).clamp(0, 1), x)
    strong = lpips((x + 0.3 * torch.randn_like(x)).clamp(0, 1), x)
    assert (mild > 0).all() and (strong > mild).all()


# ---------------------------------------------------------------- rate

def test_rate_definitions_and_conversion():
    assert bits_per_pixel(65536, 256, 256) == 1.0
    assert bandwidth_ratio(32768, 3, 256, 256) == pytest.approx(1 / 6)
    # QPSK with a rate-1/2 code carries 1 bit per channel use
    rho = bpp_to_bandwidth_ratio(0.5, channels=3, bits_per_channel_use=1.0)
    assert rho == pytest.approx(1 / 6)
    assert bandwidth_ratio_to_bpp(rho, 3, 1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------- complexity

def test_parameter_and_flop_counts_of_known_layers():
    lin = torch.nn.Linear(10, 5)
    assert count_parameters(lin) == 55
    assert count_flops(lin, torch.zeros(1, 10)) == 2 * 10 * 5
    conv = torch.nn.Conv2d(3, 8, 3)  # 30 x 30 outputs, 3*3*3 MACs each, 8 maps
    assert count_flops(conv, torch.zeros(1, 3, 32, 32)) == 2 * 8 * 30 * 30 * 27


def test_latency_reports_platform():
    lin = torch.nn.Linear(10, 5)
    out = measure_latency(lin, torch.zeros(4, 10), warmup=1, repeats=5)
    assert out["mean_ms"] > 0 and out["repeats"] == 5 and out["device"] == "cpu"
    assert out["platform"] == platform_info() and "torch" in out["platform"]
