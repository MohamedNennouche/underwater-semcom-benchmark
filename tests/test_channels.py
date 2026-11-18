import math
from pathlib import Path

import numpy as np
import pytest
import torch

from uwbench.channels import load_scenario
from uwbench.channels.awgn import AWGN
from uwbench.channels.base import normalize_power
from uwbench.channels.uac import (
    UACOFDM,
    Geometry,
    image_paths,
    impulse_response,
    noise_psd_db,
    path_loss_db,
    thorp_db_per_km,
)
from uwbench.channels.uwoc import (
    UWOC,
    beer_lambert,
    bias_and_clip,
    gamma_gamma_si,
    pointing_parameters,
    sample_pointing,
    sample_turbulence,
)

SCENARIOS = sorted((Path(__file__).resolve().parents[1] / "configs/scenarios").glob("*.yaml"))


def complex_symbols(b, n, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.standard_normal((b, n)) + 1j * rng.standard_normal((b, n)),
                           dtype=torch.complex64)


# ---------------------------------------------------------------- AWGN

@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0, 25.0])
def test_awgn_noise_variance_matches_snr(snr_db):
    x = complex_symbols(4, 250_000)
    y = AWGN()(x, snr_db, np.random.default_rng(1))
    var = (y - normalize_power(x)).abs().square().mean().item()
    assert var == pytest.approx(10 ** (-snr_db / 10), rel=0.01)


def test_awgn_numpy_mode_reproducible_and_equal_to_torch_mode():
    x = complex_symbols(2, 64)
    a = AWGN()(x.numpy(), 5.0, np.random.default_rng(3))
    b = AWGN()(x.numpy(), 5.0, np.random.default_rng(3))
    c = AWGN()(x, 5.0, np.random.default_rng(3))
    assert isinstance(a, np.ndarray) and np.array_equal(a, b)
    np.testing.assert_allclose(a, c.detach().numpy(), rtol=1e-6)


def test_awgn_per_row_snr_and_noiseless():
    x = complex_symbols(2, 200_000)
    y = AWGN()(x, torch.tensor([0.0, 20.0]), np.random.default_rng(0))
    var = (y - normalize_power(x)).abs().square().mean(dim=1)
    assert var[0].item() == pytest.approx(1.0, rel=0.02)
    assert var[1].item() == pytest.approx(0.01, rel=0.02)
    torch.testing.assert_close(AWGN()(x, None), normalize_power(x))


# ---------------------------------------------------------------- acoustic

def test_thorp_at_10_khz():
    f = 10.0  # independent evaluation of Thorp's formula, dB/km
    expected = 0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2) + 2.75e-4 * f**2 + 0.003
    assert thorp_db_per_km(f) == pytest.approx(expected, rel=1e-12)
    assert path_loss_db(2.0, f, 1.5) == pytest.approx(15 * math.log10(2) + 2 * expected)


def test_noise_psd_sums_the_four_components():
    f, s, w = 20.0, 0.3, 7.0
    comps = [17 - 30 * math.log10(f),
             40 + 20 * (s - 0.5) + 26 * math.log10(f) - 60 * math.log10(f + 0.03),
             50 + 7.5 * math.sqrt(w) + 20 * math.log10(f) - 40 * math.log10(f + 0.4),
             -15 + 20 * math.log10(f)]
    expected = 10 * math.log10(sum(10 ** (c / 10) for c in comps))
    assert noise_psd_db(f, s, w) == pytest.approx(expected, rel=1e-12)


GEOM = Geometry(range_m=1000, water_depth_m=100, tx_depth_m=30, rx_depth_m=60)


def test_image_paths_geometry():
    paths = image_paths(GEOM, 4)
    assert len(paths) == 9  # 1 direct + 2 per reflection order 1..4
    assert paths[0] == (pytest.approx(math.hypot(1000, 30)), 0, 0)  # direct
    surface = [p for p in paths if p[1:] == (1, 0)][0]
    bottom = [p for p in paths if p[1:] == (0, 1)][0]
    assert surface[0] == pytest.approx(math.hypot(1000, 30 + 60))  # mirror in the surface
    assert bottom[0] == pytest.approx(math.hypot(1000, 2 * 100 - 30 - 60))  # in the bottom


@pytest.mark.parametrize("rng", [None, np.random.default_rng(0)])
def test_impulse_response_unit_energy(rng):
    h = impulse_response(GEOM, 14e3, 8e3, rng=rng)
    assert np.sum(np.abs(h) ** 2) == pytest.approx(1.0, rel=1e-12)
    assert len(h) > 1


def ofdm(max_reflections, cfo_std=0.0):
    return UACOFDM(geometry=GEOM.__dict__, fc_hz=14e3, bandwidth_hz=8e3, n_subcarriers=256,
                   cp_length=160, max_reflections=max_reflections, cfo_std=cfo_std)


@pytest.mark.parametrize("max_reflections", [0, 2])
def test_ofdm_noiseless_recovers_symbols(max_reflections):
    x = complex_symbols(3, 1000)  # not a multiple of the subcarrier count
    y = ofdm(max_reflections)(x, None, np.random.default_rng(0))
    torch.testing.assert_close(y, normalize_power(x), atol=1e-4, rtol=1e-4)


def test_ofdm_rejects_short_cyclic_prefix():
    with pytest.raises(ValueError, match="cyclic prefix"):
        UACOFDM(geometry=GEOM.__dict__, fc_hz=14e3, bandwidth_hz=8e3, n_subcarriers=256,
                cp_length=4, max_reflections=2)


def test_ofdm_single_path_error_matches_snr():
    snr_db = 30.0
    ch = ofdm(0)
    x = complex_symbols(8, 256 * 64)
    y = ch(x, snr_db, np.random.default_rng(5))
    mse = (y - normalize_power(x)).abs().square().mean().item()
    # |H_k| = 1: error ~ sigma^2 c_k, with mean_k c_k = 1 by construction
    assert mse == pytest.approx(10 ** (-snr_db / 10), rel=0.03)
    assert ch.noise_profile.mean() == pytest.approx(1.0)


def test_residual_cfo_degrades_noiseless_link():
    x = complex_symbols(2, 2048)
    clean = ofdm(2)(x, None, np.random.default_rng(0))
    shifted = ofdm(2, cfo_std=0.05)(x, None, np.random.default_rng(0))
    assert (shifted - normalize_power(x)).abs().square().mean() > \
        100 * (clean - normalize_power(x)).abs().square().mean()


# ---------------------------------------------------------------- optical

def test_beer_lambert():
    assert beer_lambert(2.17, 3.0) == pytest.approx(math.exp(-2.17 * 3.0))


@pytest.mark.parametrize("cfg,si", [
    ({"model": "lognormal", "scintillation_index": 0.1}, 0.1),
    ({"model": "lognormal", "scintillation_index": 0.4}, 0.4),
    ({"model": "gamma_gamma", "alpha": 4.0, "beta": 4.0}, gamma_gamma_si(4.0, 4.0)),
])
def test_turbulence_mean_one_and_scintillation_index(cfg, si):
    h = sample_turbulence(cfg, 1_000_000, np.random.default_rng(0))
    assert h.mean() == pytest.approx(1.0, abs=0.005)
    assert h.var() / h.mean() ** 2 == pytest.approx(si, rel=0.02)


def test_pointing_mean_matches_closed_form():
    cfg = {"aperture_radius_m": 0.05, "beam_radius_m": 0.2, "jitter_std_m": 0.05}
    a0, w_zeq = pointing_parameters(0.05, 0.2)
    xi2 = (w_zeq / (2 * 0.05)) ** 2  # E[h_p] = A0 xi^2 / (xi^2 + 1), Farid-Hranilovic 2007
    h = sample_pointing(cfg, 1_000_000, np.random.default_rng(0))
    assert h.mean() == pytest.approx(a0 * xi2 / (xi2 + 1), rel=0.01)
    assert h.max() <= a0


def test_intensities_are_non_negative_and_negative_input_rejected():
    s = bias_and_clip(torch.randn(4, 10_000), modulation_index=1.0)
    assert (s >= 0).all()
    # the operating point cannot be moved by the latent's DC: a near-constant negative
    # latent (the collapse seen in training) still gives a unit-variance signal on the
    # bias, with Gaussian-level clipping (16 %, mean Phi(1) + phi(1) = 1.083)
    s = bias_and_clip(-5 + 1e-3 * torch.randn(4, 10_000), modulation_index=1.0)
    assert (s == 0).double().mean() < 0.2 and s.mean().item() == pytest.approx(1.08, abs=0.03)
    with pytest.raises(ValueError):
        UWOC(0.15, 10.0)(torch.tensor([[0.5, -0.1]]), 10.0, np.random.default_rng(0))


@pytest.mark.parametrize("zeta", [0.0, 2.0])
def test_uwoc_noise_matches_snr_definition(zeta):
    snr_db, c, d = 20.0, 0.305, 5.0
    ch = UWOC(c, d, shot_to_thermal=zeta)
    s = torch.ones(4, 200_000, dtype=torch.float64)
    r, v = ch.forward_csi(s, snr_db, np.random.default_rng(0))  # zero forcing
    h = math.exp(-c * d)
    expected = 10 ** (-snr_db / 10) * (1 + zeta * h) / h**2
    assert (r - 1).var().item() == pytest.approx(expected, rel=0.01)
    assert v[0, 0].item() == pytest.approx(expected)
    # forward: MMSE, (r - 1) shrunk by 1 / (1 + v); noiseless input returned unchanged
    m = ch(s, snr_db, np.random.default_rng(0))
    torch.testing.assert_close(m - 1, (r - 1) / (1 + expected))
    torch.testing.assert_close(ch(s, None, np.random.default_rng(0)), s)


def test_uwoc_mmse_output_stays_bounded_in_deep_fades():
    # 10 m of turbid harbor water: h = 4e-10, so r = y / h has noise std ~1e9; the MMSE
    # output (what learned decoders see) stays close to the DC bias
    s = 1 + 0.5 * torch.randn(2, 10_000, dtype=torch.float64).clamp(-2, 2)
    out = UWOC(2.17, 10.0)(s, 0.0, np.random.default_rng(0))
    assert torch.isfinite(out).all() and (out - 1).abs().max() < 1e-3


# ---------------------------------------------------------------- shared properties

@pytest.mark.parametrize("make,x", [
    (lambda: AWGN(), complex_symbols(2, 512)),
    (lambda: ofdm(2), complex_symbols(2, 512)),
    (lambda: UWOC(0.15, 10.0, {"model": "lognormal", "scintillation_index": 0.1}),
     torch.rand(2, 512, dtype=torch.float64) + 0.1),
])
def test_channels_are_differentiable(make, x):
    x = x.clone().requires_grad_(True)
    make()(x, 10.0, np.random.default_rng(0)).abs().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.abs()).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenarios_load_and_run(path):
    try:
        channel, grid, cfg = load_scenario(path)
    except FileNotFoundError as exc:  # measured channels need their (non-versioned) data
        pytest.skip(str(exc))
    assert grid[0] == -5 and np.all(np.diff(grid) == 5)
    assert grid[-1] == (50 if cfg["channel"] == "uwoc" else 25)
    x = torch.rand(2, 300, dtype=torch.float64) if cfg["channel"] == "uwoc" \
        else complex_symbols(2, 300)
    y = channel(x, float(grid[-1]), np.random.default_rng(0))
    assert y.shape == x.shape and torch.isfinite(y.abs()).all()
