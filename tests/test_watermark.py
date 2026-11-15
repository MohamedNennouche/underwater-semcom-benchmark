from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

from uwbench.channels import load_scenario
from uwbench.channels.base import normalize_power
from uwbench.channels.watermark import (
    WATERMARK_DIR,
    WatermarkOFDM,
    channel_files,
    load_tvir,
    replay,
)
from uwbench.data.registry import default_root

REPO = Path(__file__).resolve().parents[1]
REAL = default_root() / WATERMARK_DIR
real_data = pytest.mark.skipif(not channel_files(REAL, "NOF1"), reason="Watermark not present")


def replay_loop(x: np.ndarray, h: np.ndarray, start: int) -> np.ndarray:
    """Direct port of the sample loop of Watermark's matlab/replayfilter.m."""
    k = h.shape[1]
    xp = np.concatenate([np.zeros(k - 1, complex), x])
    hf = np.flipud(h.T)  # MATLAB: h = flipud(h.')
    y = np.zeros(len(x), complex)
    for i in range(len(x)):
        p = start + i
        f, n = p / k - p // k, p // k
        ir = (1 - f) * hf[:, n] + f * hf[:, n + 1]
        y[i] = xp[i:i + k] @ ir
    return y


def write_channel(root: Path, name: str, h: np.ndarray, fs_tau=16, bandwidth="8") -> None:
    d = root / "input" / "channels" / name / "mat"
    d.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(d / f"{name}_001.mat", {
        "h": h, "fs_tau": fs_tau, "fs_t": fs_tau / h.shape[1], "fc": 10, "V0": 0.0,
        "meta": {"bandwidth": bandwidth}})


def random_tvir(n_snap, k, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_snap, k)) + 1j * rng.standard_normal((n_snap, k))


# ---------------------------------------------------------------- synthetic files

def test_loader_reads_format_and_checks_rates(tmp_path):
    write_channel(tmp_path, "TEST", random_tvir(20, 8))
    tv = load_tvir(channel_files(tmp_path, "TEST")[0])
    assert tv.h.shape == (20, 8) and tv.fs_tau == 16 and tv.fs_t == 2 and tv.bandwidth == 8
    path = channel_files(tmp_path, "TEST")[0]
    scipy.io.savemat(path, {"h": tv.h, "fs_tau": 16, "fs_t": 3.0, "fc": 10, "V0": 0.0,
                            "meta": {"bandwidth": "8"}})
    with pytest.raises(ValueError, match="fs_t"):
        load_tvir(path)


@pytest.mark.parametrize("start,length", [(0, 30), (5, 41), (16, 8), (13, 1)])
def test_replay_matches_matlab_loop(start, length):
    h = random_tvir(12, 8)
    x = random_tvir(1, length, seed=1)[0]
    got = replay(torch.as_tensor(x), torch.as_tensor(h), start).numpy()
    np.testing.assert_allclose(got, replay_loop(x, h, start), atol=1e-9)


def test_replayed_impulse_returns_recorded_response():
    h = np.tile(random_tvir(1, 8), (6, 1))  # time-invariant
    impulse = torch.zeros(8, dtype=torch.complex128)
    impulse[0] = 1
    np.testing.assert_allclose(replay(impulse, torch.as_tensor(h), 16).numpy(), h[0])
    hv = random_tvir(6, 8)  # time-varying: tap 0 at a snapshot instant is that snapshot
    y = replay(impulse, torch.as_tensor(hv), 16).numpy()
    lag = np.arange(8) / 8
    np.testing.assert_allclose(y, (1 - lag) * hv[2] + lag * hv[3])


def synthetic_channel(tmp_path, h):
    write_channel(tmp_path, "TEST", h)
    return WatermarkOFDM("TEST", n_subcarriers=8, cp_length=8, root=tmp_path)


def test_ofdm_noiseless_time_invariant_recovers_symbols(tmp_path):
    ch = synthetic_channel(tmp_path, np.tile(random_tvir(1, 8), (40, 1)))
    x = torch.as_tensor(random_tvir(3, 20, seed=2), dtype=torch.complex64)
    y = ch(x, None, np.random.default_rng(0))
    torch.testing.assert_close(y, normalize_power(x), atol=1e-4, rtol=1e-4)


def test_ofdm_noise_matches_snr_on_flat_channel(tmp_path):
    h = np.zeros((1000, 8), complex)
    h[:, 0] = 3.0  # flat channel, normalised to unit energy by the loader
    ch = synthetic_channel(tmp_path, h)
    x = torch.as_tensor(random_tvir(4, 2000, seed=3), dtype=torch.complex64)
    y = ch(x, 20.0, np.random.default_rng(1))
    mse = (y - normalize_power(x)).abs().square().mean().item()
    assert mse == pytest.approx(0.01, rel=0.05)


def test_ofdm_rejects_short_cyclic_prefix(tmp_path):
    write_channel(tmp_path, "TEST", random_tvir(40, 8))
    with pytest.raises(ValueError, match="cp_length"):
        WatermarkOFDM("TEST", n_subcarriers=8, cp_length=4, root=tmp_path)


def test_ofdm_is_differentiable(tmp_path):
    ch = synthetic_channel(tmp_path, random_tvir(40, 8))
    x = torch.as_tensor(random_tvir(2, 20, seed=4), dtype=torch.complex64).requires_grad_(True)
    ch(x, 10.0, np.random.default_rng(0)).abs().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.abs()).all()


# ---------------------------------------------------------------- real archive

@real_data
@pytest.mark.parametrize("name,count", [("NOF1", 60), ("NCS1", 60), ("BCH1", 4),
                                        ("KAU1", 16), ("KAU2", 16)])
def test_real_channel_files(name, count):
    files = channel_files(REAL, name)
    assert len(files) == count
    tv = load_tvir(files[0])  # rate convention checked by the loader
    assert tv.h.ndim == 2 and np.iscomplexobj(tv.h)


@real_data
def test_real_replay_matches_matlab_loop():
    h = load_tvir(channel_files(REAL, "NOF1")[0]).h[:4]
    x = random_tvir(1, 3000, seed=5)[0]
    got = replay(torch.as_tensor(x), torch.as_tensor(h), 1000).numpy()
    np.testing.assert_allclose(got, replay_loop(x, h, 1000), rtol=1e-7,
                               atol=1e-6 * np.abs(h).max())


@real_data
def test_real_scenario_runs():
    channel, grid, _ = load_scenario(REPO / "configs/scenarios/uac_watermark_nof1.yaml")
    x = torch.as_tensor(random_tvir(2, 5000, seed=6), dtype=torch.complex64)
    y = channel(x, float(grid[-1]), np.random.default_rng(0))
    assert y.shape == x.shape and torch.isfinite(y.abs()).all()
