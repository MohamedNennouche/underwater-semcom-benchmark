"""Replay of measured time-varying impulse responses from the Watermark benchmark.

Source: P. van Walree, F.-X. Socheleau, R. Otnes, T. Jenserud, "The Watermark benchmark
for underwater acoustic modulation schemes", IEEE JOE 42(4), 2017 (survey ref. [248]);
data and MATLAB code: WatermarkV1.zip from https://www.ffi.no/en/research/watermark
(not redistributed; see DATASETS.md).

Channel files (verified on WatermarkV1, 2026-11-15): ``input/channels/<NAME>/mat/
<NAME>_<nnn>.mat`` with ``h`` (complex baseband TVIR, snapshots x delay taps),
``fs_tau`` (delay sampling rate, Hz), ``fs_t = fs_tau / K`` (snapshot rate, K taps),
``fc`` (centre frequency, Hz), ``V0`` and ``meta`` (``meta.bandwidth`` in Hz). Channels:
NOF1 and NCS1 (10-18 kHz, single hydrophone), BCH1 (32.5-37.5 kHz), KAU1 and KAU2
(4-8 kHz); for the array channels each file is one hydrophone (``doc/readme.txt``).

Replay (:func:`replay`) is the direct-replay filter of ``matlab/replayfilter.m``, in
baseband at ``fs_tau``: ``y[k] = sum_l x[k - l] g_p[l]`` with ``p = start + k``,
``g_p = (1 - f) h[n] + f h[n + 1]``, ``n = floor(p / K)``, ``f = p / K - n``.

:class:`WatermarkOFDM` sends the symbols with OFDM sampled at ``fs_tau``: ``N`` used
subcarriers spaced ``B / N`` around ``fc`` (``B`` = channel bandwidth, ``fs_tau / B``
must be an integer), cyclic prefix of ``cp_length >= K - 1`` samples (it covers the whole
recorded delay window, so there is no inter-symbol interference). Each row (image)
replays a random channel file from a random start time. Each file's response is
normalised to unit mean energy ``mean_n sum_l |h[n, l]|^2 = 1``. Perfect CSI: for each
OFDM symbol, the diagonal of the channel matrix, i.e. the frequency response of the TVIR
averaged over the FFT window; the variation within a symbol is left as inter-carrier
interference (NOF1 decorrelates to ~0.89 between snapshots 128 ms apart, so long
symbols suffer from strong ICI). Equalisation: one-tap
MMSE, as in :mod:`uwbench.channels.uac`. ``V0`` is not used (TODO(verify): its meaning
is documented in FFI-rapport 2016/01378, not in the archive).

SNR definition: received SNR per used subcarrier with unit-energy channel and
unit-power symbols, white noise ``CN(0, sigma^2)`` per subcarrier,
``sigma^2 = 10^(-snr_db / 10)``. The recorded ambient noise is not replayed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from uwbench.channels.base import SNR, Channel, equalize, gaussian, noise_variance, normalize_power
from uwbench.data.registry import default_root

WATERMARK_DIR = Path("watermark/extracted/WatermarkV1/Watermark")


@dataclass(frozen=True)
class TVIR:
    h: np.ndarray  # (snapshots, K) complex baseband
    fs_t: float
    fs_tau: float
    fc: float
    bandwidth: float


def load_tvir(path: Path) -> TVIR:
    m = scipy.io.loadmat(path)
    h = m["h"]
    fs_t, fs_tau = float(m["fs_t"].squeeze()), float(m["fs_tau"].squeeze())
    if not np.isclose(fs_t * h.shape[1], fs_tau):
        raise ValueError(f"{path.name}: expected fs_t = fs_tau / K (replayfilter.m convention)")
    bandwidth = float(np.ravel(m["meta"]["bandwidth"][0, 0])[0])
    return TVIR(h, fs_t, fs_tau, float(m["fc"].squeeze()), bandwidth)


def channel_files(root: Path, name: str) -> list[Path]:
    return sorted((root / "input" / "channels" / name / "mat").glob(f"{name}_*.mat"))


def average_response(h: torch.Tensor, first: int, length: int) -> torch.Tensor:
    """Mean of the impulse responses, linearly interpolated in time between snapshots, over
    samples ``first .. first + length - 1``.

    Its FFT is the diagonal of the OFDM channel matrix over that window (the exact
    perfect-CSI coefficient of a time-varying channel; the rest is inter-carrier
    interference).
    """
    k = h.shape[1]
    p = np.arange(first, first + length)
    n, f = p // k, (p % k) / k
    w = np.bincount(n, 1 - f, minlength=n.max() + 2) + np.bincount(n + 1, f, minlength=n.max() + 2)
    w = torch.as_tensor(w / length, dtype=h.real.dtype, device=h.device)
    return (w[:, None] * h[:len(w)]).sum(dim=0)


def replay(x: torch.Tensor, h: torch.Tensor, start: int) -> torch.Tensor:
    """Time-varying convolution of ``x`` (1-D, rate ``fs_tau``) with ``h`` from ``start``.

    Equal to the sample loop of ``replayfilter.m``; computed per segment of ``K`` output
    samples as ``(1 - f) (x * h[n]) + f (x * h[n + 1])`` with FFT convolutions.
    """
    k, length = h.shape[1], x.shape[0]
    j0, j1 = start // k, (start + length - 1) // k
    if j1 + 1 >= h.shape[0]:
        raise ValueError("signal longer than the remaining channel file")
    n_seg, off = j1 - j0 + 1, start - j0 * k
    xa = F.pad(x, (off + k - 1, n_seg * k - off - length))  # xa[q] <-> p = j0 K - K + 1 + q
    windows = xa.unfold(0, 2 * k - 1, k)  # (n_seg, 2K - 1)
    n_fft = 4 * k
    w = torch.fft.fft(windows, n=n_fft)
    hs = torch.fft.fft(h[j0:j1 + 2].to(x.dtype), n=n_fft)
    c0 = torch.fft.ifft(w * hs[:-1])[:, k - 1:2 * k - 1]
    c1 = torch.fft.ifft(w * hs[1:])[:, k - 1:2 * k - 1]
    f = torch.arange(k, device=x.device) / k
    return ((1 - f) * c0 + f * c1).reshape(-1)[off:off + length]


class WatermarkOFDM(Channel):
    """OFDM over replayed Watermark channels; see the module docstring."""

    def __init__(self, channel: str = "NOF1", n_subcarriers: int = 2048,
                 cp_length: int = 2048, root: Path | str | None = None):
        root = Path(root) if root else default_root() / WATERMARK_DIR
        self.files = channel_files(root, channel)
        if not self.files:
            raise FileNotFoundError(f"no {channel} channel files under {root}; "
                                    "see DATASETS.md (uwbench download watermark)")
        first = load_tvir(self.files[0])
        self.k = first.h.shape[1]
        ratio = first.fs_tau / first.bandwidth
        if not ratio.is_integer():
            raise ValueError(f"fs_tau / bandwidth = {ratio} is not an integer")
        self.n, self.cp, self.n_fft = n_subcarriers, cp_length, n_subcarriers * int(ratio)
        if cp_length < self.k - 1 or self.n_fft < self.k:
            raise ValueError(f"need cp_length >= {self.k - 1} and N * fs_tau / B >= {self.k}")
        half = n_subcarriers // 2
        self.bins = np.r_[0:n_subcarriers - half, self.n_fft - half:self.n_fft]
        self._load = functools.lru_cache(maxsize=8)(self._load_normalized)

    def _load_normalized(self, i: int) -> torch.Tensor:
        h = load_tvir(self.files[i]).h
        h = h / np.sqrt(np.mean(np.sum(np.abs(h) ** 2, axis=1)))
        return torch.as_tensor(h, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator) -> torch.Tensor:
        return self._forward(x, snr_db, rng, csi=False)

    def forward_csi(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator):
        """As :meth:`Channel.forward_csi`; the reported variance excludes the
        inter-carrier interference, which a perfect-CSI one-tap receiver cannot know."""
        return self._forward(x, snr_db, rng, csi=True)

    def _forward(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator, csi: bool):
        if not x.is_complex():
            raise TypeError("WatermarkOFDM expects complex baseband symbols")
        x = normalize_power(x)
        b, n_sym = x.shape
        s = -(-n_sym // self.n)
        bins = torch.as_tensor(self.bins, device=x.device)
        grid = torch.zeros(b, s, self.n_fft, dtype=x.dtype, device=x.device)
        grid[..., bins] = F.pad(x, (0, s * self.n - n_sym)).reshape(b, s, self.n)
        t = torch.fft.ifft(grid, norm="ortho")
        t = torch.cat([t[..., -self.cp:], t], dim=-1).reshape(b, -1)
        sym_len = self.n_fft + self.cp

        rows, responses = [], []
        for i in range(b):
            h = self._load(int(rng.integers(len(self.files)))).to(x.device)
            last = (h.shape[0] - 1) * self.k - t.shape[1]
            if last < 0:
                raise ValueError("transmission longer than one channel file")
            start = int(rng.integers(last + 1))
            rows.append(replay(t[i], h, start))
            g = torch.stack([average_response(h, start + j * sym_len + self.cp, self.n_fft)
                             for j in range(s)])
            responses.append(torch.fft.fft(g, n=self.n_fft)[:, bins])
        y = torch.fft.fft(torch.stack(rows).reshape(b, s, sym_len)[..., self.cp:],
                          norm="ortho")[..., bins]
        hf = torch.stack(responses).to(x.dtype)  # (b, s, N)
        var = None
        if snr_db is not None:
            var = noise_variance(snr_db, y)
            y = y + var.sqrt() * gaussian(tuple(y.shape), rng, y)
        out = equalize(y, hf, var, csi)

        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, s * self.n)[:, :n_sym]

        return (flat(out[0]), flat(out[1])) if csi else flat(out)
