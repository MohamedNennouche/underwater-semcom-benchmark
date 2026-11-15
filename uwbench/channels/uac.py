"""Modelled underwater acoustic channel (UAC) with an OFDM envelope.

Sources of the empirical formulas: D. Lucani, M. Medard, M. Stojanovic, "On the
relationship between transmission power and capacity of an underwater acoustic
communication channel", arXiv:0801.0426, eqs. (1)-(8), which cite M. Stojanovic, "On the
relationship between capacity and distance in an underwater acoustic communication
channel", WUWNet 2006 / ACM SIGMOBILE MC2R 11(4), 2007. Equivalent to eq. (1) of the
survey with ``k`` the spreading factor.

* Path loss ``A(l, f) = l^k a(f)^l``, ``l`` in km, ``k = 1.5`` ("practical spreading");
  ``10 log a(f)`` is Thorp's absorption in dB/km, ``f`` in kHz::

      10 log a(f) = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 2.75e-4 f^2 + 0.003

* Ambient noise PSD in dB re uPa per Hz, ``f`` in kHz, shipping activity ``s`` in [0, 1],
  wind speed ``w`` in m/s, components summed in linear power::

      10 log Nt  = 17 - 30 log f
      10 log Ns  = 40 + 20 (s - 0.5) + 26 log f - 60 log(f + 0.03)
      10 log Nw  = 50 + 7.5 w^0.5 + 20 log f - 40 log(f + 0.4)
      10 log Nth = -15 + 20 log f

* Multipath: tapped delay line from the image method in an isovelocity waveguide of
  depth ``D`` with a pressure-release surface and a reflecting bottom (constant sound
  speed, flat boundaries, specular reflections with constant coefficients, paths with at
  most ``max_reflections`` bounces). Each path ``p`` has length ``l_p``, delay
  ``l_p / c``, amplitude ``r_s^{n_s} r_b^{n_b} / sqrt(A(l_p, f_c))`` and, per
  realisation, an independent uniform random phase (surface motion). Delays are rounded
  to the nearest sample at rate ``B`` and the response is normalised to unit energy.
  **This is a simplified geometric model, not a substitute for a ray or normal-mode
  propagation model such as Bellhop**: no sound-speed profile, no frequency dependence
  per path, no time variation within a transmission.

* OFDM envelope: symbols on ``N`` subcarriers (unitary IFFT), cyclic prefix of
  ``cp_length`` samples, required to be at least the delay spread of the channel
  (checked at construction), linear convolution with the impulse response, CP removal,
  unitary FFT. The frequency-dependent attenuation and colored noise give each
  subcarrier ``k`` (frequency ``f_k``) a noise variance ``sigma^2 c_k`` with
  ``c_k = A(R, f_k) N(f_k) / mean_k[A(R, f_k) N(f_k)]``, ``R`` the range.
  Equalisation: one-tap MMSE per subcarrier with perfect channel knowledge,
  ``x_k = H_k^* Y_k / (|H_k|^2 + sigma^2 c_k)`` (zero forcing without noise).
  TODO: pilot-based channel estimation (not in version 1.0).

* Doppler: only a residual carrier frequency offset per OFDM symbol,
  ``epsilon ~ N(0, cfo_std^2)`` in units of the subcarrier spacing (inter-carrier
  interference). Wideband time compression/dilation is out of scope in version 1.0.

SNR definition: average received SNR per subcarrier,
``SNR = mean_k E|H_k X_k|^2 / mean_k(sigma^2 c_k) = 1 / sigma^2`` (unit-energy response,
unit-power symbols, ``mean_k c_k = 1``), so ``sigma^2 = 10^(-snr_db / 10)``. The path
loss magnitude is absorbed in the SNR; its frequency dependence shapes ``c_k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from uwbench.channels.base import SNR, Channel, gaussian, noise_variance, normalize_power


def thorp_db_per_km(f_khz: np.ndarray | float) -> np.ndarray | float:
    f2 = np.square(f_khz)
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def path_loss_db(l_km: np.ndarray | float, f_khz: np.ndarray | float, k: float = 1.5):
    """``10 log A(l, f) = 10 k log l + l * 10 log a(f)``."""
    return 10 * k * np.log10(l_km) + l_km * thorp_db_per_km(f_khz)


def noise_psd_db(f_khz: np.ndarray | float, shipping: float, wind_mps: float):
    """Total ambient-noise PSD in dB re uPa per Hz."""
    lf = np.log10(f_khz)
    nt = 17 - 30 * lf
    ns = 40 + 20 * (shipping - 0.5) + 26 * lf - 60 * np.log10(f_khz + 0.03)
    nw = 50 + 7.5 * math.sqrt(wind_mps) + 20 * lf - 40 * np.log10(f_khz + 0.4)
    nth = -15 + 20 * lf
    return 10 * np.log10(sum(10 ** (n / 10) for n in (nt, ns, nw, nth)))


@dataclass(frozen=True)
class Geometry:
    range_m: float
    water_depth_m: float
    tx_depth_m: float
    rx_depth_m: float


def image_paths(g: Geometry, max_reflections: int) -> list[tuple[float, int, int]]:
    """``(length_m, n_surface, n_bottom)`` of every path with at most ``max_reflections``.

    Image sources sit at depth ``2 m D + s z_tx`` (``s = +-1``); the image ``(s=+1, m)``
    has ``|m|`` surface and ``|m|`` bottom bounces, ``(s=-1, m >= 1)`` has ``m - 1``
    surface and ``m`` bottom bounces, ``(s=-1, m <= 0)`` has ``1 - m`` surface and ``-m``
    bottom bounces.
    """
    paths = []
    for m in range(-max_reflections, max_reflections + 1):
        for s in (1, -1):
            if s == 1:
                ns = nb = abs(m)
            else:
                ns, nb = (m - 1, m) if m >= 1 else (1 - m, -m)
            if ns + nb <= max_reflections:
                dz = 2 * m * g.water_depth_m + s * g.tx_depth_m - g.rx_depth_m
                paths.append((math.hypot(g.range_m, dz), ns, nb))
    return sorted(paths)


def impulse_response(g: Geometry, fc_hz: float, bandwidth_hz: float, k: float = 1.5,
                     max_reflections: int = 4, surface_coeff: float = -1.0,
                     bottom_coeff: float = 0.5, sound_speed_mps: float = 1500.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Unit-energy complex baseband taps at rate ``bandwidth_hz`` (first arrival at tap 0).

    ``rng=None`` gives the deterministic response (phases ``-2 pi f_c tau_p``); with a
    generator every path also gets an independent uniform phase.
    """
    paths = image_paths(g, max_reflections)
    length = np.array([p[0] for p in paths])
    ns = np.array([p[1] for p in paths])
    nb = np.array([p[2] for p in paths])
    delay = length / sound_speed_mps
    amp = surface_coeff**ns * bottom_coeff**nb \
        / np.sqrt(10 ** (path_loss_db(length / 1000, fc_hz / 1000, k) / 10))
    phase = np.exp(-2j * np.pi * fc_hz * delay)
    if rng is not None:
        phase = phase * np.exp(2j * np.pi * rng.random(len(paths)))
    idx = np.rint((delay - delay.min()) * bandwidth_hz).astype(int)
    taps = np.zeros(idx.max() + 1, dtype=complex)
    np.add.at(taps, idx, amp * phase)
    return taps / np.linalg.norm(taps)


class UACOFDM(Channel):
    """Modelled acoustic channel with OFDM; see the module docstring."""

    def __init__(self, geometry: dict, fc_hz: float, bandwidth_hz: float,
                 n_subcarriers: int, cp_length: int, spreading_k: float = 1.5,
                 shipping: float = 0.5, wind_mps: float = 0.0, max_reflections: int = 4,
                 surface_coeff: float = -1.0, bottom_coeff: float = 0.5,
                 sound_speed_mps: float = 1500.0, cfo_std: float = 0.0):
        self.geometry = Geometry(**geometry)
        self.n, self.cp, self.cfo_std = n_subcarriers, cp_length, cfo_std
        self.tdl = dict(g=self.geometry, fc_hz=fc_hz, bandwidth_hz=bandwidth_hz, k=spreading_k,
                        max_reflections=max_reflections, surface_coeff=surface_coeff,
                        bottom_coeff=bottom_coeff, sound_speed_mps=sound_speed_mps)
        spread = len(impulse_response(**self.tdl)) - 1
        if spread > cp_length:
            raise ValueError(f"cyclic prefix ({cp_length}) shorter than delay spread ({spread})")
        f_khz = (fc_hz + np.fft.fftfreq(n_subcarriers, d=1 / bandwidth_hz)) / 1000
        profile = 10 ** ((path_loss_db(self.geometry.range_m / 1000, f_khz, spreading_k)
                          + noise_psd_db(f_khz, shipping, wind_mps)) / 10)
        self.noise_profile = profile / profile.mean()  # c_k, FFT order

    def forward(self, x: torch.Tensor, snr_db: SNR, rng: np.random.Generator) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError("UACOFDM expects complex baseband symbols")
        x = normalize_power(x)
        b, n_sym = x.shape
        n, cp = self.n, self.cp
        s = -(-n_sym // n)
        t = torch.fft.ifft(F.pad(x, (0, s * n - n_sym)).reshape(b, s, n), norm="ortho")
        t = torch.cat([t[..., -cp:], t], dim=-1).reshape(b, s * (n + cp))

        taps = [impulse_response(**self.tdl, rng=rng) for _ in range(b)]
        h = torch.zeros(b, max(map(len, taps)), dtype=x.dtype, device=x.device)
        for i, tp in enumerate(taps):
            h[i, :len(tp)] = torch.as_tensor(tp, dtype=x.dtype)
        length = t.shape[-1] + h.shape[-1] - 1
        r = torch.fft.ifft(torch.fft.fft(t, n=length) * torch.fft.fft(h, n=length))
        r = r[..., :t.shape[-1]].reshape(b, s, n + cp)
        if self.cfo_std > 0:
            eps = torch.as_tensor(rng.normal(0, self.cfo_std, (b, s, 1)), device=x.device)
            k = torch.arange(n + cp, device=x.device)
            r = r * torch.exp(2j * torch.pi * eps * k / n).to(x.dtype)
        y = torch.fft.fft(r[..., cp:], norm="ortho")
        hf = torch.fft.fft(h, n=n).unsqueeze(1)  # (b, 1, n)
        if snr_db is None:
            xh = y / hf
        else:
            var = noise_variance(snr_db, y) * torch.as_tensor(
                self.noise_profile, dtype=y.real.dtype, device=x.device)
            y = y + var.sqrt() * gaussian(tuple(y.shape), rng, y)
            xh = hf.conj() * y / (hf.abs().square() + var)
        return xh.reshape(b, s * n)[:, :n_sym]
