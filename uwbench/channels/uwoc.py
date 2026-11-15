"""Underwater wireless optical channel (UWOC), intensity modulation / direct detection.

Model (per transmitted image, block fading)::

    y = h * s + n,      h = h_l * h_a * h_p,      r = y / h   (perfect CSI)

* ``s``: real non-negative intensities, normalised to unit mean optical power
  ``mean(s) = 1`` per row (average optical power constraint).
* ``h_l = exp(-c d)``: Beer-Lambert attenuation, eq. (2) of the survey, with ``c`` from
  Table III (``configs/uwoc_water.yaml``). Scattered photons are counted as lost, which is
  pessimistic in coastal and harbor waters (survey, Sec. II-B).
* ``h_a``: turbulence-induced scintillation, normalised to ``E[h_a] = 1``.
  - log-normal: ``h_a = exp(2X)``, ``X ~ N(-s2, s2)``, ``s2 = ln(1 + SI) / 4``
    (``SI`` = scintillation index; then ``E[h_a] = 1`` and ``Var[h_a] = SI``);
  - Gamma-Gamma: ``h_a = G1 G2``, ``G1 ~ Gamma(alpha, 1/alpha)``,
    ``G2 ~ Gamma(beta, 1/beta)``, ``SI = 1/alpha + 1/beta + 1/(alpha beta)``.
  Survey refs. [5], [85]-[87].
* ``h_p``: pointing error, Farid & Hranilovic, "Outage capacity optimization for
  free-space optical links with pointing errors", J. Lightw. Technol. 25(7), 2007:
  ``h_p = A0 exp(-2 r^2 / w_zeq^2)``, radial displacement ``r ~ Rayleigh(sigma_s)``,
  ``v = sqrt(pi) a / (sqrt(2) w_z)``, ``A0 = erf(v)^2``,
  ``w_zeq^2 = w_z^2 sqrt(pi) erf(v) / (2 v exp(-v^2))`` (``a``: aperture radius,
  ``w_z``: beam radius at the receiver). Survey Sec. II-B, ref. [28].
* ``n``: receiver noise, Gaussian, thermal plus signal-dependent shot noise:
  ``Var[n] = sigma_th^2 (1 + zeta * h s)`` where ``zeta`` is the shot-to-thermal variance
  ratio at unit received intensity (``zeta = 0``: thermal noise only).

SNR definition (electrical, at the transmitter, responsivity and mean optical power
normalised to 1): ``SNR = 1 / sigma_th^2``, so ``sigma_th^2 = 10^(-snr_db / 10)``. The
received electrical SNR of a row is ``SNR * h^2``; water type, distance, turbulence and
pointing therefore shift the curves, as in a link budget.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from uwbench.channels.base import SNR, Channel, gaussian, noise_variance, normalize_power


def beer_lambert(c_per_m: float, distance_m: float) -> float:
    return math.exp(-c_per_m * distance_m)


def lognormal_sigma2(scintillation_index: float) -> float:
    """Log-amplitude variance ``s2`` giving ``Var[exp(2X)] = SI`` with ``E = 1``."""
    return math.log1p(scintillation_index) / 4


def gamma_gamma_si(alpha: float, beta: float) -> float:
    return 1 / alpha + 1 / beta + 1 / (alpha * beta)


def sample_turbulence(cfg: dict[str, Any] | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if not cfg:
        return np.ones(n)
    if cfg["model"] == "lognormal":
        s2 = lognormal_sigma2(cfg["scintillation_index"])
        return np.exp(2 * rng.normal(-s2, math.sqrt(s2), n))
    if cfg["model"] == "gamma_gamma":
        a, b = cfg["alpha"], cfg["beta"]
        return rng.gamma(a, 1 / a, n) * rng.gamma(b, 1 / b, n)
    raise ValueError(f"unknown turbulence model {cfg['model']!r}")


def pointing_parameters(aperture_radius_m: float, beam_radius_m: float) -> tuple[float, float]:
    """Return ``(A0, w_zeq)`` of the Farid-Hranilovic model."""
    v = math.sqrt(math.pi) * aperture_radius_m / (math.sqrt(2) * beam_radius_m)
    a0 = math.erf(v) ** 2
    w_zeq2 = beam_radius_m**2 * math.sqrt(math.pi) * math.erf(v) / (2 * v * math.exp(-v * v))
    return a0, math.sqrt(w_zeq2)


def sample_pointing(cfg: dict[str, Any] | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if not cfg:
        return np.ones(n)
    a0, w_zeq = pointing_parameters(cfg["aperture_radius_m"], cfg["beam_radius_m"])
    r = rng.rayleigh(cfg["jitter_std_m"], n)
    return a0 * np.exp(-2 * r**2 / w_zeq**2)


def bias_and_clip(x: torch.Tensor, modulation_index: float) -> torch.Tensor:
    """Map real analog symbols to intensities: ``max(0, 1 + m x / rms(x))`` (DC bias 1)."""
    return torch.clamp(1 + modulation_index * normalize_power(x), min=0)


def remove_bias(r: torch.Tensor, modulation_index: float) -> torch.Tensor:
    """Inverse of :func:`bias_and_clip` away from clipping."""
    return (r - 1) / modulation_index


class UWOC(Channel):
    """IM/DD optical channel; see the module docstring for the model and SNR definition."""

    def __init__(self, attenuation_c: float, distance_m: float,
                 turbulence: dict[str, Any] | None = None,
                 pointing: dict[str, Any] | None = None, shot_to_thermal: float = 0.0):
        self.h_l = beer_lambert(attenuation_c, distance_m)
        self.turbulence, self.pointing, self.zeta = turbulence, pointing, shot_to_thermal

    def gains(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Block-fading gains ``h = h_l h_a h_p`` for ``n`` rows."""
        return self.h_l * sample_turbulence(self.turbulence, n, rng) \
            * sample_pointing(self.pointing, n, rng)

    def forward(self, s: torch.Tensor, snr_db: SNR, rng: np.random.Generator) -> torch.Tensor:
        return self.forward_csi(s, snr_db, rng)[0]

    def forward_csi(self, s: torch.Tensor, snr_db: SNR, rng: np.random.Generator):
        """As :meth:`Channel.forward_csi`. The reported variance uses the mean intensity
        (``sigma_th^2 (1 + zeta h) / h^2``): exact for ``zeta = 0`` (all scenarios),
        approximate for the signal-dependent shot noise."""
        if s.is_complex() or (s < 0).any():
            raise ValueError("UWOC expects real non-negative intensities")
        s = s / s.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        h = torch.as_tensor(self.gains(s.shape[0], rng), dtype=s.dtype, device=s.device)
        h = h.reshape(-1, *([1] * (s.ndim - 1)))
        y = h * s
        if snr_db is None:
            return y / h, torch.zeros_like(s)
        var_th = noise_variance(snr_db, s)
        var = var_th * (1 + self.zeta * y.detach().clamp_min(0))
        y = y + var.sqrt() * gaussian(tuple(s.shape), rng, s)
        return y / h, (var_th * (1 + self.zeta * h) / h**2).expand(s.shape)
