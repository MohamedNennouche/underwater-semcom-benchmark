"""Separate source-channel coding baseline and ideal separation bound.

Chain (:class:`SeparationScheme`), per image of size ``H x W x C``:

1. Budget: ``k_ch = floor(rho C H W)`` channel uses (``rho`` = bandwidth ratio, see
   :mod:`uwbench.metrics.rate`); with ``m`` bits per modulation symbol and an LDPC code
   ``(k, n)`` this gives ``N = floor(k_ch m / n)`` packets and a source budget of
   ``N (k - 32)`` bits (CRC-32 per packet).
2. Source coding at the best quality that fits the budget
   (:func:`~uwbench.baselines.source_codecs.encode_at_bpp`); the achieved bpp is kept.
3. Packets (payload + CRC), LDPC encoding, modulation, channel
   (:meth:`~uwbench.channels.base.Channel.forward_csi`: perfect CSI, per-symbol noise
   variance), exact soft demapping, belief-propagation decoding, CRC check.

Failure policy: if the budget cannot hold even the lowest-quality codestream, or if any
packet fails its CRC, or if the codestream does not decode, the image is **not decoded**.
Results keep this flag; the failure rate is reported separately from the fidelity of
decoded images. For curves that include failures, an undecoded image is replaced by a
uniform mid-grey image (value 128), documented as ``FALLBACK_VALUE``.

Ideal separation bound (:func:`ideal_separation`): the source rate equals the channel
capacity times the bandwidth ratio, ``bpp = rho C Cbar`` with ``Cbar`` the capacity per
channel use of the current realisation with perfect CSI and uniform power,
``mean_i log2(1 + 1 / v_i)`` for complex channels and ``mean_i 0.5 log2(1 + 1 / v_i)``
for the real optical channel, ``v_i`` being the per-use noise variance after equalisation
reported by the channel. The codestream is then assumed received without error. It is
not achievable (infinite block length; Gaussian inputs, which also ignore the IM/DD
non-negativity constraint; inter-carrier interference ignored on replayed channels) but
it bounds the results.

Adaptation: each (modulation, code rate) pair is a separate scheme; the "ideal
adaptation" curve takes, at each SNR, the best pair on the evaluation grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from uwbench.baselines.channel_coding import (
    CONSTELLATIONS,
    CRC_BITS,
    LDPC,
    demap_llr,
    make_packets,
    modulate,
    read_packets,
)
from uwbench.baselines.source_codecs import Encoded, decode, encode_at_bpp
from uwbench.channels.base import SNR, Channel

FALLBACK_VALUE = 128


@dataclass
class Result:
    image: np.ndarray  # (H, W, 3) uint8, fallback image when not decoded
    decoded: bool
    bpp: float  # achieved source rate (0 when nothing could be encoded)
    channel_uses: int
    packets: int = 0
    failed_packets: int = 0


def fallback(img: np.ndarray) -> np.ndarray:
    return np.full_like(img, FALLBACK_VALUE)


class SeparationScheme:
    """Source codec + LDPC + modulation; see the module docstring."""

    def __init__(self, modulation: str = "qpsk", code_rate: float = 0.5,
                 codec: str = "jpeg2000", k: int = 1024, num_iter: int = 20,
                 device: str = "cpu"):
        self.modulation, self.codec = modulation, codec
        self.m = CONSTELLATIONS[modulation].bits_per_symbol
        self.code = LDPC(k, code_rate, num_iter, device)
        if self.code.n % self.m:
            raise ValueError("codeword length must be a multiple of bits per symbol")

    @property
    def name(self) -> str:
        return f"{self.codec}+ldpc{self.code.k}/{self.code.n}+{self.modulation}"

    def budget(self, shape: tuple[int, ...], bandwidth_ratio: float) -> tuple[int, float]:
        """``(packets, target source bpp)`` for an ``(H, W, C)`` image."""
        h, w, c = shape
        uses = int(bandwidth_ratio * c * h * w)
        packets = uses * self.m // self.code.n
        return packets, packets * (self.code.k - CRC_BITS) / (h * w)

    def encode_source(self, img: np.ndarray, bandwidth_ratio: float) -> Encoded | None:
        packets, target = self.budget(img.shape, bandwidth_ratio)
        return encode_at_bpp(img, self.codec, target) if packets else None

    def transmit(self, img: np.ndarray, encoded: Encoded | None, bandwidth_ratio: float,
                 channel: Channel, snr_db: SNR, rng: np.random.Generator) -> Result:
        packets, _ = self.budget(img.shape, bandwidth_ratio)
        if encoded is None:
            return Result(fallback(img), False, 0.0, 0)
        u = torch.as_tensor(make_packets(encoded.data, packets, self.code.k))
        coded = self.code.encode(u)  # (N, n)
        symbols = modulate(coded.reshape(1, -1), self.modulation)
        y, nv = channel.forward_csi(symbols, snr_db, rng)
        llr = demap_llr(y, nv, self.modulation).reshape(packets, self.code.n)
        bits = self.code.decode(llr).round().to(torch.uint8).numpy()
        data, ok = read_packets(bits)
        uses = symbols.shape[1]
        if not ok.all():
            return Result(fallback(img), False, encoded.bpp, uses, packets, int((~ok).sum()))
        try:
            out = decode(data)
        except Exception:  # corrupt codestream despite matching CRCs
            return Result(fallback(img), False, encoded.bpp, uses, packets, 0)
        return Result(out, out.shape == img.shape, encoded.bpp, uses, packets, 0)


def capacity_per_use(channel: Channel, n_uses: int, snr_db: SNR, rng: np.random.Generator,
                     complex_symbols: bool = True) -> float:
    """Capacity per channel use (bits) of one realisation, from the noise variances that
    the channel reports for a unit-power probe (perfect CSI, uniform power)."""
    probe = torch.ones(1, n_uses, dtype=torch.complex64 if complex_symbols else torch.float64)
    _, nv = channel.forward_csi(probe, snr_db, rng)
    c = torch.log2(1 + 1 / nv.double().clamp_min(1e-30)).mean().item()
    return c if complex_symbols else c / 2


def ideal_separation(img: np.ndarray, bandwidth_ratio: float, channel: Channel, snr_db: SNR,
                     rng: np.random.Generator, codec: str = "jpeg2000",
                     complex_symbols: bool = True) -> Result:
    h, w, c = img.shape
    uses = int(bandwidth_ratio * c * h * w)
    cap = capacity_per_use(channel, uses, snr_db, rng, complex_symbols)
    encoded = encode_at_bpp(img, codec, bandwidth_ratio * c * cap)
    if encoded is None:
        return Result(fallback(img), False, 0.0, uses)
    return Result(decode(encoded.data), True, encoded.bpp, uses)
