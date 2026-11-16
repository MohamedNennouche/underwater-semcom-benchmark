"""Channel coding and modulation of the separation baseline.

* LDPC: 5G NR LDPC codes (3GPP TS 38.212, base graphs 1 and 2, rate matching) with
  belief-propagation decoding from NVIDIA Sionna (``sionna-no-rt`` >= 2.2, PyTorch,
  Apache-2.0), an optional dependency used by this module only (``pip install
  .[baseline]``). Chosen as a maintained, standard-compliant implementation; no decoder
  is re-implemented here. Default: ``k = 1024`` information bits, rate 1/2, 20 iterations
  (no early stopping in Sionna). Decoding dominates the run time of the baseline (~90 %
  on one CPU thread), so it can run on a GPU (``device="cuda"``).
* CRC: every codeword carries ``k - 32`` payload bits followed by their CRC-32
  (``zlib.crc32``, big-endian); a packet is accepted only if its CRC matches.
* Modulation, Gray-labelled, unit mean power (complex) or unit mean intensity (optical):
  BPSK, QPSK, 16-QAM (complex baseband); OOK {0, 2} and 4-PAM {0, 2/3, 4/3, 2}
  (non-negative intensities for IM/DD).
* Soft demapping: exact log-likelihood ratios, ``log p(b=1|y) / p(b=0|y)`` (Sionna's
  convention), by log-sum-exp over the constellation, with the per-symbol noise variance
  given by the channel (circular complex Gaussian, or real Gaussian for optical).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

CRC_BITS = 32
GRAY2 = [(0, 0), (0, 1), (1, 1), (1, 0)]  # Gray order of 4 amplitude levels, low to high


@dataclass(frozen=True)
class Constellation:
    points: torch.Tensor  # (M,) complex or real
    labels: torch.Tensor  # (M, m) bits of each point

    @property
    def bits_per_symbol(self) -> int:
        return self.labels.shape[1]

    @property
    def is_complex(self) -> bool:
        return self.points.is_complex()


def _qam16() -> Constellation:
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10)
    pts, labels = [], []
    for i, bi in enumerate(GRAY2):
        for q, bq in enumerate(GRAY2):
            pts.append(complex(levels[i], levels[q]))
            labels.append(bi + bq)
    return Constellation(torch.tensor(pts, dtype=torch.complex64), torch.tensor(labels))


CONSTELLATIONS: dict[str, Constellation] = {
    "bpsk": Constellation(torch.tensor([-1, 1], dtype=torch.complex64), torch.tensor([[0], [1]])),
    "qpsk": Constellation(
        torch.tensor([complex(i, q) for i in (-1, 1) for q in (-1, 1)],
                     dtype=torch.complex64) / np.sqrt(2),
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])),
    "16qam": _qam16(),
    "ook": Constellation(torch.tensor([0.0, 2.0]), torch.tensor([[0], [1]])),
    "pam4": Constellation(torch.tensor([0.0, 2 / 3, 4 / 3, 2.0]), torch.tensor(GRAY2)),
}


def modulate(bits: torch.Tensor, name: str) -> torch.Tensor:
    """Map ``(..., N m)`` bits to ``(..., N)`` symbols."""
    c = CONSTELLATIONS[name]
    m = c.bits_per_symbol
    weights = 2 ** torch.arange(m - 1, -1, -1)
    table = torch.empty_like(c.points)
    table[(c.labels * weights).sum(dim=1)] = c.points
    index = (bits.reshape(*bits.shape[:-1], -1, m).long() * weights).sum(dim=-1)
    return table[index]


def demap_llr(y: torch.Tensor, noise_var: torch.Tensor, name: str) -> torch.Tensor:
    """Exact LLRs ``log p(b=1)/p(b=0)``, shape ``(..., N m)``."""
    c = CONSTELLATIONS[name]
    var = noise_var.clamp_min(1e-12).unsqueeze(-1)
    d2 = (y.unsqueeze(-1) - c.points.to(y.device)).abs().square()
    logp = -d2 / var if c.is_complex else -d2 / (2 * var)
    one = c.labels.T.bool().to(y.device)  # (m, M)
    neg = torch.tensor(float("-inf"), device=y.device)
    lp = logp.unsqueeze(-2)  # (..., N, 1, M)
    llr = torch.logsumexp(torch.where(one, lp, neg), dim=-1) \
        - torch.logsumexp(torch.where(~one, lp, neg), dim=-1)
    return llr.reshape(*y.shape[:-1], -1)


class LDPC:
    """5G NR LDPC code from Sionna; see the module docstring."""

    def __init__(self, k: int = 1024, rate: float = 0.5, num_iter: int = 20,
                 device: str = "cpu"):
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder

        device = "cuda:0" if device == "cuda" else device  # Sionna wants an indexed device
        self.k, self.n, self.device = k, round(k / rate), device
        self._enc = LDPC5GEncoder(k=self.k, n=self.n, device=device)
        self._dec = LDPC5GDecoder(self._enc, num_iter=num_iter, hard_out=True, device=device)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Codewords on the CPU (inputs and outputs stay on the CPU whatever the device)."""
        return self._enc(u.float().to(self.device)).cpu()

    def decode(self, llr: torch.Tensor) -> torch.Tensor:
        return self._dec(llr.float().to(self.device)).cpu()


def make_packets(data: bytes, n_packets: int, k: int) -> np.ndarray:
    """Split ``data`` (zero-padded) into ``n_packets`` rows of ``k`` bits: payload + CRC-32."""
    if (k - CRC_BITS) % 8:
        raise ValueError("k - 32 must be a multiple of 8")
    size = (k - CRC_BITS) // 8
    if len(data) > n_packets * size:
        raise ValueError(f"{len(data)} bytes do not fit in {n_packets} packets of {size}")
    data = data.ljust(n_packets * size, b"\0")
    rows = []
    for i in range(n_packets):
        chunk = data[i * size:(i + 1) * size]
        rows.append(np.frombuffer(chunk + zlib.crc32(chunk).to_bytes(4, "big"), np.uint8))
    return np.unpackbits(np.stack(rows), axis=1) if rows else np.zeros((0, k), np.uint8)


def read_packets(bits: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Inverse of :func:`make_packets`: ``(payload bytes, per-packet CRC ok)``."""
    raw = np.packbits(bits.astype(np.uint8), axis=1)
    payload, ok = [], []
    for row in raw:
        chunk, crc = row[:-4].tobytes(), int.from_bytes(row[-4:].tobytes(), "big")
        payload.append(chunk)
        ok.append(zlib.crc32(chunk) == crc)
    return b"".join(payload), np.array(ok, dtype=bool)
