"""Source codecs of the separation baseline: JPEG and JPEG 2000 through Pillow.

JPEG uses libjpeg (``quality`` 1-95); JPEG 2000 uses OpenJPEG through Pillow
(irreversible 9/7 wavelet, one quality layer, ``quality_mode="rates"`` with the
compression ratio as parameter). BPG is not included in version 1.0 (no ``bpgenc``
available to test it).

Rate targeting (:func:`encode_at_bpp`): bisection on the codec parameter for the best
quality whose **actual** file size does not exceed the target, so that the codestream
always fits the channel budget. The achieved bpp (headers included) is returned and must
be the one reported. Both codecs tolerate trailing zero padding after the codestream
(checked on UIEB), so no length side information is needed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

CODECS = ("jpeg", "jpeg2000")


@dataclass(frozen=True)
class Encoded:
    data: bytes
    bpp: float
    param: float  # JPEG quality or JPEG 2000 compression ratio


def encode(img: np.ndarray, codec: str, param: float) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 image."""
    buf = io.BytesIO()
    im = Image.fromarray(img)
    if codec == "jpeg":
        im.save(buf, "JPEG", quality=int(param))
    elif codec == "jpeg2000":
        im.save(buf, "JPEG2000", quality_mode="rates", quality_layers=[float(param)],
                irreversible=True)
    else:
        raise ValueError(f"unknown codec {codec!r}")
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    """Decode to ``(H, W, 3)`` uint8; raises on a corrupt codestream."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def encode_at_bpp(img: np.ndarray, codec: str, target_bpp: float) -> Encoded | None:
    """Best-quality encoding with ``bpp <= target_bpp``; ``None`` if even the lowest
    quality does not fit."""
    pixels = img.shape[0] * img.shape[1]

    def bpp(data: bytes) -> float:
        return 8 * len(data) / pixels

    if codec == "jpeg":  # integer quality, larger is better
        lo, hi, best = 1, 95, None
        while lo <= hi:
            q = (lo + hi) // 2
            data = encode(img, codec, q)
            if bpp(data) <= target_bpp:
                best, lo = Encoded(data, bpp(data), q), q + 1
            else:
                hi = q - 1
        return best
    # JPEG 2000: compression ratio r, smaller is better. OpenJPEG's rate control gives
    # bpp ~ 8 C / r, so the bisection on log(r) starts from a bracket around that guess
    # (widened while it does not bracket the target) and stops at 0.5 % precision on r:
    # ~9 encodings instead of ~30 from the full range.
    floor, ceil, step = 0.0, np.log(10_000.0), 0.4
    guess = np.log(8 * img.shape[2] / max(target_bpp, 1e-9))
    lo, hi = max(floor, guess - step), min(ceil, guess + step)
    cache: dict[float, Encoded] = {}

    def at(log_r: float) -> Encoded:
        if log_r not in cache:
            data = encode(img, codec, np.exp(log_r))
            cache[log_r] = Encoded(data, bpp(data), float(np.exp(log_r)))
        return cache[log_r]

    while at(hi).bpp > target_bpp:  # hi must fit the budget
        if hi >= ceil:
            return None
        lo, hi = hi, min(ceil, hi + step)
    while lo > floor and at(lo).bpp <= target_bpp:  # lo must not fit (or be ratio 1)
        hi, lo = lo, max(floor, lo - step)
    if at(lo).bpp <= target_bpp:
        return at(lo)
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if at(mid).bpp <= target_bpp else (mid, hi)
    return at(hi)
