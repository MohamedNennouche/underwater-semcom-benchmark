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
    # JPEG 2000: compression ratio, smaller is better; bisection on log(ratio)
    lo, hi = np.log(1.0), np.log(10_000.0)
    worst = encode(img, codec, np.exp(hi))
    if bpp(worst) > target_bpp:
        return None
    best = Encoded(worst, bpp(worst), float(np.exp(hi)))
    for _ in range(30):
        mid = (lo + hi) / 2
        data = encode(img, codec, np.exp(mid))
        if bpp(data) <= target_bpp:
            best, hi = Encoded(data, bpp(data), float(np.exp(mid))), mid
        else:
            lo = mid
    return best
