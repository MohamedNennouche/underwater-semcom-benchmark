"""Rate metrics and the conversion between digital and analog schemes.

* Digital schemes: bits per pixel ``bpp = n_bits / (H W)`` (pixel = spatial location, all
  colour channels included), measured on the actual codestream.
* Analog schemes: bandwidth ratio ``rho = k / n``, with ``k`` the number of channel uses
  per image and ``n = C H W`` the number of source components (pixel components), as in
  Bourtsoulatze et al., IEEE TCCN 2019 (survey ref. [55]). A channel use is one complex
  symbol for complex baseband channels (AWGN, acoustic) and one real symbol for the
  optical IM/DD channel.
* Conversion: a digital scheme that carries ``eta`` information bits per channel use
  (code rate x bits per modulation symbol) needs ``k = bpp H W / eta`` channel uses, so
  ``rho = bpp / (C eta)`` and ``bpp = rho C eta``.
"""

from __future__ import annotations


def bits_per_pixel(n_bits: int, height: int, width: int) -> float:
    return n_bits / (height * width)


def bandwidth_ratio(n_channel_uses: int, channels: int, height: int, width: int) -> float:
    return n_channel_uses / (channels * height * width)


def bpp_to_bandwidth_ratio(bpp: float, channels: int, bits_per_channel_use: float) -> float:
    return bpp / (channels * bits_per_channel_use)


def bandwidth_ratio_to_bpp(rho: float, channels: int, bits_per_channel_use: float) -> float:
    return rho * channels * bits_per_channel_use
