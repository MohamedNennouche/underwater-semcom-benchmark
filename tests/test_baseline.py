import math

import numpy as np
import pytest
import torch
from PIL import Image

from uwbench.baselines.channel_coding import (
    CONSTELLATIONS,
    demap_llr,
    make_packets,
    modulate,
    read_packets,
)
from uwbench.baselines.source_codecs import decode, encode, encode_at_bpp
from uwbench.channels.awgn import AWGN
from uwbench.channels.uwoc import UWOC

sionna = pytest.importorskip("sionna")
from uwbench.baselines.channel_coding import LDPC  # noqa: E402
from uwbench.baselines.separation import (  # noqa: E402
    FALLBACK_VALUE,
    SeparationScheme,
    capacity_per_use,
    ideal_separation,
)


def q_function(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def make_image(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8))
    return np.asarray(small.resize((w, h), Image.BICUBIC))


def random_bits(n, seed=0):
    return torch.as_tensor(np.random.default_rng(seed).integers(0, 2, n))


# ---------------------------------------------------------------- modulation

@pytest.mark.parametrize("name", list(CONSTELLATIONS))
def test_noiseless_demapping_recovers_bits(name):
    bits = random_bits(CONSTELLATIONS[name].bits_per_symbol * 1000)
    sym = modulate(bits, name)
    llr = demap_llr(sym, torch.full(sym.shape, 1e-3), name)
    assert torch.equal((llr > 0).long(), bits)


@pytest.mark.parametrize("name", list(CONSTELLATIONS))
def test_constellations_are_normalised(name):
    p = CONSTELLATIONS[name].points
    if p.is_complex():
        assert p.abs().square().mean().item() == pytest.approx(1.0)
    else:
        assert p.mean().item() == pytest.approx(1.0) and (p >= 0).all()


def test_bpsk_llr_matches_closed_form():
    y = torch.tensor([0.3 + 0.2j, -1.1 - 0.5j], dtype=torch.complex64)
    var = torch.tensor([0.5, 2.0])
    torch.testing.assert_close(demap_llr(y, var, "bpsk"), 4 * y.real / var)


@pytest.mark.parametrize("name,snr_db,theory", [
    ("bpsk", 6.79, lambda s: q_function(math.sqrt(2 * s))),  # Eb/N0 = SNR
    ("qpsk", 9.80, lambda s: q_function(math.sqrt(s))),  # Eb/N0 = SNR / 2
])
def test_uncoded_ber_matches_theory(name, snr_db, theory):
    bits = random_bits(2_000_000, seed=1)
    sym = modulate(bits, name).reshape(1, -1)
    y, nv = AWGN().forward_csi(sym, snr_db, np.random.default_rng(2))
    ber = ((demap_llr(y, nv, name) > 0).long().flatten() != bits).float().mean().item()
    expected = theory(10 ** (snr_db / 10))
    assert expected == pytest.approx(1e-3, rel=0.05)  # operating point near 1e-3
    assert ber == pytest.approx(expected, rel=0.10)


# ---------------------------------------------------------------- LDPC and packets

def coded_ber(code, snr_db, n_codewords=200, seed=0):
    u = random_bits(code.k * n_codewords, seed).reshape(n_codewords, code.k)
    sym = modulate(code.encode(u).reshape(1, -1), "qpsk")
    y, nv = AWGN().forward_csi(sym, snr_db, np.random.default_rng(seed))
    llr = demap_llr(y, nv, "qpsk").reshape(n_codewords, code.n)
    return (code.decode(llr) != u).float().mean().item()


def test_ldpc_waterfall():
    code = LDPC(k=1024, rate=0.5)  # QPSK, rate 1/2: Eb/N0 = SNR
    below, above = coded_ber(code, -1.0), coded_ber(code, 3.0)
    uncoded_at_3db = q_function(math.sqrt(10 ** 0.3))  # QPSK uncoded BER, ~2e-2
    assert below > 1e-2  # below threshold the code fails
    assert above == 0.0 and uncoded_at_3db > 1e-2  # 204,800 bits, no error


def test_packets_roundtrip_and_crc_detects_errors():
    data = bytes(range(256)) * 3
    bits = make_packets(data, 7, 1024)
    out, ok = read_packets(bits)
    assert ok.all() and out[:len(data)] == data and set(out[len(data):]) <= {0}
    bits[3, 100] ^= 1
    assert read_packets(bits)[1].tolist() == [True, True, True, False, True, True, True]
    with pytest.raises(ValueError):
        make_packets(data, 1, 1024)


# ---------------------------------------------------------------- source codecs

@pytest.mark.parametrize("codec", ["jpeg", "jpeg2000"])
def test_encode_at_bpp_respects_budget(codec):
    img = make_image(192, 256)
    for target in (0.4, 0.8):  # reachable by both codecs on this small smooth image
        enc = encode_at_bpp(img, codec, target)
        assert enc is not None and enc.bpp <= target and enc.bpp > 0.5 * target
        assert enc.bpp == 8 * len(enc.data) / (192 * 256)
    assert encode_at_bpp(img, codec, 1e-4) is None


def test_codecs_decode_with_trailing_padding():
    img = make_image()
    for codec, param in (("jpeg", 50), ("jpeg2000", 20)):
        data = encode(img, codec, param)
        assert np.array_equal(decode(data), decode(data + bytes(1000)))


# ---------------------------------------------------------------- full chain

@pytest.mark.parametrize("modulation,channel", [("qpsk", AWGN()), ("16qam", AWGN()),
                                                ("ook", UWOC(0.15, 5.0))])
def test_noiseless_chain_returns_the_transmitted_codestream(modulation, channel):
    img = make_image()
    scheme = SeparationScheme(modulation, 0.5)
    enc = scheme.encode_source(img, 1 / 6)
    res = scheme.transmit(img, enc, 1 / 6, channel, None, np.random.default_rng(0))
    assert res.decoded and res.failed_packets == 0
    assert np.array_equal(res.image, decode(enc.data))  # bit-exact codestream
    packets, target = scheme.budget(img.shape, 1 / 6)
    assert enc.bpp <= target and res.channel_uses <= 1 / 6 * img.size


def test_chain_fails_below_threshold_and_uses_fallback():
    img = make_image()
    scheme = SeparationScheme("qpsk", 0.5)
    enc = scheme.encode_source(img, 1 / 6)
    res = scheme.transmit(img, enc, 1 / 6, AWGN(), -5.0, np.random.default_rng(0))
    assert not res.decoded and res.failed_packets > 0
    assert (res.image == FALLBACK_VALUE).all()


def test_capacity_probe_matches_awgn_and_optical_formulas():
    snr_db = 10.0
    cap = capacity_per_use(AWGN(), 1000, snr_db, np.random.default_rng(0))
    assert cap == pytest.approx(math.log2(1 + 10.0), rel=1e-5)
    h = math.exp(-0.15 * 5.0)
    cap_o = capacity_per_use(UWOC(0.15, 5.0), 1000, snr_db, np.random.default_rng(0),
                             complex_symbols=False)
    assert cap_o == pytest.approx(0.5 * math.log2(1 + 10.0 * h**2), rel=1e-5)


def test_ideal_bound_is_at_least_as_good_as_the_real_chain():
    img = make_image(192, 256)
    ideal = ideal_separation(img, 1 / 6, AWGN(), 10.0, np.random.default_rng(0))
    scheme = SeparationScheme("qpsk", 0.5)
    _, target = scheme.budget(img.shape, 1 / 6)
    assert ideal.decoded and ideal.bpp > target
