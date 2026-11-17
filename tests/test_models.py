import csv

import numpy as np
import pytest
import torch
from PIL import Image

from uwbench.channels.awgn import AWGN
from uwbench.channels.uwoc import UWOC
from uwbench.models import load_model
from uwbench.models.adjscc import ADJSCC
from uwbench.models.base import bandwidth_ratio, latent_channels, pad_to, transmit
from uwbench.models.deepjscc import DeepJSCC
from uwbench.models.layers import GDN
from uwbench.models.witt import WITT

torch.set_num_threads(2)
MODELS = [DeepJSCC, ADJSCC, WITT]
CHANNELS = {"complex": AWGN(), "real": UWOC(0.15, 5.0)}


def snr(*values):
    return torch.tensor(values, dtype=torch.float32)


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("symbols", ["complex", "real"])
def test_model_shapes_ratio_and_gradient(cls, symbols):
    model = cls(1 / 6, symbols)
    x = torch.rand(2, 3, 64, 80)
    z = model.encode(pad_to(x, model.downsample), snr(10, 10))
    assert not z.is_complex() and z.shape[-2:] == (64 // cls.downsample, 80 // cls.downsample)
    assert bandwidth_ratio(model, z.shape[1]) == pytest.approx(1 / 6)
    y = transmit(model, x, CHANNELS[symbols], snr(5, 15), np.random.default_rng(0))
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    torch.nn.functional.mse_loss(y, x).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("cls", MODELS)
def test_odd_sizes_are_padded_and_cropped(cls):
    model = cls(1 / 6, "complex")
    x = torch.rand(1, 3, 37, 53)
    y = transmit(model, x, AWGN(), snr(10), np.random.default_rng(0))
    assert y.shape == x.shape


@pytest.mark.parametrize("cls", [ADJSCC, WITT])
def test_snr_conditioning_changes_the_latent(cls):
    model = cls(1 / 6, "complex")
    x = torch.rand(1, 3, 32, 32)
    assert not torch.allclose(model.encode(x, snr(0)), model.encode(x, snr(20)))


def test_latent_channels_must_be_integer():
    assert latent_channels(1 / 6, 4, "complex") == 16 and latent_channels(1 / 12, 4, "real") == 4
    with pytest.raises(ValueError):
        latent_channels(0.1, 4, "complex")


def test_gdn_and_igdn_formulas():
    beta, gamma = torch.tensor([1.0, 4.0]), torch.tensor([[1.0, 0.25], [0.0, 1.0]])
    x = torch.tensor([3.0, 4.0])
    norm = torch.sqrt(beta + 1e-6 + gamma @ x**2)
    for inverse, expected in ((False, x / norm), (True, x * norm)):
        layer = GDN(2, inverse=inverse)
        with torch.no_grad():  # parameters are stored as square roots
            layer.beta.copy_(beta.sqrt())
            layer.gamma.copy_(gamma.sqrt())
        torch.testing.assert_close(layer(x.reshape(1, 2, 1, 1)).flatten(), expected)


def test_load_model_rejects_foreign_classes():
    with pytest.raises(TypeError):
        load_model({"model": "torch.nn:Identity", "params": {}}, "complex")


# ---------------------------------------------------------------- training and evaluation

def smooth_images(n, h, w, seed=0):
    rng = np.random.default_rng(seed)
    return [np.asarray(Image.fromarray(rng.integers(0, 256, (4, 5, 3), dtype=np.uint8))
                       .resize((w, h), Image.BICUBIC)) for _ in range(n)]


def test_training_then_evaluation_through_the_engine(tmp_path):
    pytest.importorskip("sionna")  # imported by the engine module
    from uwbench.eval.runner import evaluate
    from uwbench.models.train import train

    cfg = {"name": "deepjscc_test", "type": "learned",
           "model": "uwbench.models.deepjscc:DeepJSCC", "params": {"bandwidth_ratio": 1 / 6},
           "train": {"crop": 32, "batch_size": 4, "lr": 1e-3, "epochs": 2, "seed": 0,
                     "workers": 2}}
    scenario = {"name": "awgn", "channel": "awgn", "snr_db": {"start": 0, "stop": 20, "step": 10}}
    best = train(cfg, scenario, smooth_images(8, 48, 48), smooth_images(2, 48, 48, 1), tmp_path)
    assert best.exists() and (best.parent / "last.pt").exists()
    with open(best.parent / "log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(np.isfinite(float(r["val_psnr_db"])) for r in rows)

    # resume: extending to 3 epochs continues after epoch 2; two loader workers exercise
    # the per-worker crop seeding; another configuration is refused
    best = train(cfg, scenario, smooth_images(8, 48, 48), smooth_images(2, 48, 48, 1), tmp_path,
                 epochs=3)
    with open(best.parent / "log.csv", encoding="utf-8") as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["1", "2", "3"]
    other = dict(cfg, train=dict(cfg["train"], lr=1e-4))
    with pytest.raises(ValueError, match="another configuration"):
        train(other, scenario, smooth_images(8, 48, 48), smooth_images(2, 48, 48, 1), tmp_path)

    system = dict(cfg, checkpoint=str(best), device="cpu")
    images = [(f"img{i}", im) for i, im in enumerate(smooth_images(2, 176, 168, 2))]
    summary, records = evaluate(system, scenario, images, seed=0)
    assert {r["scheme"] for r in records} == {"deepjscc_test"} and len(records) == 2 * 3
    assert all(r["decoded"] == 1 and r["bpp"] == 0 for r in records)
    used = records[0]["bandwidth_ratio_used"]
    assert used == pytest.approx(1 / 6, rel=0.05)  # padding to a multiple of 4 included
    by_snr = {e["snr_db"]: e["metrics"]["psnr_all"]["mean"] for e in summary["results"]}
    assert by_snr[20.0] > by_snr[0.0]  # graceful degradation with SNR
