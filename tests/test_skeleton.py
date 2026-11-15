import numpy as np
import pytest
import torch

from uwbench import __version__
from uwbench.cli import main
from uwbench.utils import check_image_batch, seed_everything


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_seed_everything_is_reproducible():
    rng_a = seed_everything(123)
    a = (rng_a.standard_normal(4), torch.randn(4), np.random.rand(4))  # noqa: NPY002
    rng_b = seed_everything(123)
    b = (rng_b.standard_normal(4), torch.randn(4), np.random.rand(4))  # noqa: NPY002
    np.testing.assert_array_equal(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_seed_everything_rejects_negative():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_check_image_batch_accepts_valid():
    check_image_batch(torch.rand(2, 3, 8, 8))


@pytest.mark.parametrize(
    "x",
    [
        torch.rand(3, 8, 8),  # missing batch dim
        torch.rand(2, 3, 8, 8, dtype=torch.float64),  # wrong dtype
        torch.full((1, 3, 4, 4), 1.5),  # out of range
        torch.full((1, 3, 4, 4), -0.1),
    ],
)
def test_check_image_batch_rejects_invalid(x):
    with pytest.raises(ValueError):
        check_image_batch(x)
