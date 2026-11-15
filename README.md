# underwater-semcom-benchmark

Reference implementation of the evaluation protocol proposed in the survey
*Enabling Intelligence in the Deep: A Comprehensive Survey of Semantic Communications
for Underwater Wireless Systems* (Proceedings of the IEEE, under review).

> **Status: pre-release (phase 3 of 7).** The package skeleton, the shared conventions,
> the dataset registry with its download and verification commands, and the
> deduplication and frozen splits exist. This README lists only what is implemented and
> tested. Every other component of the protocol (channels, metrics, separation baseline,
> evaluation engine) will be listed here once it exists.

## Installation

Python ≥ 3.11.

```bash
git clone <repository-url>
cd underwater-semcom-benchmark
pip install -e ".[dev]"
pytest
```

For a CPU-only machine, install PyTorch first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Implemented

| Component | Module | Tested |
| --- | --- | --- |
| Global seeding (Python, NumPy, PyTorch) | `uwbench/utils/seeding.py` | yes |
| Image-batch convention check | `uwbench/utils/conventions.py` | yes |
| `uwbench --version` | `uwbench/cli.py` | yes |
| Dataset registry: UIEB, LSUI, EUVP, SUIM, TrashCan, UICD (see [DATASETS.md](DATASETS.md)) | `uwbench/data/registry.yaml` | schema |
| `uwbench download <dataset>`: Google Drive download, or manual procedure | `uwbench/data/download.py` | yes (offline) |
| `uwbench verify <dataset>`: checksums, file counts, per-file manifest | `uwbench/data/verify.py` | yes |
| `uwbench dedup`: pixel SHA-256 and pHash duplicates within and across datasets | `uwbench/data/dedup.py` | yes |
| `uwbench split`: frozen, leak-checked train/val/test and cross-dataset splits (see [PROTOCOL.md](PROTOCOL.md)) | `uwbench/data/splits.py`, `splits/` | yes |

## Conventions

- **Images:** `torch.float32`, shape `(B, C, H, W)`, values in `[0, 1]`.
- **Channel symbols:** complex baseband for acoustic and AWGN channels; real,
  non-negative intensities for optical (IM/DD) channels.
- **Randomness:** explicit seeds everywhere (`uwbench.utils.seed_everything`, or a
  `numpy.random.Generator` passed as argument).

## Data policy

This repository never contains images or files from third-party datasets, only
scripts, file-identifier lists, checksums and metadata. Downloaded datasets go into
`datasets/`, which git ignores. Each dataset keeps its own license.

## License

Code: MIT (see `LICENSE`). Datasets: their own licenses.
