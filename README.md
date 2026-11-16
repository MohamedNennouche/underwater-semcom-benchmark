# underwater-semcom-benchmark

Reference implementation of the evaluation protocol proposed in the survey
*Enabling Intelligence in the Deep: A Comprehensive Survey of Semantic Communications
for Underwater Wireless Systems* (Proceedings of the IEEE, under review).

The benchmark fixes what existing works leave incompatible. It provides:
- reference channels (acoustic and optical);
- frozen, deduplicated dataset splits;
- objective-dependent metrics;
- a separate source-channel coding baseline;
- one result format and report.

The full specification is in [PROTOCOL.md](PROTOCOL.md). This README lists only what is
implemented and tested.

## Installation

Python ≥ 3.11.

```bash
git clone <repository-url>
cd underwater-semcom-benchmark
pip install -e ".[dev,baseline]"
pytest
```

The `baseline` extra installs NVIDIA Sionna (`sionna-no-rt`) for the LDPC codes of the
separation baseline.

**PyTorch build.** Install PyTorch before the package:
- CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- With an NVIDIA GPU (recommended for full evaluations: LDPC decoding is about 30×
  faster): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`.

## Quick start (a few minutes)

```bash
uwbench download uieb                         # prints the license, then stops
uwbench download uieb --accept-license        # see the note below
uwbench verify uieb
uwbench evaluate --system configs/baselines/jpeg2000_ldpc.yaml \
                 --scenario configs/scenarios/uwoc_clear.yaml \
                 --dataset uieb --split test --limit 5
uwbench report results/jpeg2000_ldpc_uwoc_clear_uieb_test.json --out results/report
```

**Note on UIEB.** The UIEB archives are password-protected RAR files. The passwords are
published on the official page and stored in the registry. Automatic extraction calls
7-Zip or `unrar`. It has **not been tested end to end yet**, because the maintainers
extracted UIEB by hand. If it fails:
1. extract the three archives into `datasets/uieb/extracted/<archive name>/` with the
   passwords given in `uwbench/data/registry.yaml`;
2. run `uwbench verify uieb`, which checks every file against the manifest.

This writes:
- `results/jpeg2000_ldpc_uwoc_clear_uieb_test.json`, the summary with configuration and
  provenance;
- the matching `.csv`, with one row per image and SNR;
- a figure and Markdown/LaTeX tables in `results/report/`.

## Reproducing the separation baseline on UIEB

The splits are versioned in `splits/`, and their SHA-256 values are listed in
PROTOCOL.md, so `uwbench dedup` and `uwbench split` do not need to be re-run.

```bash
uwbench download uieb --accept-license && uwbench verify uieb
for sc in uac_modeled_short uwoc_clear; do
  for sy in jpeg2000_ldpc ideal_separation; do
    uwbench evaluate --system configs/baselines/$sy.yaml \
                     --scenario configs/scenarios/$sc.yaml \
                     --dataset uieb --split test --snr-step 1 --workers 3 --out results/uieb
  done
done
uwbench report results/uieb/*.json --out results/uieb/report
```

`--workers` sets the number of parallel image processes; each takes about 0.6 GB of RAM.
Results do not depend on it.

On a laptop (RTX 500, 3 workers), these four runs over the 89 test images and 1 dB
grids take about 70 minutes.

`--snr-step 1` refines the scenario grids, which use 5 dB steps; it is recorded in the
result JSON.

## Evaluating your own model

Any analog (learned) scheme can be evaluated with the same channels, data, metrics and
report as the baselines. Three steps are needed.

**1. Implement the interface** (`uwbench.models.base.JSCCModel`, a `torch.nn.Module`):

```python
import torch
from uwbench.models.base import JSCCModel, latent_channels

class MyModel(JSCCModel):
    downsample = 16                      # image side / latent side

    def __init__(self, bandwidth_ratio=1/6, symbols="complex"):
        super().__init__()
        self.symbols = symbols           # "complex" (AWGN, acoustic) or "real" (optical)
        c = latent_channels(bandwidth_ratio, self.downsample, symbols)
        ...                              # your layers, ending in c latent channels

    def encode(self, x, snr_db):         # x: (B, 3, H, W) in [0, 1]; snr_db: (B,)
        return z                         # real tensor (B, c, H/16, W/16)

    def decode(self, z_hat, snr_db):
        return x_hat                     # (B, 3, H, W) in [0, 1]
```

The benchmark does the rest, identically for every model (`uwbench.models.base.transmit`):
- it pads the image to a multiple of `downsample`, then crops it back;
- it maps the latent to channel symbols: complex pairs at unit power, or a DC bias with
  clipping for IM/DD;
- it applies the scenario's channel, power normalisation included.

The latent size sets the bandwidth ratio: k/n = c·h·w / (2·3·H·W) for complex symbols,
without the factor 2 for real ones.

**2. Describe the model in YAML**, like `configs/models/deepjscc.yaml`:

```yaml
name: my_model
type: learned
model: my_package.my_module:MyModel   # any importable class
params: {bandwidth_ratio: 0.1666667}
checkpoint: checkpoints/my_model_{scenario}/best.pt   # weights under the "model" key
```

**3. Train** (optional: you can use your own training loop and save
`{"model": state_dict}`), **then evaluate and report:**

```bash
uwbench train --model configs/models/my_model.yaml --scenario configs/scenarios/uwoc_clear.yaml
uwbench evaluate --system configs/models/my_model.yaml --scenario configs/scenarios/uwoc_clear.yaml --dataset uieb --split test
uwbench report results/*.json --out results/report     # your curves next to the baselines
```

- `--checkpoint` evaluates weights trained on another scenario, for example trained on
  `uwoc_clear` and tested on `uwoc_turbid`.
- Results follow the same JSON schema as the baselines (PROTOCOL.md §6), so they can be
  compared directly.
- Checkpoints are loaded with `weights_only=True`.

**Reference models.** Three are provided in `uwbench/models/`, with configurations in
`configs/models/`:
- **ADJSCC** (Xu et al., 2022): a direct PyTorch translation of the official TensorFlow
  code, which is MIT-licensed.
- **WITT** (Yang et al., 2023): the official PyTorch architecture, re-written. Its
  repository has no license, so the structure and hyperparameters were followed but no
  code was copied. The result has 28.3 M parameters, against 28.2 M in the paper.
- **DeepJSCC** (Bourtsoulatze et al., 2019): the paper's architecture, with the open
  choices taken from `chunbaobao/Deep-JSCC-PyTorch`, cited as inspiration.

The remaining deviations are listed in PROTOCOL.md §7. **No trained weights are
published yet.**

## Commands

| Command | Purpose |
| --- | --- |
| `uwbench download <dataset> [--accept-license]` | Download from the official source, or print the manual procedure; checks SHA-256 and extracts |
| `uwbench verify <dataset>` | Archive checksums, file counts, empty files, per-file manifest |
| `uwbench dedup` | Pixel SHA-256 and pHash duplicates within and across datasets |
| `uwbench split` | Frozen, leak-checked train/val/test and cross-dataset splits |
| `uwbench evaluate --system … --scenario … --dataset … [--split] [--realizations] [--seed] [--snr-step] [--workers] [--limit]` | Run a system over a scenario's SNR grid |
| `uwbench report <results.json …>` | Figures (PSNR, MS-SSIM, LPIPS with 95 % CI, failure rate) and Markdown/LaTeX tables |
| `uwbench train --model … --scenario … [--device] [--epochs]` | Train a learned model on one scenario (checkpoints and CSV log) |

## Implemented

| Component | Module | Tested |
| --- | --- | --- |
| Global seeding, image-batch convention check | `uwbench/utils/` | yes |
| Dataset registry: UIEB, LSUI, EUVP, SUIM, TrashCan, UICD and the Watermark channel archive (see [DATASETS.md](DATASETS.md)) | `uwbench/data/registry.yaml` | schema |
| Download (Google Drive, or manual procedure), verification, manifests | `uwbench/data/download.py`, `verify.py` | yes (offline) |
| Deduplication (pixel SHA-256, pHash ≤ 6) and frozen splits, with official test sets intact and TrashCan split by video | `uwbench/data/dedup.py`, `splits.py`, `splits/` | yes |
| Reference channels | `uwbench/channels/`, `configs/scenarios/` | yes |
| Metrics: PSNR (100 dB ceiling), MS-SSIM, LPIPS (AlexNet); bpp, k/n and their conversion; parameters, FLOPs, latency with platform | `uwbench/metrics/` | yes |
| Separation baseline | `uwbench/baselines/`, `configs/baselines/` | yes |
| Evaluation engine and result schema 1.0 | `uwbench/eval/runner.py` | yes (CI smoke test) |
| Report: figures and Markdown/LaTeX tables | `uwbench/eval/report.py` | yes (CI smoke test) |
| Model interface, `learned` systems, training (`uwbench train`), reference models DeepJSCC, ADJSCC, WITT (untrained) | `uwbench/models/`, `configs/models/` | yes (shapes, gradients, mini training on CPU) |

The reference channels are:
- complex AWGN;
- modelled acoustic OFDM: Thorp absorption, Stojanovic noise, image-method multipath,
  residual CFO;
- replay of the measured Watermark channels: an exact port of `replayfilter.m`, with
  OFDM;
- optical IM/DD: Beer–Lambert with Table III, log-normal and Gamma-Gamma turbulence,
  pointing errors, thermal and shot noise.

The separation baseline consists of:
- JPEG or JPEG 2000 at a target bpp;
- CRC-32 packets;
- 5G NR LDPC codes (Sionna, CPU or GPU);
- BPSK, QPSK and 16-QAM, or OOK and 4-PAM, with exact soft demapping;
- the failure policy and the ideal separation bound.

**Not included in this release:**
- the SUIM mIoU task metric (no PyTorch reference segmenter);
- BPG;
- pilot-based channel estimation;
- wideband Doppler;
- trained weights of the reference models.

## Conventions

- **Images:** `torch.float32`, shape `(B, C, H, W)`, values in `[0, 1]`.
- **Channel symbols:** complex baseband for acoustic and AWGN channels; real,
  non-negative intensities for optical (IM/DD) channels.
- **Randomness:** explicit seeds everywhere, recorded with every result.

## Data policy

This repository never contains images or files from third-party datasets. It holds only
scripts, file-identifier lists, checksums and metadata. Downloaded datasets go into
`datasets/`, which git ignores. Each dataset keeps its own license, and you must respect
it (see [DATASETS.md](DATASETS.md)).

## Citation

See [CITATION.cff](CITATION.cff).

## License

Code: MIT (see `LICENSE`). Datasets: their own licenses.
