# Evaluation protocol

This file specifies the reference evaluation protocol. It is the source for the
"Reference evaluation protocol" table of the survey (Section VII-C). Sections are added
as the corresponding components are implemented and tested. The separation baseline
and reporting are not specified here yet.

## 1. Datasets

See [DATASETS.md](DATASETS.md) and [`uwbench/data/registry.yaml`](uwbench/data/registry.yaml)
for sources, licenses, checksums and folder layouts. Version 1.0 covers optical images:

| Key | Task | Sample | Official split used |
| --- | --- | --- | --- |
| `uieb` | enhancement / fidelity | raw + reference (890 pairs; challenging-60 excluded) | none |
| `lsui` | enhancement / fidelity | input + reference (4279 pairs) | test (400) |
| `euvp` | enhancement / fidelity | input + reference, three paired training sets (11,435 pairs) | none |
| `suim` | semantic segmentation | image + mask (1635) | test (110) |
| `trashcan` | instance segmentation / detection | image, instance version (7212) | none (see 2.3) |
| `uicd` | captioning | image (3176), 5 captions each | none |

## 2. Deduplication and frozen splits

Commands: `uwbench dedup`, then `uwbench split`. Parameters:
[`configs/splits.yaml`](configs/splits.yaml).

### 2.1 Duplicate detection

- Each image gets two fingerprints:
  - the SHA-256 of its decoded RGB pixels;
  - its 64-bit DCT perceptual hash (pHash, `imagehash.phash`, hash size 8).
- Images are decoded by content, not by extension.
- Two images are duplicates when their pHash Hamming distance is **≤ 6**.
  - pHash thresholds against the median, so every hash has 32 set bits and distances are
    always even: 6 behaves like 7.
  - The threshold was confirmed by the maintainers on 2026-11-15 after visual inspection
    of the contact sheets (`results/dedup/`, not versioned because they show third-party
    images).
  - On those sheets, pairs at distance 6 between datasets of natural images are the
    same image (recoloured, enhanced or cropped). Pairs at distance 8 are mostly the
    same scene.
- For paired datasets, both the degraded and the reference image are hashed. Masks and
  resized copies are not.
- Two samples are linked when any of their images are duplicates. Duplicate groups are
  the connected components of these links, written to `splits/duplicates.csv`.
- TrashCan is compared with the other datasets only. Within TrashCan (dark,
  low-texture ROV footage), pHash at distance 6 linked different scenes into groups of
  up to 1245 frames across 104 videos.

Results on the six datasets (45,231 images):
- 3971 duplicate groups, of which 3365 span several datasets;
- largest group: 44 samples;
- samples in a group: UIEB 610, LSUI 2363, EUVP 2887, SUIM 1580, UICD 2173.

Most cross-dataset duplicates are EUVP–LSUI, SUIM–UICD and UIEB–UICD.

### 2.2 Split rules

1. **Official test splits are kept intact** (LSUI, SUIM).
   - Datasets with an official test split take 10% of their official training pool as
     validation.
   - Datasets without an official split are split 80/10/10.
2. **Group-aware draws.**
   - A unit is a set of samples of one dataset that share a duplicate group or a source
     video (TrashCan). A unit is never split across partitions.
   - Units are drawn in a `numpy.random.default_rng(0)` permutation of the units sorted
     by their first sample id.
   - A drawn (non-official) test split never contains a unit duplicated in another
     dataset.
3. **No train/test leakage, within or across datasets.** A train or validation sample
   is removed when its duplicate group contains a test sample of any dataset. The
   removed samples are listed in `splits/removed.csv`.
4. **Cross-dataset protocols** (`splits/cross/`).
   - Train on dataset A and test on all of dataset B.
   - The training and validation sets exclude every duplicate group present in B.

The leak check (`uwbench split`, and `tests/test_dedup_splits.py` on the versioned files)
verifies two things:
- no sample appears in two partitions of its dataset;
- no duplicate group is both in a train/val file and in a test file, for all datasets
  together and for each cross-dataset protocol.

### 2.3 TrashCan

The official train/val split of TrashCan is not used: 113 of the 115 source videos of
its validation set also appear in its training set (frame names
`vid_<video>_frame<k>`). The benchmark splits TrashCan 80/10/10 by video instead. Each
image keeps its official folder (`train/` or `val/`), which tells which official COCO
JSON holds its annotations.

### 2.4 Resulting splits

| Dataset | Train | Val | Test | Removed (copies of a test sample) |
| --- | ---: | ---: | ---: | ---: |
| UIEB | 712 | 88 | 89 | 1 |
| LSUI | 3425 | 378 | 400 (official) | 76 |
| EUVP | 8926 | 1092 | 1144 | 273 |
| SUIM | 1317 | 146 | 110 (official) | 62 |
| TrashCan | 5725 | 734 | 753 | 0 |
| UICD | 2403 | 299 | 318 | 156 |

| Cross-dataset protocol | Train | Val | Test |
| --- | ---: | ---: | ---: |
| `lsui_to_uieb` | 3425 | 378 | 889 (all of UIEB) |
| `euvp_to_uieb` | 8924 | 1090 | 889 (all of UIEB) |

Each split file has a `sample_id` column and one path column per role, relative to
`datasets/<key>/extracted/`. Cross-dataset files also have a `dataset` column.

### 2.5 Checksums of the versioned split files

Generated by `uwbench split`. `tests/test_dedup_splits.py` checks that this table
matches the files in `splits/`.

| File | SHA-256 |
| --- | --- |
| `cross/euvp_to_uieb_test.csv` | `4c2b395b32c98e1f5d12657c6312c1d9b3c7962e33f7319ca5353912ee9dd749` |
| `cross/euvp_to_uieb_train.csv` | `2005932a0fc68bc43f3baaa267e5c1439c787d291ba26256f15b2c3c45a41c94` |
| `cross/euvp_to_uieb_val.csv` | `becdfc5b795fec2dd681ae85d456cc652fafb4a5cb0736a045ba335c1b5c7291` |
| `cross/lsui_to_uieb_test.csv` | `4c2b395b32c98e1f5d12657c6312c1d9b3c7962e33f7319ca5353912ee9dd749` |
| `cross/lsui_to_uieb_train.csv` | `0fdb6ed2e78048f1fc8f5e22fe548be9e90700c7879474b2b2b7156497ee8dfb` |
| `cross/lsui_to_uieb_val.csv` | `6c61f6c51afe881c92cef01b47080a2aafc55eab072e771896af674729feff0f` |
| `duplicates.csv` | `3521519249ee5fe25fd891d86d11ded1c855180ef6add4099b1bde1a7bf78060` |
| `euvp_test.csv` | `7d58dc38bf346c4b1a15850b79aadf1b5caf8dd01d48cb208483e8c41c6700cc` |
| `euvp_train.csv` | `ac25e32fdf98bceffa9e130f9e60ba3334a12eaa490f76f9feb0a5bbddcd16f2` |
| `euvp_val.csv` | `e72519bb6ec645b364f200cd8278e340cf893d89d5c2856efa7d8f1a3700648e` |
| `lsui_test.csv` | `e109153539accb9526b01184a585da7b024cb28a8ad4e47bccd56709c1153b49` |
| `lsui_train.csv` | `f41a588ba8965aa6df92d1bbe47ec9f5dece30225e456b60450842a5f172927d` |
| `lsui_val.csv` | `4c06445861a23585bc409aff05bab4963456e8d5775f927a18a106bbca29e13a` |
| `removed.csv` | `c306325f2aad3afa8475c55448653645b4b965aad214386ea8748041b036bda9` |
| `suim_test.csv` | `e832d9f033d0426393f33b9309cf95678f3efde6a4cdcc2d9a5a0e34160f5d43` |
| `suim_train.csv` | `e80fdb4f040cd5f3fdfcb2a5739464b7b81b5cce0d790097abfa29d760d6450e` |
| `suim_val.csv` | `7d41dff4659ac4ef3b0ccc97194353d3b83c7d2d2e012bebb65abfb72a154b53` |
| `trashcan_test.csv` | `655bd3dbf3509d24c70af8151a02b926b8ecff9e99c4776fb6b93389de2a1bb3` |
| `trashcan_train.csv` | `66c0899f60d7c70b36fbaac55fb659be5c9980326e7c28cd807d04174b2364b0` |
| `trashcan_val.csv` | `b947d0121effaf359fbfcb10936c043b6f4d1bbbea2c87a18ccbc63674b33394` |
| `uicd_test.csv` | `b3afd10dac168b670b7c8ff408e8a0cb5da6bb4dd3c93dfe1a978373bb3d9362` |
| `uicd_train.csv` | `7695d5ee14020a34e4b6ce9e3284bb5c4d7704bbd888155de232f8decfdc41ed` |
| `uicd_val.csv` | `01720c6286ff0ea5f5964f640820e53e3544f5d27dd5750437694d5a1660bf79` |
| `uieb_test.csv` | `5a114ba0691874e3140b3f50ca0edffc3c45689837743e9f0cf265bb0caff6c5` |
| `uieb_train.csv` | `2b122792f42064dfbca70923972b4d2b2a71f7f3b09c1a638ed416011b49fadf` |
| `uieb_val.csv` | `1197a13afafce683f9653927734d3c0ff6eee396cf33b64dfe3402ee8d876bbb` |

## 3. Reference channels

Code: [`uwbench/channels/`](uwbench/channels/). Scenarios:
[`configs/scenarios/`](configs/scenarios/). Each channel implements
`forward(x, snr_db, rng) -> y` once in PyTorch. It is differentiable for training, and
calling it on NumPy arrays gives the evaluation mode. All randomness comes from an
explicit `numpy.random.Generator`.

**Conventions**
- Symbols are `(B, n)`, one row per image.
- Acoustic and AWGN channels take complex baseband symbols, normalised to `E|x|² = 1`
  per row.
- Optical channels take real non-negative intensities, normalised to unit mean optical
  power per row.
- Channels return symbols after perfect-CSI compensation.
- `snr_db` may be a scalar, one value per row, or `None` (noiseless).

### 3.1 AWGN (control)

- Model: y = x + n, with n ~ CN(0, σ²).
- SNR = E|x|² / σ² = 1/σ².

### 3.2 Modelled acoustic channel with OFDM (`uac_ofdm`)

Formulas: Lucani, Médard and Stojanovic, arXiv:0801.0426, eqs. (1)–(8), citing
Stojanovic, WUWNet 2006 / ACM MC2R 2007. They are consistent with eq. (1) of the survey.

- **Path loss:** A(l, f) = l^k a(f)^l, with l in km and k = 1.5.
- **Absorption (Thorp):** 10 log a(f) = 0.11 f²/(1+f²) + 44 f²/(4100+f²) + 2.75·10⁻⁴ f²
  + 0.003, in dB/km, with f in kHz.
- **Ambient noise:** in dB re µPa/Hz, the four components below are summed in linear
  power:
  - 10 log N_t = 17 − 30 log f;
  - 10 log N_s = 40 + 20(s − 0.5) + 26 log f − 60 log(f + 0.03);
  - 10 log N_w = 50 + 7.5 √w + 20 log f − 40 log(f + 0.4);
  - 10 log N_th = −15 + 20 log f.
- **Multipath:** image method in an isovelocity waveguide (pressure-release surface,
  reflecting bottom).
  - Paths have at most `max_reflections` bounces.
  - Each path has amplitude r_s^{n_s} r_b^{n_b} / √A(l_p, f_c), delay l_p / c and an
    independent uniform phase per realisation.
  - Delays are rounded to the nearest sample, and the response is normalised to unit
    energy.
  - **This simplified model does not replace Bellhop**: there is no sound-speed profile,
    no per-path frequency dependence, and no variation within a transmission.
- **OFDM:**
  - N subcarriers with a unitary IFFT/FFT.
  - The cyclic prefix must be at least the delay spread; this is checked when the channel
    is built.
  - Subcarrier k has noise variance σ² c_k, where c_k = A(R, f_k)N(f_k) /
    mean(A N).
  - Equalisation is a one-tap MMSE with perfect CSI. Pilot-based estimation is a `TODO`.
- **Doppler:** only a residual frequency offset per OFDM symbol, ε ~ N(0, cfo_std²) in
  subcarrier spacings. Wideband time compression is out of scope in version 1.0.
- **SNR:** the average received SNR per subcarrier, 1/σ². The magnitude of the path
  loss is absorbed in the SNR; its frequency dependence shapes c_k.

### 3.3 Optical IM/DD channel (`uwoc`)

y = h s + n, with h = h_l h_a h_p (block fading per image). The receiver returns
r = y / h (perfect CSI).

- **Attenuation h_l = exp(−c d):** Beer–Lambert, eq. (2) of the survey. The coefficient
  c comes from Table III ([`configs/uwoc_water.yaml`](configs/uwoc_water.yaml), 532 nm):
  clear ocean 0.15, coastal ocean 0.305, turbid harbor 2.17 m⁻¹. Pure water is not in
  Table III.
- **Turbulence h_a, with E[h_a] = 1:**
  - log-normal: h_a = exp(2X), X ~ N(−σ², σ²), σ² = ln(1 + SI)/4;
  - Gamma-Gamma: SI = 1/α + 1/β + 1/(αβ).
- **Pointing h_p:** Farid and Hranilovic (JLT 2007), h_p = A₀ exp(−2r²/w²_zeq), with a
  Rayleigh-distributed radial displacement r.
- **Receiver noise:** Gaussian, with variance σ²_th (1 + ζ h s). Here ζ is the ratio of
  shot to thermal noise variance; ζ = 0 means thermal noise only.
- **IM/DD constraints:**
  - the input must be non-negative, and the channel rejects it otherwise;
  - the mean optical power is normalised to 1;
  - analog symbols are mapped by DC bias and clipping, `bias_and_clip`.
- **SNR:** the transmit electrical SNR, 1/σ²_th. The received SNR is SNR · h², so water
  type, distance, turbulence and pointing shift the curves as in a link budget.

### 3.4 Scenarios

| Scenario | Channel | Main parameters | SNR grid (dB) |
| --- | --- | --- | --- |
| `awgn` | AWGN | — | −5 : 5 : 25 |
| `uac_modeled_short` | `uac_ofdm` | 1 km, 10–18 kHz band (f_c 14 kHz, B 8 kHz), N 1024, CP 512, depth 100 m, wind 5 m/s, s 0.5, CFO std 0.01 | −5 : 5 : 25 |
| `uac_modeled_long` | `uac_ofdm` | 5 km, 4–8 kHz band (f_c 6 kHz, B 4 kHz), N 256, CP 64, otherwise as above | −5 : 5 : 25 |
| `uwoc_clear` | `uwoc` | clear ocean, 20 m (Beer–Lambert loss 26.1 dB) | −5 : 5 : 50 |
| `uwoc_coastal` | `uwoc` | coastal ocean, 10 m (26.5 dB) | −5 : 5 : 50 |
| `uwoc_turbid` | `uwoc` | turbid harbor, 2 m (37.7 dB) | −5 : 5 : 50 |
| `uwoc_turbulence_weak` | `uwoc` | clear ocean, 20 m, log-normal SI 0.1 | −5 : 5 : 50 |
| `uwoc_turbulence_moderate` | `uwoc` | clear ocean, 20 m, Gamma-Gamma α = β = 4 (SI 0.5625) | −5 : 5 : 50 |
| `uwoc_pointing` | `uwoc` | clear ocean, 20 m, aperture radius 5 cm, beam radius 20 cm, jitter 5 cm | −5 : 5 : 50 |

The acoustic bands are the three Watermark bands (van Walree et al., IEEE JOE 2017).

`TODO(verify)`, to be confirmed by the maintainers:
- the scenario parameters that come neither from the survey nor from cited formulas:
  distances, geometry, wind, reflection coefficients, residual CFO, scintillation
  indices, pointing parameters;
- the final SNR grids;
- the Table III coefficients.

### 3.5 Watermark replay (`watermark_ofdm`)

Source: van Walree et al., IEEE JOE 2017 (survey ref. [248]). The data, `WatermarkV1.zip`
from FFI, is not redistributed (see DATASETS.md).

- **Channel files.** Each `.mat` holds a complex baseband time-varying impulse response
  `h`, of size snapshots × K taps, with `fs_t = fs_tau / K`, plus `fc`, `V0` and
  `meta.bandwidth`. This format was verified on the archive.
- **Replay.** The replay ports `replayfilter.m` exactly:
  y[k] = Σ_l x[k−l] g_p[l], with g_p = (1−f) h[n] + f h[n+1], n = ⌊p/K⌋ and f = p/K − n.
  The tests check it against the MATLAB sample loop, on synthetic and real files.
- **OFDM.**
  - The signal is sampled at `fs_tau`, with N subcarriers in the channel band.
  - The cyclic prefix is at least K − 1 samples, covering the whole recorded delay
    window.
  - Each image uses a random file and a random start time.
- **Perfect CSI.** It is the diagonal of the channel matrix: the frequency response of
  the TVIR averaged over each FFT window. Equalisation is a one-tap MMSE.
- **Normalisation and noise.**
  - Each file is normalised to unit mean energy.
  - White noise is added at the requested SNR, defined as for 3.2. The recorded ambient
    noise is not replayed.
  - `V0` is not used, which is `TODO(verify)`: it is described in FFI-rapport
    2016/01378.
- **Time variation, measured.**
  - NOF1 snapshots correlate at 0.89 at 128 ms; NCS1 snapshots at 0.78 at 32 ms.
  - With the shortest symbols the NOF1 window allows (N = 1024, 128 ms), OFDM with
    perfect CSI reaches an NMSE of −9.2 dB without noise and −11.6 dB at 25 dB SNR.
  - This interference floor is a property of the measured channel. It is the reason a
    replayed channel is harder than the modelled one.

| Scenario | Channel | Main parameters | SNR grid (dB) |
| --- | --- | --- | --- |
| `uac_watermark_nof1` | `watermark_ofdm` | NOF1 (Oslofjord, 10–18 kHz), N 1024, CP 2048 samples at 16 kHz | −5 : 5 : 25 |

## 4. Metrics

Code: [`uwbench/metrics/`](uwbench/metrics/). Every metric returns one value per image.
The dataset aggregate is the mean over images; its 95 % confidence interval comes with
the evaluation engine.

### 4.1 Fidelity (`fidelity.py`)

Reconstructions are clamped to [0, 1] and scored in float, without 8-bit quantisation.

- **PSNR**
  - Per image: 10 log₁₀(1 / MSE), with the MSE over all pixels and colour channels and a
    data range of 1. Averaged over images, not computed from the mean MSE.
  - Identical images give the documented ceiling of **100 dB** instead of infinity, so
    the means stay finite.
- **MS-SSIM**
  - Wang, Simoncelli and Bovik, Asilomar 2003, via `pytorch_msssim` with its defaults:
    window 11, σ = 1.5, and the five scale weights of the paper.
  - The five scales need both image sides to be at least **161 px**, and smaller images
    are rejected. All test images of the six datasets meet this; the smallest side found
    is 168 px, in UIEB.
- **LPIPS**
  - Zhang et al., CVPR 2018, `lpips` 0.1.4 with the **AlexNet** backbone, as in the
    literature. Inputs are mapped to [−1, 1]. Lower is better.

### 4.2 Rate (`rate.py`)

- **Digital schemes:** bits per pixel, bpp = n_bits / (H W), measured on the actual
  codestream.
- **Analog schemes:** bandwidth ratio ρ = k / n, where n = C H W source components and k
  channel uses per image. This follows Bourtsoulatze et al., TCCN 2019 (survey ref.
  [55]).
  - A channel use is one complex symbol on the AWGN and acoustic channels.
  - It is one real symbol on the optical IM/DD channel.
- **Conversion:** a digital scheme carrying η information bits per channel use (code
  rate × bits per modulation symbol) needs k = bpp H W / η channel uses. Hence
  **ρ = bpp / (C η)** and **bpp = ρ C η**. For example, 0.5 bpp with QPSK and a rate-1/2
  code on RGB images gives ρ = 1/6.

### 4.3 Complexity (`complexity.py`)

- **Parameters:** the number of model parameters.
- **FLOPs per image:** from `torch.utils.flop_counter.FlopCounterMode`, where one
  multiply-add counts as 2 FLOPs. Matrix products and convolutions are covered;
  elementwise operations are not.
- **Latency:** the wall-clock inference time, as mean, standard deviation and median
  over repeated runs after warm-up, with CUDA synchronisation. It is always reported
  with the platform: OS, Python, PyTorch, CPU, threads and GPU.

### 4.4 Task metric (deferred)

The optional mIoU on SUIM needs a frozen, public, pre-trained segmenter. The only
official one found is SUIM-Net. Its checkpoints ship as Keras/TensorFlow `.hdf5` files
in the SUIM archive (`Checkpoint_Data/`). Using them would add a TensorFlow dependency,
and no verified PyTorch port exists. The task metric is therefore **not part of version
1.0**.
