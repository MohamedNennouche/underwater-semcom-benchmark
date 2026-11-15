# Evaluation protocol

This file specifies the reference evaluation protocol. It is the source for the
"Reference evaluation protocol" table of the survey (Section VII-C). Sections are added
as the corresponding components are implemented and tested. Channels, metrics, the
separation baseline and reporting are not specified here yet.

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
