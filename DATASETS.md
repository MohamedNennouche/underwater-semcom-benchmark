# Datasets

> **You must read and respect the license of every dataset you download.** This
> repository never redistributes images or annotation files. It only records where the
> official copies are, how to check them (SHA-256 checksums, file counts) and how to split
> them. All licenses below are quoted from the official pages and remain marked
> `TODO(verify)` until confirmed by the maintainers.

Version 1.0 covers optical images only. Videos, sonar and acoustic recordings are not
part of version 1.0.

## Summary

| Key | Dataset | Task | Survey ref. | Access | Declared license (official page) |
| --- | --- | --- | --- | --- | --- |
| `uieb` | UIEB | enhancement, paired | [212] | automatic (Google Drive) | non-commercial, academic use only; redistribution forbidden |
| `lsui` | LSUI | enhancement, paired | [213] | automatic (Google Drive) | none stated; "please cite our paper" |
| `euvp` | EUVP (paired subsets) | enhancement, paired | [211] | **manual** (official link dead; unofficial mirror) | none stated |
| `suim` | SUIM | semantic segmentation | [219] | automatic (Google Drive) | none stated (code is MIT) |
| `trashcan` | TrashCan 1.0 | instance segmentation / detection | [222] | **manual** (bot check) | free for academic teaching/research use; JAMSTEC permission for commercial use |
| `uicd` | UICD | image captioning | [228] | automatic (Google Drive) | images provided exclusively for academic and research purposes |

Exact wording, Baidu mirrors, sizes, and the date each source was checked are in
[`uwbench/data/registry.yaml`](uwbench/data/registry.yaml).

## Procedure

```bash
uwbench download uieb                   # prints the license and stops
uwbench download uieb --accept-license  # downloads and extracts
uwbench verify uieb                     # checks checksums and folder structure
```

Files go to `datasets/<key>/` (override with `--root` or the `UWBENCH_DATA` environment
variable):

```
datasets/<key>/archives/    archives as downloaded
datasets/<key>/extracted/   one sub-folder per archive, or your manual deposit
```

`datasets/` is ignored by git.

**UIEB archives are password-protected RAR files.** The passwords are the ones published
on the official page and are stored in the registry. Extracting them requires
[7-Zip](https://www.7-zip.org/) (`7z`, found on `PATH` or in `C:\Program Files\7-Zip`)
or `unrar`. libarchive (`bsdtar`, the Windows `tar`) cannot decrypt RAR: it would write
empty files, which `verify` reports.

### Manual datasets

- **EUVP.** The Google Drive folder linked from the
  [official page](https://irvlab.cs.umn.edu/resources/euvp-dataset) returned HTTP 404 on
  2026-11-15. The reference copy whose SHA-256 is recorded in the registry is the
  **unofficial** Kaggle mirror
  [`pamuduranasinghe/euvp-dataset`](https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset).
  Its paired folder counts equal the official statistics. The CC0-1.0 label shown on
  Kaggle was set by the uploader and is not the authors' license. Save the archive as
  `datasets/euvp/archives/euvp.zip`.
- **TrashCan.** Open <https://hdl.handle.net/11299/214865> in a browser, read
  `LICENSE.txt`, and download the dataset archive (images and labels, not the
  checkpoints). Save it as `datasets/trashcan/archives/TrashCan.zip`.

Then run `uwbench download <key> --accept-license`, which checks the SHA-256 and extracts
the archive, followed by `uwbench verify <key>`.

## What `verify` checks

1. The SHA-256 of each archive, if the archive is still present and its checksum is
   recorded in the registry.
2. The number of files in each expected directory.
3. The SHA-256 of every extracted file, against `uwbench/data/manifests/<key>.csv` if that
   manifest exists. It reports missing, modified and unexpected files.

A dataset with no recorded layout and no manifest never passes.

For maintainers: after a first verified download, record the archive checksums and the
`expected` paths in `registry.yaml`, then create the manifest with
`uwbench verify <key> --write-manifest`.
