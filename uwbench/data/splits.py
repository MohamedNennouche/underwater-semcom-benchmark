"""``uwbench split``: frozen train/val/test partitions that preserve official test sets.

Rules (``configs/splits.yaml``, decided by the maintainers on 2026-11-15):

1. **Official test splits are kept intact** (LSUI, SUIM). Datasets without one get a
   seeded ``1 - val - test`` / ``val`` / ``test`` partition (80/10/10); datasets with one
   get ``val_fraction`` of their official training pool as validation.
2. **Group-aware draws.** A *unit* is the set of a dataset's samples sharing a duplicate
   group (``splits/duplicates.csv``) or a ``split_unit`` key (e.g. the source video of a
   TrashCan frame). Each unit goes to a single split, so near-duplicates never straddle
   splits of the same dataset. Units are drawn in a ``numpy.random.default_rng(seed)``
   permutation of the units sorted by their first ``sample_id``. A drawn (non-official)
   test split excludes units duplicated in another dataset.
3. **No train/test leakage across datasets.** A train or val sample is removed when its
   duplicate group contains a test sample of *any* dataset (``splits/removed.csv``).
4. **Cross-dataset protocols** (train on A, test on all of B) get their own files in
   ``splits/cross/``; their train/val sets exclude every group present in their test set.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from uwbench.data.dedup import Sample, SampleKey
from uwbench.data.verify import sha256_file

SPLITS = ("train", "val", "test")
Parts = dict[str, list[Sample]]


def _key(s: Sample) -> SampleKey:
    return s.dataset, s.sample_id


def _units(samples: list[Sample], groups: dict[SampleKey, str],
           unit_pattern: str | None) -> list[list[Sample]]:
    """Samples joined by a shared duplicate group or a shared ``unit_pattern`` match."""
    parent = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for i, s in enumerate(samples):
        tags = [("group", groups[_key(s)])] if _key(s) in groups else []
        if unit_pattern and (m := re.match(unit_pattern, s.sample_id)):
            tags.append(("unit", m.group(1)))
        for tag in tags:
            parent[find(i)] = find(first.setdefault(tag, i))
    units: dict[int, list[Sample]] = {}
    for i, s in enumerate(samples):
        units.setdefault(find(i), []).append(s)
    return sorted((sorted(u, key=lambda s: s.sample_id) for u in units.values()),
                  key=lambda u: u[0].sample_id)


def assign_splits(samples: list[Sample], groups: dict[SampleKey, str], seed: int,
                  val_fraction: float, test_fraction: float,
                  unit_pattern: str | None = None,
                  cross_groups: frozenset[str] | set[str] = frozenset()) -> Parts:
    """Rules 1 and 2 for one dataset (before the cross-dataset removal of rule 3).

    Units belonging to a ``cross_groups`` group (duplicates shared with another dataset)
    are never drawn into a test split, so a drawn test set never removes training
    samples from another dataset. Official test splits are not affected.
    """
    official = {s.split for s in samples}
    pool = [s for s in samples if s.split != "test"]
    if official <= {None}:
        target = {"test": round(test_fraction * len(pool)), "val": round(val_fraction * len(pool))}
    elif official <= {"trainval", "test"}:
        target = {"test": 0, "val": round(val_fraction * len(pool))}
    else:
        raise ValueError(f"mixed or unknown official splits: {official}")
    parts: Parts = {"train": [], "val": [], "test": [s for s in samples if s.split == "test"]}
    drawn = {"test": 0, "val": 0}
    units = _units(pool, groups, unit_pattern)
    for i in np.random.default_rng(seed).permutation(len(units)):
        shared = any(groups.get(_key(s)) in cross_groups for s in units[i])
        split = next((k for k in ("test", "val")
                      if drawn[k] < target[k] and not (k == "test" and shared)), "train")
        parts[split] += units[i]
        if split != "train":
            drawn[split] += len(units[i])
    return {k: sorted(v, key=lambda s: s.sample_id) for k, v in parts.items()}


def _test_groups(samples: Iterable[Sample], groups: dict[SampleKey, str]) -> set[str]:
    return {groups[_key(s)] for s in samples if _key(s) in groups}


def remove_test_duplicates(parts_by_ds: dict[str, Parts], groups: dict[SampleKey, str]
                           ) -> list[tuple[str, str, str, str]]:
    """Rule 3, in place. Returns ``(dataset, sample_id, split, group)`` of removed samples."""
    banned = _test_groups((s for p in parts_by_ds.values() for s in p["test"]), groups)
    removed = []
    for parts in parts_by_ds.values():
        for split in ("train", "val"):
            keep = []
            for s in parts[split]:
                if groups.get(_key(s)) in banned:
                    removed.append((s.dataset, s.sample_id, split, groups[_key(s)]))
                else:
                    keep.append(s)
            parts[split] = keep
    return removed


def cross_dataset(proto: dict[str, list[str]], parts_by_ds: dict[str, Parts],
                  groups: dict[SampleKey, str]) -> Parts:
    """Rule 4: ``proto`` maps train/val/test to lists of ``<dataset>_<split>`` names."""
    def collect(names: list[str]) -> list[Sample]:
        return [s for n in names for s in parts_by_ds[n.rsplit("_", 1)[0]][n.rsplit("_", 1)[1]]]

    test = collect(proto["test"])
    banned = _test_groups(test, groups)
    return {"test": test, **{k: [s for s in collect(proto[k]) if groups.get(_key(s)) not in banned]
                             for k in ("train", "val")}}


def _write(path: Path, samples: list[Sample], with_dataset: bool) -> None:
    roles = list(samples[0].paths) if samples else []
    if any(list(s.paths) != roles for s in samples):
        raise ValueError(f"{path.name}: samples with different roles")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow((["dataset"] if with_dataset else []) + ["sample_id", *roles])
        w.writerows(([s.dataset] if with_dataset else []) + [s.sample_id, *s.paths.values()]
                    for s in samples)


def write_splits(key: str, parts: Parts, out_dir: Path) -> None:
    for split in SPLITS:
        _write(out_dir / f"{key}_{split}.csv", parts[split], with_dataset=False)


def write_cross(name: str, parts: Parts, out_dir: Path) -> None:
    for split in SPLITS:
        _write(out_dir / "cross" / f"{name}_{split}.csv", parts[split], with_dataset=True)


def write_removed(removed: list[tuple[str, str, str, str]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "sample_id", "from_split", "group"])
        w.writerows(sorted(removed))


def build_splits(keys: list[str], root: Path, cfg: dict[str, Any], split_dir: Path,
                 registry: dict[str, Any] | None = None
                 ) -> tuple[dict[str, Parts], list[tuple[str, str, str, str]]]:
    """Apply rules 1-4 to ``keys`` and write every file of ``split_dir``."""
    from uwbench.data.dedup import list_samples, read_groups

    groups = read_groups(split_dir / "duplicates.csv")
    datasets_of: dict[str, set[str]] = {}
    for (ds, _), g in groups.items():
        datasets_of.setdefault(g, set()).add(ds)
    cross_groups = {g for g, d in datasets_of.items() if len(d) > 1}
    parts_by_ds = {k: assign_splits(list_samples(k, root, registry), groups, cfg["seed"],
                                    cfg["val_fraction"], cfg["test_fraction"],
                                    cfg.get("split_unit", {}).get(k), cross_groups)
                   for k in keys}
    removed = remove_test_duplicates(parts_by_ds, groups)
    for k, parts in parts_by_ds.items():
        write_splits(k, parts, split_dir)
    write_removed(removed, split_dir / "removed.csv")
    for name, proto in (cfg.get("cross_dataset") or {}).items():
        names = [n.rsplit("_", 1)[0] for part in proto.values() for n in part]
        if all(n in parts_by_ds for n in names):
            write_cross(name, cross_dataset(proto, parts_by_ds, groups), split_dir)
    return parts_by_ds, removed


def _read(path: Path, dataset: str | None) -> list[SampleKey]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(dataset or r["dataset"], r["sample_id"]) for r in csv.DictReader(f)]


def check_no_leak(split_dir: Path, cross: dict[str, Any] | None = None) -> list[str]:
    """Problems: a sample in two splits of its dataset, a duplicate group in both a
    train/val file and a test file (all datasets together, and within each cross-dataset
    protocol), or a missing cross-dataset file."""
    from uwbench.data.dedup import read_groups

    groups = read_groups(split_dir / "duplicates.csv")
    problems: list[str] = []

    def leak(train: list[SampleKey], test: list[SampleKey], where: str) -> None:
        def of(keys: list[SampleKey]) -> set[str]:
            return {groups[k] for k in keys if k in groups}

        shared = of(train) & of(test)
        if shared:
            problems.append(f"{where}: {len(shared)} duplicate group(s) in train/val and test, "
                            f"e.g. {sorted(shared)[:3]}")

    seen: dict[SampleKey, str] = {}
    fit: list[SampleKey] = []
    test: list[SampleKey] = []
    for path in sorted(split_dir.glob("*_*.csv")):
        key, split = path.stem.rsplit("_", 1)
        if split not in SPLITS:
            continue
        for k in _read(path, key):
            if k in seen:
                problems.append(f"{k[1]} in both {seen[k]} and {path.name}")
            seen[k] = path.name
            (test if split == "test" else fit).append(k)
    leak(fit, test, "all datasets")

    for name in cross or {}:
        files = {s: split_dir / "cross" / f"{name}_{s}.csv" for s in SPLITS}
        if missing := [p.name for p in files.values() if not p.exists()]:
            problems.append(f"cross-dataset {name}: missing {missing}")
            continue
        leak(_read(files["train"], None) + _read(files["val"], None), _read(files["test"], None),
             f"cross-dataset {name}")
    return problems


def split_checksums(split_dir: Path) -> dict[str, str]:
    """SHA-256 of every versioned split file (listed in PROTOCOL.md)."""
    return {p.relative_to(split_dir).as_posix(): sha256_file(p)
            for p in sorted(split_dir.rglob("*.csv"))}
