"""``uwbench dedup``: exact and perceptual duplicates within and across datasets.

Each image gets two fingerprints:

* ``pixel_sha256``: SHA-256 of the decoded RGB pixels (shape included), so that two
  encodings of the same pixels match, whatever the file name or format;
* ``phash``: 64-bit DCT perceptual hash (``imagehash.phash``, hash_size 8; C. Zauner,
  "Implementation and benchmarking of perceptual image hash functions", 2010).

Two images are duplicates when ``hamming(phash_a, phash_b) <= threshold``; equal pixels
give distance 0. A *sample* (one row of a split, e.g. a raw/reference pair) is linked to
another sample when any of their hashed images are duplicates; for paired datasets both
the degraded and the reference image are hashed. Linked samples form duplicate groups
(connected components), written to ``splits/duplicates.csv``; each group is labelled by
its first member in the order (oldest dataset by ``published``, then ``sample_id``).
How groups constrain the splits is described in :mod:`uwbench.data.splits`.

Datasets listed in ``skip_intra_dedup`` are only compared with other datasets (pHash is
unreliable on low-texture ROV footage; their splits are made by video instead).

Images are decoded by content (Pillow), not by extension.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from uwbench.data.registry import get_entry
from uwbench.data.verify import _files

HASHED_ROLES = ("input", "reference", "image")  # masks and resized copies are not hashed
SampleKey = tuple[str, str]  # (dataset, sample_id)


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    split: str | None  # "test", "trainval" (official pools) or None (no official split)
    paths: dict[str, str] = field(compare=False)  # role -> path relative to extracted/


def list_samples(key: str, root: Path, registry: dict[str, Any] | None = None) -> list[Sample]:
    """Enumerate the samples of ``key``; roles are paired by file stem."""
    base = root / key / "extracted"
    samples: list[Sample] = []
    for src in get_entry(key, registry)["samples"]:
        by_role = {r: {p.stem: p for p in _files(base / d, recursive=False)}
                   for r, d in src["roles"].items()}
        anchor = next(iter(by_role.values()))  # the first role defines the samples
        for stem in sorted(anchor):
            if missing := [r for r, files in by_role.items() if stem not in files]:
                raise ValueError(f"{key}: sample {stem!r} has no {missing}")
            sid = f"{src['prefix']}/{stem}" if src.get("prefix") else stem
            paths = {r: files[stem].relative_to(base).as_posix() for r, files in by_role.items()}
            samples.append(Sample(key, sid, src.get("split"), paths))
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError(f"{key}: duplicate sample ids")
    return samples


def image_fingerprint(path: Path) -> tuple[str, str]:
    """Return ``(pixel_sha256, phash_hex)`` for one image file."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    px = np.asarray(rgb)
    digest = hashlib.sha256(str(px.shape).encode() + px.tobytes()).hexdigest()
    return digest, str(imagehash.phash(rgb))


def index_dataset(key: str, root: Path, samples: list[Sample], workers: int | None = None
                  ) -> list[dict[str, str]]:
    """Fingerprint every hashed image of ``samples``; cached in ``<root>/<key>/index.csv``.

    shortcut: the cache is keyed by path only; delete it if files change (``verify``
    detects changed files through the manifest).
    """
    base = root / key / "extracted"
    cache_path = root / key / "index.csv"
    cache: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        with open(cache_path, newline="", encoding="utf-8") as f:
            cache = {r["path"]: r for r in csv.DictReader(f)}
    wanted = [(s.sample_id, r, p) for s in samples for r, p in s.paths.items()
              if r in HASHED_ROLES]
    todo = sorted({p for _, _, p in wanted} - set(cache))
    if todo:
        files = [base / p for p in todo]
        if workers == 1:
            results = list(map(image_fingerprint, files))
        else:
            # Each worker imports scipy (imagehash); unbounded OpenBLAS threads per worker
            # exhausted memory with 22 workers. Children inherit this environment.
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as ex:
                results = list(ex.map(image_fingerprint, files, chunksize=16))
        for p, (sha, ph) in zip(todo, results, strict=True):
            cache[p] = {"path": p, "pixel_sha256": sha, "phash": ph}
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["path", "pixel_sha256", "phash"], lineterminator="\n")
            w.writeheader()
            w.writerows(cache[p] for p in sorted(cache))
    return [{"dataset": key, "sample_id": sid, "role": r, "path": p, **cache[p]}
            for sid, r, p in wanted]


def near_pairs(phashes: np.ndarray, max_distance: int, chunk: int = 256) -> np.ndarray:
    """All ``(i, j, d)`` with ``i < j`` and Hamming distance ``d <= max_distance``.

    Parameters
    ----------
    phashes : numpy.ndarray of uint64, shape (N,)
    """
    out = []
    for a in range(0, len(phashes), chunk):
        d = np.bitwise_count(phashes[a:a + chunk, None] ^ phashes[None, a:])
        i, j = np.nonzero(d <= max_distance)
        keep = j > i  # column c is global index a + c, row r is a + r
        out.append(np.stack([i[keep] + a, j[keep] + a, d[i[keep], j[keep]]], axis=1))
    return np.concatenate(out).astype(np.int64) if out else np.empty((0, 3), np.int64)


def group_duplicates(rows: list[dict[str, str]], pairs: Iterable[tuple[int, int]],
                     rank: dict[str, int]) -> list[list[SampleKey]]:
    """Connected components of samples linked by image ``pairs`` (indices into ``rows``).

    Each group is sorted keeper first: oldest dataset (lowest ``rank``), then sample id.
    Only groups of two or more samples are returned.
    """
    parent: dict[SampleKey, SampleKey] = {}

    def find(x: SampleKey) -> SampleKey:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        a = find((rows[i]["dataset"], rows[i]["sample_id"]))
        b = find((rows[j]["dataset"], rows[j]["sample_id"]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[SampleKey, list[SampleKey]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)

    def order(s: SampleKey) -> tuple[int, str]:
        return rank[s[0]], s[1]

    return sorted((sorted(g, key=order) for g in groups.values() if len(g) > 1),
                  key=lambda g: order(g[0]))


def write_duplicates(groups: list[list[SampleKey]], path: Path) -> None:
    """One row per grouped sample; ``group`` is the first member's ``dataset:sample_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["group", "dataset", "sample_id"])
        for g in groups:
            w.writerows([f"{g[0][0]}:{g[0][1]}", ds, sid] for ds, sid in g)


def read_groups(path: Path) -> dict[SampleKey, str]:
    """Map each grouped sample to its group label; ungrouped samples are absent."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["dataset"], r["sample_id"]): r["group"] for r in csv.DictReader(f)}


def contact_sheet(pairs: list[tuple[Path, Path, str]], out: Path, thumb: int = 160) -> None:
    """Side-by-side thumbnails of image pairs with a caption, one pair per row.

    Written under results/ (git-ignored) because it shows third-party images.
    """
    sheet = Image.new("RGB", (2 * thumb + 10, max(1, len(pairs)) * (thumb + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (a, b, caption) in enumerate(pairs):
        y = k * (thumb + 18)
        for col, p in enumerate((a, b)):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (col * (thumb + 10), y))
        draw.text((2, y + thumb + 2), caption, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def run_dedup(keys: list[str], root: Path, cfg: dict[str, Any], out_csv: Path,
              report_dir: Path, registry: dict[str, Any] | None = None,
              workers: int | None = None) -> dict[str, Any]:
    """Index ``keys``, write ``out_csv`` and contact sheets; return summary statistics."""
    published = {k: str(get_entry(k, registry)["published"]) for k in keys}
    order = sorted(keys, key=lambda k: (published[k], k))
    rank = {k: i for i, k in enumerate(order)}
    rows: list[dict[str, str]] = []
    for k in keys:
        rows += index_dataset(k, root, list_samples(k, root, registry), workers)
    ph = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    t, margin = cfg["phash_threshold"], cfg["contact_sheet_margin"]
    pairs = near_pairs(ph, t + margin)
    skip = set(cfg.get("skip_intra_dedup", []))
    links = [(i, j) for i, j, d in pairs.tolist() if d <= t and not (
        rows[i]["dataset"] == rows[j]["dataset"] and rows[i]["dataset"] in skip)]
    groups = group_duplicates(rows, links, rank)
    write_duplicates(groups, out_csv)

    # Near pairs between different samples, and contact sheets around the threshold.
    def key(r: dict[str, str]) -> SampleKey:
        return r["dataset"], r["sample_id"]

    between = [(i, j, d) for i, j, d in pairs.tolist() if key(rows[i]) != key(rows[j])]
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "near_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distance", "dataset_a", "path_a", "dataset_b", "path_b"])
        for i, j, d in between:
            a, b = rows[i], rows[j]
            w.writerow([d, a["dataset"], a["path"], b["dataset"], b["path"]])
    # One sheet per (distance, dataset pair) so that no pair of datasets hides the others.
    rng = np.random.default_rng(cfg["seed"])
    strata: dict[tuple[int, str, str], list[tuple[int, int]]] = {}
    for i, j, d in between:
        a, b = sorted((rows[i]["dataset"], rows[j]["dataset"]))
        strata.setdefault((d, a, b), []).append((i, j))
    for (d, a, b), members in sorted(strata.items()):
        pick = sorted(rng.choice(len(members), min(len(members), cfg["contact_sheet_pairs"]),
                                 replace=False))
        sheet = []
        for k in pick:
            ra, rb = rows[members[k][0]], rows[members[k][1]]
            sheet.append((root / ra["dataset"] / "extracted" / ra["path"],
                          root / rb["dataset"] / "extracted" / rb["path"],
                          f"d={d}  {ra['dataset']}:{ra['sample_id']}  |  "
                          f"{rb['dataset']}:{rb['sample_id']}"))
        contact_sheet(sheet, report_dir / f"contact_d{d}_{a}-{b}.png")

    grouped = [s for g in groups for s in g]
    return {
        "images": len(rows),
        "pairs_between_samples_by_distance": {
            d: sum(1 for p in between if p[2] == d) for d in range(t + margin + 1)},
        "groups": len(groups),
        "largest_group": max((len(g) for g in groups), default=0),
        "cross_dataset_groups": sum(1 for g in groups if len({ds for ds, _ in g}) > 1),
        "grouped_samples_by_dataset": {k: sum(1 for ds, _ in grouped if ds == k) for k in keys},
        "group_label_order": order,
    }
