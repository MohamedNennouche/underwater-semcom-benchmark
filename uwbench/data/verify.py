"""``uwbench verify <dataset>``: check archives, folder structure and per-file checksums.

Three independent checks, each only when its reference exists:

1. archive SHA-256 against ``registry.yaml`` (archives may be deleted after extraction);
2. file counts of the ``expected`` directories (non-recursive, hidden files ignored);
3. per-file SHA-256 against ``manifests/<key>.csv`` (relative path, digest), reporting
   missing, modified and unexpected files.

A dataset with neither a recorded structure nor a manifest never passes.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uwbench.data.registry import MANIFEST_DIR, get_entry


@dataclass
class VerifyReport:
    dataset: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        lines = [f"[{'OK' if self.ok else 'FAIL'}] {self.dataset}"]
        lines += [f"  problem: {p}" for p in self.problems]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _files(directory: Path, recursive: bool) -> list[Path]:
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and not p.name.startswith("."))


def write_manifest(key: str, root: Path, manifest_dir: Path = MANIFEST_DIR) -> Path:
    """Record ``relative_path,sha256`` for every extracted file of ``key``."""
    extracted = root / key / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(f"{extracted} does not exist")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{key}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256"])
        for p in _files(extracted, recursive=True):
            w.writerow([p.relative_to(extracted).as_posix(), sha256_file(p)])
    return out


def verify(
    key: str,
    root: Path,
    registry: dict[str, dict[str, Any]] | None = None,
    manifest_dir: Path = MANIFEST_DIR,
) -> VerifyReport:
    entry = get_entry(key, registry)
    report = VerifyReport(entry["name"])
    target = root / key

    for f in entry["files"]:
        archive = target / "archives" / f["name"]
        if not archive.exists():
            report.notes.append(f"archive {f['name']} absent (skipped)")
        elif f.get("sha256") and sha256_file(archive) != f["sha256"]:
            report.problems.append(f"archive {f['name']}: SHA-256 differs from registry")
        elif not f.get("sha256"):
            report.notes.append(f"archive {f['name']}: no SHA-256 recorded yet")

    extracted = target / "extracted"
    if not extracted.is_dir():
        report.problems.append(f"{extracted} not found; run: uwbench download {key}")
        return report

    checked = False
    for e in entry["expected"]:
        if e["path"] is None:
            report.notes.append(f"layout of '{e['subset']}' not recorded yet, TODO(verify)")
            continue
        checked = True
        d = extracted / e["path"]
        if not d.is_dir():
            report.problems.append(f"missing directory {e['path']}")
        else:
            files = _files(d, recursive=False)
            if len(files) != e["count"]:
                report.problems.append(f"{e['path']}: {len(files)} files, expected {e['count']}")
            if empty := [p.name for p in files if p.stat().st_size == 0]:
                # e.g. an encrypted archive extracted without its password
                report.problems.append(f"{e['path']}: {len(empty)} empty file(s), e.g. {empty[:3]}")

    manifest = manifest_dir / f"{key}.csv"
    if manifest.exists():
        checked = True
        with open(manifest, newline="", encoding="utf-8") as f:
            expected = {row["path"]: row["sha256"] for row in csv.DictReader(f)}
        files = _files(extracted, recursive=True)
        present = {p.relative_to(extracted).as_posix(): p for p in files}
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        modified = sorted(k for k in set(expected) & set(present)
                          if sha256_file(present[k]) != expected[k])
        for label, items in (("missing", missing), ("unexpected", extra), ("modified", modified)):
            if items:
                report.problems.append(f"{len(items)} {label} file(s), e.g. {items[:3]}")
    else:
        report.notes.append(f"no manifest {manifest.name} yet")

    if not checked:
        report.problems.append("nothing to check against: no recorded layout and no manifest")
    return report
