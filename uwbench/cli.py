"""Command-line entry point ``uwbench``.

Sub-commands are registered here as their phases are implemented and tested.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from uwbench import __version__
from uwbench.data.registry import default_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uwbench", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"uwbench {__version__}")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="download a dataset or print its manual procedure")
    dl.add_argument("dataset")
    dl.add_argument("--root", type=Path, default=None, help="data root (default ./datasets)")
    dl.add_argument("--accept-license", action="store_true",
                    help="confirm you read and accept the dataset license")

    vf = sub.add_parser("verify", help="check checksums and folder structure of a dataset")
    vf.add_argument("dataset")
    vf.add_argument("--root", type=Path, default=None)
    vf.add_argument("--write-manifest", action="store_true",
                    help="record per-file SHA-256 of the current extraction (maintainers)")

    for name, text in (("dedup", "find duplicates within and across datasets"),
                       ("split", "write the frozen train/val/test partitions")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--root", type=Path, default=None)
        p.add_argument("--config", type=Path, default=Path("configs/splits.yaml"))
        p.add_argument("--splits-dir", type=Path, default=Path("splits"))
        p.add_argument("--datasets", nargs="+", default=None, help="default: whole registry")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = getattr(args, "root", None) or default_root()
    if args.command == "download":
        from uwbench.data.download import download

        try:
            download(args.dataset, root, accept_license=args.accept_license)
        except PermissionError as exc:
            print(exc)
            return 2
        return 0
    if args.command == "verify":
        from uwbench.data.verify import verify, write_manifest

        if args.write_manifest:
            print(f"wrote {write_manifest(args.dataset, root)}")
        report = verify(args.dataset, root)
        print(report)
        return 0 if report.ok else 1
    if args.command in ("dedup", "split"):
        import yaml

        from uwbench.data.registry import load_registry

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        keys = args.datasets or list(load_registry())
        if args.command == "dedup":
            from uwbench.data.dedup import run_dedup

            stats = run_dedup(keys, root, cfg, args.splits_dir / "duplicates.csv",
                              Path("results/dedup"))
            print(yaml.safe_dump(stats, sort_keys=False))
            print("Contact sheets and near pairs: results/dedup/ (not versioned)")
            return 0
        from uwbench.data.splits import build_splits, check_no_leak, split_checksums

        parts_by_ds, removed = build_splits(keys, root, cfg, args.splits_dir)
        for key, parts in parts_by_ds.items():
            n_removed = sum(1 for r in removed if r[0] == key)
            print(f"{key}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items())
                  + f" (removed as duplicates of a test sample: {n_removed})")
        if problems := check_no_leak(args.splits_dir, cfg.get("cross_dataset")):
            print("\n".join(problems))
            return 1
        print("\nNo leakage found.\n\n| File | SHA-256 |\n| --- | --- |")
        print("\n".join(f"| `{n}` | `{h}` |" for n, h in split_checksums(args.splits_dir).items()))
        return 0
    build_parser().print_help()
    return 0
