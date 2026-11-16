"""Command-line entry point ``uwbench``.

Sub-commands are registered here as their phases are implemented and tested.
"""

from __future__ import annotations

import argparse
import os
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

    ev = sub.add_parser("evaluate", help="evaluate a system on a scenario and dataset split")
    ev.add_argument("--system", type=Path, required=True, help="configs/baselines/*.yaml")
    ev.add_argument("--scenario", type=Path, required=True, help="configs/scenarios/*.yaml")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--split", default="test")
    ev.add_argument("--realizations", type=int, default=1, help="channel draws per image")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ev.add_argument("--snr-step", type=float, default=None,
                    help="override the scenario's SNR step (same range)")
    ev.add_argument("--limit", type=int, default=None, help="first N images only")
    ev.add_argument("--root", type=Path, default=None)
    ev.add_argument("--out", type=Path, default=Path("results"))

    rp = sub.add_parser("report", help="figures and tables from evaluate results")
    rp.add_argument("results", type=Path, nargs="+", help="result JSON files")
    rp.add_argument("--out", type=Path, default=Path("results/report"))

    tr = sub.add_parser("train", help="train a learned model on one scenario")
    tr.add_argument("--model", type=Path, required=True, help="configs/models/*.yaml")
    tr.add_argument("--scenario", type=Path, required=True)
    tr.add_argument("--device", default="cuda" if _cuda() else "cpu")
    tr.add_argument("--epochs", type=int, default=None, help="override the config")
    tr.add_argument("--limit", type=int, default=None, help="first N train/val images only")
    tr.add_argument("--root", type=Path, default=None)
    tr.add_argument("--out", type=Path, default=Path("checkpoints"))
    ev.add_argument("--checkpoint", type=Path, default=None,
                    help="learned systems: weights to use (default: the system's checkpoint)")
    return parser


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


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
        # image datasets only: entries without samples (e.g. Watermark channels) are skipped
        keys = args.datasets or [k for k, e in load_registry().items() if "samples" in e]
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
    if args.command == "evaluate":
        import yaml

        from uwbench.eval.runner import evaluate, split_images, write

        system = yaml.safe_load(args.system.read_text(encoding="utf-8"))
        scenario = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
        if system["type"] == "learned":
            system["checkpoint"] = str(args.checkpoint or system["checkpoint"].format(
                scenario=scenario["name"]))
        images = split_images(args.dataset, args.split, root)[:args.limit]
        summary, records = evaluate(system, scenario, images, seed=args.seed,
                                    realizations=args.realizations, workers=args.workers,
                                    snr_step=args.snr_step)
        summary.update(dataset=args.dataset, split=args.split, limit=args.limit)
        stem = f"{system['name']}_{scenario['name']}_{args.dataset}_{args.split}"
        print(f"wrote {write(summary, records, args.out, stem)}")
        return 0
    if args.command == "train":
        import yaml

        from uwbench.eval.runner import split_images
        from uwbench.models.train import train

        cfg = yaml.safe_load(args.model.read_text(encoding="utf-8"))
        scenario = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
        sets = cfg["train"].get("datasets", ["lsui", "euvp", "uieb"])
        tr = [p for d in sets for _, p in split_images(d, "train", root)][:args.limit]
        va = [p for d in sets for _, p in split_images(d, "val", root)][:args.limit]
        best = train(cfg, scenario, tr, va, args.out, args.device, args.epochs)
        print(f"best checkpoint: {best}")
        return 0
    if args.command == "report":
        from uwbench.eval.report import report

        for path in report(args.results, args.out):
            print(f"wrote {path}")
        return 0
    build_parser().print_help()
    return 0
