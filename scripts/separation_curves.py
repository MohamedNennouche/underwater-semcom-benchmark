"""PSNR-SNR curves of the separation baseline on one dataset split and one scenario.

Example (phase 6 check):

    python scripts/separation_curves.py --dataset uieb --split test \
        --scenario configs/scenarios/uwoc_clear.yaml --ratio 0.1667

For every MCS of the channel family and every SNR of the scenario grid, each image is
sent once (seeded) through the full chain. Writes ``results/separation/<scenario>_*.csv``
(one row per image and setting) and a PNG with, per MCS, the mean PSNR including
failures (mid-grey fallback image), the ideal-adaptation envelope, the ideal separation
bound and the failure rates. The transmitted image is the dataset's ``input`` (paired
datasets) or ``image`` role, scored against itself. Superseded by ``uwbench evaluate``
once phase 7 exists.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uwbench.baselines.separation import SeparationScheme, ideal_separation
from uwbench.channels import load_scenario
from uwbench.data.registry import default_root
from uwbench.metrics.fidelity import psnr

MCS = {"complex": [("bpsk", 0.5), ("qpsk", 0.5), ("16qam", 0.5)],
       "real": [("ook", 0.5), ("pam4", 0.5)]}


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.tensor(img).permute(2, 0, 1)[None].float() / 255


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="uieb")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--ratio", type=float, default=1 / 6, help="bandwidth ratio k/n")
    ap.add_argument("--codec", default="jpeg2000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--out", type=Path, default=Path("results/separation"))
    args = ap.parse_args()

    channel, grid, cfg = load_scenario(args.scenario)
    family = "real" if cfg["channel"] == "uwoc" else "complex"
    with open(f"splits/{args.dataset}_{args.split}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    role = "input" if "input" in rows[0] else "image"
    base = default_root() / args.dataset / "extracted"
    images = [np.asarray(Image.open(base / r[role]).convert("RGB")) for r in rows[:args.limit]]
    schemes = [SeparationScheme(m, r, args.codec) for m, r in MCS[family]]
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['name']}_{args.dataset}_{args.split}_r{args.ratio:.4f}"
    records = []
    for i, img in enumerate(images):
        ref = to_tensor(img)
        for s in schemes:
            enc = s.encode_source(img, args.ratio)  # independent of the SNR
            for snr in grid:
                res = s.transmit(img, enc, args.ratio, channel, float(snr), rng)
                records.append([s.name, snr, i, int(res.decoded), res.bpp,
                                psnr(to_tensor(res.image), ref).item()])
        for snr in grid:
            res = ideal_separation(img, args.ratio, channel, float(snr), rng, args.codec,
                                   complex_symbols=family == "complex")
            records.append(["ideal_separation", snr, i, int(res.decoded), res.bpp,
                            psnr(to_tensor(res.image), ref).item()])
        print(f"{i + 1}/{len(images)}", flush=True)

    with open(args.out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["scheme", "snr_db", "image", "decoded", "bpp", "psnr_db"])
        w.writerows(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = np.array([r[1:] for r in records], dtype=float)
    names = np.array([r[0] for r in records])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    means = {}
    for name in dict.fromkeys(names):
        sel = names == name
        m = [rec[sel & (rec[:, 0] == snr), 4].mean() for snr in grid]
        f = [1 - rec[sel & (rec[:, 0] == snr), 2].mean() for snr in grid]
        means[name] = m
        style = "k--" if name == "ideal_separation" else "-o"
        ax.plot(grid, m, style, label=name, markersize=3)
        if name != "ideal_separation":
            ax2.plot(grid, f, "-o", label=name, markersize=3)
    envelope = np.max([v for k, v in means.items() if k != "ideal_separation"], axis=0)
    ax.plot(grid, envelope, "r:", linewidth=2, label="ideal adaptation (best MCS)")
    ax.set(xlabel="SNR (dB)", ylabel="mean PSNR incl. failures (dB)",
           title=f"{cfg['name']} - {args.dataset} {args.split} (n={len(images)}), "
                 f"k/n={args.ratio:.3f}")
    ax2.set(xlabel="SNR (dB)", ylabel="failure rate", yscale="log", ylim=(1e-3, 1.1))
    ax.legend(fontsize=7)
    ax2.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / f"{stem}.png", dpi=130)
    print(f"wrote {args.out / stem}.csv/.png")


if __name__ == "__main__":
    main()
