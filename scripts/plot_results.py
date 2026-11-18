"""One metric versus SNR for the reference systems, one panel per scenario.

Example (the figures of the README):

    python scripts/plot_results.py --metric psnr --out docs/figures/psnr
    python scripts/plot_results.py --metric lpips --out docs/figures/lpips

Reads the result files ``<system>_<scenario>_uieb_test.json`` of ``--results`` (default:
the versioned reference results) and writes ``<out>.png`` and ``<out>.pdf``. Unlike
``uwbench report``, which draws every scheme of one scenario, this script compares the
four reference curves across scenarios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PANELS = [("uac_modeled_short", "(a) Acoustic, 1 km, 10–18 kHz"),
          ("uwoc_clear", "(b) Optical, clear ocean, 20 m"),
          ("uwoc_turbulence_moderate", "(c) Optical, clear ocean, 20 m, turbulence"),
          ("uwoc_turbid", "(d) Optical, turbid harbor, 2 m (mismatch)")]
# (file prefix, scheme inside the file, legend label, style)
CURVES = [("ideal_separation", "ideal_separation", "Ideal separation bound",
           {"color": "0.4", "linestyle": "--"}),
          ("jpeg2000_ldpc", "ideal_adaptation", "JPEG 2000 + LDPC (ideal adaptation)",
           {"color": "tab:red", "marker": "s"}),
          ("deepjscc", "deepjscc", "DeepJSCC", {"color": "tab:orange", "marker": "o"}),
          ("adjscc", "adjscc", "ADJSCC", {"color": "tab:blue", "marker": "^"})]
LABELS = {"psnr": "PSNR (dB), incl. failures", "ms_ssim": "MS-SSIM, incl. failures",
          "lpips": "LPIPS, incl. failures (lower is better)"}


def curve(results: Path, system: str, scenario: str, scheme: str, metric: str):
    d = json.loads((results / f"{system}_{scenario}_uieb_test.json").read_text(encoding="utf-8"))
    rows = [r for r in d["results"] if r["scheme"] == scheme]
    if not rows:
        raise ValueError(f"no scheme {scheme!r} in {system}_{scenario}")
    stats = [r["metrics"][f"{metric}_all"] for r in rows]
    return [r["snr_db"] for r in rows], [s["mean"] for s in stats], [s["ci95"] for s in stats]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("reference_results/uieb_test"))
    ap.add_argument("--metric", choices=sorted(LABELS), default="psnr")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/psnr"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, (scenario, title) in zip(axes, PANELS, strict=True):
        for system, scheme, label, style in CURVES:
            snr, mean, ci = curve(args.results, system, scenario, scheme, args.metric)
            ax.plot(snr, mean, label=label, markersize=3, linewidth=1.2, **style)
            ax.fill_between(snr, [c[0] for c in ci], [c[1] for c in ci],
                            color=style["color"], alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Transmitted SNR (dB)" if scenario.startswith("uwoc") else "SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(LABELS[args.metric])
    axes[0].legend(fontsize=7, loc="upper right" if args.metric == "lpips" else "lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.png and .pdf")


if __name__ == "__main__":
    main()
