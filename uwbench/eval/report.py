"""``uwbench report``: figures and summary tables from ``uwbench evaluate`` JSON files.

Results are grouped by (scenario, dataset, split): every system / bandwidth ratio /
scheme becomes one curve of the group's figure, so several rates and systems can be
compared on the same axes. Per group, a PNG with four panels versus SNR: PSNR, MS-SSIM
and LPIPS including failures (mean with 95 % CI band) and the failure rate (linear axis,
so that a drop to zero is drawn where it happens). Tables (Markdown and LaTeX): mean
PSNR including failures, and failure rate, per curve and SNR.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

Series = dict[str, list[dict[str, Any]]]  # label -> entries sorted by SNR
PANELS = (("psnr_all", "PSNR incl. failures (dB)"), ("ms_ssim_all", "MS-SSIM incl. failures"),
          ("lpips_all", "LPIPS incl. failures (lower is better)"))


def load(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def grouped(results: list[dict[str, Any]]) -> dict[str, Series]:
    groups: dict[str, Series] = defaultdict(dict)
    for res in results:
        key = f"{res['scenario']['name']}_{res.get('dataset', 'data')}_{res.get('split', 'all')}"
        system = res["system"]
        rho = system.get("bandwidth_ratio", system.get("params", {}).get("bandwidth_ratio"))
        for e in res["results"]:
            label = f"{system['name']} (k/n={rho:.3g}): {e['scheme']}" if rho is not None \
                else f"{system['name']}: {e['scheme']}"
            groups[key].setdefault(label, []).append(e)
    return {k: {lab: sorted(v, key=lambda e: e["snr_db"]) for lab, v in s.items()}
            for k, s in groups.items()}


def plot(key: str, series: Series, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, entries in series.items():
        snr = [e["snr_db"] for e in entries]
        style = {"linestyle": "--"} if "ideal" in label else {"marker": "o", "markersize": 3}
        for ax, (metric, _) in zip(axes.flat[:3], PANELS, strict=True):
            m = [e["metrics"][metric]["mean"] for e in entries]
            line, = ax.plot(snr, m, label=label, **style)
            ci = [e["metrics"][metric]["ci95"] for e in entries]
            if all(c and c[0] == c[0] for c in ci):  # skip NaN intervals (n = 1)
                ax.fill_between(snr, [c[0] for c in ci], [c[1] for c in ci],
                                color=line.get_color(), alpha=0.15)
        axes.flat[3].plot(snr, [e["failure_rate"]["mean"] for e in entries], label=label,
                          **style)
    for ax, (_, ylabel) in zip(axes.flat, (*PANELS, (None, "failure rate")), strict=True):
        ax.set(xlabel="SNR (dB)", ylabel=ylabel)
        ax.grid(alpha=0.3)
    axes.flat[3].set_ylim(-0.02, 1.02)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(key)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{100 * v:.0f}" if pct else f"{v:.1f}"


def tables(groups: dict[str, Series]) -> tuple[str, str]:
    """Markdown and LaTeX tables of mean PSNR incl. failures and failure rate (%)."""
    md, tex = [], []
    for key, series in groups.items():
        snrs = sorted({e["snr_db"] for es in series.values() for e in es})
        head = [f"{s:g}" for s in snrs]
        for title, get, pct in (
                ("PSNR incl. failures (dB)", lambda e: e["metrics"]["psnr_all"]["mean"], False),
                ("Failure rate (%)", lambda e: e["failure_rate"]["mean"], True)):
            md += [f"### {key}: {title}", "", "| Scheme | " + " | ".join(head) + " |",
                   "| --- |" + " ---: |" * len(snrs)]
            tex += [f"% {key}: {title}", r"\begin{tabular}{l" + "r" * len(snrs) + "}",
                    r"\toprule", "Scheme & " + " & ".join(head) + r" \\", r"\midrule"]
            for label, entries in series.items():
                by_snr = {e["snr_db"]: get(e) for e in entries}
                cells = [_fmt(by_snr.get(s), pct) for s in snrs]
                md.append(f"| {label} | " + " | ".join(cells) + " |")
                tex.append(label.replace("_", r"\_").replace("&", r"\&") + " & "
                           + " & ".join(cells) + r" \\")
            md.append("")
            tex += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(md), "\n".join(tex)


def report(paths: list[Path], out_dir: Path) -> list[Path]:
    groups = grouped(load(paths))
    written = [plot(k, s, out_dir) for k, s in groups.items()]
    md, tex = tables(groups)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.tex").write_text(tex, encoding="utf-8")
    return [*written, out_dir / "report.md", out_dir / "report.tex"]
