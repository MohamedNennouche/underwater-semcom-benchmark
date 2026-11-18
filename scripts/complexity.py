"""Parameters, FLOPs and latency of the learned models, for the complexity table.

Example (run on the platform the paper reports, here an RTX 3090):

    python scripts/complexity.py --size 256 --device cuda

For each model config and symbol type (complex: acoustic, real: optical) it counts the
parameters, the FLOPs of encode + decode for one ``size x size`` RGB image (one multiply-add
= 2 FLOPs, see :mod:`uwbench.metrics.complexity`) and the latency of encode + decode at
batch size 1. The channel is not included. Random weights are used: neither the FLOPs nor
the latency depend on the trained values. Writes ``results/complexity.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from uwbench.metrics.complexity import count_flops, count_parameters, measure_latency
from uwbench.models import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["deepjscc", "adjscc"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/complexity.json"))
    args = ap.parse_args()

    rows = []
    for name in args.models:
        cfg = yaml.safe_load(Path(f"configs/models/{name}.yaml").read_text(encoding="utf-8"))
        for symbols in ("complex", "real"):
            model = load_model(cfg, symbols, device=args.device)
            x = torch.rand(1, 3, args.size, args.size, device=args.device)
            snr = torch.full((1,), 10.0, device=args.device)

            def run(img: torch.Tensor) -> torch.Tensor:
                return model.decode(model.encode(img, snr), snr)

            lat = measure_latency(run, x, repeats=50)
            rows.append({"model": name, "symbols": symbols, "size": args.size,
                         "parameters": count_parameters(model),
                         "flops_per_image": count_flops(run, x), **lat})
            print(f"{name:9s} {symbols:7s} params {rows[-1]['parameters'] / 1e6:6.2f} M  "
                  f"GFLOPs {rows[-1]['flops_per_image'] / 1e9:7.2f}  "
                  f"latency {lat['median_ms']:7.2f} ms ({lat['device']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
