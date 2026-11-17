"""``uwbench evaluate``: one system over a scenario's SNR grid on a dataset split.

For every image, scheme of the system, SNR of the grid and channel realisation
(``realizations`` per image), the image is transmitted and scored (PSNR, MS-SSIM, LPIPS
against the transmitted image). Results:

* ``<out>/<stem>.csv``: one row per (scheme, SNR, realisation, image);
* ``<out>/<stem>.json``: summary (schema :data:`SCHEMA_VERSION`), see :func:`summarize`,
  with the full system and scenario configurations, seeds and provenance (git commit,
  library versions, platform).

Systems (YAML): ``type: separation`` (``configs/baselines/``; one scheme per modulation
and code rate, plus the derived ``ideal_adaptation``: best scheme per SNR; LDPC decoding
on ``device``, ``auto`` = CUDA when available), ``type: ideal_separation``, or
``type: learned`` (``configs/models/``; any :class:`uwbench.models.base.JSCCModel` given
by ``model: "package.module:Class"``, ``params`` and ``checkpoint``; the whole image is
transmitted, padded to a multiple of the model's downsampling factor).

Long runs: every finished image is saved at once to ``<stem>.partial.jsonl`` and an
interrupted run resumes from it; one progress line is printed per image.

Speed: images can be processed in parallel worker processes (one thread each); every
image has its own generator ``default_rng([seed, image_index])``, so results do not
depend on the number of workers. With LDPC decoding on a small GPU, one worker is
usually best (several workers compete for the same GPU and memory). The source encoding
of an image is computed once per scheme and reused across SNRs and realisations;
metrics are cached per distinct reconstruction (all failures share the fallback image;
a decoded codestream gives the same image at every SNR).
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCHEMA_VERSION = "1.0"
MCS_DEFAULT = {"complex": [["bpsk", 0.5], ["qpsk", 0.5], ["16qam", 0.5]],
               "real": [["ook", 0.5], ["pam4", 0.5]]}
METRICS = ("psnr", "ms_ssim", "lpips")
LIBRARIES = ("numpy", "scipy", "torch", "torchvision", "pillow", "pytorch-msssim", "lpips",
             "sionna-no-rt", "imagehash", "pyyaml")
REPO = Path(__file__).resolve().parents[2]


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.tensor(img).permute(2, 0, 1)[None].float() / 255


def split_images(dataset: str, split: str, root: Path) -> list[tuple[str, Path]]:
    """``(sample_id, path)`` of the transmitted image of every sample of a split file:
    role ``input`` for paired datasets, ``image`` otherwise."""
    with open(REPO / "splits" / f"{dataset}_{split}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    role = "input" if "input" in rows[0] else "image"
    return [(r["sample_id"], root / dataset / "extracted" / r[role]) for r in rows]


def snr_values(spec: dict[str, float], step: float | None = None) -> list[float]:
    step = step or spec["step"]
    return [float(v) for v in np.arange(spec["start"], spec["stop"] + step / 2, step)]


# ---------------------------------------------------------------- per-image worker

def _init_worker() -> None:
    torch.set_num_threads(1)


def evaluate_image(job: dict[str, Any]) -> list[dict[str, Any]]:
    """All records of one image (run in a worker process)."""
    from uwbench.baselines.separation import SeparationScheme, ideal_separation
    from uwbench.channels import build_channel
    from uwbench.metrics.fidelity import lpips, ms_ssim, psnr

    system, scenario = job["system"], job["scenario"]
    img = job["image"] if isinstance(job["image"], np.ndarray) else \
        np.asarray(Image.open(job["image"]).convert("RGB"))
    ref = to_tensor(img)
    channel = build_channel(scenario["channel"], scenario.get("params"))
    family = "real" if scenario["channel"] == "uwoc" else "complex"
    rng = np.random.default_rng([job["seed"], job["index"]])
    rho, codec = system.get("bandwidth_ratio"), system.get("codec", "jpeg2000")
    cache: dict[str, dict[str, float]] = {}

    def scores(out: np.ndarray | torch.Tensor) -> dict[str, float]:
        if isinstance(out, torch.Tensor):  # analog output: every reconstruction differs
            return {"psnr": psnr(out, ref).item(), "ms_ssim": ms_ssim(out, ref).item(),
                    "lpips": lpips(out, ref).item()}
        key = hashlib.sha256(out.tobytes()).hexdigest()
        if key not in cache:
            x = to_tensor(out)
            cache[key] = {"psnr": psnr(x, ref).item(), "ms_ssim": ms_ssim(x, ref).item(),
                          "lpips": lpips(x, ref).item()}
        return cache[key]

    records = []

    def add(scheme: str, snr: float, r: int, res) -> None:
        records.append({"scheme": scheme, "snr_db": snr, "realization": r,
                        "image_index": job["index"], "sample_id": job["sample_id"],
                        "decoded": int(res.decoded), "bpp": res.bpp,
                        "bandwidth_ratio_used": res.channel_uses / img.size,
                        **scores(res.image)})

    if system["type"] == "learned":
        from uwbench.baselines.separation import Result
        from uwbench.models import load_model
        from uwbench.models.base import pad_to, transmit

        device = system.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(system, family, system["checkpoint"], device)
        x = ref.to(device)
        with torch.no_grad():
            latent = model.encode(pad_to(x, model.downsample), torch.zeros(1, device=device))
        uses = latent.numel() // (2 if family == "complex" else 1)
        for snr in job["snr_db"]:
            for r in range(job["realizations"]):
                with torch.no_grad():
                    out = transmit(model, x, channel, torch.full((1,), snr, device=device),
                                   rng, system.get("modulation_index", 1.0))
                add(system["name"], snr, r, Result(out.clamp(0, 1).cpu(), True, 0.0, uses))
        return records

    if system["type"] == "separation":
        device = system.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        for mod, rate in system.get("mcs") or MCS_DEFAULT[family]:
            scheme = SeparationScheme(mod, rate, codec, **system.get("ldpc", {}),
                                      device=device)
            enc = scheme.encode_source(img, rho)
            for snr in job["snr_db"]:
                for r in range(job["realizations"]):
                    add(scheme.name, snr, r, scheme.transmit(img, enc, rho, channel, snr, rng))
    elif system["type"] == "ideal_separation":
        for snr in job["snr_db"]:
            for r in range(job["realizations"]):
                add("ideal_separation", snr, r, ideal_separation(
                    img, rho, channel, snr, rng, codec, complex_symbols=family == "complex"))
    else:
        raise ValueError(f"unknown system type {system['type']!r}")
    return records


# ---------------------------------------------------------------- statistics

def mean_ci(values: Sequence[float]) -> dict[str, Any]:
    """Mean and 95 % confidence interval (Student t on the per-sample values)."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = float(v.mean())
    half = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)) \
        if v.size > 1 else float("nan")
    return {"mean": m, "ci95": [m - half, m + half], "n": int(v.size)}


def wilson(k: int, n: int) -> dict[str, Any]:
    """Proportion ``k / n`` with its 95 % Wilson score interval."""
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    z, p = 1.959963984540054, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"mean": p, "ci95": [max(0.0, centre - half), min(1.0, centre + half)], "n": n}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (scheme, SNR).

    ``metrics.<m>_all``: every transmission, undecoded images scored on the fallback
    image; ``metrics.<m>_decoded``: decoded images only; ``failure_rate``: Wilson
    interval; ``bpp``: achieved source rate of decoded transmissions;
    ``bandwidth_ratio_used``: channel uses actually sent per source component. For a
    system with several schemes, ``ideal_adaptation`` picks at each SNR the scheme with
    the best mean ``psnr_all`` (genie selection over the grid).
    """
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r["scheme"], r["snr_db"])].append(r)
    schemes = list(dict.fromkeys(r["scheme"] for r in records))
    if len(schemes) > 1:
        for snr in sorted({k[1] for k in groups}):
            best = max(schemes, key=lambda s: np.mean([r["psnr"] for r in groups[(s, snr)]]))
            groups[("ideal_adaptation", snr)] = [dict(r, selected=best)
                                                 for r in groups[(best, snr)]]
    out = []
    for (scheme, snr), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        dec = [r for r in rows if r["decoded"]]
        entry = {"scheme": scheme, "snr_db": snr, "n": len(rows),
                 "failure_rate": wilson(len(rows) - len(dec), len(rows)),
                 "bpp": mean_ci([r["bpp"] for r in dec]),
                 "bandwidth_ratio_used": mean_ci([r["bandwidth_ratio_used"] for r in rows]),
                 "metrics": {f"{m}_{k}": mean_ci([r[m] for r in sel])
                             for m in METRICS for k, sel in (("all", rows), ("decoded", dec))}}
        if scheme == "ideal_adaptation":
            entry["selected_scheme"] = rows[0]["selected"]
        out.append(entry)
    return out


# ---------------------------------------------------------------- provenance and run

def provenance() -> dict[str, Any]:
    from uwbench import __version__
    from uwbench.metrics.complexity import platform_info

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    libs = {}
    for name in LIBRARIES:
        try:
            libs[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "uwbench_version": __version__, "libraries": libs, "platform": platform_info(),
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv)}


def _load_partial(path: Path, key: str) -> dict[int, list[dict[str, Any]]]:
    """Records of the images already done by an interrupted run with the same ``key``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]).get("key") != key:
        raise ValueError(f"{path} belongs to another configuration; delete it to start over")
    done, valid = {}, lines[:1]
    for line in lines[1:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:  # last line cut by the interruption
            break
        done[entry["index"]] = entry["records"]
        valid.append(line)
    # drop the cut line, otherwise the next append would be glued to it
    path.write_text("\n".join(valid) + "\n", encoding="utf-8")
    return done


def evaluate(system: dict[str, Any], scenario: dict[str, Any],
             images: Sequence[tuple[str, Path | np.ndarray]], *, seed: int = 0,
             realizations: int = 1, workers: int = 1, snr_step: float | None = None,
             partial: Path | None = None, progress: bool = False
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate ``system`` on ``images``.

    With ``partial``, every finished image is appended to that JSON-lines file at once,
    and a run interrupted with the same configuration resumes from it (images already in
    the file are not recomputed). ``progress`` prints one line per finished image.
    """
    grid = snr_values(scenario["snr_db"], snr_step)
    jobs = [{"index": i, "sample_id": sid, "image": img, "system": system,
             "scenario": scenario, "snr_db": grid, "realizations": realizations,
             "seed": seed} for i, (sid, img) in enumerate(images)]
    key = hashlib.sha256(json.dumps(
        {"system": system, "scenario": scenario, "snr_db": grid, "realizations": realizations,
         "seed": seed, "images": [sid for sid, _ in images]}, sort_keys=True, default=str
    ).encode()).hexdigest()
    done: dict[int, list[dict[str, Any]]] = {}
    if partial is not None:
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists():
            done = _load_partial(partial, key)
        else:
            partial.write_text(json.dumps({"key": key}) + "\n", encoding="utf-8")
    todo = [j for j in jobs if j["index"] not in done]
    t0 = time.time()

    def finished(job: dict[str, Any], recs: list[dict[str, Any]]) -> None:
        done[job["index"]] = recs
        if partial is not None:
            with open(partial, "a", encoding="utf-8") as f:
                f.write(json.dumps({"index": job["index"], "records": recs}) + "\n")
        if progress:
            n_new = len(done) - (len(jobs) - len(todo))
            eta = (time.time() - t0) / n_new * (len(jobs) - len(done))
            print(f"[{len(done)}/{len(jobs)}] {job['sample_id']}  "
                  f"elapsed {time.time() - t0:.0f} s, remaining ~{eta:.0f} s", flush=True)

    if progress and len(todo) < len(jobs):
        print(f"resuming: {len(jobs) - len(todo)} of {len(jobs)} images already done",
              flush=True)
    if workers > 1 and todo:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
            futures = {ex.submit(evaluate_image, j): j for j in todo}
            try:
                for fut in as_completed(futures):
                    finished(futures[fut], fut.result())
            except BaseException:  # error or Ctrl+C: do not run the queued images unsaved
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    else:
        for j in todo:
            finished(j, evaluate_image(j))
    records = [r for i in sorted(done) for r in done[i]]
    summary = {"schema_version": SCHEMA_VERSION, "system": system, "scenario": scenario,
               "n_images": len(images), "snr_db": grid, "realizations": realizations,
               "seed": seed, "results": summarize(records), "provenance": provenance()}
    return summary, records


def write(summary: dict[str, Any], records: list[dict[str, Any]], out_dir: Path,
          stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    summary = dict(summary, per_image_csv=f"{stem}.csv")
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path
