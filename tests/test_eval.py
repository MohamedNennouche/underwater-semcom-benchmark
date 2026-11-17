"""Smoke test of the evaluation engine and the report on a synthetic mini-dataset
generated on the fly (no download), as run in CI."""

import json
import math

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("sionna")
from uwbench.eval.report import report  # noqa: E402
from uwbench.eval.runner import SCHEMA_VERSION, evaluate, mean_ci, wilson, write  # noqa: E402

SCENARIO = {"name": "awgn_smoke", "channel": "awgn",
            "snr_db": {"start": -5, "stop": 25, "step": 10}}
SEPARATION = {"name": "j2k_ldpc", "type": "separation", "codec": "jpeg2000",
              "bandwidth_ratio": 1 / 6, "mcs": [["bpsk", 0.5], ["qpsk", 0.5]]}
IDEAL = {"name": "ideal", "type": "ideal_separation", "codec": "jpeg2000",
         "bandwidth_ratio": 1 / 6}


def synthetic_images(n=3, h=192, w=200):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        small = Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8))
        out.append((f"synthetic_{i}", np.asarray(small.resize((w, h), Image.BICUBIC))))
    return out


def by_key(summary):
    return {(e["scheme"], e["snr_db"]): e for e in summary["results"]}


@pytest.fixture(scope="module")
def separation_run():
    return evaluate(SEPARATION, SCENARIO, synthetic_images(), seed=1)


def test_summary_schema_and_provenance(separation_run):
    summary, records = separation_run
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["snr_db"] == [-5.0, 5.0, 15.0, 25.0] and summary["n_images"] == 3
    prov = summary["provenance"]
    assert {"git_commit", "git_dirty", "libraries", "platform", "timestamp_utc"} <= prov.keys()
    assert "torch" in prov["libraries"] and "sionna-no-rt" in prov["libraries"]
    assert len(records) == 2 * 4 * 3  # schemes x SNRs x images
    e = by_key(summary)[("ideal_adaptation", 25.0)]
    assert {"psnr_all", "psnr_decoded", "ms_ssim_all", "lpips_all"} <= e["metrics"].keys()
    assert e["selected_scheme"].endswith("qpsk")  # more bits survive at high SNR


def test_cliff_behaviour(separation_run):
    res = by_key(separation_run[0])
    for scheme in ("jpeg2000+ldpc1024/2048+bpsk", "jpeg2000+ldpc1024/2048+qpsk"):
        low, high = res[(scheme, -5.0)], res[(scheme, 25.0)]
        assert low["failure_rate"]["mean"] == 1.0 and high["failure_rate"]["mean"] == 0.0
        assert high["metrics"]["psnr_all"]["mean"] > low["metrics"]["psnr_all"]["mean"] + 10
        assert low["metrics"]["psnr_decoded"]["mean"] is None  # nothing decoded


def test_results_do_not_depend_on_worker_count(separation_run):
    summary2, records2 = evaluate(SEPARATION, SCENARIO, synthetic_images(), seed=1, workers=2)
    metrics = ("psnr", "ms_ssim", "lpips")  # float sums may differ with the thread count
    for a, b in zip(records2, separation_run[1], strict=True):
        assert {k: v for k, v in a.items() if k not in metrics} == \
            {k: v for k, v in b.items() if k not in metrics}  # every decision is identical
        # abs: LPIPS can be ~1e-4, where GPU summation order alone moves the 7th digit
        assert [a[m] for m in metrics] == pytest.approx([b[m] for m in metrics], rel=1e-6,
                                                        abs=1e-6)
    assert len(summary2["results"]) == len(separation_run[0]["results"])


def test_interrupted_run_resumes_from_partial_file(tmp_path, separation_run, monkeypatch):
    from uwbench.eval import runner

    partial = tmp_path / "run.partial.jsonl"
    images = synthetic_images()
    full, _ = evaluate(SEPARATION, SCENARIO, images, seed=1, partial=partial)
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 3  # key + one line per image
    partial.write_text("\n".join(lines[:3]) + "\n" + lines[3][:50], encoding="utf-8")  # cut
    calls = []
    real = runner.evaluate_image
    monkeypatch.setattr(runner, "evaluate_image", lambda job: calls.append(job["index"])
                        or real(job))
    resumed, records = evaluate(SEPARATION, SCENARIO, images, seed=1, partial=partial)
    assert calls == [2]  # only the image whose line was cut is recomputed
    assert resumed["results"] == full["results"]
    resaved = partial.read_text(encoding="utf-8").splitlines()
    assert len(resaved) == 1 + 3 and all(json.loads(line) for line in resaved)  # cut line gone
    with pytest.raises(ValueError, match="another configuration"):
        evaluate(SEPARATION, SCENARIO, images, seed=2, partial=partial)


def test_worker_error_is_raised_with_several_workers(tmp_path):
    small = synthetic_images(1, 100, 100)  # too small for MS-SSIM: the worker raises
    system = dict(SEPARATION, mcs=[["bpsk", 0.5]])
    with pytest.raises(ValueError, match="MS-SSIM"):
        evaluate(system, SCENARIO, small + synthetic_images(3), seed=1, workers=2,
                 partial=tmp_path / "run.partial.jsonl")


def test_ideal_bound_and_report(tmp_path, separation_run):
    ideal, ideal_records = evaluate(IDEAL, SCENARIO, synthetic_images(), seed=1)
    sep = by_key(separation_run[0])
    for snr in (15.0, 25.0):  # the bound is above the best real scheme
        assert by_key(ideal)[("ideal_separation", snr)]["metrics"]["psnr_all"]["mean"] >= \
            sep[("ideal_adaptation", snr)]["metrics"]["psnr_all"]["mean"]
    paths = [write(dict(s, dataset="synthetic", split="test"), r, tmp_path, name)
             for name, (s, r) in (("sep", separation_run), ("ideal", (ideal, ideal_records)))]
    loaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert loaded["per_image_csv"] == "sep.csv" and (tmp_path / "sep.csv").exists()
    written = report(paths, tmp_path / "report")
    assert (tmp_path / "report" / "awgn_smoke_synthetic_test.png").exists()
    md = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "PSNR incl. failures" in md and "ideal_separation" in md
    assert r"\toprule" in (tmp_path / "report" / "report.tex").read_text(encoding="utf-8")
    assert len(written) == 3


def test_statistics():
    s = mean_ci([1.0, 2.0, 3.0, 4.0])
    half = 3.182446305284263 * math.sqrt(5 / 3) / 2  # t(0.975, 3) * sd / sqrt(n)
    assert s["mean"] == 2.5 and s["ci95"] == pytest.approx([2.5 - half, 2.5 + half])
    assert mean_ci([])["mean"] is None
    w = wilson(0, 10)
    assert w["mean"] == 0 and w["ci95"][0] == 0
    assert w["ci95"][1] == pytest.approx(0.2775, abs=1e-3)
