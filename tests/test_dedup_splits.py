import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from uwbench.data.dedup import (
    Sample,
    group_duplicates,
    image_fingerprint,
    near_pairs,
    read_groups,
    run_dedup,
)
from uwbench.data.splits import (
    assign_splits,
    build_splits,
    check_no_leak,
    cross_dataset,
    remove_test_duplicates,
    split_checksums,
)

REPO = Path(__file__).resolve().parents[1]


def smooth_image(rng, size=64):
    """Low-frequency random image: JPEG keeps it perceptually close."""
    small = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
    return small.resize((size, size), Image.BICUBIC)


def popcount(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------- primitives

def test_near_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 40, dtype=np.uint64)
    ph = np.concatenate([base, base ^ np.uint64(0b111)])  # 40 pairs at distance 3
    got = {tuple(p) for p in near_pairs(ph, 5, chunk=7).tolist()}  # chunk not dividing N
    want = {(i, j, popcount(int(ph[i]) ^ int(ph[j])))
            for i in range(len(ph)) for j in range(i + 1, len(ph))
            if popcount(int(ph[i]) ^ int(ph[j])) <= 5}
    assert got == want and len(want) >= 40


def test_fingerprint_is_by_pixels_and_content(tmp_path):
    im = smooth_image(np.random.default_rng(1))
    im.save(tmp_path / "a.png")
    im.save(tmp_path / "b.bmp")
    im.save(tmp_path / "c.png", format="JPEG", quality=90)  # JPEG data, .png name
    a, b, c = (image_fingerprint(tmp_path / n) for n in ("a.png", "b.bmp", "c.png"))
    assert a == b  # same pixels, different format: identical fingerprints
    assert a[0] != c[0]  # lossy copy: different pixels
    assert popcount(int(a[1], 16) ^ int(c[1], 16)) <= 6  # but perceptually close


def test_groups_are_transitive_and_labelled_by_oldest_dataset():
    rows = [{"dataset": d, "sample_id": s} for d, s in
            [("new", "a"), ("old", "z"), ("old", "b"), ("new", "c"), ("new", "d")]]
    # chain new:a - old:z - old:b ; separate pair new:c - new:d
    groups = group_duplicates(rows, [(0, 1), (1, 2), (3, 4)], {"old": 0, "new": 1})
    assert groups == [[("old", "b"), ("old", "z"), ("new", "a")], [("new", "c"), ("new", "d")]]


def _samples(n, split=None, ds="toy", fmt="s{:03d}"):
    return [Sample(ds, fmt.format(i), split, {"image": f"{i}.png"}) for i in range(n)]


def _ids(samples):
    return {s.sample_id for s in samples}


def test_assign_splits_80_10_10_deterministic():
    samples = _samples(100)
    a = assign_splits(samples, {}, seed=0, val_fraction=0.1, test_fraction=0.1)
    assert [len(a[k]) for k in ("train", "val", "test")] == [80, 10, 10]
    assert a == assign_splits(samples, {}, seed=0, val_fraction=0.1, test_fraction=0.1)
    assert a["test"] != assign_splits(samples, {}, 1, 0.1, 0.1)["test"]


def test_assign_splits_keeps_duplicate_groups_and_units_together():
    samples = _samples(60, fmt="vid_{:02d}_frame0")  # 60 distinct ids...
    samples += [Sample("toy", f"vid_{i:02d}_frame1", None, {"image": "x"}) for i in range(60)]
    groups = {("toy", "vid_00_frame0"): "g", ("toy", "vid_05_frame1"): "g"}
    parts = assign_splits(samples, groups, 0, 0.1, 0.1, unit_pattern=r"^(vid_\d+)_frame")
    where = {s.sample_id: k for k, v in parts.items() for s in v}
    for i in range(60):  # both frames of each video in the same split
        assert where[f"vid_{i:02d}_frame0"] == where[f"vid_{i:02d}_frame1"]
    assert where["vid_00_frame0"] == where["vid_05_frame0"]  # joined through group g


def test_drawn_test_split_avoids_samples_shared_with_other_datasets():
    samples = _samples(100)
    groups = {("toy", f"s{i:03d}"): f"g{i}" for i in range(50)}
    parts = assign_splits(samples, groups, 0, 0.1, 0.1,
                          cross_groups={f"g{i}" for i in range(50)})
    assert len(parts["test"]) == 10
    assert all(int(s.sample_id[1:]) >= 50 for s in parts["test"])


def test_assign_splits_keeps_official_test_intact():
    samples = _samples(50, "trainval") + _samples(7, "test", fmt="t{}")
    parts = assign_splits(samples, {}, 0, 0.1, 0.1)
    assert _ids(parts["test"]) == {f"t{i}" for i in range(7)}
    assert len(parts["val"]) == 5 and len(parts["train"]) == 45


def test_train_copies_of_any_test_sample_are_removed():
    a = {"train": _samples(3, ds="a"), "val": [], "test": _samples(1, ds="a", fmt="t{}")}
    b = {"train": _samples(2, ds="b"), "val": [], "test": []}
    groups = {("a", "t0"): "g1", ("b", "s001"): "g1", ("a", "s002"): "g1",  # copies of test
              ("a", "s000"): "g2", ("b", "s000"): "g2"}  # train-only group: harmless
    removed = remove_test_duplicates({"a": a, "b": b}, groups)
    assert sorted((d, s) for d, s, _, _ in removed) == [("a", "s002"), ("b", "s001")]
    assert _ids(a["train"]) == {"s000", "s001"} and _ids(b["train"]) == {"s000"}


def test_cross_dataset_excludes_train_copies_of_any_test_split():
    parts = {"a": {"train": _samples(3, ds="a"), "val": [], "test": []},
             "b": {"train": _samples(2, ds="b", fmt="b{}"), "val": [], "test": []}}
    groups = {("a", "s001"): "g", ("b", "b1"): "g"}  # a's train copies b's *train* sample
    cross = cross_dataset({"train": ["a_train"], "val": ["a_val"],
                           "test": ["b_train", "b_val", "b_test"]}, parts, groups)
    assert _ids(cross["train"]) == {"s000", "s002"} and len(cross["test"]) == 2


# ---------------------------------------------------------------- end to end

@pytest.fixture
def two_datasets(tmp_path):
    """'old' (2019) and 'new' (2023); new contains an exact copy and a JPEG copy of old."""
    rng = np.random.default_rng(2)
    root = tmp_path / "data"
    imgs = [smooth_image(rng) for _ in range(40)]
    for sub in ("old/extracted/raw", "old/extracted/ref", "new/extracted/img"):
        (root / sub).mkdir(parents=True)
    for i, im in enumerate(imgs[:20]):
        im.save(root / f"old/extracted/raw/o{i:02d}.png")
        im.transpose(Image.FLIP_LEFT_RIGHT).save(root / f"old/extracted/ref/o{i:02d}.png")
    for i, im in enumerate(imgs[20:]):
        im.save(root / f"new/extracted/img/n{i:02d}.png")
    for i in range(10):  # old o00-o09 also appear in new, half of them re-encoded
        if i % 2:
            imgs[i].save(root / f"new/extracted/img/copy{i:02d}.jpg", quality=85)
        else:
            imgs[i].save(root / f"new/extracted/img/copy{i:02d}.png")
    registry = {
        "old": {"published": "2019-01",
                "samples": [{"roles": {"input": "raw", "reference": "ref"}}]},
        "new": {"published": "2023-01", "samples": [{"roles": {"image": "img"}}]},
    }
    cfg = {"seed": 0, "phash_threshold": 6, "contact_sheet_margin": 2,
           "contact_sheet_pairs": 4, "val_fraction": 0.1, "test_fraction": 0.1,
           "cross_dataset": {"new_to_old": {"train": ["new_train"], "val": ["new_val"],
                                            "test": ["old_train", "old_val", "old_test"]}}}
    return root, registry, cfg, tmp_path


def test_dedup_and_splits_end_to_end(two_datasets):
    root, registry, cfg, tmp = two_datasets
    split_dir = tmp / "splits"
    stats = run_dedup(["new", "old"], root, cfg, split_dir / "duplicates.csv", tmp / "report",
                      registry, workers=1)
    assert stats["group_label_order"] == ["old", "new"]
    groups = read_groups(split_dir / "duplicates.csv")
    assert all(groups.get(("new", f"copy{i:02d}")) == f"old:o{i:02d}" for i in range(10))

    parts, removed = build_splits(["old", "new"], root, cfg, split_dir, registry)
    assert check_no_leak(split_dir, cfg["cross_dataset"]) == []
    # drawn test splits only hold samples unique to their dataset: nothing to remove
    assert removed == []
    assert all(int(s.sample_id[1:]) >= 10 for s in parts["old"]["test"])
    assert not any(s.sample_id.startswith("copy") for s in parts["new"]["test"])
    # in the cross protocol, new's train holds no copy of any old sample
    cross_train = (split_dir / "cross/new_to_old_train.csv").read_text().splitlines()[1:]
    assert cross_train and not any(",copy" in line for line in cross_train)

    # A duplicate group spanning a train split and a test split is detected.
    test_id = parts["old"]["test"][0].sample_id
    train_id = parts["new"]["train"][0].sample_id
    with open(split_dir / "duplicates.csv", "a", encoding="utf-8") as f:
        f.write(f"old:{test_id},old,{test_id}\nold:{test_id},new,{train_id}\n")
    assert any("train/val and test" in p for p in check_no_leak(split_dir))


def test_leak_check_flags_sample_in_two_splits(tmp_path):
    (tmp_path / "duplicates.csv").write_text("group,dataset,sample_id\n")
    for split, ids in (("train", ["a", "b"]), ("val", []), ("test", ["b"])):
        (tmp_path / f"toy_{split}.csv").write_text("sample_id\n" + "".join(f"{i}\n" for i in ids))
    assert check_no_leak(tmp_path) == ["b in both toy_test.csv and toy_train.csv"]


# ---------------------------------------------------------------- versioned files

SPLITS_DIR = REPO / "splits"
built = pytest.mark.skipif(not (SPLITS_DIR / "removed.csv").exists(),
                           reason="splits not built yet")


@built
def test_versioned_splits_have_no_leak():
    cfg = yaml.safe_load((REPO / "configs/splits.yaml").read_text(encoding="utf-8"))
    assert check_no_leak(SPLITS_DIR, cfg["cross_dataset"]) == []


@built
def test_protocol_lists_current_split_checksums():
    text = (REPO / "PROTOCOL.md").read_text(encoding="utf-8")
    listed = dict(re.findall(r"\| `([\w./-]+\.csv)` \| `([0-9a-f]{64})` \|", text))
    assert listed == split_checksums(SPLITS_DIR)
