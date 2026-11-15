import zipfile
from pathlib import Path

import pytest

from uwbench.cli import main
from uwbench.data.download import download, extract
from uwbench.data.registry import load_registry
from uwbench.data.verify import sha256_file, verify, write_manifest

# ---------------------------------------------------------------- registry schema

REQUIRED = {"name", "task", "homepage", "license_page", "license", "license_status",
            "access", "sources", "checked", "files", "expected"}


@pytest.mark.parametrize("key,entry", sorted(load_registry().items()))
def test_registry_entry_schema(key, entry):
    assert REQUIRED <= entry.keys(), REQUIRED - entry.keys()
    assert entry["access"] in {"gdrive", "manual"}
    assert entry["sources"] and all(s.startswith("https://") for s in entry["sources"])
    assert entry["license_status"]  # TODO(verify) until confirmed by the maintainer
    if entry["access"] == "gdrive":
        assert entry["files"] and all(f["gdrive_id"] and f["name"] for f in entry["files"])
    else:
        assert entry["manual_instructions"]
    for e in entry["expected"]:
        assert isinstance(e["count"], int) and e["count"] > 0


# ---------------------------------------------------------------- fake dataset

def _zip_with(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


@pytest.fixture
def fake(tmp_path):
    """A one-archive gdrive dataset whose 'download' copies a local zip."""
    src = tmp_path / "src.zip"
    _zip_with(src, {"imgs/a.png": b"a", "imgs/b.png": b"b", "masks/a.png": b"m"})
    registry = {"toy": {
        "name": "Toy", "full_name": "toy", "license": "test-only", "license_status": "ok",
        "license_page": "https://example.org", "access": "gdrive",
        "files": [{"name": "toy.zip", "gdrive_id": "ID", "sha256": None}],
        "expected": [{"subset": "imgs", "path": "toy/imgs", "count": 2},
                     {"subset": "masks", "path": "toy/masks", "count": 1}],
    }}
    calls = []

    def fetch(file_id, output):
        calls.append(file_id)
        output.write_bytes(src.read_bytes())

    return registry, fetch, calls, tmp_path / "data", src


def test_download_requires_license_acceptance(fake):
    registry, fetch, calls, root, _ = fake
    with pytest.raises(PermissionError):
        download("toy", root, accept_license=False, registry=registry, fetch=fetch)
    assert calls == []


def test_download_extracts_and_verify_passes(fake, tmp_path):
    registry, fetch, calls, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)  # idempotent
    assert calls == ["ID"]
    assert (root / "toy/extracted/toy/imgs/a.png").read_bytes() == b"a"
    report = verify("toy", root, registry, manifest_dir=tmp_path / "manifests")
    assert report.ok, str(report)


def test_download_rejects_wrong_archive_checksum(fake):
    registry, fetch, _, root, _ = fake
    registry["toy"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256"):
        download("toy", root, accept_license=True, registry=registry, fetch=fetch)


def test_manual_access_fetches_nothing(fake, capsys):
    registry, fetch, calls, root, _ = fake
    registry["toy"].update(access="manual", manual_instructions="Ask the authors.")
    download("toy", root, registry=registry, fetch=fetch)
    assert calls == [] and "Ask the authors." in capsys.readouterr().out


def test_manual_access_extracts_deposited_archive(fake):
    registry, fetch, calls, root, src = fake
    registry["toy"].update(access="manual", manual_instructions="Ask the authors.")
    registry["toy"]["files"][0]["sha256"] = sha256_file(src)
    (root / "toy/archives").mkdir(parents=True)
    (root / "toy/archives/toy.zip").write_bytes(src.read_bytes())
    with pytest.raises(PermissionError):
        download("toy", root, registry=registry, fetch=fetch)
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    assert calls == [] and (root / "toy/extracted/toy/imgs/a.png").exists()


def test_download_skips_extracted_dataset_without_archive(fake):
    registry, fetch, calls, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    (root / "toy/archives/toy.zip").unlink()
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    assert calls == ["ID"]  # not fetched again


def test_verify_reports_count_mismatch_and_missing_dir(fake, tmp_path):
    registry, fetch, _, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    (root / "toy/extracted/toy/imgs/b.png").unlink()
    registry["toy"]["expected"].append({"subset": "x", "path": "toy/nope", "count": 1})
    report = verify("toy", root, registry, manifest_dir=tmp_path / "m")
    assert not report.ok
    assert any("1 files, expected 2" in p for p in report.problems)
    assert any("missing directory toy/nope" in p for p in report.problems)


def test_verify_archive_checksum(fake, tmp_path):
    registry, fetch, _, root, src = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    registry["toy"]["files"][0]["sha256"] = sha256_file(src)
    assert verify("toy", root, registry, manifest_dir=tmp_path / "m").ok
    registry["toy"]["files"][0]["sha256"] = "f" * 64
    assert not verify("toy", root, registry, manifest_dir=tmp_path / "m").ok


def test_manifest_detects_modified_missing_and_extra(fake, tmp_path):
    registry, fetch, _, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    mdir = tmp_path / "manifests"
    write_manifest("toy", root, mdir)
    registry["toy"]["expected"] = []  # rely on the manifest alone
    assert verify("toy", root, registry, manifest_dir=mdir).ok

    base = root / "toy/extracted/toy"
    (base / "imgs/a.png").write_bytes(b"changed")
    (base / "masks/a.png").unlink()
    (base / "new.txt").write_bytes(b"x")
    problems = " ".join(verify("toy", root, registry, manifest_dir=mdir).problems)
    assert "1 modified" in problems and "1 missing" in problems and "1 unexpected" in problems


def test_verify_fails_without_reference(fake, tmp_path):
    registry, fetch, _, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    for e in registry["toy"]["expected"]:
        e["path"] = None
    assert not verify("toy", root, registry, manifest_dir=tmp_path / "m").ok


def test_verify_missing_extraction(fake, tmp_path):
    registry, *_, root, _ = fake
    report = verify("toy", root, registry, manifest_dir=tmp_path / "m")
    assert not report.ok and "not found" in report.problems[0]


def test_extract_rejects_unknown_type(tmp_path):
    p = tmp_path / "x.7z"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        extract(p, tmp_path / "out")


def test_verify_rejects_empty_files(fake, tmp_path):
    registry, fetch, _, root, _ = fake
    download("toy", root, accept_license=True, registry=registry, fetch=fetch)
    (root / "toy/extracted/toy/imgs/a.png").write_bytes(b"")
    problems = verify("toy", root, registry, manifest_dir=tmp_path / "m").problems
    assert any("1 empty file" in p for p in problems)


def test_encrypted_rar_refuses_libarchive(monkeypatch, tmp_path):
    from uwbench.data import download as dl

    monkeypatch.setattr(dl.shutil, "which", lambda name: "bsdtar" if name == "bsdtar" else None)
    monkeypatch.setattr(dl, "SEVEN_ZIP_WINDOWS", tmp_path / "absent.exe")
    assert dl._rar_command(tmp_path / "a.rar", tmp_path, password="pw") is None
    monkeypatch.setattr(dl.shutil, "which", lambda name: "7z" if name == "7z" else None)
    assert "-ppw" in dl._rar_command(tmp_path / "a.rar", tmp_path, password="pw")


def test_failed_extraction_leaves_no_destination(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        extract(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # a later download retries instead of skipping


def test_cli_download_without_license_exits_2(tmp_path, capsys):
    assert main(["download", "uieb", "--root", str(tmp_path)]) == 2
    assert "--accept-license" in capsys.readouterr().out


def test_cli_verify_missing_dataset_exits_1(tmp_path):
    assert main(["verify", "suim", "--root", str(tmp_path)]) == 1
