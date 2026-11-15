"""``uwbench download <dataset>``: fetch official archives or print the manual procedure.

Automatic download is only used when the registry gives an official Google Drive file
(``access: gdrive``, fetched with ``gdown``). For ``access: manual`` the procedure and the
target folder are printed and nothing is fetched. Downloading requires the user to state
that they read the dataset license (``accept_license=True`` / ``--accept-license``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from uwbench.data.registry import get_entry
from uwbench.data.verify import sha256_file

Fetch = Callable[[str, Path], None]


def gdown_fetch(file_id: str, output: Path) -> None:
    """Download one public Google Drive file to ``output``."""
    import gdown

    # resume=True continues an interrupted transfer from gdown's own "<output>*.part" file.
    if gdown.download(id=file_id, output=str(output), quiet=False, resume=True) is None:
        raise RuntimeError(f"gdown could not download Google Drive file {file_id}")


SEVEN_ZIP_WINDOWS = Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "7-Zip", "7z.exe")


def _rar_command(archive: Path, dest: Path, password: str | None = None) -> list[str] | None:
    # shortcut: relies on an external extractor; GNU tar cannot read RAR.
    pw = [f"-p{password}"] if password else []
    if exe := shutil.which("7z") or (
        str(SEVEN_ZIP_WINDOWS) if SEVEN_ZIP_WINDOWS.exists() else None
    ):
        return [exe, "x", "-y", *pw, f"-o{dest}", str(archive)]
    if exe := shutil.which("unrar"):
        return [exe, "x", "-o+", *pw, str(archive), f"{dest}{Path('/')}"]
    if password:
        return None  # libarchive cannot decrypt RAR: it writes empty files instead
    # Windows' own tar is bsdtar; named by full path because PATH may hold GNU tar (no RAR).
    # Anaconda's bsdtar 3.7.7 failed on UIEB's RAR files where Windows' 3.8.9 succeeded.
    system_tar = Path(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
    if os.name == "nt" and system_tar.exists():
        return [str(system_tar), "-xf", str(archive), "-C", str(dest)]
    if exe := shutil.which("bsdtar"):
        return [exe, "-xf", str(archive), "-C", str(dest)]
    return None


def extract(archive: Path, dest: Path, password: str | None = None) -> None:
    """Extract a ``.zip`` (stdlib) or ``.rar`` (7z, unrar or bsdtar) archive into ``dest``.

    Extraction goes to ``<dest>.partial`` and is renamed only on success, so ``dest``
    exists only for a complete extraction.
    """
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)  # zipfile strips absolute paths and '..' components
    elif suffix == ".rar":
        cmd = _rar_command(archive, tmp, password)
        if cmd is None:
            need = "7-Zip or unrar (encrypted archive)" if password else "7z, unrar or bsdtar"
            raise RuntimeError(f"no RAR extractor found: install {need}")
        subprocess.run(cmd, check=True)
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")
    tmp.replace(dest)


def download(
    key: str,
    root: Path,
    accept_license: bool = False,
    registry: dict[str, dict[str, Any]] | None = None,
    fetch: Fetch = gdown_fetch,
) -> Path:
    """Download and extract dataset ``key`` into ``root/key``; return that folder."""
    entry = get_entry(key, registry)
    target = root / key
    print(f"{entry['name']} - {entry.get('full_name', '')}\nLicense ({entry['license_status']}): "
          f"{entry['license']}\nLicense page: {entry['license_page']}\n")

    def pending(f: dict[str, Any]) -> bool:  # neither extracted nor deposited
        dest = target / "extracted" / Path(f["name"]).stem
        return not dest.exists() and not (target / "archives" / f["name"]).exists()

    if entry["access"] == "manual" and (not entry["files"] or any(map(pending, entry["files"]))):
        print(f"Manual access required.\n{entry['manual_instructions']}\n"
              f"Archive folder: {(target / 'archives').resolve()}")
        return target
    if not accept_license:
        raise PermissionError("read the license above, then re-run with --accept-license")

    for f in entry["files"]:
        archive = target / "archives" / f["name"]
        dest = target / "extracted" / Path(f["name"]).stem
        if dest.exists():  # complete extraction (extract() is atomic); archive may be deleted
            continue
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            part = archive.with_name(archive.name + ".part")
            fetch(f["gdrive_id"], part)
            part.replace(archive)  # only complete downloads get the final name
        digest = sha256_file(archive)
        if f.get("sha256") and digest != f["sha256"]:
            raise RuntimeError(f"{archive.name}: SHA-256 {digest} != registry {f['sha256']}")
        if not f.get("sha256"):
            print(f"{archive.name}: SHA-256 {digest} (not yet recorded in registry.yaml)")
        extract(archive, dest, f.get("password"))
    print(f"Done. Run: uwbench verify {key}")
    return target
