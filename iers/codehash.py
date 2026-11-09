"""Code hash stored in every result record (the project does not use version control)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import ROOT

EXTENSIONS = {".py", ".yaml", ".toml"}
EXCLUDED_TOP = {"results", "logs"}


def project_files(root: Path = ROOT) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        if path.relative_to(root).parts[0] in EXCLUDED_TOP:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def write_manifest(root: Path = ROOT) -> str:
    """The code hash, plus logs/code_manifests/<hash>.json with the SHA256 of every file under it.

    Any two hashes in the records can then be compared file by file. Returns the hash.
    """
    import json
    import os

    chash = code_hash(root)
    out = root / "logs" / "code_manifests" / f"{chash}.json"
    if not out.exists():
        files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in project_files(root)}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"code_hash": chash, "files": files}, indent=1), encoding="utf-8")
        os.replace(tmp, out)
    return chash


def code_hash(root: Path = ROOT) -> str:
    """SHA256 over the relative path and bytes of every .py, .yaml and .toml file."""
    digest = hashlib.sha256()
    for path in project_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
