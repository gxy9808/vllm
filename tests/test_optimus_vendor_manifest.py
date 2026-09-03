# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import hashlib
from pathlib import Path, PurePosixPath

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "OPTIMUS_JIT_VENDOR_MANIFEST.tsv"
VENDORED_ROOTS = (
    REPO_ROOT / "vllm" / "models" / "step4" / "nvidia" / "ops" / "cute_dsl",
    REPO_ROOT / "vllm" / "models" / "step4" / "nvidia" / "ops" / "triton",
    REPO_ROOT / "csrc" / "libtorch_stable" / "step4",
)
VENDORED_PREFIXES = tuple(
    PurePosixPath(root.relative_to(REPO_ROOT).as_posix()) for root in VENDORED_ROOTS
)
LEGACY_VENDORED_PATHS = (
    REPO_ROOT / "optimus_cutedsl",
    REPO_ROOT / "optimus_triton",
    REPO_ROOT / "optimus_jit_compile_log.py",
)
IGNORED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so"}
NON_RUNTIME_PARTS = {
    "benchmark",
    "benchmarks",
    "experiment",
    "experimental",
    "testing",
    "tests",
    "training",
}
MANIFEST_COLUMNS = (
    "component",
    "vendored_path",
    "source_repository",
    "source_revision",
    "source_path",
    "source_mode",
    "source_git_blob",
    "source_sha256",
    "vendored_sha256",
    "state",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest() -> list[dict[str, str]]:
    rows = []
    with MANIFEST_PATH.open(newline="") as file:
        lines = (line for line in file if line.strip() and not line.startswith("#"))
        for values in csv.reader(lines, delimiter="\t"):
            assert len(values) == len(MANIFEST_COLUMNS)
            rows.append(dict(zip(MANIFEST_COLUMNS, values, strict=True)))
    return rows


def _source_inventory() -> set[str]:
    for root in VENDORED_ROOTS:
        assert root.is_dir(), f"missing vendored source root: {root}"

    return {
        path.relative_to(REPO_ROOT).as_posix()
        for root in VENDORED_ROOTS
        for path in root.rglob("*")
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.parts)
        and path.suffix not in IGNORED_SUFFIXES
    }


def _is_under_approved_root(path: PurePosixPath) -> bool:
    return any(path.is_relative_to(prefix) for prefix in VENDORED_PREFIXES)


def test_optimus_vendor_manifest_matches_source_tree():
    rows = _read_manifest()
    paths = [row["vendored_path"] for row in rows]

    assert len(paths) == len(set(paths)), "manifest contains duplicate paths"
    assert set(paths) == _source_inventory()

    for row in rows:
        relative_path = PurePosixPath(row["vendored_path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert _is_under_approved_root(relative_path)
        assert not NON_RUNTIME_PARTS.intersection(relative_path.parts)
        assert "_bwd" not in relative_path.name
        assert "backward" not in relative_path.name
        assert relative_path.name != "optimus_jit_compile_log.py"

        if relative_path.is_relative_to(VENDORED_PREFIXES[-1]):
            assert row["component"] == "step4_native"
        else:
            assert row["component"] == "optimus_jit"

        vendored_path = REPO_ROOT.joinpath(*relative_path.parts)
        assert vendored_path.is_file()
        assert HEX40.fullmatch(row["source_revision"])
        assert HEX40.fullmatch(row["source_git_blob"])
        assert HEX64.fullmatch(row["source_sha256"])
        assert HEX64.fullmatch(row["vendored_sha256"])
        assert row["source_mode"] in {"100644", "100755"}
        assert row["state"] in {"exact", "modified"}

        current_sha256 = _sha256(vendored_path)
        assert row["vendored_sha256"] == current_sha256
        assert (row["source_sha256"] == current_sha256) == (row["state"] == "exact")


def test_legacy_optimus_packages_are_not_shipped():
    for path in LEGACY_VENDORED_PATHS:
        assert not path.exists(), f"legacy top-level vendored path remains: {path}"
