"""Dynamic builder/loader for the Step4 ``_signature`` C extension.

When the optional extension is missing, it is compiled under the vLLM Step4
cache root and imported from there. Cache keys mix source, revision,
Python/platform, and compiler identities so stale binaries are not reused.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Optional

CACHE_ENV_VAR = "VLLM_STEP4_JIT_CACHE_DIR"
LEGACY_CACHE_ENV_VAR = "OPTIMUS_JIT_CACHE_DIR"
MODULE_NAME = "vllm.models.step4.nvidia.ops.triton.utils._signature"
SOURCE_FILE = "_signature.cpp"
EXT_SUBDIR = ("extensions", "step4_signature")
PY_LIMITED_API_VALUE = "0x03100000"
_LOCK_TIMEOUT_SECS = 300

__all__ = ["load_or_build_signature_module"]


def load_or_build_signature_module() -> ModuleType | None:
    """Return the compiled `_signature` module, building it if needed."""

    source_path = Path(__file__).with_name(SOURCE_FILE)
    try:
        compiler_cmd = _compiler_command()
        cache_dir, metadata = _determine_cache_dir(source_path, compiler_cmd)
        suffix = _extension_suffix()
        target_path = cache_dir / f"_signature{suffix}"
        if not target_path.exists():
            _build_extension(source_path, target_path, cache_dir, metadata, compiler_cmd)
            print(f"Built _signature extension in {target_path}")
        module = _load_extension(target_path)
        return module
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"Failed to build/load _signature extension: {exc}", RuntimeWarning)
        return None


def _extension_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix:
        return suffix
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if not suffixes:
        raise RuntimeError("No known extension suffix")
    return suffixes[0]


def _determine_cache_dir(source_path: Path, compiler_cmd: List[str]) -> tuple[Path, dict]:
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    for subdir in EXT_SUBDIR:
        cache_root = cache_root / subdir
        cache_root.mkdir(parents=True, exist_ok=True)

    source_hex = _sha256_path(source_path)
    git_rev = _git_rev(source_path)
    git_key = (git_rev or "nogit")[:12]
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = f"{platform.system().lower()}-{platform.machine().lower()}"
    compiler_id = _compiler_identity(compiler_cmd)
    compiler_key = _safe_fragment(compiler_id.split()[0] if compiler_id else "compiler")
    command_repr = " ".join(compiler_cmd)

    key_seed = "|".join(
        [source_hex, git_rev or "nogit", platform_tag, py_tag, compiler_id, command_repr]
    )
    digest = hashlib.sha256(key_seed.encode("utf-8")).hexdigest()[:16]
    dir_name = f"{py_tag}-{platform_tag}-{compiler_key}-{git_key}-{source_hex[:12]}-{digest}"
    cache_dir = cache_root / dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(source_path),
        "source_sha256": source_hex,
        "git_revision": git_rev,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": compiler_id,
        "compiler_command": compiler_cmd,
        "cache_dir": str(cache_dir),
    }
    return cache_dir, metadata


def _cache_root() -> Path:
    env_value = os.environ.get(CACHE_ENV_VAR) or os.environ.get(
        LEGACY_CACHE_ENV_VAR
    )
    if env_value:
        return Path(env_value).expanduser()
    try:
        return Path.home().expanduser() / ".cache" / "vllm" / "step4"
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "vllm_step4"


def _build_extension(
    source_path: Path,
    target_path: Path,
    cache_dir: Path,
    metadata: dict,
    compiler_cmd: List[str],
) -> None:
    lock_path = cache_dir / ".build.lock"
    with _acquire_lock(lock_path):
        if target_path.exists():
            return
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        cmd = list(compiler_cmd)
        cmd.extend(_python_cflags())
        cmd.extend(["-std=c++17", "-O3", "-fPIC", "-shared"])
        cmd.append(f"-DPy_LIMITED_API={PY_LIMITED_API_VALUE}")
        for include_dir in _include_dirs():
            cmd.append(f"-I{include_dir}")
        cmd.extend(["-o", str(tmp_path), str(source_path)])
        cmd.extend(_python_link_args())
        cmd.extend(_python_ldflags())

        metadata_path = cache_dir / "metadata.json"
        metadata = {
            **metadata,
            "build_command": cmd,
            "built_at": time.time(),
        }
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise RuntimeError(
                f"Failed to build _signature extension (exit code {exc.returncode}):\n"
                f"STDOUT: {exc.stdout}\nSTDERR: {exc.stderr}"
            ) from exc
        else:
            metadata["compiler_stdout"] = result.stdout.strip()
            metadata["compiler_stderr"] = result.stderr.strip()
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, target_path)


def _compiler_command() -> List[str]:
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        return shlex.split(cxx_env)
    for candidate in ("c++", "g++", "clang++"):
        compiler = shutil.which(candidate)
        if compiler:
            return [compiler]
    raise RuntimeError("No suitable C++ compiler found; set the CXX environment variable")


def _python_cflags() -> List[str]:
    flags = _split_flags(sysconfig.get_config_var("CFLAGS"))
    return [flag for flag in flags if flag]


def _include_dirs() -> Iterable[str]:
    include = sysconfig.get_path("include")
    if include:
        yield include
    plat_include = sysconfig.get_path("platinclude")
    if plat_include and plat_include != include:
        yield plat_include


def _python_link_args() -> List[str]:
    args: List[str] = []
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        args.append(f"-L{libdir}")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if ldlibrary:
        libname = _lib_name(ldlibrary)
        if libname:
            args.append(f"-l{libname}")
        elif libdir:
            args.append(str(Path(libdir) / ldlibrary))
    libs = _split_flags(sysconfig.get_config_var("LIBS"))
    syslibs = _split_flags(sysconfig.get_config_var("SYSLIBS"))
    return args + libs + syslibs


def _python_ldflags() -> List[str]:
    return _split_flags(sysconfig.get_config_var("LDFLAGS"))


def _split_flags(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return shlex.split(value)


def _lib_name(ldlibrary: str) -> Optional[str]:
    if not ldlibrary.startswith("lib"):
        return None
    name = ldlibrary[3:]
    for suffix in (".so", ".a", ".dylib"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or None


@contextlib.contextmanager
def _acquire_lock(lock_path: Path):
    deadline = time.time() + _LOCK_TIMEOUT_SECS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="ascii") as lock_file:
                lock_file.write(str(os.getpid()))
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for build lock {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(lock_path)


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {MODULE_NAME} from {path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[MODULE_NAME] = module
    return module


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev(source_path: Path) -> Optional[str]:
    current = source_path.resolve()
    for ancestor in current.parents:
        if (ancestor / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(ancestor),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except Exception:
                return None
            else:
                return result.stdout.strip()
    return None


def _compiler_identity(cmd: List[str]) -> str:
    compiler_binary = _compiler_binary_from_cmd(cmd)
    try:
        result = subprocess.run(
            [compiler_binary, "--version"], capture_output=True, text=True, check=True
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else compiler_binary
        return first_line.strip()
    except Exception:
        return compiler_binary


def _compiler_binary_from_cmd(cmd: List[str]) -> str:
    wrappers = {"ccache", "sccache", "distcc", "icecc"}
    for token in cmd:
        name = Path(token).name
        if name in wrappers:
            continue
        if token.startswith("-"):
            continue
        return token
    return cmd[0]


def _safe_fragment(value: str) -> str:
    filtered = [c for c in value if c.isalnum() or c in ("-", "_")]
    return "".join(filtered) or "na"
