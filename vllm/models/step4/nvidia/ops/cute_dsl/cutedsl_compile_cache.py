"""Persistent AOT cache for Optimus CuTeDSL compiled kernels."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections import namedtuple
from collections.abc import Hashable, Iterator, MutableMapping
from errno import EACCES, EAGAIN
from functools import lru_cache, wraps
from getpass import getuser
from importlib import metadata
from pathlib import Path
from typing import Any


_CACHE_ENABLED_ENV = "OPTIMUS_CUTEDSL_AOT_CACHE"
_CACHE_DIR_ENV = "OPTIMUS_CUTEDSL_AOT_CACHE_DIR"
_CACHE_TARGET_ARCH_ENV = "OPTIMUS_CUTEDSL_AOT_CACHE_ARCH"
_CUTE_DSL_ARCH_ENV = "CUTE_DSL_ARCH"
_EXPORT_FUNCTION_NAME_PREFIX = "optimus_cutedsl_aot_"
_METADATA_SUFFIX = ".meta.pkl"
_MANIFEST_SUFFIX = ".manifest.json"
_LOCK_TIMEOUT_SECONDS = 60.0
_CACHE_FORMAT_VERSION = 2
_SOURCE_PACKAGES = (
    "torch",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "cuda-python",
    "apache-tvm-ffi",
)
_LOADED_MODULES_LOCK = threading.Lock()
_LOADED_AOT_MODULES: dict[tuple[int, str, str], Any] = {}
_CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def cutedsl_aot_cache_enabled() -> bool:
    value = os.getenv(_CACHE_ENABLED_ENV)
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_cutedsl_aot_cache_path() -> Path:
    configured = os.getenv(_CACHE_DIR_ENV)
    if configured:
        cache_dir = Path(configured)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "optimus_cutedsl_aot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class _FileLock:
    def __init__(self, path: Path, *, exclusive: bool, timeout: float) -> None:
        self._path = path
        self._exclusive = exclusive
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o666)
        lock_type = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if exc.errno not in {EACCES, EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out waiting for CuTeDSL cache lock: {self._path}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class CutedslJitCache(MutableMapping[Hashable, Any]):
    def __init__(self, namespace: str | None = None) -> None:
        self._cache: dict[Hashable, Any] = {}
        self._namespace = namespace

    def __getitem__(self, key: Hashable) -> Any:
        return self._cache[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._cache[key] = value

    def __delitem__(self, key: Hashable) -> None:
        del self._cache[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache


class CutedslAotCache(CutedslJitCache):
    def __init__(self, namespace: str | None = None) -> None:
        super().__init__(namespace=namespace)
        self._cache_base_path = (
            get_cutedsl_aot_cache_path()
            / f"v{_CACHE_FORMAT_VERSION}"
            / _source_fingerprint()
        )
        self._resolved_cache_path: Path | None = None

    @property
    def _cache_path(self) -> Path:
        if self._resolved_cache_path is None:
            cache_path = self._cache_base_path / _target_fingerprint()
            if self._namespace:
                cache_path = cache_path / self._namespace
            cache_path.mkdir(parents=True, exist_ok=True)
            self._resolved_cache_path = cache_path
        return self._resolved_cache_path

    def __contains__(self, key: object) -> bool:
        if key in self._cache:
            return True
        if not isinstance(key, Hashable):
            return False
        return self._try_load_from_disk(key)

    def __getitem__(self, key: Hashable) -> Any:
        if key not in self:
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: Hashable, default: Any = None) -> Any:
        if key in self:
            return super().__getitem__(key)
        return default

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._try_export_to_disk(key, value)
        super().__setitem__(key, value)

    def clear(self) -> None:
        super().clear()
        for child in self._cache_path.iterdir():
            if child.is_file():
                child.unlink()

    def _try_load_from_disk(self, key: Hashable) -> bool:
        digest = _key_digest(key)
        shared_lib_path = self._cache_path / f"{digest}.so"
        with _FileLock(self._cache_path / f"{digest}.lock", exclusive=False, timeout=_LOCK_TIMEOUT_SECONDS):
            if not shared_lib_path.exists():
                return False
            manifest = self._validate_manifest(digest)
            _preload_cutedsl_runtime_libraries()
            import cutlass.cute as cute

            module = _get_or_load_cutedsl_module(
                cute,
                shared_lib_path,
                manifest["artifact_sha256"],
            )
            value = getattr(module, manifest["export_function_name"])
            metadata_path = self._metadata_path(digest)
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    value = (value, *pickle.load(f))
            super().__setitem__(key, value)
            return True

    def _try_export_to_disk(self, key: Hashable, value: Any) -> None:
        exportable_value, metadata = self._split_exportable_value(value)
        export_to_c = getattr(exportable_value, "export_to_c", None)
        if export_to_c is None:
            raise TypeError(
                "CuTeDSL AOT cache only supports JitCompiledFunction values "
                f"with export_to_c(); got {type(value)!r}"
            )
        digest = _key_digest(key)
        shared_lib_path = self._cache_path / f"{digest}.so"
        with _FileLock(self._cache_path / f"{digest}.lock", exclusive=True, timeout=_LOCK_TIMEOUT_SECONDS):
            if shared_lib_path.exists():
                self._validate_manifest(digest)
                return
            object_path = shared_lib_path.with_suffix(".o")
            tmp_shared_lib_path = shared_lib_path.with_suffix(".so.tmp")
            try:
                _export_to_object_file(
                    export_to_c,
                    object_path,
                    self._export_function_name(digest),
                )
                _link_shared_library(object_path, tmp_shared_lib_path)
                os.replace(tmp_shared_lib_path, shared_lib_path)
                self._write_metadata(digest, metadata)
                self._write_manifest(digest)
            finally:
                object_path.unlink(missing_ok=True)
                tmp_shared_lib_path.unlink(missing_ok=True)

    def _metadata_path(self, digest: str) -> Path:
        return self._cache_path / f"{digest}{_METADATA_SUFFIX}"

    def _manifest_path(self, digest: str) -> Path:
        return self._cache_path / f"{digest}{_MANIFEST_SUFFIX}"

    @staticmethod
    def _split_exportable_value(value: Any) -> tuple[Any, tuple[Any, ...] | None]:
        if isinstance(value, tuple) and value:
            return value[0], value[1:]
        return value, None

    def _write_metadata(self, digest: str, metadata: tuple[Any, ...] | None) -> None:
        metadata_path = self._metadata_path(digest)
        if metadata is None:
            if metadata_path.exists():
                metadata_path.unlink()
            return
        tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(metadata, f)
        os.replace(tmp_path, metadata_path)

    def _write_manifest(self, digest: str) -> None:
        manifest_path = self._manifest_path(digest)
        manifest = self._expected_manifest(digest)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
        os.replace(tmp_path, manifest_path)

    def _expected_manifest(self, digest: str) -> dict[str, Any]:
        metadata_path = self._metadata_path(digest)
        return {
            "format_version": _CACHE_FORMAT_VERSION,
            "source_fingerprint": _source_fingerprint(),
            "target_fingerprint": _target_fingerprint(),
            "namespace": self._namespace or "",
            "key_digest": digest,
            "export_function_name": self._export_function_name(digest),
            "artifact_sha256": _file_sha256(self._cache_path / f"{digest}.so"),
            "metadata_sha256": (
                _file_sha256(metadata_path) if metadata_path.exists() else None
            ),
        }

    def _export_function_name(self, digest: str) -> str:
        return _aot_export_function_name(self._namespace, digest)

    def _validate_manifest(self, digest: str) -> dict[str, Any]:
        manifest_path = self._manifest_path(digest)
        if not manifest_path.exists():
            raise RuntimeError(
                "Refusing to load a CuTeDSL AOT artifact without a v2 compatibility "
                f"manifest: {self._cache_path / f'{digest}.so'}. The cache entry is "
                "legacy or incomplete and must be rebuilt by the current Optimus JIT."
            )
        try:
            with open(manifest_path, encoding="utf-8") as f:
                actual = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid CuTeDSL AOT compatibility manifest: {manifest_path}: {exc}"
            ) from exc
        expected = self._expected_manifest(digest)
        mismatches = {
            field: {"expected": expected[field], "actual": actual.get(field)}
            for field in expected
            if actual.get(field) != expected[field]
        }
        if mismatches:
            raise RuntimeError(
                "Refusing to load an incompatible or corrupted CuTeDSL AOT artifact "
                f"at {self._cache_path / f'{digest}.so'}: {mismatches}"
            )
        return expected


def get_cutedsl_jit_cache(namespace: str | None = None) -> CutedslJitCache:
    if cutedsl_aot_cache_enabled():
        return CutedslAotCache(namespace=namespace)
    return CutedslJitCache(namespace=namespace)


def get_cutedsl_jit_cache_for_callable(fn: Any) -> CutedslJitCache:
    return get_cutedsl_jit_cache(_callable_cache_namespace(fn))


def cached_compile_function(fn: Any) -> Any:
    cache = get_cutedsl_jit_cache_for_callable(fn)
    hits = 0
    misses = 0

    @wraps(fn)
    def _wrapped(*args: Hashable, **kwargs: Hashable) -> Any:
        nonlocal hits, misses
        key = (args, tuple(sorted(kwargs.items())))
        cached = cache.get(key)
        if cached is not None:
            hits += 1
            return cached
        misses += 1
        value = fn(*args, **kwargs)
        cache[key] = value
        return value

    def _cache_clear() -> None:
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def _cache_info() -> Any:
        return _CacheInfo(hits, misses, None, len(cache))

    _wrapped.cache_clear = _cache_clear  # type: ignore[attr-defined]
    _wrapped.cache_info = _cache_info  # type: ignore[attr-defined]
    return _wrapped


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    h.update(f"python={sys.version_info.major}.{sys.version_info.minor}".encode())
    for package in _SOURCE_PACKAGES:
        h.update(f"{package}.version={_package_version(package)}".encode())
        h.update(f"{package}.record={_package_record_digest(package)}".encode())
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        data = src.read_bytes()
        h.update(src.relative_to(root).as_posix().encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _target_fingerprint() -> str:
    h = hashlib.sha256()
    for key, value in _target_fields():
        h.update(f"{key}={value}".encode())
    return h.hexdigest()


def _target_fields() -> tuple[tuple[str, str], ...]:
    libc_name, libc_version = platform.libc_ver()
    return (
        (_CACHE_TARGET_ARCH_ENV, os.getenv(_CACHE_TARGET_ARCH_ENV, "")),
        (_CUTE_DSL_ARCH_ENV, os.getenv(_CUTE_DSL_ARCH_ENV, "")),
        *_torch_cuda_target_fields(),
        ("cuda.driver_version", _cuda_driver_version()),
        ("platform.machine", platform.machine()),
        ("platform.libc", f"{libc_name}:{libc_version}"),
        ("python.soabi", str(sysconfig.get_config_var("SOABI") or "unknown")),
    )


def _torch_cuda_target_fields() -> tuple[tuple[str, str], ...]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return (("torch.import", f"unavailable:{type(exc).__name__}"),)

    torch_package_version = getattr(torch, "__version__", "unknown")
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "unknown"
    capability = "unknown"
    device_name = "unknown"
    arch_list = "unknown"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                device = cuda.current_device()
                major, minor = cuda.get_device_capability(device)
                capability = f"sm_{major}{minor}"
                device_name = str(cuda.get_device_name(device))
                arch_list = ",".join(str(arch) for arch in cuda.get_arch_list())
        except Exception as exc:
            capability = f"unavailable:{type(exc).__name__}"
    return (
        ("torch.version", str(torch_package_version)),
        ("torch.version.cuda", str(torch_cuda_version)),
        ("torch.cuda.device_capability", capability),
        ("torch.cuda.device_name", device_name),
        ("torch.cuda.arch_list", arch_list),
    )


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _package_record_digest(package: str) -> str:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return "missing"
    record = distribution.read_text("RECORD")
    if record is None:
        record = distribution.read_text("METADATA")
    if record is None:
        return "missing"
    return hashlib.sha256(record.encode()).hexdigest()


def _cuda_driver_version() -> str:
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        result = libcuda.cuDriverGetVersion(ctypes.byref(version))
    except (AttributeError, OSError) as exc:
        return f"unavailable:{type(exc).__name__}"
    if result != 0:
        return f"error:{result}"
    return str(version.value)


def _key_digest(key: Hashable) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_or_load_cutedsl_module(cute: Any, path: Path, artifact_sha256: str) -> Any:
    key = (os.getpid(), str(path.resolve()), artifact_sha256)
    with _LOADED_MODULES_LOCK:
        module = _LOADED_AOT_MODULES.get(key)
        if module is None:
            module = cute.runtime.load_module(str(path), enable_tvm_ffi=True)
            # Note(wangbojun/codex): A TVM FFI Function does not promise to
            # retain the dlopen/CUDA module that owns its function handle.
            _LOADED_AOT_MODULES[key] = module
        return module


def _callable_cache_namespace(fn: Any) -> str:
    code = getattr(fn, "__code__", None)
    raw_file = getattr(code, "co_filename", None)
    root = Path(__file__).resolve().parent
    if raw_file:
        try:
            src = Path(raw_file).resolve().relative_to(root).with_suffix("").as_posix()
        except ValueError:
            src = Path(raw_file).resolve().stem
    else:
        src = "unknown"
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))
    first_line = getattr(code, "co_firstlineno", 0)
    raw = f"{src}:{qualname}:{first_line}"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"auto/{safe[:96]}_{digest}"


def _aot_export_function_name(namespace: str | None, digest: str) -> str:
    identity = f"{namespace or ''}\0{digest}".encode()
    return _EXPORT_FUNCTION_NAME_PREFIX + hashlib.sha256(identity).hexdigest()


def _export_to_object_file(
    export_to_c: Any,
    obj_path: Path,
    function_name: str,
) -> None:
    parameters = inspect.signature(export_to_c).parameters
    if "object_file_path" in parameters:
        export_to_c(object_file_path=str(obj_path), function_name=function_name)
        return
    if "file_path" in parameters:
        export_to_c(
            file_path=str(obj_path.parent),
            file_name=obj_path.stem,
            function_prefix=function_name,
        )
        return
    raise TypeError(f"Unsupported CuTeDSL export_to_c signature: {inspect.signature(export_to_c)}")


def _link_shared_library(object_path: Path, shared_lib_path: Path) -> None:
    try:
        subprocess.run(
            ["cc", "-shared", "-o", str(shared_lib_path), str(object_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "CuTeDSL AOT cache needs a C compiler named `cc` to link its exported object file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CuTeDSL AOT cache failed to link its exported object file: "
            f"{exc.stderr.strip()}"
        ) from exc


@lru_cache(maxsize=1)
def _preload_cutedsl_runtime_libraries() -> None:
    import cutlass.cute as cute

    for lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        path = Path(lib_path)
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
