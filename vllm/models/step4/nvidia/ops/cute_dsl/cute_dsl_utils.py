# Copyright (c) 2025, Tri Dao.

import os
import pathlib
import inspect
from typing import Any, Callable, Mapping, MutableMapping, Tuple, TypeVar
from functools import lru_cache, partial
from dataclasses import dataclass, fields

import torch
from vllm.models.step4.nvidia.ops.cute_dsl._cutlass_compat import apply_patches

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

_CacheKeyT = TypeVar("_CacheKeyT")
_CacheValueT = TypeVar("_CacheValueT")


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

fake_compile_torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int8: cutlass.Int8,
    torch.uint8: cutlass.Uint8,
    torch.int16: cutlass.Int16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.bool: cutlass.Boolean,
}
if hasattr(torch, "float8_e4m3fn") and hasattr(cutlass, "Float8E4M3FN"):
    fake_compile_torch2cute_dtype_map[torch.float8_e4m3fn] = cutlass.Float8E4M3FN
if hasattr(torch, "float8_e5m2") and hasattr(cutlass, "Float8E5M2"):
    fake_compile_torch2cute_dtype_map[torch.float8_e5m2] = cutlass.Float8E5M2


@dataclass(frozen=True)
class CompileTensorSpec:
    dynamic_mode: int | None = None
    leading_dim: int | None = None
    divisibility: int = 1
    assumed_align: int | None = None
    stride_order: Tuple[int, ...] | None = None
    symbolic_stride_dims: Tuple[int, ...] = ()


def _fake_tensor_stride(fake_tensor: Any) -> list[Any]:
    return list(fake_tensor.stride)


def _set_fake_tensor_stride(fake_tensor: Any, stride: list[Any]) -> None:
    typed_tensor = getattr(fake_tensor, "_typed_tensor", None)
    if typed_tensor is not None:
        typed_tensor._stride = tuple(stride)
        return
    if hasattr(fake_tensor, "_stride"):
        fake_tensor._stride = tuple(stride)
        return
    raise AttributeError(
        "Unsupported CUTLASS _FakeTensor stride representation; "
        "expected _typed_tensor or _stride."
    )


DYNAMIC_DIM0 = CompileTensorSpec(dynamic_mode=0, symbolic_stride_dims=(0,))


def dynamic_dim0_specs(*arg_names: str) -> dict[str, CompileTensorSpec]:
    return {name: DYNAMIC_DIM0 for name in arg_names}


def normalize_runtime_tensor_arg(arg: Any) -> Any:
    # Note(wangbojun/codex): CuTeDSL runtime conversion only needs the raw tensor
    # storage/metadata. Unwrapping Parameter avoids subclass-specific dispatch
    # such as vLLM's ModelWeightParameter on the TVM FFI path.
    if isinstance(arg, torch.nn.Parameter):
        return arg.detach()
    return arg


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
                types.extend(obj_types)
                self._values_pos.append(len(obj_types))
            else:
                self._values_pos.append(0)
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    pathlib.Path(filepath).write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def _normalize_compile_options(options: Any) -> Any:
    return options


def cute_compile_patched(*args, **kwargs):
    """Call ``cute.compile`` and optionally dump the loaded cubin/SASS.

    Resolve ``cute.compile`` at call time so vLLM's standard JIT monitor can
    observe compilations even when this helper module was imported before the
    monitor was activated.
    """
    if "options" in kwargs:
        kwargs = dict(kwargs)
        options = _normalize_compile_options(kwargs.get("options", ""))
        if options is None:
            kwargs.pop("options")
        else:
            kwargs["options"] = options
    cubin_path = os.getenv("CUTE_CUBIN_PATH", None)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=cubin_path
        )
    try:
        output = cute.compile(*args, **kwargs)
    finally:
        if cubin_path is not None:
            cutlass.base_dsl.runtime.cuda.load_cubin_module_data = (
                load_cubin_module_data_og
            )
    if cubin_path is not None and pathlib.Path(cubin_path).exists():
        if extract is not None:
            try:
                sass = extract(cubin_path, None)
                pathlib.Path(cubin_path).with_suffix(".annotated.sass").write_text(sass)
            except Exception as exc:
                print(
                    f"[cute_compile_patched] cubin dumped to {cubin_path}; "
                    f"SASS extraction skipped: {exc}",
                    flush=True,
                )
    elif cubin_path is not None:
        print(
            f"[cute_compile_patched] CUTE_CUBIN_PATH was set but no cubin was "
            f"loaded through the patched runtime hook: {cubin_path}",
            flush=True,
        )
    return output


def _infer_assumed_align_bytes(tensor: torch.Tensor, max_align: int = 16) -> int:
    try:
        ptr = int(tensor.data_ptr())
    except Exception:
        ptr = 0

    align = 1
    while align < max_align and ptr % (align * 2) == 0:
        align *= 2

    try:
        element_size = max(1, int(tensor.element_size()))
    except Exception:
        element_size = 1
    return max(align, element_size)


def _infer_leading_dim(tensor: torch.Tensor) -> int | None:
    if tensor.ndim == 0:
        return None
    stride_ones = [idx for idx, value in enumerate(tensor.stride()) if value == 1]
    if len(stride_ones) == 1:
        return stride_ones[0]
    return None


def _infer_stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    try:
        return tuple(tensor.dim_order())
    except Exception:
        indexed = list(enumerate(tensor.stride()))
        indexed.sort(key=lambda pair: pair[1], reverse=True)
        return tuple(idx for idx, _ in indexed)


def make_fake_tensor_like(
    tensor: torch.Tensor,
    spec: CompileTensorSpec | None = None,
) -> cute.Tensor:
    from cutlass.cute.typing import SymInt, sym_int64

    apply_patches()
    try:
        from cutlass.cute.runtime import make_fake_tensor
    except ImportError as err:
        raise RuntimeError(
            "nvidia-cutlass-dsl<4.3.3 does not provide make_fake_tensor; "
            "this fake-tensor compile path requires nvidia-cutlass-dsl>=4.3.3."
        ) from err

    cute_dtype = fake_compile_torch2cute_dtype_map.get(tensor.dtype)
    if cute_dtype is None:
        raise AssertionError(
            f"Unsupported dtype for fake compile tensor: {tensor.dtype}."
        )

    if spec is None:
        spec = CompileTensorSpec()

    fake_tensor = make_fake_tensor(
        cute_dtype,
        tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        assumed_align=(
            _infer_assumed_align_bytes(tensor)
            if spec.assumed_align is None
            else spec.assumed_align
        ),
    )

    if spec.dynamic_mode is None:
        if spec.symbolic_stride_dims:
            stride = _fake_tensor_stride(fake_tensor)
            for dim in spec.symbolic_stride_dims:
                if not (0 <= dim < len(stride)):
                    raise AssertionError(
                        f"symbolic stride dim {dim} out of range for rank {len(stride)}"
                    )
                if not isinstance(stride[dim], SymInt):
                    stride[dim] = sym_int64(divisibility=spec.divisibility)
            _set_fake_tensor_stride(fake_tensor, stride)
        return fake_tensor

    leading_dim = spec.leading_dim
    if leading_dim is None:
        leading_dim = _infer_leading_dim(tensor)
    if leading_dim is None:
        raise AssertionError(
            "Dynamic fake tensor compile requires an explicit leading_dim or a unique stride-1 dimension."
        )

    fake_tensor = fake_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    stride_order = spec.stride_order if spec.stride_order is not None else _infer_stride_order(tensor)
    try:
        fake_tensor = fake_tensor.mark_compact_shape_dynamic(
            mode=spec.dynamic_mode,
            stride_order=stride_order,
            divisibility=spec.divisibility,
        )
    except RuntimeError as err:
        if "stride_order" not in str(err):
            raise
        fake_tensor = fake_tensor.mark_compact_shape_dynamic(
            mode=spec.dynamic_mode,
            divisibility=spec.divisibility,
        )
    if spec.symbolic_stride_dims:
        stride = _fake_tensor_stride(fake_tensor)
        for dim in spec.symbolic_stride_dims:
            if not (0 <= dim < len(stride)):
                raise AssertionError(
                    f"symbolic stride dim {dim} out of range for rank {len(stride)}"
                )
            if not isinstance(stride[dim], SymInt):
                stride[dim] = sym_int64(divisibility=spec.divisibility)
        _set_fake_tensor_stride(fake_tensor, stride)
    return fake_tensor


def make_fake_compile_args(
    *args: Any,
    tensor_specs: Mapping[int, CompileTensorSpec] | None = None,
) -> tuple[Any, ...]:
    specs = tensor_specs or {}
    converted = []
    for idx, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            converted.append(make_fake_tensor_like(arg, specs.get(idx)))
        else:
            converted.append(arg)
    return tuple(converted)


def _resolve_named_tensor_specs(
    func: Any,
    args: tuple[Any, ...],
    tensor_specs_by_name: Mapping[str, CompileTensorSpec],
) -> dict[int, CompileTensorSpec]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        call = getattr(func, "__call__", None)
        if call is None:
            raise AssertionError(
                "Failed to inspect compile callable; pass positional tensor_specs instead."
            )
        signature = inspect.signature(call)

    positional_names = [
        param.name
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if positional_names and positional_names[0] == "self":
        positional_names = positional_names[1:]

    resolved: dict[int, CompileTensorSpec] = {}
    unknown_names = sorted(set(tensor_specs_by_name) - set(positional_names[: len(args)]))
    if unknown_names:
        raise AssertionError(
            f"Unknown tensor spec names {unknown_names} for compile callable. "
            f"Available positional args: {positional_names[: len(args)]}"
        )

    for idx, name in enumerate(positional_names[: len(args)]):
        spec = tensor_specs_by_name.get(name)
        if spec is not None:
            resolved[idx] = spec
    return resolved


def cute_compile_with_spec(
    func: Any,
    *args: Any,
    cache: MutableMapping[_CacheKeyT, _CacheValueT] | None = None,
    cache_key: _CacheKeyT | None = None,
    tensor_specs: Mapping[int, CompileTensorSpec] | None = None,
    tensor_specs_by_name: Mapping[str, CompileTensorSpec] | None = None,
    **kwargs: Any,
):
    if tensor_specs is not None and tensor_specs_by_name is not None:
        raise AssertionError("Pass either tensor_specs or tensor_specs_by_name, not both.")

    resolved_specs = tensor_specs
    if tensor_specs_by_name is not None:
        resolved_specs = _resolve_named_tensor_specs(func, args, tensor_specs_by_name)

    def _compile():
        return cute_compile_patched(
            func,
            *make_fake_compile_args(*args, tensor_specs=resolved_specs),
            **kwargs,
        )

    if cache is None:
        return _compile()
    if cache_key is None:
        raise AssertionError("cache_key is required when cache is provided.")
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    value = _compile()
    cache[cache_key] = value
    return value


def is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def assert_cuda_graph_cache_miss_allowed(
    op_name: str,
    phase: str,
) -> None:
    if is_current_stream_capturing():
        raise AssertionError(
            f"{op_name} first {phase} cannot run inside CUDA Graph capture. "
            "Warm up the same public API once outside CUDA Graph capture before capturing."
        )


def cache_get_or_create(
    cache: MutableMapping[_CacheKeyT, _CacheValueT],
    key: _CacheKeyT,
    factory: Callable[[], _CacheValueT],
    *,
    op_name: str,
    phase: str,
) -> _CacheValueT:
    cached = cache.get(key)
    if cached is not None:
        return cached
    assert_cuda_graph_cache_miss_allowed(op_name, phase)
    value = factory()
    cache[key] = value
    return value


def allocate_group_quant_scales(
    rows: int,
    groups_per_row: int,
    device: torch.device,
    column_major_scales: bool,
    out_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    expected_shape = (rows, groups_per_row)
    if out_scales is None:
        if column_major_scales:
            return torch.empty_strided(
                expected_shape,
                (1, rows),
                device=device,
                dtype=torch.float32,
            )
        return torch.empty(expected_shape, device=device, dtype=torch.float32)

    assert out_scales.shape == expected_shape, "`out_scales` must match (rows, groups_per_row)."
    assert out_scales.device == device, "`out_scales` must be on the same device as the inputs."
    assert out_scales.dtype == torch.float32, "`out_scales` must use torch.float32."
    if column_major_scales:
        assert out_scales.stride() == (1, rows), (
            "`out_scales` must use column-major `(1, rows)` stride when "
            "`column_major_scales=True`."
        )
    else:
        assert out_scales.is_contiguous(), (
            "`out_scales` must be contiguous when `column_major_scales=False`."
        )
    return out_scales


def tensor_cache_signature(
    tensor: torch.Tensor | None,
) -> tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype, str, int | None] | None:
    if tensor is None:
        return None
    device = tensor.device
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        device.type,
        device.index,
    )
