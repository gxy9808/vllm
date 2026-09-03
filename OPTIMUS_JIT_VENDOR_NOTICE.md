# Vendored Component Notice: Step4 Optimus Inference Kernels

This notice describes provenance, packaging, and release constraints for the
model-scoped Optimus inference kernel closure used by Step4 on the vLLM
`v0.27.0` baseline. It is an informational notice only. **It is not a
copyright license, a sublicense, or an authorization to redistribute any
file.** Inclusion of this notice in package metadata does not change the
license of the referenced files.

## Scope and provenance

Only the Step4 inference closure copied from the internal `optimus_jit`
repository is vendored. It is owned by the Step4 NVIDIA model integration at:

- `vllm/models/step4/nvidia/ops/cute_dsl/`;
- `vllm/models/step4/nvidia/ops/triton/`.

The source snapshot used for this integration was:

- repository: `https://gitlab.basemind.com/wangbojun/optimus_jit.git`;
- branch at copy time: `yxl-sparse`;
- immutable source commit:
  `503ba0df4d7f69333f1bbbbd421a6e9c5921f635`;
- source package roots: `src/optimus_cutedsl/` and
  `src/optimus_triton/`.

The vendored paths are deliberately not complete copies of those source
package roots. Training, backward-only, benchmark, testing, experimental, and
unrelated Optimus operators are omitted. Imports were rewritten for the
model-scoped namespace, and some files were renamed; the manifest preserves
each file's original `source_path` rather than treating the new path as its
origin. In particular:

- `cute_dsl/_cutlass_compat.py` originated from
  `src/optimus_cutedsl/_cutlass_monkeypatch.py`;
- `triton/router_bias.py` originated from
  `src/optimus_triton/router_bias_topk.py`.

`OPTIMUS_JIT_VENDOR_MANIFEST.tsv` inventories every file in the two approved
vendored kernel roots. It records the source repository and revision, source
Git blob and SHA-256 digest, current vendored SHA-256 digest, and whether the
current file differs from the source snapshot. Model-owned integration files
such as `vllm/models/step4/nvidia/ops/__init__.py` and its README are not
represented as Optimus source. The manifest records provenance only; it is not
a license or redistribution authorization.

The related CUDA integration under `csrc/libtorch_stable/step4/` builds
Step4-specific native operators into vLLM's stable libtorch extension. The
model-scoped directory follows the extension's ownership boundary; the
existing `torch.ops._C.optimus_*` names remain compatibility ABI. Its manifest
baseline is internal vLLM commit
`201aa57ab1ac6b785b8cf704b35c711c61bf0d07`. The corresponding history begins
with commit
`f17cbb914fbde3d1313fcd329172bcb9c629d1ca` for the isolated QK-norm/RoPE
extension and commit `050fb2ffc35f3c2f56b3a8f421d9e40fb242b4d9` for the fused
add/RMSNorm source. All four current files differ from the manifest baseline;
their current checksums are recorded in the manifest. Their redistribution
rights must still be covered by the release audit; an internal commit history
and SPDX header are provenance evidence, not authorization by themselves.

## Model-scoped ownership

The vendored Python kernels are implementation details of the Step4 NVIDIA
backend. They are not general top-level Python packages and do not define a
cross-project Optimus compatibility API. Callers must use the lazy Step4
facade/dispatcher so registry and config discovery do not import CuTeDSL leaf
modules or initialize the CUDA driver.

The standalone `optimus_jit_compile_log.py` hook is not part of this closure.
Step4 uses vLLM's shared `vllm.utils.jit_monitor` and model-executor warmup
infrastructure instead of installing a second CuTeDSL or Triton monkey patch.
Do not reintroduce the top-level `optimus_cutedsl`, `optimus_triton`, or
`optimus_jit_compile_log` modules into release artifacts.

If the larger Optimus library must be shared by training, vLLM, or other
serving systems, it should be released as a separately versioned package with
its own license review, compatibility matrix, build, tests, and provenance
metadata. Do not expand the model-scoped closure without confirming that the
new file is required for Step4 inference and recording its origin and license
status.

## Current packaging wiring

The source tree wires the model-scoped closure through:

- `pyproject.toml`, whose normal `vllm*` package discovery includes the nested
  Step4 packages without publishing top-level Optimus packages;
- `MANIFEST.in` and `setup.py`, which include the retained Triton signature C++
  helper under `vllm.models.step4.nvidia.ops.triton.utils`;
- `OPTIMUS_JIT_VENDOR_MANIFEST.tsv`, which is included in source and wheel
  license metadata; and
- `CMakeLists.txt`, which adds `csrc/libtorch_stable/step4/` only to CUDA builds
  and registers the Step4 operators in `torch.ops._C`.

The inference-only Triton closure intentionally omits the old tensor-data-swap
helper and loader because Step4 does not call that API. No training,
experimental, benchmark, testing, or backward-only subtree is copied and no
packaging prune rule is used as a substitute for a minimal source inventory.

These declarations describe what a new build should contain. They do not prove
that any previously built sdist or wheel contains the current files. Release
artifacts must be rebuilt from the final reviewed source. An empty-device wheel
is useful for a Python-content audit, but it does not prove that the CUDA
extension was compiled or that the StepFun operators were registered. A CUDA
wheel must also be built, installed into a clean environment, and checked for:

- `torch.ops._C.optimus_fused_qknorm_rope_cache`;
- `torch.ops._C.optimus_fused_qknorm_rope_cache_bitwise`; and
- `torch.ops._C.optimus_fused_add_rms_norm`.

The source distribution and wheel inventory must additionally verify that the
approved model-scoped roots are present and that top-level `optimus_*`,
training, experimental, benchmark, testing, and backward-only files are
absent.

## License status: P0 public-release gate

License coverage remains incomplete across the retained closure. At the time
of this notice, the manifest covers 54 files: 50 retained `optimus_jit` files
and four StepFun native sources. A mechanical header scan finds 27 retained
`optimus_jit` files with an `Apache-2.0` SPDX identifier and 23 with no SPDX
license identifier; five of the latter carry a copyright notice. All four
StepFun native files carry an `Apache-2.0` SPDX identifier. These counts are
inventory evidence only and do not establish that any header was authorized by
the relevant rightsholder.

The repository-level Apache-2.0 declaration does not automatically relicense
these files. In particular, copyright-only notices attributed to Wentao Guo,
Ted Zadouri, Tri Dao, or other authors must retain their attribution and must be
mapped to their actual upstream source and license. Do not guess that license,
replace the notice, or add an invented SPDX identifier.

**Public source distribution and public wheel publication are blocked** until
all of the following are complete:

1. Record the origin revision and ownership for every retained file.
2. Obtain redistribution authorization for StepFun-owned or otherwise
   internally owned files that currently lack a clear license.
3. Identify the exact upstream source and license for every third-party-derived
   file, preserving all required notices and license texts.
4. Add SPDX identifiers only when supported by authoritative ownership and
   license evidence.
5. Run a final source and binary license scan and obtain the required human/legal
   release approval.
6. Rebuild and inspect the sdist and CUDA wheel after the approved license
   metadata is present.

Until those steps are complete, this notice must remain a warning and
provenance record; it must not be presented as license clearance. Moving or
renaming a file does not cure a missing redistribution grant.
