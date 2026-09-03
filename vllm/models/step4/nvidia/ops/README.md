# Step4 NVIDIA operators

This directory contains the NVIDIA-only inference kernel closure required by
the Step4 model:

- `cute_dsl/`: QK/indexer norm + RoPE, DSA indexer/summary kernels, sparse GQA
  prefill/decode, and MTP sparse-attention kernels;
- `triton/`: the Step4 router-bias kernel and its driver-launch helpers.

It intentionally excludes Optimus training, backward, benchmark, accuracy
guard, experimental, DeepGEMM/MoE, quantization, and unrelated model kernels.
Native libtorch-stable CUDA operators remain under
`csrc/libtorch_stable/step4/`.

The Python package roots are lazy and must not initialize CUDA during model
registry import.  Vendored-file provenance is recorded in
`OPTIMUS_JIT_VENDOR_MANIFEST.tsv`; redistribution remains subject to the
license gate documented in `OPTIMUS_JIT_VENDOR_NOTICE.md`.
