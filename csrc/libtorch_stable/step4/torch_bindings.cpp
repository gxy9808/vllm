// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "ops.h"

#include <torch/csrc/stable/library.h>

// Keep Step4 registrations in a dedicated fragment. The namespace and
// existing operator names remain stable for Python compatibility.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, step4_ops) {
  step4_ops.def(
      "optimus_fused_qknorm_rope_cache("
      "Tensor! q_out, Tensor! k_out, Tensor! v_out, Tensor qkv, "
      "Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, "
      "Tensor positions, Tensor slot_mapping, Tensor! k_cache, "
      "Tensor! v_cache, int head_dim, int num_q_heads, int num_kv_heads, "
      "int rotary_pairs, float epsilon, float norm_weight_bias) -> ()");
  step4_ops.impl("optimus_fused_qknorm_rope_cache",
                 TORCH_BOX(&vllm::optimus_fused_qknorm_rope_cache));

  step4_ops.def(
      "optimus_fused_qknorm_rope_cache_bitwise("
      "Tensor! q_out, Tensor! k_out, Tensor! v_out, Tensor qkv, "
      "Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, "
      "Tensor positions, Tensor slot_mapping, Tensor! k_cache, "
      "Tensor! v_cache, int head_dim, int num_q_heads, int num_kv_heads, "
      "int rotary_pairs, float epsilon, float norm_weight_bias) -> ()");
  step4_ops.impl("optimus_fused_qknorm_rope_cache_bitwise",
                 TORCH_BOX(&vllm::optimus_fused_qknorm_rope_cache_bitwise));

  step4_ops.def(
      "optimus_fused_add_rms_norm("
      "Tensor! output, Tensor! residual_out, Tensor input, Tensor residual, "
      "Tensor weight, float epsilon, bool zero_centered) -> ()");
  step4_ops.impl("optimus_fused_add_rms_norm",
                 TORCH_BOX(&vllm::optimus_fused_add_rms_norm));
}
