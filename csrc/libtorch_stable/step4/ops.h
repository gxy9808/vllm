// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstdint>

#include <torch/csrc/stable/tensor.h>

// Step4-specific declarations used by the registration fragment. This
// extension contains the QK-norm+RoPE(+KV cache) and fused add+RMSNorm
// kernels; custom all-reduce collectives are outside its scope.
//
// These sources are added to VLLM_STABLE_EXT_SRC only for CUDA; the kernels are
// hand-written for NVIDIA and are not HIP-portable.

namespace vllm {

void optimus_fused_qknorm_rope_cache(
    torch::stable::Tensor& q_out, torch::stable::Tensor& k_out,
    torch::stable::Tensor& v_out, torch::stable::Tensor& qkv,
    torch::stable::Tensor& q_weight, torch::stable::Tensor& k_weight,
    torch::stable::Tensor& cos, torch::stable::Tensor& sin,
    torch::stable::Tensor& positions, torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& k_cache, torch::stable::Tensor& v_cache,
    int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    int64_t rotary_pairs, double epsilon, double norm_weight_bias);

void optimus_fused_qknorm_rope_cache_bitwise(
    torch::stable::Tensor& q_out, torch::stable::Tensor& k_out,
    torch::stable::Tensor& v_out, torch::stable::Tensor& qkv,
    torch::stable::Tensor& q_weight, torch::stable::Tensor& k_weight,
    torch::stable::Tensor& cos, torch::stable::Tensor& sin,
    torch::stable::Tensor& positions, torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& k_cache, torch::stable::Tensor& v_cache,
    int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    int64_t rotary_pairs, double epsilon, double norm_weight_bias);

void optimus_fused_add_rms_norm(torch::stable::Tensor& output,
                                torch::stable::Tensor& residual_out,
                                torch::stable::Tensor& input,
                                torch::stable::Tensor& residual,
                                torch::stable::Tensor& weight, double epsilon,
                                bool zero_centered);

}  // namespace vllm
