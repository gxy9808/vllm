// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Step4 bitwise fused residual-add + Optimus-compatible RMSNorm kernel.

// This kernel is built into `_C_stable_libtorch` and uses the libtorch stable
// ABI (torch::stable::Tensor / STD_TORCH_CHECK) instead of ATen and c10.
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/type_convert.cuh"

#include <torch/csrc/stable/macros.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace vllm {
namespace {

constexpr unsigned int kFullWarpMask = 0xffffffff;

template <typename T>
struct alignas(sizeof(T) * 4) Vec4 {
  T x;
  T y;
  T z;
  T w;
};

template <typename scalar_t>
__device__ __forceinline__ float4
load_vector(const typename _typeConvert<scalar_t>::hip_type* ptr) {
  using Converter = _typeConvert<scalar_t>;
  using Scalar = typename Converter::hip_type;
  const Vec4<Scalar> value = *reinterpret_cast<const Vec4<Scalar>*>(ptr);
  return make_float4(Converter::convert(value.x), Converter::convert(value.y),
                     Converter::convert(value.z), Converter::convert(value.w));
}

template <typename scalar_t>
__device__ __forceinline__ void store_vector(
    typename _typeConvert<scalar_t>::hip_type* ptr, const float4 value) {
  using Converter = _typeConvert<scalar_t>;
  using Scalar = typename Converter::hip_type;
  Vec4<Scalar> converted{
      Converter::convert(value.x),
      Converter::convert(value.y),
      Converter::convert(value.z),
      Converter::convert(value.w),
  };
  *reinterpret_cast<Vec4<Scalar>*>(ptr) = converted;
}

template <typename scalar_t>
__device__ __forceinline__ float round_to_scalar_type(const float value) {
  using Converter = _typeConvert<scalar_t>;
  // This explicit downcast/upcast is required for bitwise equivalence with
  // the unfused aten.add -> Optimus RMSNorm graph. aten.add materializes the
  // activation-typed residual tensor before RMSNorm reads it.
  return Converter::convert(Converter::convert(value));
}

template <typename scalar_t>
__device__ __forceinline__ float4 round_to_scalar_type(float4 value) {
  value.x = round_to_scalar_type<scalar_t>(value.x);
  value.y = round_to_scalar_type<scalar_t>(value.y);
  value.z = round_to_scalar_type<scalar_t>(value.z);
  value.w = round_to_scalar_type<scalar_t>(value.w);
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, mask, 32);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum_all(float value) {
  static __shared__ float shared[32];
  const int lane = threadIdx.x & 0x1f;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  return warp_reduce_sum(lane < (blockDim.x >> 5) ? shared[lane] : 0.0f);
}

template <typename scalar_t, typename weight_t, int NUM, bool ZERO_CENTERED>
__global__ void optimus_fused_add_rms_norm_kernel(
    const void* input_void, const void* residual_void, const void* weight_void,
    void* output_void, void* residual_out_void, const float reciprocal_hidden,
    const float epsilon) {
  using Scalar = typename _typeConvert<scalar_t>::hip_type;
  using Weight = typename _typeConvert<weight_t>::hip_type;
  const auto* input = static_cast<const Scalar*>(input_void);
  const auto* residual = static_cast<const Scalar*>(residual_void);
  const auto* weight = static_cast<const Weight*>(weight_void);
  auto* output = static_cast<Scalar*>(output_void);
  auto* residual_out = static_cast<Scalar*>(residual_out_void);

  const int block_offset = (NUM > 1 ? 256 : blockDim.x) * 4;
  const int offset = blockIdx.x * block_offset * NUM + threadIdx.x * 4;

  float4 values[NUM];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NUM; ++i) {
    const int vector_offset = offset + i * block_offset;
    values[i] = load_vector<scalar_t>(input + vector_offset);
    const float4 residual_value =
        load_vector<scalar_t>(residual + vector_offset);
    values[i].x += residual_value.x;
    values[i].y += residual_value.y;
    values[i].z += residual_value.z;
    values[i].w += residual_value.w;

    // The residual output and the RMSNorm input must be the exact same
    // activation-typed value. Without this explicit round-trip, RMSNorm would
    // consume the pre-rounding FP32 registers and would not be bitwise with
    // the original aten.add -> Optimus RMSNorm sequence.
    values[i] = round_to_scalar_type<scalar_t>(values[i]);
    store_vector<scalar_t>(residual_out + vector_offset, values[i]);
    sum += values[i].x * values[i].x + values[i].y * values[i].y +
           values[i].z * values[i].z + values[i].w * values[i].w;
  }

  sum = block_reduce_sum_all(sum);
  const float inverse_rms = rsqrtf(sum * reciprocal_hidden + epsilon);

#pragma unroll
  for (int i = 0; i < NUM; ++i) {
    const int vector_offset = offset + i * block_offset;
    float4 gamma =
        load_vector<weight_t>(weight + threadIdx.x * 4 + i * block_offset);
    if constexpr (ZERO_CENTERED) {
      gamma.x += 1.0f;
      gamma.y += 1.0f;
      gamma.z += 1.0f;
      gamma.w += 1.0f;
    }
    float4 normalized;
    normalized.x = values[i].x * inverse_rms * gamma.x;
    normalized.y = values[i].y * inverse_rms * gamma.y;
    normalized.z = values[i].z * inverse_rms * gamma.z;
    normalized.w = values[i].w * inverse_rms * gamma.w;
    store_vector<scalar_t>(output + vector_offset, normalized);
  }
}

template <typename scalar_t, typename weight_t, bool ZERO_CENTERED>
__launch_bounds__(32, 1) __global__ void optimus_fused_add_rms_norm_192_kernel(
    const void* input_void, const void* residual_void, const void* weight_void,
    void* output_void, void* residual_out_void, const float reciprocal_hidden,
    const float epsilon) {
  using Scalar = typename _typeConvert<scalar_t>::hip_type;
  using Weight = typename _typeConvert<weight_t>::hip_type;
  const auto* input = static_cast<const Scalar*>(input_void);
  const auto* residual = static_cast<const Scalar*>(residual_void);
  const auto* weight = static_cast<const Weight*>(weight_void);
  auto* output = static_cast<Scalar*>(output_void);
  auto* residual_out = static_cast<Scalar*>(residual_out_void);

  constexpr int kHiddenSize = 192;
  constexpr int kVectorSize = 4;
  constexpr int kMainOffset = 32 * kVectorSize;
  constexpr int kTailLanes = (kHiddenSize - kMainOffset) / kVectorSize;

  const int lane_offset = threadIdx.x * kVectorSize;
  const int row_offset = blockIdx.x * kHiddenSize;

  float4 value0 = load_vector<scalar_t>(input + row_offset + lane_offset);
  const float4 residual0 =
      load_vector<scalar_t>(residual + row_offset + lane_offset);
  value0.x += residual0.x;
  value0.y += residual0.y;
  value0.z += residual0.z;
  value0.w += residual0.w;
  // See the generic kernel: this round-trip preserves the materialized add
  // boundary required for bitwise equivalence.
  value0 = round_to_scalar_type<scalar_t>(value0);
  store_vector<scalar_t>(residual_out + row_offset + lane_offset, value0);

  float sum = value0.x * value0.x + value0.y * value0.y + value0.z * value0.z +
              value0.w * value0.w;
  const bool has_tail = threadIdx.x < kTailLanes;
  float4 value1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (has_tail) {
    const int tail_offset = row_offset + kMainOffset + lane_offset;
    value1 = load_vector<scalar_t>(input + tail_offset);
    const float4 residual1 = load_vector<scalar_t>(residual + tail_offset);
    value1.x += residual1.x;
    value1.y += residual1.y;
    value1.z += residual1.z;
    value1.w += residual1.w;
    value1 = round_to_scalar_type<scalar_t>(value1);
    store_vector<scalar_t>(residual_out + tail_offset, value1);
    sum += value1.x * value1.x + value1.y * value1.y + value1.z * value1.z +
           value1.w * value1.w;
  }

  sum = warp_reduce_sum(sum);
  const float inverse_rms = rsqrtf(sum * reciprocal_hidden + epsilon);

  float4 gamma0 = load_vector<weight_t>(weight + lane_offset);
  if constexpr (ZERO_CENTERED) {
    gamma0.x += 1.0f;
    gamma0.y += 1.0f;
    gamma0.z += 1.0f;
    gamma0.w += 1.0f;
  }
  float4 normalized0;
  normalized0.x = value0.x * inverse_rms * gamma0.x;
  normalized0.y = value0.y * inverse_rms * gamma0.y;
  normalized0.z = value0.z * inverse_rms * gamma0.z;
  normalized0.w = value0.w * inverse_rms * gamma0.w;
  store_vector<scalar_t>(output + row_offset + lane_offset, normalized0);

  if (has_tail) {
    const int tail_offset = row_offset + kMainOffset + lane_offset;
    float4 gamma1 = load_vector<weight_t>(weight + kMainOffset + lane_offset);
    if constexpr (ZERO_CENTERED) {
      gamma1.x += 1.0f;
      gamma1.y += 1.0f;
      gamma1.z += 1.0f;
      gamma1.w += 1.0f;
    }
    float4 normalized1;
    normalized1.x = value1.x * inverse_rms * gamma1.x;
    normalized1.y = value1.y * inverse_rms * gamma1.y;
    normalized1.z = value1.z * inverse_rms * gamma1.z;
    normalized1.w = value1.w * inverse_rms * gamma1.w;
    store_vector<scalar_t>(output + tail_offset, normalized1);
  }
}

template <typename scalar_t, typename weight_t, int NUM, bool ZERO_CENTERED>
void launch_generic(const torch::stable::Tensor& input,
                    const torch::stable::Tensor& residual,
                    const torch::stable::Tensor& weight,
                    torch::stable::Tensor& output,
                    torch::stable::Tensor& residual_out, int rows,
                    int hidden_size, float epsilon, cudaStream_t stream) {
  const int threads = hidden_size / 4 / NUM;
  optimus_fused_add_rms_norm_kernel<scalar_t, weight_t, NUM, ZERO_CENTERED>
      <<<rows, threads, 0, stream>>>(
          input.data_ptr(), residual.data_ptr(), weight.data_ptr(),
          output.data_ptr(), residual_out.data_ptr(),
          1.0f / static_cast<float>(hidden_size), epsilon);
}

template <typename scalar_t, typename weight_t, bool ZERO_CENTERED>
void launch(const torch::stable::Tensor& input,
            const torch::stable::Tensor& residual,
            const torch::stable::Tensor& weight, torch::stable::Tensor& output,
            torch::stable::Tensor& residual_out, int rows, int hidden_size,
            float epsilon, cudaStream_t stream) {
  if (hidden_size == 192) {
    optimus_fused_add_rms_norm_192_kernel<scalar_t, weight_t, ZERO_CENTERED>
        <<<rows, 32, 0, stream>>>(
            input.data_ptr(), residual.data_ptr(), weight.data_ptr(),
            output.data_ptr(), residual_out.data_ptr(),
            1.0f / static_cast<float>(hidden_size), epsilon);
    return;
  }

#define LAUNCH_GENERIC(NUM)                                             \
  launch_generic<scalar_t, weight_t, NUM, ZERO_CENTERED>(               \
      input, residual, weight, output, residual_out, rows, hidden_size, \
      epsilon, stream)

  if (hidden_size < 4096) {
    LAUNCH_GENERIC(1);
  } else {
    switch (hidden_size / 1024) {
      case 4:
        LAUNCH_GENERIC(4);
        break;
      case 5:
        LAUNCH_GENERIC(5);
        break;
      case 6:
        LAUNCH_GENERIC(6);
        break;
      case 7:
        LAUNCH_GENERIC(7);
        break;
      case 8:
        LAUNCH_GENERIC(8);
        break;
      case 9:
        LAUNCH_GENERIC(9);
        break;
      case 10:
        LAUNCH_GENERIC(10);
        break;
      case 12:
        LAUNCH_GENERIC(12);
        break;
      case 16:
        LAUNCH_GENERIC(16);
        break;
      default:
        STD_TORCH_CHECK(false, "Unsupported hidden size: ", hidden_size);
    }
  }
#undef LAUNCH_GENERIC
}

template <typename scalar_t, typename weight_t>
void dispatch_zero_centered(const torch::stable::Tensor& input,
                            const torch::stable::Tensor& residual,
                            const torch::stable::Tensor& weight,
                            torch::stable::Tensor& output,
                            torch::stable::Tensor& residual_out, int rows,
                            int hidden_size, float epsilon, bool zero_centered,
                            cudaStream_t stream) {
  if (zero_centered) {
    launch<scalar_t, weight_t, true>(input, residual, weight, output,
                                     residual_out, rows, hidden_size, epsilon,
                                     stream);
  } else {
    launch<scalar_t, weight_t, false>(input, residual, weight, output,
                                      residual_out, rows, hidden_size, epsilon,
                                      stream);
  }
}

void check_same_device(const torch::stable::Tensor& reference,
                       const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.get_device_index() == reference.get_device_index(),
                  name, " must be on the same CUDA device as input");
}

void check_same_shape(const torch::stable::Tensor& reference,
                      const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.dim() == reference.dim(), "input and ", name,
                  " must have the same shape");
  for (int64_t d = 0; d < reference.dim(); ++d) {
    STD_TORCH_CHECK(tensor.size(d) == reference.size(d), "input and ", name,
                    " must have the same shape");
  }
}

}  // namespace

void optimus_fused_add_rms_norm(torch::stable::Tensor& output,
                                torch::stable::Tensor& residual_out,
                                torch::stable::Tensor& input,
                                torch::stable::Tensor& residual,
                                torch::stable::Tensor& weight, double epsilon,
                                bool zero_centered) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  STD_TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  STD_TORCH_CHECK(residual_out.is_cuda(), "residual_out must be a CUDA tensor");
  check_same_device(input, residual, "residual");
  check_same_device(input, weight, "weight");
  check_same_device(input, output, "output");
  check_same_device(input, residual_out, "residual_out");

  STD_TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  check_same_shape(input, residual, "residual");
  check_same_shape(input, output, "output");
  check_same_shape(input, residual_out, "residual_out");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                  "input and residual must have the same dtype");
  STD_TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                  "input and output must have the same dtype");
  STD_TORCH_CHECK(input.scalar_type() == residual_out.scalar_type(),
                  "input and residual_out must have the same dtype");
  STD_TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  STD_TORCH_CHECK(weight.numel() == input.size(input.dim() - 1),
                  "weight size must match input hidden size");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  STD_TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  STD_TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  STD_TORCH_CHECK(residual_out.is_contiguous(),
                  "residual_out must be contiguous");

  const int64_t hidden_size_64 = input.size(input.dim() - 1);
  STD_TORCH_CHECK(
      hidden_size_64 > 0 && hidden_size_64 <= std::numeric_limits<int>::max(),
      "hidden size is out of range");
  const int hidden_size = static_cast<int>(hidden_size_64);
  const bool supported_hidden_size =
      hidden_size == 192 || (hidden_size < 4096 && hidden_size % 128 == 0) ||
      (hidden_size >= 4096 && hidden_size % 1024 == 0 &&
       (hidden_size / 1024 == 4 || hidden_size / 1024 == 5 ||
        hidden_size / 1024 == 6 || hidden_size / 1024 == 7 ||
        hidden_size / 1024 == 8 || hidden_size / 1024 == 9 ||
        hidden_size / 1024 == 10 || hidden_size / 1024 == 12 ||
        hidden_size / 1024 == 16));
  STD_TORCH_CHECK(supported_hidden_size,
                  "Optimus fused RMSNorm does not support hidden size ",
                  hidden_size);

  if (input.numel() == 0) {
    return;
  }
  const int64_t rows_64 = input.numel() / hidden_size;
  STD_TORCH_CHECK(rows_64 <= std::numeric_limits<int>::max(),
                  "row count is out of range");
  const int rows = static_cast<int>(rows_64);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "optimus_fused_add_rms_norm_input", [&] {
        using input_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            weight.scalar_type(), "optimus_fused_add_rms_norm_weight", [&] {
              using weight_t = scalar_t;
              dispatch_zero_centered<input_t, weight_t>(
                  input, residual, weight, output, residual_out, rows,
                  hidden_size, static_cast<float>(epsilon), zero_centered,
                  stream);
            });
      });
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm
