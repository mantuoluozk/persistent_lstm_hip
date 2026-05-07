#include "persistent_lstm_hip.h"

#include <stdexcept>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
  hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), shmem, stream, __VA_ARGS__)
#define GPU_GET_LAST_ERROR() hipGetLastError()
#define GPU_SUCCESS hipSuccess
#define GPU_GET_ERROR_STRING(err) hipGetErrorString(err)
using gpu_half = half;
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
  kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#define GPU_GET_LAST_ERROR() cudaGetLastError()
#define GPU_SUCCESS cudaSuccess
#define GPU_GET_ERROR_STRING(err) cudaGetErrorString(err)
using gpu_half = half;
#endif

namespace {

constexpr int kHiddenSize = 128;
constexpr int kInputSize = 5;
constexpr int kThreads = 128;
constexpr int kGateCols = 4 * kHiddenSize;

__device__ inline float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ inline float half_to_float(gpu_half v) {
  return __half2float(v);
}

__device__ inline gpu_half float_to_half(float v) {
  return __float2half(v);
}

__device__ inline void accumulate_packed_pairs(
    const gpu_half* __restrict__ values,
    int value_count,
    const gpu_half* __restrict__ weights_packed,
    int out_idx,
    float& i_acc,
    float& f_acc,
    float& g_acc,
    float& o_acc) {
  const int pair_count = (value_count + 1) / 2;
  for (int pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
    const int k0 = pair_idx * 2;
    const int k1 = k0 + 1;
    const float v0 = k0 < value_count ? half_to_float(values[k0]) : 0.0f;
    const float v1 = k1 < value_count ? half_to_float(values[k1]) : 0.0f;
    const int pair_base = pair_idx * (kGateCols * 2);
    const int col_base = out_idx * 2;
    i_acc += v0 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base + 0]) +
             v1 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base + 1]);
    f_acc += v0 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base + 0]) +
             v1 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base + 1]);
    g_acc += v0 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base + 0]) +
             v1 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base + 1]);
    o_acc += v0 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base + 0]) +
             v1 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base + 1]);
  }
}

__global__ void persistent_lstm_pair01_kernel(
    const gpu_half* __restrict__ x,
    const gpu_half* __restrict__ weight_ih_l0_t,
    const gpu_half* __restrict__ weight_hh_l0_t,
    const gpu_half* __restrict__ bias_l0,
    const gpu_half* __restrict__ weight_ih_l1_t,
    const gpu_half* __restrict__ weight_hh_l1_t,
    const gpu_half* __restrict__ bias_l1,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;

  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half h0_cur[kHiddenSize];
  __shared__ gpu_half h1_cur[kHiddenSize];
  float c0_reg = 0.0f;
  float c1_reg = 0.0f;

  h0_cur[h] = float_to_half(0.0f);
  h1_cur[h] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    const int x_base = (b * seq_len + t) * kInputSize;

    float i0 = half_to_float(bias_l0[0 * kHiddenSize + h]);
    float f0 = half_to_float(bias_l0[1 * kHiddenSize + h]);
    float g0 = half_to_float(bias_l0[2 * kHiddenSize + h]);
    float o0 = half_to_float(bias_l0[3 * kHiddenSize + h]);

    #pragma unroll
    for (int k = 0; k < kInputSize; ++k) {
      const float xv = half_to_float(x[x_base + k]);
      const int row = k * (4 * kHiddenSize);
      i0 += xv * half_to_float(weight_ih_l0_t[row + 0 * kHiddenSize + h]);
      f0 += xv * half_to_float(weight_ih_l0_t[row + 1 * kHiddenSize + h]);
      g0 += xv * half_to_float(weight_ih_l0_t[row + 2 * kHiddenSize + h]);
      o0 += xv * half_to_float(weight_ih_l0_t[row + 3 * kHiddenSize + h]);
    }

    #pragma unroll 8
    for (int k = 0; k < kHiddenSize; ++k) {
      const float h_prev = half_to_float(h0_cur[k]);
      const int row = k * (4 * kHiddenSize);
      i0 += h_prev * half_to_float(weight_hh_l0_t[row + 0 * kHiddenSize + h]);
      f0 += h_prev * half_to_float(weight_hh_l0_t[row + 1 * kHiddenSize + h]);
      g0 += h_prev * half_to_float(weight_hh_l0_t[row + 2 * kHiddenSize + h]);
      o0 += h_prev * half_to_float(weight_hh_l0_t[row + 3 * kHiddenSize + h]);
    }

    const float i0_gate = sigmoidf_fast(i0);
    const float f0_gate = sigmoidf_fast(f0);
    const float g0_gate = tanhf(g0);
    const float o0_gate = sigmoidf_fast(o0);
    c0_reg = f0_gate * c0_reg + i0_gate * g0_gate;
    const float c0_val = c0_reg;
    const float h0_val = o0_gate * tanhf(c0_val);

    h0_cur[h] = float_to_half(h0_val);
    __syncthreads();

    float i1 = half_to_float(bias_l1[0 * kHiddenSize + h]);
    float f1 = half_to_float(bias_l1[1 * kHiddenSize + h]);
    float g1 = half_to_float(bias_l1[2 * kHiddenSize + h]);
    float o1 = half_to_float(bias_l1[3 * kHiddenSize + h]);

    #pragma unroll 8
    for (int k = 0; k < kHiddenSize; ++k) {
      const float h0v = half_to_float(h0_cur[k]);
      const float h1v = half_to_float(h1_cur[k]);
      const int row = k * (4 * kHiddenSize);
      i1 += h0v * half_to_float(weight_ih_l1_t[row + 0 * kHiddenSize + h]) +
            h1v * half_to_float(weight_hh_l1_t[row + 0 * kHiddenSize + h]);
      f1 += h0v * half_to_float(weight_ih_l1_t[row + 1 * kHiddenSize + h]) +
            h1v * half_to_float(weight_hh_l1_t[row + 1 * kHiddenSize + h]);
      g1 += h0v * half_to_float(weight_ih_l1_t[row + 2 * kHiddenSize + h]) +
            h1v * half_to_float(weight_hh_l1_t[row + 2 * kHiddenSize + h]);
      o1 += h0v * half_to_float(weight_ih_l1_t[row + 3 * kHiddenSize + h]) +
            h1v * half_to_float(weight_hh_l1_t[row + 3 * kHiddenSize + h]);
    }

    const float i1_gate = sigmoidf_fast(i1);
    const float f1_gate = sigmoidf_fast(f1);
    const float g1_gate = tanhf(g1);
    const float o1_gate = sigmoidf_fast(o1);
    c1_reg = f1_gate * c1_reg + i1_gate * g1_gate;
    const float c1_val = c1_reg;
    const float h1_val = o1_gate * tanhf(c1_val);

    h1_cur[h] = float_to_half(h1_val);
    out[(b * seq_len + t) * kHiddenSize + h] = h1_cur[h];
    __syncthreads();
  }
}

__global__ void persistent_lstm_pair23_last_kernel(
    const gpu_half* __restrict__ x,
    const gpu_half* __restrict__ weight_ih_l2_t,
    const gpu_half* __restrict__ weight_hh_l2_t,
    const gpu_half* __restrict__ bias_l2,
    const gpu_half* __restrict__ weight_ih_l3_t,
    const gpu_half* __restrict__ weight_hh_l3_t,
    const gpu_half* __restrict__ bias_l3,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;

  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half h2_cur[kHiddenSize];
  __shared__ gpu_half h3_cur[kHiddenSize];
  float c2_reg = 0.0f;
  float c3_reg = 0.0f;

  h2_cur[h] = float_to_half(0.0f);
  h3_cur[h] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    const int x_base = (b * seq_len + t) * kHiddenSize;

    float i2 = half_to_float(bias_l2[0 * kHiddenSize + h]);
    float f2 = half_to_float(bias_l2[1 * kHiddenSize + h]);
    float g2 = half_to_float(bias_l2[2 * kHiddenSize + h]);
    float o2 = half_to_float(bias_l2[3 * kHiddenSize + h]);

    #pragma unroll 8
    for (int k = 0; k < kHiddenSize; ++k) {
      const float xv = half_to_float(x[x_base + k]);
      const float h_prev = half_to_float(h2_cur[k]);
      const int row = k * (4 * kHiddenSize);
      i2 += xv * half_to_float(weight_ih_l2_t[row + 0 * kHiddenSize + h]) +
            h_prev * half_to_float(weight_hh_l2_t[row + 0 * kHiddenSize + h]);
      f2 += xv * half_to_float(weight_ih_l2_t[row + 1 * kHiddenSize + h]) +
            h_prev * half_to_float(weight_hh_l2_t[row + 1 * kHiddenSize + h]);
      g2 += xv * half_to_float(weight_ih_l2_t[row + 2 * kHiddenSize + h]) +
            h_prev * half_to_float(weight_hh_l2_t[row + 2 * kHiddenSize + h]);
      o2 += xv * half_to_float(weight_ih_l2_t[row + 3 * kHiddenSize + h]) +
            h_prev * half_to_float(weight_hh_l2_t[row + 3 * kHiddenSize + h]);
    }

    const float i2_gate = sigmoidf_fast(i2);
    const float f2_gate = sigmoidf_fast(f2);
    const float g2_gate = tanhf(g2);
    const float o2_gate = sigmoidf_fast(o2);
    c2_reg = f2_gate * c2_reg + i2_gate * g2_gate;
    const float c2_val = c2_reg;
    const float h2_val = o2_gate * tanhf(c2_val);

    h2_cur[h] = float_to_half(h2_val);
    __syncthreads();

    float i3 = half_to_float(bias_l3[0 * kHiddenSize + h]);
    float f3 = half_to_float(bias_l3[1 * kHiddenSize + h]);
    float g3 = half_to_float(bias_l3[2 * kHiddenSize + h]);
    float o3 = half_to_float(bias_l3[3 * kHiddenSize + h]);

    #pragma unroll 8
    for (int k = 0; k < kHiddenSize; ++k) {
      const float h2v = half_to_float(h2_cur[k]);
      const float h3v = half_to_float(h3_cur[k]);
      const int row = k * (4 * kHiddenSize);
      i3 += h2v * half_to_float(weight_ih_l3_t[row + 0 * kHiddenSize + h]) +
            h3v * half_to_float(weight_hh_l3_t[row + 0 * kHiddenSize + h]);
      f3 += h2v * half_to_float(weight_ih_l3_t[row + 1 * kHiddenSize + h]) +
            h3v * half_to_float(weight_hh_l3_t[row + 1 * kHiddenSize + h]);
      g3 += h2v * half_to_float(weight_ih_l3_t[row + 2 * kHiddenSize + h]) +
            h3v * half_to_float(weight_hh_l3_t[row + 2 * kHiddenSize + h]);
      o3 += h2v * half_to_float(weight_ih_l3_t[row + 3 * kHiddenSize + h]) +
            h3v * half_to_float(weight_hh_l3_t[row + 3 * kHiddenSize + h]);
    }

    const float i3_gate = sigmoidf_fast(i3);
    const float f3_gate = sigmoidf_fast(f3);
    const float g3_gate = tanhf(g3);
    const float o3_gate = sigmoidf_fast(o3);
    c3_reg = f3_gate * c3_reg + i3_gate * g3_gate;
    const float c3_val = c3_reg;
    const float h3_val = o3_gate * tanhf(c3_val);

    h3_cur[h] = float_to_half(h3_val);
    __syncthreads();
  }

  out[b * kHiddenSize + h] = h3_cur[h];
}

__global__ void persistent_lstm_pair01_interleaved_kernel(
    const gpu_half* __restrict__ x,
    const gpu_half* __restrict__ weight_ih_l0_packed,
    const gpu_half* __restrict__ weight_hh_l0_packed,
    const gpu_half* __restrict__ bias_l0,
    const gpu_half* __restrict__ weight_ih_l1_packed,
    const gpu_half* __restrict__ weight_hh_l1_packed,
    const gpu_half* __restrict__ bias_l1,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;
  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half x_step[kInputSize];
  __shared__ gpu_half h0_cur[kHiddenSize];
  __shared__ gpu_half h1_cur[kHiddenSize];
  float c0_reg = 0.0f;
  float c1_reg = 0.0f;

  if (h < kInputSize) {
    x_step[h] = float_to_half(0.0f);
  }
  h0_cur[h] = float_to_half(0.0f);
  h1_cur[h] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    if (h < kInputSize) {
      x_step[h] = x[(b * seq_len + t) * kInputSize + h];
    }
    __syncthreads();

    float i0 = half_to_float(bias_l0[0 * kHiddenSize + h]);
    float f0 = half_to_float(bias_l0[1 * kHiddenSize + h]);
    float g0 = half_to_float(bias_l0[2 * kHiddenSize + h]);
    float o0 = half_to_float(bias_l0[3 * kHiddenSize + h]);
    accumulate_packed_pairs(x_step, kInputSize, weight_ih_l0_packed, h, i0, f0, g0, o0);
    accumulate_packed_pairs(h0_cur, kHiddenSize, weight_hh_l0_packed, h, i0, f0, g0, o0);

    const float i0_gate = sigmoidf_fast(i0);
    const float f0_gate = sigmoidf_fast(f0);
    const float g0_gate = tanhf(g0);
    const float o0_gate = sigmoidf_fast(o0);
    c0_reg = f0_gate * c0_reg + i0_gate * g0_gate;
    h0_cur[h] = float_to_half(o0_gate * tanhf(c0_reg));
    __syncthreads();

    float i1 = half_to_float(bias_l1[0 * kHiddenSize + h]);
    float f1 = half_to_float(bias_l1[1 * kHiddenSize + h]);
    float g1 = half_to_float(bias_l1[2 * kHiddenSize + h]);
    float o1 = half_to_float(bias_l1[3 * kHiddenSize + h]);
    accumulate_packed_pairs(h0_cur, kHiddenSize, weight_ih_l1_packed, h, i1, f1, g1, o1);
    accumulate_packed_pairs(h1_cur, kHiddenSize, weight_hh_l1_packed, h, i1, f1, g1, o1);

    const float i1_gate = sigmoidf_fast(i1);
    const float f1_gate = sigmoidf_fast(f1);
    const float g1_gate = tanhf(g1);
    const float o1_gate = sigmoidf_fast(o1);
    c1_reg = f1_gate * c1_reg + i1_gate * g1_gate;
    h1_cur[h] = float_to_half(o1_gate * tanhf(c1_reg));
    out[(b * seq_len + t) * kHiddenSize + h] = h1_cur[h];
    __syncthreads();
  }
}

__global__ void persistent_lstm_pair23_last_interleaved_kernel(
    const gpu_half* __restrict__ x,
    const gpu_half* __restrict__ weight_ih_l2_packed,
    const gpu_half* __restrict__ weight_hh_l2_packed,
    const gpu_half* __restrict__ bias_l2,
    const gpu_half* __restrict__ weight_ih_l3_packed,
    const gpu_half* __restrict__ weight_hh_l3_packed,
    const gpu_half* __restrict__ bias_l3,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;
  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half x_step[kHiddenSize];
  __shared__ gpu_half h2_cur[kHiddenSize];
  __shared__ gpu_half h3_cur[kHiddenSize];
  float c2_reg = 0.0f;
  float c3_reg = 0.0f;

  x_step[h] = float_to_half(0.0f);
  h2_cur[h] = float_to_half(0.0f);
  h3_cur[h] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    x_step[h] = x[(b * seq_len + t) * kHiddenSize + h];
    __syncthreads();

    float i2 = half_to_float(bias_l2[0 * kHiddenSize + h]);
    float f2 = half_to_float(bias_l2[1 * kHiddenSize + h]);
    float g2 = half_to_float(bias_l2[2 * kHiddenSize + h]);
    float o2 = half_to_float(bias_l2[3 * kHiddenSize + h]);
    accumulate_packed_pairs(x_step, kHiddenSize, weight_ih_l2_packed, h, i2, f2, g2, o2);
    accumulate_packed_pairs(h2_cur, kHiddenSize, weight_hh_l2_packed, h, i2, f2, g2, o2);

    const float i2_gate = sigmoidf_fast(i2);
    const float f2_gate = sigmoidf_fast(f2);
    const float g2_gate = tanhf(g2);
    const float o2_gate = sigmoidf_fast(o2);
    c2_reg = f2_gate * c2_reg + i2_gate * g2_gate;
    h2_cur[h] = float_to_half(o2_gate * tanhf(c2_reg));
    __syncthreads();

    float i3 = half_to_float(bias_l3[0 * kHiddenSize + h]);
    float f3 = half_to_float(bias_l3[1 * kHiddenSize + h]);
    float g3 = half_to_float(bias_l3[2 * kHiddenSize + h]);
    float o3 = half_to_float(bias_l3[3 * kHiddenSize + h]);
    accumulate_packed_pairs(h2_cur, kHiddenSize, weight_ih_l3_packed, h, i3, f3, g3, o3);
    accumulate_packed_pairs(h3_cur, kHiddenSize, weight_hh_l3_packed, h, i3, f3, g3, o3);

    const float i3_gate = sigmoidf_fast(i3);
    const float f3_gate = sigmoidf_fast(f3);
    const float g3_gate = tanhf(g3);
    const float o3_gate = sigmoidf_fast(o3);
    c3_reg = f3_gate * c3_reg + i3_gate * g3_gate;
    h3_cur[h] = float_to_half(o3_gate * tanhf(c3_reg));
    __syncthreads();
  }

  out[b * kHiddenSize + h] = h3_cur[h];
}

bool can_use_specialized_kernel(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3,
    const torch::Tensor& bias_l3) {
  return (
      x.scalar_type() == torch::kFloat16 &&
      x.dim() == 3 &&
      x.size(2) == kInputSize &&
      weight_ih_l0.dim() == 2 && weight_ih_l0.size(0) == 4 * kHiddenSize && weight_ih_l0.size(1) == kInputSize &&
      weight_hh_l0.dim() == 2 && weight_hh_l0.size(0) == 4 * kHiddenSize && weight_hh_l0.size(1) == kHiddenSize &&
      bias_l0.dim() == 1 && bias_l0.size(0) == 4 * kHiddenSize &&
      weight_ih_l1.dim() == 2 && weight_ih_l1.size(0) == 4 * kHiddenSize && weight_ih_l1.size(1) == kHiddenSize &&
      weight_hh_l1.dim() == 2 && weight_hh_l1.size(0) == 4 * kHiddenSize && weight_hh_l1.size(1) == kHiddenSize &&
      bias_l1.dim() == 1 && bias_l1.size(0) == 4 * kHiddenSize &&
      weight_ih_l2.dim() == 2 && weight_ih_l2.size(0) == 4 * kHiddenSize && weight_ih_l2.size(1) == kHiddenSize &&
      weight_hh_l2.dim() == 2 && weight_hh_l2.size(0) == 4 * kHiddenSize && weight_hh_l2.size(1) == kHiddenSize &&
      bias_l2.dim() == 1 && bias_l2.size(0) == 4 * kHiddenSize &&
      weight_ih_l3.dim() == 2 && weight_ih_l3.size(0) == 4 * kHiddenSize && weight_ih_l3.size(1) == kHiddenSize &&
      weight_hh_l3.dim() == 2 && weight_hh_l3.size(0) == 4 * kHiddenSize && weight_hh_l3.size(1) == kHiddenSize &&
      bias_l3.dim() == 1 && bias_l3.size(0) == 4 * kHiddenSize
  );
}

void check_last_error() {
  auto err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    throw std::runtime_error(GPU_GET_ERROR_STRING(err));
  }
}

bool can_use_interleaved_kernel(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_packed,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_packed,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_packed,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_packed,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3) {
  return (
      x.scalar_type() == torch::kFloat16 &&
      x.dim() == 3 &&
      x.size(2) == kInputSize &&
      weight_ih_l0_packed.dim() == 3 && weight_ih_l0_packed.size(0) == 3 && weight_ih_l0_packed.size(1) == kGateCols && weight_ih_l0_packed.size(2) == 2 &&
      weight_hh_l0_packed.dim() == 3 && weight_hh_l0_packed.size(0) == 64 && weight_hh_l0_packed.size(1) == kGateCols && weight_hh_l0_packed.size(2) == 2 &&
      bias_l0.dim() == 1 && bias_l0.size(0) == kGateCols &&
      weight_ih_l1_packed.dim() == 3 && weight_ih_l1_packed.size(0) == 64 && weight_ih_l1_packed.size(1) == kGateCols && weight_ih_l1_packed.size(2) == 2 &&
      weight_hh_l1_packed.dim() == 3 && weight_hh_l1_packed.size(0) == 64 && weight_hh_l1_packed.size(1) == kGateCols && weight_hh_l1_packed.size(2) == 2 &&
      bias_l1.dim() == 1 && bias_l1.size(0) == kGateCols &&
      weight_ih_l2_packed.dim() == 3 && weight_ih_l2_packed.size(0) == 64 && weight_ih_l2_packed.size(1) == kGateCols && weight_ih_l2_packed.size(2) == 2 &&
      weight_hh_l2_packed.dim() == 3 && weight_hh_l2_packed.size(0) == 64 && weight_hh_l2_packed.size(1) == kGateCols && weight_hh_l2_packed.size(2) == 2 &&
      bias_l2.dim() == 1 && bias_l2.size(0) == kGateCols &&
      weight_ih_l3_packed.dim() == 3 && weight_ih_l3_packed.size(0) == 64 && weight_ih_l3_packed.size(1) == kGateCols && weight_ih_l3_packed.size(2) == 2 &&
      weight_hh_l3_packed.dim() == 3 && weight_hh_l3_packed.size(0) == 64 && weight_hh_l3_packed.size(1) == kGateCols && weight_hh_l3_packed.size(2) == 2 &&
      bias_l3.dim() == 1 && bias_l3.size(0) == kGateCols
  );
}

}  // namespace

torch::Tensor persistent_lstm4_forward_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias) {
  if (!can_use_specialized_kernel(
          x,
          weight_ih_l0,
          weight_hh_l0,
          bias_l0,
          weight_ih_l1,
          weight_hh_l1,
          bias_l1,
          weight_ih_l2,
          weight_hh_l2,
          bias_l2,
          weight_ih_l3,
          weight_hh_l3,
          bias_l3)) {
    return persistent_lstm4_forward_reference(
        x,
        weight_ih_l0,
        weight_hh_l0,
        bias_l0,
        weight_ih_l1,
        weight_hh_l1,
        bias_l1,
        weight_ih_l2,
        weight_hh_l2,
        bias_l2,
        weight_ih_l3,
        weight_hh_l3,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto x_c = x.contiguous();
  auto wih0 = weight_ih_l0.contiguous();
  auto whh0 = weight_hh_l0.contiguous();
  auto b0 = bias_l0.contiguous();
  auto wih1 = weight_ih_l1.contiguous();
  auto whh1 = weight_hh_l1.contiguous();
  auto b1 = bias_l1.contiguous();
  auto wih2 = weight_ih_l2.contiguous();
  auto whh2 = weight_hh_l2.contiguous();
  auto b2 = bias_l2.contiguous();
  auto wih3 = weight_ih_l3.contiguous();
  auto whh3 = weight_hh_l3.contiguous();
  auto b3 = bias_l3.contiguous();

  const int batch_size = static_cast<int>(x_c.size(0));
  const int seq_len = static_cast<int>(x_c.size(1));

  auto seq01 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());
  auto last = torch::empty({batch_size, kHiddenSize}, x_c.options());

  const gpu_half* x_ptr = reinterpret_cast<const gpu_half*>(x_c.data_ptr<at::Half>());
  const gpu_half* wih0_ptr = reinterpret_cast<const gpu_half*>(wih0.data_ptr<at::Half>());
  const gpu_half* whh0_ptr = reinterpret_cast<const gpu_half*>(whh0.data_ptr<at::Half>());
  const gpu_half* b0_ptr = reinterpret_cast<const gpu_half*>(b0.data_ptr<at::Half>());
  const gpu_half* wih1_ptr = reinterpret_cast<const gpu_half*>(wih1.data_ptr<at::Half>());
  const gpu_half* whh1_ptr = reinterpret_cast<const gpu_half*>(whh1.data_ptr<at::Half>());
  const gpu_half* b1_ptr = reinterpret_cast<const gpu_half*>(b1.data_ptr<at::Half>());
  const gpu_half* wih2_ptr = reinterpret_cast<const gpu_half*>(wih2.data_ptr<at::Half>());
  const gpu_half* whh2_ptr = reinterpret_cast<const gpu_half*>(whh2.data_ptr<at::Half>());
  const gpu_half* b2_ptr = reinterpret_cast<const gpu_half*>(b2.data_ptr<at::Half>());
  const gpu_half* wih3_ptr = reinterpret_cast<const gpu_half*>(wih3.data_ptr<at::Half>());
  const gpu_half* whh3_ptr = reinterpret_cast<const gpu_half*>(whh3.data_ptr<at::Half>());
  const gpu_half* b3_ptr = reinterpret_cast<const gpu_half*>(b3.data_ptr<at::Half>());
  gpu_half* seq01_ptr = reinterpret_cast<gpu_half*>(seq01.data_ptr<at::Half>());
  gpu_half* last_ptr = reinterpret_cast<gpu_half*>(last.data_ptr<at::Half>());

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair01_kernel,
      batch_size,
      kThreads,
      0,
      0,
      x_ptr,
      wih0_ptr,
      whh0_ptr,
      b0_ptr,
      wih1_ptr,
      whh1_ptr,
      b1_ptr,
      seq01_ptr,
      batch_size,
      seq_len);
  check_last_error();

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair23_last_kernel,
      batch_size,
      kThreads,
      0,
      0,
      seq01_ptr,
      wih2_ptr,
      whh2_ptr,
      b2_ptr,
      wih3_ptr,
      whh3_ptr,
      b3_ptr,
      last_ptr,
      batch_size,
      seq_len);
  check_last_error();

  return torch::matmul(last, linear_weight.transpose(0, 1)) + linear_bias;
}

torch::Tensor persistent_lstm4_forward_packed_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_t,
    const torch::Tensor& weight_hh_l0_t,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_t,
    const torch::Tensor& weight_hh_l1_t,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_t,
    const torch::Tensor& weight_hh_l2_t,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_t,
    const torch::Tensor& weight_hh_l3_t,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias) {
  if (!(x.scalar_type() == torch::kFloat16 &&
        x.dim() == 3 &&
        x.size(2) == kInputSize &&
        weight_ih_l0_t.dim() == 2 && weight_ih_l0_t.size(0) == kInputSize && weight_ih_l0_t.size(1) == 4 * kHiddenSize &&
        weight_hh_l0_t.dim() == 2 && weight_hh_l0_t.size(0) == kHiddenSize && weight_hh_l0_t.size(1) == 4 * kHiddenSize &&
        weight_ih_l1_t.dim() == 2 && weight_ih_l1_t.size(0) == kHiddenSize && weight_ih_l1_t.size(1) == 4 * kHiddenSize &&
        weight_hh_l1_t.dim() == 2 && weight_hh_l1_t.size(0) == kHiddenSize && weight_hh_l1_t.size(1) == 4 * kHiddenSize &&
        weight_ih_l2_t.dim() == 2 && weight_ih_l2_t.size(0) == kHiddenSize && weight_ih_l2_t.size(1) == 4 * kHiddenSize &&
        weight_hh_l2_t.dim() == 2 && weight_hh_l2_t.size(0) == kHiddenSize && weight_hh_l2_t.size(1) == 4 * kHiddenSize &&
        weight_ih_l3_t.dim() == 2 && weight_ih_l3_t.size(0) == kHiddenSize && weight_ih_l3_t.size(1) == 4 * kHiddenSize &&
        weight_hh_l3_t.dim() == 2 && weight_hh_l3_t.size(0) == kHiddenSize && weight_hh_l3_t.size(1) == 4 * kHiddenSize)) {
    return persistent_lstm4_forward_reference(
        x,
        weight_ih_l0_t.transpose(0, 1).contiguous(),
        weight_hh_l0_t.transpose(0, 1).contiguous(),
        bias_l0,
        weight_ih_l1_t.transpose(0, 1).contiguous(),
        weight_hh_l1_t.transpose(0, 1).contiguous(),
        bias_l1,
        weight_ih_l2_t.transpose(0, 1).contiguous(),
        weight_hh_l2_t.transpose(0, 1).contiguous(),
        bias_l2,
        weight_ih_l3_t.transpose(0, 1).contiguous(),
        weight_hh_l3_t.transpose(0, 1).contiguous(),
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto x_c = x.contiguous();
  auto wih0 = weight_ih_l0_t.contiguous();
  auto whh0 = weight_hh_l0_t.contiguous();
  auto b0 = bias_l0.contiguous();
  auto wih1 = weight_ih_l1_t.contiguous();
  auto whh1 = weight_hh_l1_t.contiguous();
  auto b1 = bias_l1.contiguous();
  auto wih2 = weight_ih_l2_t.contiguous();
  auto whh2 = weight_hh_l2_t.contiguous();
  auto b2 = bias_l2.contiguous();
  auto wih3 = weight_ih_l3_t.contiguous();
  auto whh3 = weight_hh_l3_t.contiguous();
  auto b3 = bias_l3.contiguous();

  const int batch_size = static_cast<int>(x_c.size(0));
  const int seq_len = static_cast<int>(x_c.size(1));

  auto seq01 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());
  auto last = torch::empty({batch_size, kHiddenSize}, x_c.options());

  const gpu_half* x_ptr = reinterpret_cast<const gpu_half*>(x_c.data_ptr<at::Half>());
  const gpu_half* wih0_ptr = reinterpret_cast<const gpu_half*>(wih0.data_ptr<at::Half>());
  const gpu_half* whh0_ptr = reinterpret_cast<const gpu_half*>(whh0.data_ptr<at::Half>());
  const gpu_half* b0_ptr = reinterpret_cast<const gpu_half*>(b0.data_ptr<at::Half>());
  const gpu_half* wih1_ptr = reinterpret_cast<const gpu_half*>(wih1.data_ptr<at::Half>());
  const gpu_half* whh1_ptr = reinterpret_cast<const gpu_half*>(whh1.data_ptr<at::Half>());
  const gpu_half* b1_ptr = reinterpret_cast<const gpu_half*>(b1.data_ptr<at::Half>());
  const gpu_half* wih2_ptr = reinterpret_cast<const gpu_half*>(wih2.data_ptr<at::Half>());
  const gpu_half* whh2_ptr = reinterpret_cast<const gpu_half*>(whh2.data_ptr<at::Half>());
  const gpu_half* b2_ptr = reinterpret_cast<const gpu_half*>(b2.data_ptr<at::Half>());
  const gpu_half* wih3_ptr = reinterpret_cast<const gpu_half*>(wih3.data_ptr<at::Half>());
  const gpu_half* whh3_ptr = reinterpret_cast<const gpu_half*>(whh3.data_ptr<at::Half>());
  const gpu_half* b3_ptr = reinterpret_cast<const gpu_half*>(b3.data_ptr<at::Half>());
  gpu_half* seq01_ptr = reinterpret_cast<gpu_half*>(seq01.data_ptr<at::Half>());
  gpu_half* last_ptr = reinterpret_cast<gpu_half*>(last.data_ptr<at::Half>());

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair01_kernel,
      batch_size,
      kThreads,
      0,
      0,
      x_ptr,
      wih0_ptr,
      whh0_ptr,
      b0_ptr,
      wih1_ptr,
      whh1_ptr,
      b1_ptr,
      seq01_ptr,
      batch_size,
      seq_len);
  check_last_error();

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair23_last_kernel,
      batch_size,
      kThreads,
      0,
      0,
      seq01_ptr,
      wih2_ptr,
      whh2_ptr,
      b2_ptr,
      wih3_ptr,
      whh3_ptr,
      b3_ptr,
      last_ptr,
      batch_size,
      seq_len);
  check_last_error();

  return torch::matmul(last, linear_weight.transpose(0, 1)) + linear_bias;
}

torch::Tensor persistent_lstm4_forward_interleaved_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_packed,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_packed,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_packed,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_packed,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias) {
  if (!can_use_interleaved_kernel(
          x,
          weight_ih_l0_packed,
          weight_hh_l0_packed,
          bias_l0,
          weight_ih_l1_packed,
          weight_hh_l1_packed,
          bias_l1,
          weight_ih_l2_packed,
          weight_hh_l2_packed,
          bias_l2,
          weight_ih_l3_packed,
          weight_hh_l3_packed,
          bias_l3)) {
    auto unpack = [](const torch::Tensor& packed, int64_t original_k) {
      auto contiguous = packed.contiguous();
      const auto pairs = contiguous.size(0);
      const auto out = contiguous.size(1);
      const auto restored = contiguous.permute({0, 2, 1}).contiguous().view({pairs * 2, out});
      return restored.narrow(0, 0, original_k).transpose(0, 1).contiguous();
    };
    return persistent_lstm4_forward_reference(
        x,
        unpack(weight_ih_l0_packed, kInputSize),
        unpack(weight_hh_l0_packed, kHiddenSize),
        bias_l0,
        unpack(weight_ih_l1_packed, kHiddenSize),
        unpack(weight_hh_l1_packed, kHiddenSize),
        bias_l1,
        unpack(weight_ih_l2_packed, kHiddenSize),
        unpack(weight_hh_l2_packed, kHiddenSize),
        bias_l2,
        unpack(weight_ih_l3_packed, kHiddenSize),
        unpack(weight_hh_l3_packed, kHiddenSize),
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto x_c = x.contiguous();
  auto wih0 = weight_ih_l0_packed.contiguous();
  auto whh0 = weight_hh_l0_packed.contiguous();
  auto b0 = bias_l0.contiguous();
  auto wih1 = weight_ih_l1_packed.contiguous();
  auto whh1 = weight_hh_l1_packed.contiguous();
  auto b1 = bias_l1.contiguous();
  auto wih2 = weight_ih_l2_packed.contiguous();
  auto whh2 = weight_hh_l2_packed.contiguous();
  auto b2 = bias_l2.contiguous();
  auto wih3 = weight_ih_l3_packed.contiguous();
  auto whh3 = weight_hh_l3_packed.contiguous();
  auto b3 = bias_l3.contiguous();

  const int batch_size = static_cast<int>(x_c.size(0));
  const int seq_len = static_cast<int>(x_c.size(1));

  auto seq01 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());
  auto last = torch::empty({batch_size, kHiddenSize}, x_c.options());

  const gpu_half* x_ptr = reinterpret_cast<const gpu_half*>(x_c.data_ptr<at::Half>());
  const gpu_half* wih0_ptr = reinterpret_cast<const gpu_half*>(wih0.data_ptr<at::Half>());
  const gpu_half* whh0_ptr = reinterpret_cast<const gpu_half*>(whh0.data_ptr<at::Half>());
  const gpu_half* b0_ptr = reinterpret_cast<const gpu_half*>(b0.data_ptr<at::Half>());
  const gpu_half* wih1_ptr = reinterpret_cast<const gpu_half*>(wih1.data_ptr<at::Half>());
  const gpu_half* whh1_ptr = reinterpret_cast<const gpu_half*>(whh1.data_ptr<at::Half>());
  const gpu_half* b1_ptr = reinterpret_cast<const gpu_half*>(b1.data_ptr<at::Half>());
  const gpu_half* wih2_ptr = reinterpret_cast<const gpu_half*>(wih2.data_ptr<at::Half>());
  const gpu_half* whh2_ptr = reinterpret_cast<const gpu_half*>(whh2.data_ptr<at::Half>());
  const gpu_half* b2_ptr = reinterpret_cast<const gpu_half*>(b2.data_ptr<at::Half>());
  const gpu_half* wih3_ptr = reinterpret_cast<const gpu_half*>(wih3.data_ptr<at::Half>());
  const gpu_half* whh3_ptr = reinterpret_cast<const gpu_half*>(whh3.data_ptr<at::Half>());
  const gpu_half* b3_ptr = reinterpret_cast<const gpu_half*>(b3.data_ptr<at::Half>());
  gpu_half* seq01_ptr = reinterpret_cast<gpu_half*>(seq01.data_ptr<at::Half>());
  gpu_half* last_ptr = reinterpret_cast<gpu_half*>(last.data_ptr<at::Half>());

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair01_interleaved_kernel,
      batch_size,
      kThreads,
      0,
      0,
      x_ptr,
      wih0_ptr,
      whh0_ptr,
      b0_ptr,
      wih1_ptr,
      whh1_ptr,
      b1_ptr,
      seq01_ptr,
      batch_size,
      seq_len);
  check_last_error();

  GPU_LAUNCH_KERNEL(
      persistent_lstm_pair23_last_interleaved_kernel,
      batch_size,
      kThreads,
      0,
      0,
      seq01_ptr,
      wih2_ptr,
      whh2_ptr,
      b2_ptr,
      wih3_ptr,
      whh3_ptr,
      b3_ptr,
      last_ptr,
      batch_size,
      seq_len);
  check_last_error();

  return torch::matmul(last, linear_weight.transpose(0, 1)) + linear_bias;
}
