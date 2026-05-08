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
constexpr int kWaveThreads = 64;
constexpr int kGateCols = 4 * kHiddenSize;
constexpr int kHiddenPairs = kHiddenSize / 2;

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

__device__ inline void accumulate_packed_pairs_dual(
    const gpu_half* __restrict__ values,
    int value_count,
    const gpu_half* __restrict__ weights_packed,
    int out_idx0,
    int out_idx1,
    float& i0_acc,
    float& f0_acc,
    float& g0_acc,
    float& o0_acc,
    float& i1_acc,
    float& f1_acc,
    float& g1_acc,
    float& o1_acc) {
  const int pair_count = (value_count + 1) / 2;
  const int col_base0 = out_idx0 * 2;
  const int col_base1 = out_idx1 * 2;
  for (int pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
    const int k0 = pair_idx * 2;
    const int k1 = k0 + 1;
    const float v0 = k0 < value_count ? half_to_float(values[k0]) : 0.0f;
    const float v1 = k1 < value_count ? half_to_float(values[k1]) : 0.0f;
    const int pair_base = pair_idx * (kGateCols * 2);
    i0_acc += v0 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base0 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base0 + 1]);
    f0_acc += v0 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base0 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base0 + 1]);
    g0_acc += v0 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base0 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base0 + 1]);
    o0_acc += v0 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base0 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base0 + 1]);
    i1_acc += v0 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base1 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base1 + 1]);
    f1_acc += v0 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base1 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base1 + 1]);
    g1_acc += v0 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base1 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base1 + 1]);
    o1_acc += v0 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base1 + 0]) +
              v1 * half_to_float(weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base1 + 1]);
  }
}

__device__ inline void load_recurrent_cache_for_thread(
    const gpu_half* __restrict__ weights_packed,
    int out_idx,
    gpu_half* __restrict__ cache) {
  const int col_base = out_idx * 2;
  for (int pair_idx = 0; pair_idx < kHiddenPairs; ++pair_idx) {
    const int pair_base = pair_idx * (kGateCols * 2);
    const int cache_base = pair_idx * 8;
    cache[cache_base + 0] = weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base + 0];
    cache[cache_base + 1] = weights_packed[pair_base + (0 * kHiddenSize * 2) + col_base + 1];
    cache[cache_base + 2] = weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base + 0];
    cache[cache_base + 3] = weights_packed[pair_base + (1 * kHiddenSize * 2) + col_base + 1];
    cache[cache_base + 4] = weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base + 0];
    cache[cache_base + 5] = weights_packed[pair_base + (2 * kHiddenSize * 2) + col_base + 1];
    cache[cache_base + 6] = weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base + 0];
    cache[cache_base + 7] = weights_packed[pair_base + (3 * kHiddenSize * 2) + col_base + 1];
  }
}

__device__ inline void accumulate_cached_recurrent(
    const gpu_half* __restrict__ h_values,
    const gpu_half* __restrict__ hh_cache,
    float& i_acc,
    float& f_acc,
    float& g_acc,
    float& o_acc) {
  for (int pair_idx = 0; pair_idx < kHiddenPairs; ++pair_idx) {
    const int k0 = pair_idx * 2;
    const int k1 = k0 + 1;
    const float v0 = half_to_float(h_values[k0]);
    const float v1 = half_to_float(h_values[k1]);
    const int cache_base = pair_idx * 8;
    i_acc += v0 * half_to_float(hh_cache[cache_base + 0]) + v1 * half_to_float(hh_cache[cache_base + 1]);
    f_acc += v0 * half_to_float(hh_cache[cache_base + 2]) + v1 * half_to_float(hh_cache[cache_base + 3]);
    g_acc += v0 * half_to_float(hh_cache[cache_base + 4]) + v1 * half_to_float(hh_cache[cache_base + 5]);
    o_acc += v0 * half_to_float(hh_cache[cache_base + 6]) + v1 * half_to_float(hh_cache[cache_base + 7]);
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
  const int lane = threadIdx.x;
  if (b >= batch_size || lane >= kWaveThreads) {
    return;
  }
  const int h0 = lane;
  const int h1 = lane + kWaveThreads;

  __shared__ gpu_half x_step[kInputSize];
  __shared__ gpu_half h0_cur[kHiddenSize];
  __shared__ gpu_half h1_cur[kHiddenSize];
  float c0_reg0 = 0.0f;
  float c0_reg1 = 0.0f;
  float c1_reg0 = 0.0f;
  float c1_reg1 = 0.0f;

  if (lane < kInputSize) {
    x_step[lane] = float_to_half(0.0f);
  }
  h0_cur[h0] = float_to_half(0.0f);
  h0_cur[h1] = float_to_half(0.0f);
  h1_cur[h0] = float_to_half(0.0f);
  h1_cur[h1] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    if (lane < kInputSize) {
      x_step[lane] = x[(b * seq_len + t) * kInputSize + lane];
    }
    __syncthreads();

    float i0_0 = half_to_float(bias_l0[0 * kHiddenSize + h0]);
    float f0_0 = half_to_float(bias_l0[1 * kHiddenSize + h0]);
    float g0_0 = half_to_float(bias_l0[2 * kHiddenSize + h0]);
    float o0_0 = half_to_float(bias_l0[3 * kHiddenSize + h0]);
    float i0_1 = half_to_float(bias_l0[0 * kHiddenSize + h1]);
    float f0_1 = half_to_float(bias_l0[1 * kHiddenSize + h1]);
    float g0_1 = half_to_float(bias_l0[2 * kHiddenSize + h1]);
    float o0_1 = half_to_float(bias_l0[3 * kHiddenSize + h1]);
    accumulate_packed_pairs_dual(
        x_step, kInputSize, weight_ih_l0_packed, h0, h1,
        i0_0, f0_0, g0_0, o0_0, i0_1, f0_1, g0_1, o0_1);
    accumulate_packed_pairs_dual(
        h0_cur, kHiddenSize, weight_hh_l0_packed, h0, h1,
        i0_0, f0_0, g0_0, o0_0, i0_1, f0_1, g0_1, o0_1);

    const float i0_gate0 = sigmoidf_fast(i0_0);
    const float f0_gate0 = sigmoidf_fast(f0_0);
    const float g0_gate0 = tanhf(g0_0);
    const float o0_gate0 = sigmoidf_fast(o0_0);
    const float i0_gate1 = sigmoidf_fast(i0_1);
    const float f0_gate1 = sigmoidf_fast(f0_1);
    const float g0_gate1 = tanhf(g0_1);
    const float o0_gate1 = sigmoidf_fast(o0_1);
    c0_reg0 = f0_gate0 * c0_reg0 + i0_gate0 * g0_gate0;
    c0_reg1 = f0_gate1 * c0_reg1 + i0_gate1 * g0_gate1;
    h0_cur[h0] = float_to_half(o0_gate0 * tanhf(c0_reg0));
    h0_cur[h1] = float_to_half(o0_gate1 * tanhf(c0_reg1));
    __syncthreads();

    float i1_0 = half_to_float(bias_l1[0 * kHiddenSize + h0]);
    float f1_0 = half_to_float(bias_l1[1 * kHiddenSize + h0]);
    float g1_0 = half_to_float(bias_l1[2 * kHiddenSize + h0]);
    float o1_0 = half_to_float(bias_l1[3 * kHiddenSize + h0]);
    float i1_1 = half_to_float(bias_l1[0 * kHiddenSize + h1]);
    float f1_1 = half_to_float(bias_l1[1 * kHiddenSize + h1]);
    float g1_1 = half_to_float(bias_l1[2 * kHiddenSize + h1]);
    float o1_1 = half_to_float(bias_l1[3 * kHiddenSize + h1]);
    accumulate_packed_pairs_dual(
        h0_cur, kHiddenSize, weight_ih_l1_packed, h0, h1,
        i1_0, f1_0, g1_0, o1_0, i1_1, f1_1, g1_1, o1_1);
    accumulate_packed_pairs_dual(
        h1_cur, kHiddenSize, weight_hh_l1_packed, h0, h1,
        i1_0, f1_0, g1_0, o1_0, i1_1, f1_1, g1_1, o1_1);

    const float i1_gate0 = sigmoidf_fast(i1_0);
    const float f1_gate0 = sigmoidf_fast(f1_0);
    const float g1_gate0 = tanhf(g1_0);
    const float o1_gate0 = sigmoidf_fast(o1_0);
    const float i1_gate1 = sigmoidf_fast(i1_1);
    const float f1_gate1 = sigmoidf_fast(f1_1);
    const float g1_gate1 = tanhf(g1_1);
    const float o1_gate1 = sigmoidf_fast(o1_1);
    c1_reg0 = f1_gate0 * c1_reg0 + i1_gate0 * g1_gate0;
    c1_reg1 = f1_gate1 * c1_reg1 + i1_gate1 * g1_gate1;
    h1_cur[h0] = float_to_half(o1_gate0 * tanhf(c1_reg0));
    h1_cur[h1] = float_to_half(o1_gate1 * tanhf(c1_reg1));
    out[(b * seq_len + t) * kHiddenSize + h0] = h1_cur[h0];
    out[(b * seq_len + t) * kHiddenSize + h1] = h1_cur[h1];
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
  const int lane = threadIdx.x;
  if (b >= batch_size || lane >= kWaveThreads) {
    return;
  }
  const int h0 = lane;
  const int h1 = lane + kWaveThreads;

  __shared__ gpu_half x_step[kHiddenSize];
  __shared__ gpu_half h2_cur[kHiddenSize];
  __shared__ gpu_half h3_cur[kHiddenSize];
  float c2_reg0 = 0.0f;
  float c2_reg1 = 0.0f;
  float c3_reg0 = 0.0f;
  float c3_reg1 = 0.0f;

  x_step[h0] = float_to_half(0.0f);
  x_step[h1] = float_to_half(0.0f);
  h2_cur[h0] = float_to_half(0.0f);
  h2_cur[h1] = float_to_half(0.0f);
  h3_cur[h0] = float_to_half(0.0f);
  h3_cur[h1] = float_to_half(0.0f);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    x_step[h0] = x[(b * seq_len + t) * kHiddenSize + h0];
    x_step[h1] = x[(b * seq_len + t) * kHiddenSize + h1];
    __syncthreads();

    float i2_0 = half_to_float(bias_l2[0 * kHiddenSize + h0]);
    float f2_0 = half_to_float(bias_l2[1 * kHiddenSize + h0]);
    float g2_0 = half_to_float(bias_l2[2 * kHiddenSize + h0]);
    float o2_0 = half_to_float(bias_l2[3 * kHiddenSize + h0]);
    float i2_1 = half_to_float(bias_l2[0 * kHiddenSize + h1]);
    float f2_1 = half_to_float(bias_l2[1 * kHiddenSize + h1]);
    float g2_1 = half_to_float(bias_l2[2 * kHiddenSize + h1]);
    float o2_1 = half_to_float(bias_l2[3 * kHiddenSize + h1]);
    accumulate_packed_pairs_dual(
        x_step, kHiddenSize, weight_ih_l2_packed, h0, h1,
        i2_0, f2_0, g2_0, o2_0, i2_1, f2_1, g2_1, o2_1);
    accumulate_packed_pairs_dual(
        h2_cur, kHiddenSize, weight_hh_l2_packed, h0, h1,
        i2_0, f2_0, g2_0, o2_0, i2_1, f2_1, g2_1, o2_1);

    const float i2_gate0 = sigmoidf_fast(i2_0);
    const float f2_gate0 = sigmoidf_fast(f2_0);
    const float g2_gate0 = tanhf(g2_0);
    const float o2_gate0 = sigmoidf_fast(o2_0);
    const float i2_gate1 = sigmoidf_fast(i2_1);
    const float f2_gate1 = sigmoidf_fast(f2_1);
    const float g2_gate1 = tanhf(g2_1);
    const float o2_gate1 = sigmoidf_fast(o2_1);
    c2_reg0 = f2_gate0 * c2_reg0 + i2_gate0 * g2_gate0;
    c2_reg1 = f2_gate1 * c2_reg1 + i2_gate1 * g2_gate1;
    h2_cur[h0] = float_to_half(o2_gate0 * tanhf(c2_reg0));
    h2_cur[h1] = float_to_half(o2_gate1 * tanhf(c2_reg1));
    __syncthreads();

    float i3_0 = half_to_float(bias_l3[0 * kHiddenSize + h0]);
    float f3_0 = half_to_float(bias_l3[1 * kHiddenSize + h0]);
    float g3_0 = half_to_float(bias_l3[2 * kHiddenSize + h0]);
    float o3_0 = half_to_float(bias_l3[3 * kHiddenSize + h0]);
    float i3_1 = half_to_float(bias_l3[0 * kHiddenSize + h1]);
    float f3_1 = half_to_float(bias_l3[1 * kHiddenSize + h1]);
    float g3_1 = half_to_float(bias_l3[2 * kHiddenSize + h1]);
    float o3_1 = half_to_float(bias_l3[3 * kHiddenSize + h1]);
    accumulate_packed_pairs_dual(
        h2_cur, kHiddenSize, weight_ih_l3_packed, h0, h1,
        i3_0, f3_0, g3_0, o3_0, i3_1, f3_1, g3_1, o3_1);
    accumulate_packed_pairs_dual(
        h3_cur, kHiddenSize, weight_hh_l3_packed, h0, h1,
        i3_0, f3_0, g3_0, o3_0, i3_1, f3_1, g3_1, o3_1);

    const float i3_gate0 = sigmoidf_fast(i3_0);
    const float f3_gate0 = sigmoidf_fast(f3_0);
    const float g3_gate0 = tanhf(g3_0);
    const float o3_gate0 = sigmoidf_fast(o3_0);
    const float i3_gate1 = sigmoidf_fast(i3_1);
    const float f3_gate1 = sigmoidf_fast(f3_1);
    const float g3_gate1 = tanhf(g3_1);
    const float o3_gate1 = sigmoidf_fast(o3_1);
    c3_reg0 = f3_gate0 * c3_reg0 + i3_gate0 * g3_gate0;
    c3_reg1 = f3_gate1 * c3_reg1 + i3_gate1 * g3_gate1;
    h3_cur[h0] = float_to_half(o3_gate0 * tanhf(c3_reg0));
    h3_cur[h1] = float_to_half(o3_gate1 * tanhf(c3_reg1));
    __syncthreads();
  }

  out[b * kHiddenSize + h0] = h3_cur[h0];
  out[b * kHiddenSize + h1] = h3_cur[h1];
}

__global__ void persistent_lstm_projected_full_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh_packed,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;
  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half h_cur[kHiddenSize];
  gpu_half hh_cache[kHiddenPairs * 8];
  float c_reg = 0.0f;

  h_cur[h] = float_to_half(0.0f);
  load_recurrent_cache_for_thread(weight_hh_packed, h, hh_cache);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    const int gate_base = (b * seq_len + t) * kGateCols;
    float i_acc = half_to_float(gate_proj[gate_base + 0 * kHiddenSize + h]);
    float f_acc = half_to_float(gate_proj[gate_base + 1 * kHiddenSize + h]);
    float g_acc = half_to_float(gate_proj[gate_base + 2 * kHiddenSize + h]);
    float o_acc = half_to_float(gate_proj[gate_base + 3 * kHiddenSize + h]);
    accumulate_cached_recurrent(h_cur, hh_cache, i_acc, f_acc, g_acc, o_acc);

    const float i_gate = sigmoidf_fast(i_acc);
    const float f_gate = sigmoidf_fast(f_acc);
    const float g_gate = tanhf(g_acc);
    const float o_gate = sigmoidf_fast(o_acc);
    c_reg = f_gate * c_reg + i_gate * g_gate;
    h_cur[h] = float_to_half(o_gate * tanhf(c_reg));
    out[(b * seq_len + t) * kHiddenSize + h] = h_cur[h];
    __syncthreads();
  }
}

__global__ void persistent_lstm_projected_last_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh_packed,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  const int b = blockIdx.x;
  const int h = threadIdx.x;
  if (b >= batch_size || h >= kHiddenSize) {
    return;
  }

  __shared__ gpu_half h_cur[kHiddenSize];
  gpu_half hh_cache[kHiddenPairs * 8];
  float c_reg = 0.0f;

  h_cur[h] = float_to_half(0.0f);
  load_recurrent_cache_for_thread(weight_hh_packed, h, hh_cache);
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    const int gate_base = (b * seq_len + t) * kGateCols;
    float i_acc = half_to_float(gate_proj[gate_base + 0 * kHiddenSize + h]);
    float f_acc = half_to_float(gate_proj[gate_base + 1 * kHiddenSize + h]);
    float g_acc = half_to_float(gate_proj[gate_base + 2 * kHiddenSize + h]);
    float o_acc = half_to_float(gate_proj[gate_base + 3 * kHiddenSize + h]);
    accumulate_cached_recurrent(h_cur, hh_cache, i_acc, f_acc, g_acc, o_acc);

    const float i_gate = sigmoidf_fast(i_acc);
    const float f_gate = sigmoidf_fast(f_acc);
    const float g_gate = tanhf(g_acc);
    const float o_gate = sigmoidf_fast(o_acc);
    c_reg = f_gate * c_reg + i_gate * g_gate;
    h_cur[h] = float_to_half(o_gate * tanhf(c_reg));
    __syncthreads();
  }

  out[b * kHiddenSize + h] = h_cur[h];
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

bool can_use_projected_kernel(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3) {
  return (
      x.scalar_type() == torch::kFloat16 &&
      x.dim() == 3 &&
      x.size(2) == kInputSize &&
      weight_ih_l0.dim() == 2 && weight_ih_l0.size(0) == kGateCols && weight_ih_l0.size(1) == kInputSize &&
      weight_hh_l0_packed.dim() == 3 && weight_hh_l0_packed.size(0) == kHiddenPairs && weight_hh_l0_packed.size(1) == kGateCols && weight_hh_l0_packed.size(2) == 2 &&
      bias_l0.dim() == 1 && bias_l0.size(0) == kGateCols &&
      weight_ih_l1.dim() == 2 && weight_ih_l1.size(0) == kGateCols && weight_ih_l1.size(1) == kHiddenSize &&
      weight_hh_l1_packed.dim() == 3 && weight_hh_l1_packed.size(0) == kHiddenPairs && weight_hh_l1_packed.size(1) == kGateCols && weight_hh_l1_packed.size(2) == 2 &&
      bias_l1.dim() == 1 && bias_l1.size(0) == kGateCols &&
      weight_ih_l2.dim() == 2 && weight_ih_l2.size(0) == kGateCols && weight_ih_l2.size(1) == kHiddenSize &&
      weight_hh_l2_packed.dim() == 3 && weight_hh_l2_packed.size(0) == kHiddenPairs && weight_hh_l2_packed.size(1) == kGateCols && weight_hh_l2_packed.size(2) == 2 &&
      bias_l2.dim() == 1 && bias_l2.size(0) == kGateCols &&
      weight_ih_l3.dim() == 2 && weight_ih_l3.size(0) == kGateCols && weight_ih_l3.size(1) == kHiddenSize &&
      weight_hh_l3_packed.dim() == 3 && weight_hh_l3_packed.size(0) == kHiddenPairs && weight_hh_l3_packed.size(1) == kGateCols && weight_hh_l3_packed.size(2) == 2 &&
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
      kWaveThreads,
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
      kWaveThreads,
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

torch::Tensor persistent_lstm4_forward_projected_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias) {
  if (!can_use_projected_kernel(
          x,
          weight_ih_l0,
          weight_hh_l0_packed,
          bias_l0,
          weight_ih_l1,
          weight_hh_l1_packed,
          bias_l1,
          weight_ih_l2,
          weight_hh_l2_packed,
          bias_l2,
          weight_ih_l3,
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
        weight_ih_l0.contiguous(),
        unpack(weight_hh_l0_packed, kHiddenSize),
        bias_l0,
        weight_ih_l1.contiguous(),
        unpack(weight_hh_l1_packed, kHiddenSize),
        bias_l1,
        weight_ih_l2.contiguous(),
        unpack(weight_hh_l2_packed, kHiddenSize),
        bias_l2,
        weight_ih_l3.contiguous(),
        unpack(weight_hh_l3_packed, kHiddenSize),
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto x_c = x.contiguous();
  auto wih0 = weight_ih_l0.contiguous();
  auto whh0 = weight_hh_l0_packed.contiguous();
  auto b0 = bias_l0.contiguous();
  auto wih1 = weight_ih_l1.contiguous();
  auto whh1 = weight_hh_l1_packed.contiguous();
  auto b1 = bias_l1.contiguous();
  auto wih2 = weight_ih_l2.contiguous();
  auto whh2 = weight_hh_l2_packed.contiguous();
  auto b2 = bias_l2.contiguous();
  auto wih3 = weight_ih_l3.contiguous();
  auto whh3 = weight_hh_l3_packed.contiguous();
  auto b3 = bias_l3.contiguous();

  const int64_t batch_size = x_c.size(0);
  const int64_t seq_len = x_c.size(1);

  auto x_2d = x_c.view({batch_size * seq_len, kInputSize});
  auto gate0 = (torch::matmul(x_2d, wih0.transpose(0, 1)) + b0).view({batch_size, seq_len, kGateCols}).contiguous();
  auto seq0 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());

  GPU_LAUNCH_KERNEL(
      persistent_lstm_projected_full_kernel,
      static_cast<int>(batch_size),
      kThreads,
      0,
      0,
      reinterpret_cast<const gpu_half*>(gate0.data_ptr<at::Half>()),
      reinterpret_cast<const gpu_half*>(whh0.data_ptr<at::Half>()),
      reinterpret_cast<gpu_half*>(seq0.data_ptr<at::Half>()),
      static_cast<int>(batch_size),
      static_cast<int>(seq_len));
  check_last_error();

  auto gate1 = (torch::matmul(seq0.view({batch_size * seq_len, kHiddenSize}), wih1.transpose(0, 1)) + b1)
                   .view({batch_size, seq_len, kGateCols})
                   .contiguous();
  auto seq1 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());
  GPU_LAUNCH_KERNEL(
      persistent_lstm_projected_full_kernel,
      static_cast<int>(batch_size),
      kThreads,
      0,
      0,
      reinterpret_cast<const gpu_half*>(gate1.data_ptr<at::Half>()),
      reinterpret_cast<const gpu_half*>(whh1.data_ptr<at::Half>()),
      reinterpret_cast<gpu_half*>(seq1.data_ptr<at::Half>()),
      static_cast<int>(batch_size),
      static_cast<int>(seq_len));
  check_last_error();

  auto gate2 = (torch::matmul(seq1.view({batch_size * seq_len, kHiddenSize}), wih2.transpose(0, 1)) + b2)
                   .view({batch_size, seq_len, kGateCols})
                   .contiguous();
  auto seq2 = torch::empty({batch_size, seq_len, kHiddenSize}, x_c.options());
  GPU_LAUNCH_KERNEL(
      persistent_lstm_projected_full_kernel,
      static_cast<int>(batch_size),
      kThreads,
      0,
      0,
      reinterpret_cast<const gpu_half*>(gate2.data_ptr<at::Half>()),
      reinterpret_cast<const gpu_half*>(whh2.data_ptr<at::Half>()),
      reinterpret_cast<gpu_half*>(seq2.data_ptr<at::Half>()),
      static_cast<int>(batch_size),
      static_cast<int>(seq_len));
  check_last_error();

  auto gate3 = (torch::matmul(seq2.view({batch_size * seq_len, kHiddenSize}), wih3.transpose(0, 1)) + b3)
                   .view({batch_size, seq_len, kGateCols})
                   .contiguous();
  auto last = torch::empty({batch_size, kHiddenSize}, x_c.options());
  GPU_LAUNCH_KERNEL(
      persistent_lstm_projected_last_kernel,
      static_cast<int>(batch_size),
      kThreads,
      0,
      0,
      reinterpret_cast<const gpu_half*>(gate3.data_ptr<at::Half>()),
      reinterpret_cast<const gpu_half*>(whh3.data_ptr<at::Half>()),
      reinterpret_cast<gpu_half*>(last.data_ptr<at::Half>()),
      static_cast<int>(batch_size),
      static_cast<int>(seq_len));
  check_last_error();

  return torch::matmul(last, linear_weight.transpose(0, 1)) + linear_bias;
}
