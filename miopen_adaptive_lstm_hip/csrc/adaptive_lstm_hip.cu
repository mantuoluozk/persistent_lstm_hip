#include "adaptive_lstm_hip.h"
#include "adaptive_lstm_pipeline.h"

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
#include <ATen/cuda/CUDAContext.h>
#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#elif __has_include(<hipblas.h>)
#include <hipblas.h>
#else
#error "MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS requires a hipBLAS header"
#endif
#endif

#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
#if defined(HIPBLAS_V2)
#define ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define ADAPTIVE_LSTM_HIPBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#else
#define ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F HIPBLAS_R_32F
#define ADAPTIVE_LSTM_HIPBLAS_COMPUTE_16F HIPBLAS_R_16F
#endif
using adaptive_lstm_hipblas_compute_type = decltype(ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F);
#if defined(HIPBLAS_GEMM_DEFAULT_TENSOR_OP)
#define ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO HIPBLAS_GEMM_DEFAULT_TENSOR_OP
#else
#define ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO HIPBLAS_GEMM_DEFAULT
#endif
#endif

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

__device__ inline float half_to_float(gpu_half v) {
  return __half2float(v);
}

__device__ inline gpu_half float_to_half(float v) {
  return __float2half(v);
}

__device__ inline float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <bool FastAct>
__device__ inline float lstm_sigmoid(float x) {
  if constexpr (FastAct) {
    return 1.0f / (1.0f + exp2f(-1.4426950408889634f * x));
  } else {
    return sigmoidf_fast(x);
  }
}

template <bool FastAct>
__device__ inline float lstm_tanh(float x) {
  if constexpr (FastAct) {
    const float y = 2.0f * lstm_sigmoid<true>(2.0f * x) - 1.0f;
    return fminf(1.0f, fmaxf(-1.0f, y));
  } else {
    return tanhf(x);
  }
}

__device__ inline float shfl_down_float(float v, int delta) {
#ifdef __HIP_PLATFORM_AMD__
  return __shfl_down(v, delta);
#else
  return __shfl_down_sync(0xffffffff, v, delta);
#endif
}

template <int Partitions>
__device__ inline float reduce_partition_sum(float v) {
  for (int offset = 1; offset < Partitions; offset <<= 1) {
    v += shfl_down_float(v, offset);
  }
  return v;
}

void check_last_error() {
  auto err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    throw std::runtime_error(GPU_GET_ERROR_STRING(err));
  }
}

void check_last_error_if(bool enabled) {
  if (enabled) {
    check_last_error();
  }
}

bool env_flag_enabled(const char* name, bool default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  const std::string value(raw);
  if (value == "0" || value == "false" || value == "False" || value == "off" || value == "OFF") {
    return false;
  }
  return true;
}

#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
adaptive_lstm_hipblas_compute_type recurrent_compute_type() {
  const char* raw = std::getenv("MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE");
  if (raw == nullptr) {
    return ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F;
  }
  const std::string value(raw);
  if (value == "fp16" || value == "FP16" || value == "16" || value == "half") {
    return ADAPTIVE_LSTM_HIPBLAS_COMPUTE_16F;
  }
  if (value == "" || value == "fp32" || value == "FP32" || value == "32" || value == "float") {
    return ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F;
  }
  if (value.rfind("fp16_layers:", 0) == 0 || value == "fp16_except_last" ||
      value == "fp16_first" || value == "auto_fast" || value == "fast" ||
      value == "auto_balanced" || value == "balanced" ||
      value == "auto_aggressive" || value == "aggressive") {
    return ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F;
  }
  throw std::invalid_argument(
      "MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE must be fp32, fp16, fp16_first, fp16_except_last, auto_fast, auto_balanced, auto_aggressive, or fp16_layers:<ids>");
}

hipblasHandle_t adaptive_lstm_hipblas_handle(const char* context) {
  static thread_local hipblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    const hipblasStatus_t create_status = hipblasCreate(&handle);
    if (create_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("hipblasCreate failed for ") + context);
    }
  }
  return handle;
}

void adaptive_lstm_set_hipblas_stream(hipblasHandle_t handle, const char* context) {
  const hipblasStatus_t stream_status =
      hipblasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());
  if (stream_status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("hipblasSetStream failed for ") + context);
  }
}
#endif

void h128_recurrent_gemm(
    torch::Tensor& recur,
    const torch::Tensor& h_state,
    const torch::Tensor& whh_t,
    int batch_size) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true)) {
    hipblasHandle_t handle = adaptive_lstm_hipblas_handle("adaptive H128 GEMM scan");
    adaptive_lstm_set_hipblas_stream(handle, "adaptive H128 GEMM scan");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t gemm_status = hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        512,
        batch_size,
        128,
        &alpha,
        static_cast<const void*>(whh_t.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        512,
        static_cast<const void*>(h_state.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        128,
        &beta,
        static_cast<void*>(recur.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        512,
        recurrent_compute_type(),
        ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO);
    if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasGemmEx failed for adaptive H128 GEMM scan");
    }
    return;
  }
#endif
  at::mm_out(recur, h_state, whh_t);
}

void h128_recurrent_gemm_accumulate_gate(
    torch::Tensor& gate,
    const torch::Tensor& h_state,
    const torch::Tensor& whh_t,
    int batch_size,
    int t) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true)) {
    hipblasHandle_t handle = adaptive_lstm_hipblas_handle("adaptive H128 seqmajor accum");
    adaptive_lstm_set_hipblas_stream(handle, "adaptive H128 seqmajor accum");

    const float alpha = 1.0f;
    const float beta = 1.0f;
    gpu_half* gate_t =
        reinterpret_cast<gpu_half*>(gate.data_ptr<at::Half>()) + static_cast<size_t>(t) * batch_size * 512;
    const hipblasStatus_t gemm_status = hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        512,
        batch_size,
        128,
        &alpha,
        static_cast<const void*>(whh_t.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        512,
        static_cast<const void*>(h_state.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        128,
        &beta,
        static_cast<void*>(gate_t),
        HIPBLAS_R_16F,
        512,
        recurrent_compute_type(),
        ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO);
    if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasGemmEx failed for adaptive H128 seqmajor accum");
    }
    return;
  }
#endif
  auto gate_t = gate.select(0, t);
  at::addmm_out(gate_t, gate_t, h_state, whh_t, 1.0, 1.0);
}

void batchmajor_recurrent_gemm_accumulate_gate(
    torch::Tensor& gate,
    const torch::Tensor& h_state,
    const torch::Tensor& whh_t,
    int batch_size,
    int hidden_size,
    int seq_len,
    int t) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true)) {
    hipblasHandle_t handle = adaptive_lstm_hipblas_handle("adaptive batch-major gate accum");
    adaptive_lstm_set_hipblas_stream(handle, "adaptive batch-major gate accum");

    const int gate_size = 4 * hidden_size;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    const hipblasStatus_t gemm_status = hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        gate_size,
        batch_size,
        hidden_size,
        &alpha,
        static_cast<const void*>(whh_t.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        gate_size,
        static_cast<const void*>(h_state.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        hidden_size,
        &beta,
        static_cast<void*>(gate.data_ptr<at::Half>() + t * gate_size),
        HIPBLAS_R_16F,
        seq_len * gate_size,
        recurrent_compute_type(),
        ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO);
    if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasGemmEx failed for adaptive batch-major gate accum");
    }
    return;
  }
#endif
  auto gate_t = gate.select(1, t);
  at::addmm_out(gate_t, gate_t, h_state, whh_t, 1.0, 1.0);
}

void generic_recurrent_gemm(
    torch::Tensor& recur,
    const torch::Tensor& h_state,
    const torch::Tensor& whh_t,
    int batch_size,
    int hidden_size) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true)) {
    hipblasHandle_t handle = adaptive_lstm_hipblas_handle("adaptive generic GEMM scan");
    adaptive_lstm_set_hipblas_stream(handle, "adaptive generic GEMM scan");

    const int gate_size = 4 * hidden_size;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t gemm_status = hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        gate_size,
        batch_size,
        hidden_size,
        &alpha,
        static_cast<const void*>(whh_t.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        gate_size,
        static_cast<const void*>(h_state.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        hidden_size,
        &beta,
        static_cast<void*>(recur.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        gate_size,
        recurrent_compute_type(),
        ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO);
    if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasGemmEx failed for adaptive generic GEMM scan");
    }
    return;
  }
#endif
  at::mm_out(recur, h_state, whh_t);
}

#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
void recurrent_gemm_with_handle(
    hipblasHandle_t handle,
    torch::Tensor& recur,
    const torch::Tensor& h_state,
    const torch::Tensor& whh_t,
    int batch_size,
    int hidden_size,
    adaptive_lstm_hipblas_compute_type compute_type,
    const char* context) {
  const int gate_size = 4 * hidden_size;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const hipblasStatus_t gemm_status = hipblasGemmEx(
      handle,
      HIPBLAS_OP_N,
      HIPBLAS_OP_N,
      gate_size,
      batch_size,
      hidden_size,
      &alpha,
      static_cast<const void*>(whh_t.data_ptr<at::Half>()),
      HIPBLAS_R_16F,
      gate_size,
      static_cast<const void*>(h_state.data_ptr<at::Half>()),
      HIPBLAS_R_16F,
      hidden_size,
      &beta,
      static_cast<void*>(recur.data_ptr<at::Half>()),
      HIPBLAS_R_16F,
      gate_size,
      compute_type,
      ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO);
  if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("hipblasGemmEx failed for ") + context);
  }
}
#endif

void input_projection_gemm(
    torch::Tensor& gate,
    const torch::Tensor& input_2d,
    const torch::Tensor& wih_t,
    int item_count,
    int input_size,
    int gate_size) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true)) {
    hipblasHandle_t handle = adaptive_lstm_hipblas_handle("adaptive input projection GEMM");
    adaptive_lstm_set_hipblas_stream(handle, "adaptive input projection GEMM");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto gemm_algo =
        (input_size % 8 == 0 && gate_size % 8 == 0) ? ADAPTIVE_LSTM_HIPBLAS_GEMM_ALGO
                                                     : HIPBLAS_GEMM_DEFAULT;
    const hipblasStatus_t gemm_status = hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        gate_size,
        item_count,
        input_size,
        &alpha,
        static_cast<const void*>(wih_t.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        gate_size,
        static_cast<const void*>(input_2d.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        input_size,
        &beta,
        static_cast<void*>(gate.data_ptr<at::Half>()),
        HIPBLAS_R_16F,
        gate_size,
        ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F,
        gemm_algo);
    if (gemm_status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasGemmEx failed for adaptive input projection GEMM");
    }
    return;
  }
#endif
  at::mm_out(gate, input_2d, wih_t);
}

template <int ReadBlock, int LocalSize>
__global__ void adaptive_lstm_hidden_update_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int hidden_size,
    bool write_sequence) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int h0 = tid * ReadBlock;
  const int hidden_stride = blockDim.x * ReadBlock;
  extern __shared__ unsigned char shared_raw[];
  gpu_half* h_prev = reinterpret_cast<gpu_half*>(shared_raw);
  gpu_half* h_next = h_prev + hidden_size;
  float* c_cur = reinterpret_cast<float*>(h_next + hidden_size);

  for (int base_h = h0; base_h < hidden_size; base_h += hidden_stride) {
    for (int r = 0; r < ReadBlock; ++r) {
      const int h = base_h + r;
      if (h < hidden_size) {
        h_prev[h] = float_to_half(0.0f);
        h_next[h] = float_to_half(0.0f);
        c_cur[h] = 0.0f;
      }
    }
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    for (int base_h = h0; base_h < hidden_size; base_h += hidden_stride) {
      for (int r = 0; r < ReadBlock; ++r) {
        const int h = base_h + r;
        if (h >= hidden_size) {
          continue;
        }

        const int gate_base = (b * seq_len + t) * (4 * hidden_size);
        float i_acc = half_to_float(gate_proj[gate_base + 0 * hidden_size + h]) +
                      half_to_float(bias[0 * hidden_size + h]);
        float f_acc = half_to_float(gate_proj[gate_base + 1 * hidden_size + h]) +
                      half_to_float(bias[1 * hidden_size + h]);
        float g_acc = half_to_float(gate_proj[gate_base + 2 * hidden_size + h]) +
                      half_to_float(bias[2 * hidden_size + h]);
        float o_acc = half_to_float(gate_proj[gate_base + 3 * hidden_size + h]) +
                      half_to_float(bias[3 * hidden_size + h]);

        for (int k = 0; k < hidden_size; ++k) {
          const float hv = half_to_float(h_prev[k]);
          i_acc += hv * half_to_float(weight_hh[(0 * hidden_size + h) * hidden_size + k]);
          f_acc += hv * half_to_float(weight_hh[(1 * hidden_size + h) * hidden_size + k]);
          g_acc += hv * half_to_float(weight_hh[(2 * hidden_size + h) * hidden_size + k]);
          o_acc += hv * half_to_float(weight_hh[(3 * hidden_size + h) * hidden_size + k]);
        }

        const float i_gate = sigmoidf_fast(i_acc);
        const float f_gate = sigmoidf_fast(f_acc);
        const float g_gate = tanhf(g_acc);
        const float o_gate = sigmoidf_fast(o_acc);
        const float c_next = f_gate * c_cur[h] + i_gate * g_gate;
        c_cur[h] = c_next;
        h_next[h] = float_to_half(o_gate * tanhf(c_next));
        if (write_sequence) {
          out[(b * seq_len + t) * hidden_size + h] = h_next[h];
        }
      }
    }
    __syncthreads();
    for (int base_h = h0; base_h < hidden_size; base_h += hidden_stride) {
      for (int r = 0; r < ReadBlock; ++r) {
        const int h = base_h + r;
        if (h < hidden_size) {
          h_prev[h] = h_next[h];
        }
      }
    }
    __syncthreads();
  }

  if (!write_sequence) {
    for (int base_h = h0; base_h < hidden_size; base_h += hidden_stride) {
      for (int r = 0; r < ReadBlock; ++r) {
        const int h = base_h + r;
        if (h < hidden_size) {
          out[b * hidden_size + h] = h_prev[h];
        }
      }
    }
  }
}

template <bool WriteSequence, int ReadBlock>
__global__ void __launch_bounds__(256) adaptive_lstm_h128_gemm_scan_pointwise_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ recur,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state,
    float* __restrict__ c_state,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int t) {
  constexpr int kH = 128;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = (batch_size * kH) / ReadBlock;
  if (item >= total_items) {
    return;
  }

  const int flat = item * ReadBlock;
  const int b = flat / kH;
  const int h_base = flat - b * kH;
  const int gate_base = (b * seq_len + t) * (4 * kH);
  const int recur_base = b * (4 * kH);
  const int state_base = b * kH;

#pragma unroll
  for (int r = 0; r < ReadBlock; ++r) {
    const int h = h_base + r;
    const int idx = state_base + h;
    const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) +
                        half_to_float(recur[recur_base + 0 * kH + h]) +
                        half_to_float(bias[0 * kH + h]);
    const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) +
                        half_to_float(recur[recur_base + 1 * kH + h]) +
                        half_to_float(bias[1 * kH + h]);
    const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) +
                        half_to_float(recur[recur_base + 2 * kH + h]) +
                        half_to_float(bias[2 * kH + h]);
    const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) +
                        half_to_float(recur[recur_base + 3 * kH + h]) +
                        half_to_float(bias[3 * kH + h]);

    const float i_gate = sigmoidf_fast(i_acc);
    const float f_gate = sigmoidf_fast(f_acc);
    const float g_gate = tanhf(g_acc);
    const float o_gate = sigmoidf_fast(o_acc);
    const float c_next = f_gate * c_state[idx] + i_gate * g_gate;
    c_state[idx] = c_next;
    const gpu_half h_next = float_to_half(o_gate * tanhf(c_next));
    h_state[idx] = h_next;
    if (WriteSequence) {
      out[(b * seq_len + t) * kH + h] = h_next;
    }
  }
}

template <bool WriteSequence, int ReadBlock>
__global__ void __launch_bounds__(256) adaptive_lstm_generic_gemm_scan_pointwise_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ recur,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state,
    float* __restrict__ c_state,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int hidden_size,
    int t) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = (batch_size * hidden_size) / ReadBlock;
  if (item >= total_items) {
    return;
  }

  const int flat = item * ReadBlock;
  const int b = flat / hidden_size;
  const int h_base = flat - b * hidden_size;
  const int gate_size = 4 * hidden_size;
  const int gate_base = (b * seq_len + t) * gate_size;
  const int recur_base = b * gate_size;
  const int state_base = b * hidden_size;

#pragma unroll
  for (int r = 0; r < ReadBlock; ++r) {
    const int h = h_base + r;
    const int idx = state_base + h;
    const float i_acc = half_to_float(gate_proj[gate_base + 0 * hidden_size + h]) +
                        half_to_float(recur[recur_base + 0 * hidden_size + h]) +
                        half_to_float(bias[0 * hidden_size + h]);
    const float f_acc = half_to_float(gate_proj[gate_base + 1 * hidden_size + h]) +
                        half_to_float(recur[recur_base + 1 * hidden_size + h]) +
                        half_to_float(bias[1 * hidden_size + h]);
    const float g_acc = half_to_float(gate_proj[gate_base + 2 * hidden_size + h]) +
                        half_to_float(recur[recur_base + 2 * hidden_size + h]) +
                        half_to_float(bias[2 * hidden_size + h]);
    const float o_acc = half_to_float(gate_proj[gate_base + 3 * hidden_size + h]) +
                        half_to_float(recur[recur_base + 3 * hidden_size + h]) +
                        half_to_float(bias[3 * hidden_size + h]);

    const float i_gate = sigmoidf_fast(i_acc);
    const float f_gate = sigmoidf_fast(f_acc);
    const float g_gate = tanhf(g_acc);
    const float o_gate = sigmoidf_fast(o_acc);
    const float c_next = f_gate * c_state[idx] + i_gate * g_gate;
    c_state[idx] = c_next;
    const gpu_half h_next = float_to_half(o_gate * tanhf(c_next));
    h_state[idx] = h_next;
    if (WriteSequence) {
      out[(b * seq_len + t) * hidden_size + h] = h_next;
    }
  }
}

template <int HiddenSize, bool WriteSequence, int ReadBlock, bool FastAct = false>
__global__ void __launch_bounds__(256) adaptive_lstm_fixed_gemm_scan_pointwise_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ recur,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state,
    float* __restrict__ c_state,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int t) {
  constexpr int kH = HiddenSize;
  constexpr int kGateSize = 4 * HiddenSize;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = (batch_size * kH) / ReadBlock;
  if (item >= total_items) {
    return;
  }

  const int flat = item * ReadBlock;
  const int b = flat / kH;
  const int h_base = flat - b * kH;
  const int gate_base = (b * seq_len + t) * kGateSize;
  const int recur_base = b * kGateSize;
  const int state_base = b * kH;

#pragma unroll
  for (int r = 0; r < ReadBlock; ++r) {
    const int h = h_base + r;
    const int idx = state_base + h;
    const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) +
                        half_to_float(recur[recur_base + 0 * kH + h]) +
                        half_to_float(bias[0 * kH + h]);
    const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) +
                        half_to_float(recur[recur_base + 1 * kH + h]) +
                        half_to_float(bias[1 * kH + h]);
    const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) +
                        half_to_float(recur[recur_base + 2 * kH + h]) +
                        half_to_float(bias[2 * kH + h]);
    const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) +
                        half_to_float(recur[recur_base + 3 * kH + h]) +
                        half_to_float(bias[3 * kH + h]);

    const float i_gate = lstm_sigmoid<FastAct>(i_acc);
    const float f_gate = lstm_sigmoid<FastAct>(f_acc);
    const float g_gate = lstm_tanh<FastAct>(g_acc);
    const float o_gate = lstm_sigmoid<FastAct>(o_acc);
    const float c_next = f_gate * c_state[idx] + i_gate * g_gate;
    c_state[idx] = c_next;
    const gpu_half h_next = float_to_half(o_gate * lstm_tanh<FastAct>(c_next));
    h_state[idx] = h_next;
    if (WriteSequence) {
      out[(b * seq_len + t) * kH + h] = h_next;
    }
  }
}

template <int HiddenSize, bool WriteSequence, int ReadBlock, bool FastAct = false>
__global__ void __launch_bounds__(256) adaptive_lstm_fixed_gate_accum_pointwise_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state,
    float* __restrict__ c_state,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int t) {
  constexpr int kH = HiddenSize;
  constexpr int kGateSize = 4 * HiddenSize;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = (batch_size * kH) / ReadBlock;
  if (item >= total_items) {
    return;
  }

  const int flat = item * ReadBlock;
  const int b = flat / kH;
  const int h_base = flat - b * kH;
  const int gate_base = (b * seq_len + t) * kGateSize;
  const int state_base = b * kH;

#pragma unroll
  for (int r = 0; r < ReadBlock; ++r) {
    const int h = h_base + r;
    const int idx = state_base + h;
    const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) +
                        half_to_float(bias[0 * kH + h]);
    const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) +
                        half_to_float(bias[1 * kH + h]);
    const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) +
                        half_to_float(bias[2 * kH + h]);
    const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) +
                        half_to_float(bias[3 * kH + h]);

    const float i_gate = lstm_sigmoid<FastAct>(i_acc);
    const float f_gate = lstm_sigmoid<FastAct>(f_acc);
    const float g_gate = lstm_tanh<FastAct>(g_acc);
    const float o_gate = lstm_sigmoid<FastAct>(o_acc);
    const float c_next = f_gate * c_state[idx] + i_gate * g_gate;
    c_state[idx] = c_next;
    const gpu_half h_next = float_to_half(o_gate * lstm_tanh<FastAct>(c_next));
    h_state[idx] = h_next;
    if (WriteSequence) {
      out[(b * seq_len + t) * kH + h] = h_next;
    }
  }
}

template <bool WriteSequence, int ReadBlock>
__global__ void __launch_bounds__(256) adaptive_lstm_h128_seqmajor_update_kernel(
    const gpu_half* __restrict__ gate_seq,
    gpu_half* __restrict__ h_state,
    float* __restrict__ c_state,
    gpu_half* __restrict__ seq_out,
    int batch_size,
    int seq_len,
    int t) {
  constexpr int kH = 128;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = (batch_size * kH) / ReadBlock;
  if (item >= total_items) {
    return;
  }

  const int flat = item * ReadBlock;
  const int b = flat / kH;
  const int h_base = flat - b * kH;
  const int gate_base = (t * batch_size + b) * (4 * kH);
  const int state_base = b * kH;

#pragma unroll
  for (int r = 0; r < ReadBlock; ++r) {
    const int h = h_base + r;
    const int idx = state_base + h;
    const float i_acc = half_to_float(gate_seq[gate_base + 0 * kH + h]);
    const float f_acc = half_to_float(gate_seq[gate_base + 1 * kH + h]);
    const float g_acc = half_to_float(gate_seq[gate_base + 2 * kH + h]);
    const float o_acc = half_to_float(gate_seq[gate_base + 3 * kH + h]);

    const float i_gate = sigmoidf_fast(i_acc);
    const float f_gate = sigmoidf_fast(f_acc);
    const float g_gate = tanhf(g_acc);
    const float o_gate = sigmoidf_fast(o_acc);
    const float c_next = f_gate * c_state[idx] + i_gate * g_gate;
    c_state[idx] = c_next;
    const gpu_half h_next = float_to_half(o_gate * tanhf(c_next));
    h_state[idx] = h_next;
    if (WriteSequence) {
      seq_out[(t * batch_size + b) * kH + h] = h_next;
    }
  }
}

template <int Partitions, int LocalSize>
__global__ void adaptive_lstm_hidden_update_partitioned_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len,
    int hidden_size,
    bool write_sequence) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int h0 = tid / Partitions;
  const int partition = tid - h0 * Partitions;
  const int hidden_stride = blockDim.x / Partitions;
  extern __shared__ unsigned char shared_raw[];
  gpu_half* h_prev = reinterpret_cast<gpu_half*>(shared_raw);
  gpu_half* h_next = h_prev + hidden_size;
  float* c_cur = reinterpret_cast<float*>(h_next + hidden_size);

  for (int h = h0; h < hidden_size; h += hidden_stride) {
    if (partition == 0) {
      h_prev[h] = float_to_half(0.0f);
      h_next[h] = float_to_half(0.0f);
      c_cur[h] = 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    for (int h = h0; h < hidden_size; h += hidden_stride) {
      float i_recur = 0.0f;
      float f_recur = 0.0f;
      float g_recur = 0.0f;
      float o_recur = 0.0f;

      for (int k = partition; k < hidden_size; k += Partitions) {
        const float hv = half_to_float(h_prev[k]);
        i_recur += hv * half_to_float(weight_hh[(0 * hidden_size + h) * hidden_size + k]);
        f_recur += hv * half_to_float(weight_hh[(1 * hidden_size + h) * hidden_size + k]);
        g_recur += hv * half_to_float(weight_hh[(2 * hidden_size + h) * hidden_size + k]);
        o_recur += hv * half_to_float(weight_hh[(3 * hidden_size + h) * hidden_size + k]);
      }

      i_recur = reduce_partition_sum<Partitions>(i_recur);
      f_recur = reduce_partition_sum<Partitions>(f_recur);
      g_recur = reduce_partition_sum<Partitions>(g_recur);
      o_recur = reduce_partition_sum<Partitions>(o_recur);

      if (partition == 0) {
        const int gate_base = (b * seq_len + t) * (4 * hidden_size);
        const float i_acc = half_to_float(gate_proj[gate_base + 0 * hidden_size + h]) +
                            half_to_float(bias[0 * hidden_size + h]) + i_recur;
        const float f_acc = half_to_float(gate_proj[gate_base + 1 * hidden_size + h]) +
                            half_to_float(bias[1 * hidden_size + h]) + f_recur;
        const float g_acc = half_to_float(gate_proj[gate_base + 2 * hidden_size + h]) +
                            half_to_float(bias[2 * hidden_size + h]) + g_recur;
        const float o_acc = half_to_float(gate_proj[gate_base + 3 * hidden_size + h]) +
                            half_to_float(bias[3 * hidden_size + h]) + o_recur;

        const float i_gate = sigmoidf_fast(i_acc);
        const float f_gate = sigmoidf_fast(f_acc);
        const float g_gate = tanhf(g_acc);
        const float o_gate = sigmoidf_fast(o_acc);
        const float c_next = f_gate * c_cur[h] + i_gate * g_gate;
        c_cur[h] = c_next;
        h_next[h] = float_to_half(o_gate * tanhf(c_next));
        if (write_sequence) {
          out[(b * seq_len + t) * hidden_size + h] = h_next[h];
        }
      }
    }
    __syncthreads();
    for (int h = h0; h < hidden_size; h += hidden_stride) {
      if (partition == 0) {
        h_prev[h] = h_next[h];
      }
    }
    __syncthreads();
  }

  if (!write_sequence) {
    for (int h = h0; h < hidden_size; h += hidden_stride) {
      if (partition == 0) {
        out[b * hidden_size + h] = h_prev[h];
      }
    }
  }
}

template <bool CheckTail, bool WriteSequence>
__global__ void __launch_bounds__(adaptive_lstm::H128CachedB2Traits::kBlockSize, 1)
adaptive_lstm_h128_cached_b2_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  constexpr int kH = adaptive_lstm::H128CachedB2Traits::kHiddenSize;
  constexpr int kPartitions = adaptive_lstm::H128CachedB2Traits::kPartitions;
  constexpr int kPerPartition = kH / kPartitions;
  constexpr int kBatchTile = adaptive_lstm::H128CachedB2Traits::kBatchTile;
  const int b0 = blockIdx.x * kBatchTile;
  const int b1 = b0 + 1;
  const bool has_b1 = !CheckTail || b1 < batch_size;
  const int tid = threadIdx.x;
  const int h = tid / kPartitions;
  const int partition = tid - h * kPartitions;

  __shared__ float h_prev[kBatchTile * kH];
  __shared__ float h_next[kBatchTile * kH];

  gpu_half i_w[kPerPartition];
  gpu_half f_w[kPerPartition];
  gpu_half g_w[kPerPartition];
  gpu_half o_w[kPerPartition];

  const gpu_half i_bias = bias[0 * kH + h];
  const gpu_half f_bias = bias[1 * kH + h];
  const gpu_half g_bias = bias[2 * kH + h];
  const gpu_half o_bias = bias[3 * kH + h];

#pragma unroll
  for (int idx = 0; idx < kPerPartition; ++idx) {
    const int k = partition + idx * kPartitions;
    i_w[idx] = weight_hh[(0 * kH + h) * kH + k];
    f_w[idx] = weight_hh[(1 * kH + h) * kH + k];
    g_w[idx] = weight_hh[(2 * kH + h) * kH + k];
    o_w[idx] = weight_hh[(3 * kH + h) * kH + k];
  }

  float c0_reg = 0.0f;
  float c1_reg = 0.0f;
  if (partition == 0) {
    h_prev[h] = 0.0f;
    h_next[h] = 0.0f;
    h_prev[kH + h] = 0.0f;
    h_next[kH + h] = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    float i0_recur = 0.0f;
    float f0_recur = 0.0f;
    float g0_recur = 0.0f;
    float o0_recur = 0.0f;
    float i1_recur = 0.0f;
    float f1_recur = 0.0f;
    float g1_recur = 0.0f;
    float o1_recur = 0.0f;

#pragma unroll
    for (int idx = 0; idx < kPerPartition; ++idx) {
      const int k = partition + idx * kPartitions;
      const float h0v = h_prev[k];
      const float iw = half_to_float(i_w[idx]);
      const float fw = half_to_float(f_w[idx]);
      const float gw = half_to_float(g_w[idx]);
      const float ow = half_to_float(o_w[idx]);
      i0_recur += h0v * iw;
      f0_recur += h0v * fw;
      g0_recur += h0v * gw;
      o0_recur += h0v * ow;
      if (has_b1) {
        const float h1v = h_prev[kH + k];
        i1_recur += h1v * iw;
        f1_recur += h1v * fw;
        g1_recur += h1v * gw;
        o1_recur += h1v * ow;
      }
    }

    i0_recur = reduce_partition_sum<kPartitions>(i0_recur);
    f0_recur = reduce_partition_sum<kPartitions>(f0_recur);
    g0_recur = reduce_partition_sum<kPartitions>(g0_recur);
    o0_recur = reduce_partition_sum<kPartitions>(o0_recur);
    i1_recur = reduce_partition_sum<kPartitions>(i1_recur);
    f1_recur = reduce_partition_sum<kPartitions>(f1_recur);
    g1_recur = reduce_partition_sum<kPartitions>(g1_recur);
    o1_recur = reduce_partition_sum<kPartitions>(o1_recur);

    if (partition == 0) {
      const int gate0_base = (b0 * seq_len + t) * (4 * kH);
      const float i0_acc = half_to_float(gate_proj[gate0_base + 0 * kH + h]) +
                           half_to_float(i_bias) + i0_recur;
      const float f0_acc = half_to_float(gate_proj[gate0_base + 1 * kH + h]) +
                           half_to_float(f_bias) + f0_recur;
      const float g0_acc = half_to_float(gate_proj[gate0_base + 2 * kH + h]) +
                           half_to_float(g_bias) + g0_recur;
      const float o0_acc = half_to_float(gate_proj[gate0_base + 3 * kH + h]) +
                           half_to_float(o_bias) + o0_recur;
      const float i0_gate = sigmoidf_fast(i0_acc);
      const float f0_gate = sigmoidf_fast(f0_acc);
      const float g0_gate = tanhf(g0_acc);
      const float o0_gate = sigmoidf_fast(o0_acc);
      c0_reg = f0_gate * c0_reg + i0_gate * g0_gate;
      h_next[h] = o0_gate * tanhf(c0_reg);
      if (WriteSequence) {
        out[(b0 * seq_len + t) * kH + h] = float_to_half(h_next[h]);
      }

      if (has_b1) {
        const int gate1_base = (b1 * seq_len + t) * (4 * kH);
        const float i1_acc = half_to_float(gate_proj[gate1_base + 0 * kH + h]) +
                             half_to_float(i_bias) + i1_recur;
        const float f1_acc = half_to_float(gate_proj[gate1_base + 1 * kH + h]) +
                             half_to_float(f_bias) + f1_recur;
        const float g1_acc = half_to_float(gate_proj[gate1_base + 2 * kH + h]) +
                             half_to_float(g_bias) + g1_recur;
        const float o1_acc = half_to_float(gate_proj[gate1_base + 3 * kH + h]) +
                             half_to_float(o_bias) + o1_recur;
        const float i1_gate = sigmoidf_fast(i1_acc);
        const float f1_gate = sigmoidf_fast(f1_acc);
        const float g1_gate = tanhf(g1_acc);
        const float o1_gate = sigmoidf_fast(o1_acc);
        c1_reg = f1_gate * c1_reg + i1_gate * g1_gate;
        h_next[kH + h] = o1_gate * tanhf(c1_reg);
        if (WriteSequence) {
          out[(b1 * seq_len + t) * kH + h] = float_to_half(h_next[kH + h]);
        }
      }
    }
    __syncthreads();
    if (partition == 0) {
      h_prev[h] = h_next[h];
      if (has_b1) {
        h_prev[kH + h] = h_next[kH + h];
      }
    }
    __syncthreads();
  }

  if (!WriteSequence && partition == 0) {
    out[b0 * kH + h] = float_to_half(h_prev[h]);
    if (has_b1) {
      out[b1 * kH + h] = float_to_half(h_prev[kH + h]);
    }
  }
}

template <bool CheckTail, bool WriteSequence>
__global__ void __launch_bounds__(adaptive_lstm::H128CachedB4Traits::kBlockSize, 1)
adaptive_lstm_h128_cached_b4_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  constexpr int kH = adaptive_lstm::H128CachedB4Traits::kHiddenSize;
  constexpr int kPartitions = adaptive_lstm::H128CachedB4Traits::kPartitions;
  constexpr int kPerPartition = kH / kPartitions;
  constexpr int kBatchTile = adaptive_lstm::H128CachedB4Traits::kBatchTile;
  const int b_base = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int h = tid / kPartitions;
  const int partition = tid - h * kPartitions;

  __shared__ float h_prev[kBatchTile * kH];
  __shared__ float h_next[kBatchTile * kH];

  gpu_half i_w[kPerPartition];
  gpu_half f_w[kPerPartition];
  gpu_half g_w[kPerPartition];
  gpu_half o_w[kPerPartition];

  const gpu_half i_bias = bias[0 * kH + h];
  const gpu_half f_bias = bias[1 * kH + h];
  const gpu_half g_bias = bias[2 * kH + h];
  const gpu_half o_bias = bias[3 * kH + h];

#pragma unroll
  for (int idx = 0; idx < kPerPartition; ++idx) {
    const int k = partition + idx * kPartitions;
    i_w[idx] = weight_hh[(0 * kH + h) * kH + k];
    f_w[idx] = weight_hh[(1 * kH + h) * kH + k];
    g_w[idx] = weight_hh[(2 * kH + h) * kH + k];
    o_w[idx] = weight_hh[(3 * kH + h) * kH + k];
  }

  float c_reg[kBatchTile];
#pragma unroll
  for (int bt = 0; bt < kBatchTile; ++bt) {
    c_reg[bt] = 0.0f;
    if (partition == 0) {
      h_prev[bt * kH + h] = 0.0f;
      h_next[bt * kH + h] = 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    float i_recur[kBatchTile];
    float f_recur[kBatchTile];
    float g_recur[kBatchTile];
    float o_recur[kBatchTile];
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = 0.0f;
      f_recur[bt] = 0.0f;
      g_recur[bt] = 0.0f;
      o_recur[bt] = 0.0f;
    }

#pragma unroll
    for (int idx = 0; idx < kPerPartition; ++idx) {
      const int k = partition + idx * kPartitions;
      const float iw = half_to_float(i_w[idx]);
      const float fw = half_to_float(f_w[idx]);
      const float gw = half_to_float(g_w[idx]);
      const float ow = half_to_float(o_w[idx]);
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const float hv = h_prev[bt * kH + k];
          i_recur[bt] += hv * iw;
          f_recur[bt] += hv * fw;
          g_recur[bt] += hv * gw;
          o_recur[bt] += hv * ow;
        }
      }
    }

#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = reduce_partition_sum<kPartitions>(i_recur[bt]);
      f_recur[bt] = reduce_partition_sum<kPartitions>(f_recur[bt]);
      g_recur[bt] = reduce_partition_sum<kPartitions>(g_recur[bt]);
      o_recur[bt] = reduce_partition_sum<kPartitions>(o_recur[bt]);
    }

    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const int gate_base = (b * seq_len + t) * (4 * kH);
          const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) +
                              half_to_float(i_bias) + i_recur[bt];
          const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) +
                              half_to_float(f_bias) + f_recur[bt];
          const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) +
                              half_to_float(g_bias) + g_recur[bt];
          const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) +
                              half_to_float(o_bias) + o_recur[bt];
          const float i_gate = sigmoidf_fast(i_acc);
          const float f_gate = sigmoidf_fast(f_acc);
          const float g_gate = tanhf(g_acc);
          const float o_gate = sigmoidf_fast(o_acc);
          c_reg[bt] = f_gate * c_reg[bt] + i_gate * g_gate;
          h_next[bt * kH + h] = o_gate * tanhf(c_reg[bt]);
          if (WriteSequence) {
            out[(b * seq_len + t) * kH + h] = float_to_half(h_next[bt * kH + h]);
          }
        }
      }
    }
    __syncthreads();
    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          h_prev[bt * kH + h] = h_next[bt * kH + h];
        }
      }
    }
    __syncthreads();
  }

  if (!WriteSequence && partition == 0) {
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      const int b = b_base + bt;
      if (!CheckTail || b < batch_size) {
        out[b * kH + h] = float_to_half(h_prev[bt * kH + h]);
      }
    }
  }
}

template <typename Traits, bool CheckTail, bool WriteSequence>
__global__ void __launch_bounds__(Traits::kBlockSize, 1)
adaptive_lstm_h128_cached_bt_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {
  constexpr int kH = Traits::kHiddenSize;
  constexpr int kPartitions = Traits::kPartitions;
  constexpr int kPerPartition = kH / kPartitions;
  constexpr int kBatchTile = Traits::kBatchTile;
  const int b_base = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int h = tid / kPartitions;
  const int partition = tid - h * kPartitions;

  __shared__ float h_prev[kBatchTile * kH];
  __shared__ float h_next[kBatchTile * kH];

  gpu_half i_w[kPerPartition];
  gpu_half f_w[kPerPartition];
  gpu_half g_w[kPerPartition];
  gpu_half o_w[kPerPartition];

  const gpu_half i_bias = bias[0 * kH + h];
  const gpu_half f_bias = bias[1 * kH + h];
  const gpu_half g_bias = bias[2 * kH + h];
  const gpu_half o_bias = bias[3 * kH + h];

#pragma unroll
  for (int idx = 0; idx < kPerPartition; ++idx) {
    const int k = partition + idx * kPartitions;
    i_w[idx] = weight_hh[(0 * kH + h) * kH + k];
    f_w[idx] = weight_hh[(1 * kH + h) * kH + k];
    g_w[idx] = weight_hh[(2 * kH + h) * kH + k];
    o_w[idx] = weight_hh[(3 * kH + h) * kH + k];
  }

  float c_reg[kBatchTile];
#pragma unroll
  for (int bt = 0; bt < kBatchTile; ++bt) {
    c_reg[bt] = 0.0f;
    if (partition == 0) {
      h_prev[bt * kH + h] = 0.0f;
      h_next[bt * kH + h] = 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    float i_recur[kBatchTile];
    float f_recur[kBatchTile];
    float g_recur[kBatchTile];
    float o_recur[kBatchTile];
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = 0.0f;
      f_recur[bt] = 0.0f;
      g_recur[bt] = 0.0f;
      o_recur[bt] = 0.0f;
    }

#pragma unroll
    for (int idx = 0; idx < kPerPartition; ++idx) {
      const int k = partition + idx * kPartitions;
      const float iw = half_to_float(i_w[idx]);
      const float fw = half_to_float(f_w[idx]);
      const float gw = half_to_float(g_w[idx]);
      const float ow = half_to_float(o_w[idx]);
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const float hv = h_prev[bt * kH + k];
          i_recur[bt] += hv * iw;
          f_recur[bt] += hv * fw;
          g_recur[bt] += hv * gw;
          o_recur[bt] += hv * ow;
        }
      }
    }

#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = reduce_partition_sum<kPartitions>(i_recur[bt]);
      f_recur[bt] = reduce_partition_sum<kPartitions>(f_recur[bt]);
      g_recur[bt] = reduce_partition_sum<kPartitions>(g_recur[bt]);
      o_recur[bt] = reduce_partition_sum<kPartitions>(o_recur[bt]);
    }

    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const int gate_base = (b * seq_len + t) * (4 * kH);
          const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) +
                              half_to_float(i_bias) + i_recur[bt];
          const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) +
                              half_to_float(f_bias) + f_recur[bt];
          const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) +
                              half_to_float(g_bias) + g_recur[bt];
          const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) +
                              half_to_float(o_bias) + o_recur[bt];
          const float i_gate = sigmoidf_fast(i_acc);
          const float f_gate = sigmoidf_fast(f_acc);
          const float g_gate = tanhf(g_acc);
          const float o_gate = sigmoidf_fast(o_acc);
          c_reg[bt] = f_gate * c_reg[bt] + i_gate * g_gate;
          h_next[bt * kH + h] = o_gate * tanhf(c_reg[bt]);
          if (WriteSequence) {
            out[(b * seq_len + t) * kH + h] = float_to_half(h_next[bt * kH + h]);
          }
        }
      }
    }
    __syncthreads();
    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          h_prev[bt * kH + h] = h_next[bt * kH + h];
        }
      }
    }
    __syncthreads();
  }

  if (!WriteSequence && partition == 0) {
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      const int b = b_base + bt;
      if (!CheckTail || b < batch_size) {
        out[b * kH + h] = float_to_half(h_prev[bt * kH + h]);
      }
    }
  }
}

template <int ReadBlock>
void launch_for_local_size(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor& out,
    int batch_size,
    int seq_len,
    int hidden_size,
    bool write_sequence,
    int local_size) {
  const int shared_bytes =
      hidden_size * (2 * static_cast<int>(sizeof(gpu_half)) + static_cast<int>(sizeof(float)));
  if (local_size <= 64) {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_kernel<ReadBlock, 64>),
        batch_size,
        64,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  } else if (local_size <= 128) {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_kernel<ReadBlock, 128>),
        batch_size,
        128,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  } else {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_kernel<ReadBlock, 256>),
        batch_size,
        256,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  }
}

template <int Partitions>
void launch_partitioned_for_local_size(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor& out,
    int batch_size,
    int seq_len,
    int hidden_size,
    bool write_sequence,
    int local_size) {
  const int shared_bytes =
      hidden_size * (2 * static_cast<int>(sizeof(gpu_half)) + static_cast<int>(sizeof(float)));
  if (local_size <= 256) {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_partitioned_kernel<Partitions, 256>),
        batch_size,
        256,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  } else if (local_size <= 512) {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_partitioned_kernel<Partitions, 512>),
        batch_size,
        512,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  } else {
    GPU_LAUNCH_KERNEL(
        (adaptive_lstm_hidden_update_partitioned_kernel<Partitions, 1024>),
        batch_size,
        1024,
        shared_bytes,
        0,
        reinterpret_cast<const gpu_half*>(gate_proj.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(weight_hh.data_ptr<at::Half>()),
        reinterpret_cast<const gpu_half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        hidden_size,
        write_sequence);
  }
}

}  // namespace

torch::Tensor adaptive_lstm_hidden_update_partitioned_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t partitions) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive partitioned hidden update expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive partitioned hidden update currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,4H], weight_hh [4H,H], bias [4H]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const int hidden_size = static_cast<int>(gate.size(2) / 4);
  if (gate.size(2) != 4 * hidden_size ||
      whh.size(0) != 4 * hidden_size ||
      whh.size(1) != hidden_size ||
      b.size(0) != 4 * hidden_size) {
    throw std::invalid_argument("incompatible adaptive partitioned hidden update shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, hidden_size}, gate.options())
      : torch::empty({batch_size, hidden_size}, gate.options());

  const int parts = static_cast<int>(partitions);
  int local_size = 1;
  while (local_size < hidden_size * parts) {
    local_size <<= 1;
  }
  local_size = std::min(local_size, 1024);

  if (parts == 8) {
    launch_partitioned_for_local_size<8>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  } else if (parts == 2) {
    launch_partitioned_for_local_size<2>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  } else {
    launch_partitioned_for_local_size<4>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  }
  check_last_error();
  return out;
}

torch::Tensor adaptive_lstm_hidden_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t read_block,
    int64_t items_per_group) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive hidden update expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive hidden update currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,4H], weight_hh [4H,H], bias [4H]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const int hidden_size = static_cast<int>(gate.size(2) / 4);
  if (gate.size(2) != 4 * hidden_size ||
      whh.size(0) != 4 * hidden_size ||
      whh.size(1) != hidden_size ||
      b.size(0) != 4 * hidden_size) {
    throw std::invalid_argument("incompatible adaptive hidden update shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, hidden_size}, gate.options())
      : torch::empty({batch_size, hidden_size}, gate.options());

  const int rb = static_cast<int>(read_block);
  const int local_size = static_cast<int>(items_per_group);
  if (rb == 4) {
    launch_for_local_size<4>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  } else if (rb == 2) {
    launch_for_local_size<2>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  } else {
    launch_for_local_size<1>(gate, whh, b, out, batch_size, seq_len, hidden_size, write_sequence, local_size);
  }
  check_last_error();
  return out;
}

torch::Tensor adaptive_lstm_h128_cached_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive H128 cached update expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive H128 cached update currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,512], weight_hh [512,128] or [128,512], bias [512]");
  }
  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  if (gate.size(2) != 512 || whh.size(0) != 512 || whh.size(1) != 128 || b.size(0) != 512) {
    throw std::invalid_argument("adaptive H128 cached update received incompatible shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, 128}, gate.options())
      : torch::empty({batch_size, 128}, gate.options());

  const bool check_tail = (batch_size & 1) != 0;
  const int grid = (batch_size + 1) / 2;
  if (write_sequence) {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b2_kernel<true, true>),
          grid,
          512,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b2_kernel<false, true>),
          grid,
          512,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  } else {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b2_kernel<true, false>),
          grid,
          512,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b2_kernel<false, false>),
          grid,
          512,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  }
  check_last_error();
  return out;
}

torch::Tensor adaptive_lstm_h128_cached_b4_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive H128 cached B4 update expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive H128 cached B4 update currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,512], weight_hh [512,128], bias [512]");
  }
  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  if (gate.size(2) != 512 || whh.size(0) != 512 || whh.size(1) != 128 || b.size(0) != 512) {
    throw std::invalid_argument("adaptive H128 cached B4 update received incompatible shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, 128}, gate.options())
      : torch::empty({batch_size, 128}, gate.options());

  constexpr int kBatchTile = adaptive_lstm::H128CachedB4Traits::kBatchTile;
  const bool check_tail = (batch_size % kBatchTile) != 0;
  const int grid = (batch_size + kBatchTile - 1) / kBatchTile;
  if (write_sequence) {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b4_kernel<true, true>),
          grid,
          adaptive_lstm::H128CachedB4Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b4_kernel<false, true>),
          grid,
          adaptive_lstm::H128CachedB4Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  } else {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b4_kernel<true, false>),
          grid,
          adaptive_lstm::H128CachedB4Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_b4_kernel<false, false>),
          grid,
          adaptive_lstm::H128CachedB4Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  }
  check_last_error();
  return out;
}

torch::Tensor adaptive_lstm_h128_cached_b8_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive H128 cached B8 update expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive H128 cached B8 update currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,512], weight_hh [512,128], bias [512]");
  }
  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  if (gate.size(2) != 512 || whh.size(0) != 512 || whh.size(1) != 128 || b.size(0) != 512) {
    throw std::invalid_argument("adaptive H128 cached B8 update received incompatible shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, 128}, gate.options())
      : torch::empty({batch_size, 128}, gate.options());

  constexpr int kBatchTile = adaptive_lstm::H128CachedB8Traits::kBatchTile;
  const bool check_tail = (batch_size % kBatchTile) != 0;
  const int grid = (batch_size + kBatchTile - 1) / kBatchTile;
  if (write_sequence) {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_bt_kernel<adaptive_lstm::H128CachedB8Traits, true, true>),
          grid,
          adaptive_lstm::H128CachedB8Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_bt_kernel<adaptive_lstm::H128CachedB8Traits, false, true>),
          grid,
          adaptive_lstm::H128CachedB8Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  } else {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_bt_kernel<adaptive_lstm::H128CachedB8Traits, true, false>),
          grid,
          adaptive_lstm::H128CachedB8Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_cached_bt_kernel<adaptive_lstm::H128CachedB8Traits, false, false>),
          grid,
          adaptive_lstm::H128CachedB8Traits::kBlockSize,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len);
    }
  }
  check_last_error();
  return out;
}

torch::Tensor adaptive_lstm_h128_gemm_scan_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t read_block) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda())) {
    throw std::invalid_argument("adaptive H128 GEMM scan expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive H128 GEMM scan currently expects FP16 tensors");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1)) {
    throw std::invalid_argument("expected gate [B,T,512], weight_hh [512,128], bias [512]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const bool whh_is_pretransposed = whh.size(0) == 128 && whh.size(1) == 512;
  const bool whh_is_native = whh.size(0) == 512 && whh.size(1) == 128;
  if (gate.size(2) != 512 || !(whh_is_pretransposed || whh_is_native) || b.size(0) != 512) {
    throw std::invalid_argument("adaptive H128 GEMM scan received incompatible shapes");
  }

  auto out = write_sequence
      ? torch::empty({batch_size, seq_len, 128}, gate.options())
      : torch::empty({batch_size, 128}, gate.options());
  auto h_state = torch::zeros({batch_size, 128}, gate.options());
  auto c_state = torch::zeros({batch_size, 128}, gate.options().dtype(torch::kFloat32));
  auto recur = torch::empty({batch_size, 512}, gate.options());
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4)) {
    throw std::invalid_argument("adaptive H128 GEMM scan read_block must be 1, 2, or 4");
  }
  const int pointwise_items = (batch_size * 128) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  const bool use_direct_blas = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true);
  hipblasHandle_t recurrent_handle = nullptr;
  if (use_direct_blas) {
    recurrent_handle = adaptive_lstm_hipblas_handle("adaptive H128 GEMM scan");
    adaptive_lstm_set_hipblas_stream(recurrent_handle, "adaptive H128 GEMM scan");
  }
#endif

  for (int t = 0; t < seq_len; ++t) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
    if (use_direct_blas) {
      recurrent_gemm_with_handle(
          recurrent_handle, recur, h_state, whh_t, batch_size, 128, recurrent_compute_type(), "adaptive H128 GEMM scan");
    } else
#endif
    {
    h128_recurrent_gemm(recur, h_state, whh_t, batch_size);
    }
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

torch::Tensor adaptive_lstm_h128_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda() &&
        h_state.is_cuda() && c_state.is_cuda() && recur.is_cuda() && out.is_cuda())) {
    throw std::invalid_argument("adaptive H128 GEMM scan workspace expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16 &&
        h_state.scalar_type() == torch::kFloat16 &&
        recur.scalar_type() == torch::kFloat16 &&
        out.scalar_type() == torch::kFloat16 &&
        c_state.scalar_type() == torch::kFloat32)) {
    throw std::invalid_argument("adaptive H128 GEMM scan workspace expects FP16 data and FP32 cell state");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1 &&
        h_state.dim() == 2 && c_state.dim() == 2 && recur.dim() == 2)) {
    throw std::invalid_argument(
        "expected gate [B,T,512], weight_hh [128,512] or [512,128], state [B,128], recur [B,512]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const bool whh_is_pretransposed = whh.size(0) == 128 && whh.size(1) == 512;
  const bool whh_is_native = whh.size(0) == 512 && whh.size(1) == 128;
  const bool out_shape_ok = write_sequence
      ? (out.dim() == 3 && out.size(0) == batch_size && out.size(1) == seq_len && out.size(2) == 128)
      : (out.dim() == 2 && out.size(0) == batch_size && out.size(1) == 128);
  if (gate.size(2) != 512 || !(whh_is_pretransposed || whh_is_native) || b.size(0) != 512 ||
      h_state.size(0) != batch_size || h_state.size(1) != 128 ||
      c_state.size(0) != batch_size || c_state.size(1) != 128 ||
      recur.size(0) != batch_size || recur.size(1) != 512 || !out_shape_ok) {
    throw std::invalid_argument("adaptive H128 GEMM scan workspace received incompatible shapes");
  }
  if (!(h_state.is_contiguous() && c_state.is_contiguous() && recur.is_contiguous() &&
        out.is_contiguous())) {
    throw std::invalid_argument("adaptive H128 GEMM scan workspace tensors must be contiguous");
  }

  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4)) {
    throw std::invalid_argument("adaptive H128 GEMM scan workspace read_block must be 1, 2, or 4");
  }

  h_state.zero_();
  c_state.zero_();
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int pointwise_items = (batch_size * 128) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  const bool use_direct_blas = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true);
  hipblasHandle_t recurrent_handle = nullptr;
  if (use_direct_blas) {
    recurrent_handle = adaptive_lstm_hipblas_handle("adaptive H128 GEMM scan workspace");
    adaptive_lstm_set_hipblas_stream(recurrent_handle, "adaptive H128 GEMM scan workspace");
  }
#endif

  for (int t = 0; t < seq_len; ++t) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
    if (use_direct_blas) {
      recurrent_gemm_with_handle(
          recurrent_handle,
          recur,
          h_state,
          whh_t,
          batch_size,
          128,
          recurrent_compute_type(),
          "adaptive H128 GEMM scan workspace");
    } else
#endif
    {
    h128_recurrent_gemm(recur, h_state, whh_t, batch_size);
    }
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<true, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_gemm_scan_pointwise_kernel<false, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

torch::Tensor adaptive_lstm_h128_seqmajor_accum_update_forward(
    const torch::Tensor& gate_seq,
    const torch::Tensor& weight_hh,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor seq_out,
    bool write_sequence,
    int64_t read_block) {
  c10::InferenceMode inference_mode;
  if (!(gate_seq.is_cuda() && weight_hh.is_cuda() && h_state.is_cuda() &&
        c_state.is_cuda() && seq_out.is_cuda())) {
    throw std::invalid_argument("adaptive H128 seqmajor accum expects CUDA/HIP tensors");
  }
  if (!(gate_seq.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        h_state.scalar_type() == torch::kFloat16 &&
        seq_out.scalar_type() == torch::kFloat16 &&
        c_state.scalar_type() == torch::kFloat32)) {
    throw std::invalid_argument("adaptive H128 seqmajor accum expects FP16 data and FP32 cell state");
  }
  if (!(gate_seq.dim() == 3 && weight_hh.dim() == 2 && h_state.dim() == 2 &&
        c_state.dim() == 2)) {
    throw std::invalid_argument("expected gate [T,B,512], weight_hh [128,512] or [512,128], state [B,128]");
  }

  auto gate = gate_seq.contiguous();
  auto whh = weight_hh.contiguous();
  const int seq_len = static_cast<int>(gate.size(0));
  const int batch_size = static_cast<int>(gate.size(1));
  const bool whh_is_pretransposed = whh.size(0) == 128 && whh.size(1) == 512;
  const bool whh_is_native = whh.size(0) == 512 && whh.size(1) == 128;
  const bool out_shape_ok = write_sequence
      ? (seq_out.dim() == 3 && seq_out.size(0) == seq_len && seq_out.size(1) == batch_size &&
         seq_out.size(2) == 128)
      : (seq_out.dim() == 2 && seq_out.size(0) == batch_size && seq_out.size(1) == 128);
  if (gate.size(2) != 512 || !(whh_is_pretransposed || whh_is_native) ||
      h_state.size(0) != batch_size || h_state.size(1) != 128 ||
      c_state.size(0) != batch_size || c_state.size(1) != 128 || !out_shape_ok) {
    throw std::invalid_argument("adaptive H128 seqmajor accum received incompatible shapes");
  }
  if (!(h_state.is_contiguous() && c_state.is_contiguous() && seq_out.is_contiguous())) {
    throw std::invalid_argument("adaptive H128 seqmajor accum workspace tensors must be contiguous");
  }

  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4)) {
    throw std::invalid_argument("adaptive H128 seqmajor accum read_block must be 1, 2, or 4");
  }

  h_state.zero_();
  c_state.zero_();
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int pointwise_items = (batch_size * 128) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);

  for (int t = 0; t < seq_len; ++t) {
    h128_recurrent_gemm_accumulate_gate(gate, h_state, whh_t, batch_size, t);
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<true, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<true, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<true, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<false, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<false, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_h128_seqmajor_update_kernel<false, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(seq_out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return seq_out;
}

torch::Tensor adaptive_lstm_input_gemm_forward_workspace(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight_ih_t,
    torch::Tensor gate) {
  c10::InferenceMode inference_mode;
  if (!(input_2d.is_cuda() && weight_ih_t.is_cuda() && gate.is_cuda())) {
    throw std::invalid_argument("adaptive input projection GEMM expects CUDA/HIP tensors");
  }
  if (!(input_2d.scalar_type() == torch::kFloat16 &&
        weight_ih_t.scalar_type() == torch::kFloat16 &&
        gate.scalar_type() == torch::kFloat16)) {
    throw std::invalid_argument("adaptive input projection GEMM expects FP16 tensors");
  }
  if (!(input_2d.dim() == 2 && weight_ih_t.dim() == 2 && gate.dim() == 2)) {
    throw std::invalid_argument("expected input [N,K], weight_ih_t [K,4H], gate [N,4H]");
  }
  auto input = input_2d.contiguous();
  auto weight = weight_ih_t.contiguous();
  if (input.size(1) != weight.size(0) || gate.size(0) != input.size(0) ||
      gate.size(1) != weight.size(1) || !gate.is_contiguous()) {
    throw std::invalid_argument("adaptive input projection GEMM received incompatible shapes");
  }
  input_projection_gemm(
      gate,
      input,
      weight,
      static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)),
      static_cast<int>(gate.size(1)));
  return gate;
}

template <int HiddenSize, bool FastAct>
torch::Tensor adaptive_lstm_fixed_gemm_scan_update_forward_workspace_impl(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t recurrent_compute) {
  c10::InferenceMode inference_mode;
  constexpr int kH = HiddenSize;
  constexpr int kGateSize = 4 * HiddenSize;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda() &&
        h_state.is_cuda() && c_state.is_cuda() && recur.is_cuda() && out.is_cuda())) {
    throw std::invalid_argument("adaptive fixed GEMM scan workspace expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16 &&
        h_state.scalar_type() == torch::kFloat16 &&
        recur.scalar_type() == torch::kFloat16 &&
        out.scalar_type() == torch::kFloat16 &&
        c_state.scalar_type() == torch::kFloat32)) {
    throw std::invalid_argument("adaptive fixed GEMM scan workspace expects FP16 data and FP32 cell state");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1 &&
        h_state.dim() == 2 && c_state.dim() == 2 && recur.dim() == 2)) {
    throw std::invalid_argument(
        "expected gate [B,T,4H], weight_hh [H,4H] or [4H,H], state [B,H], recur [B,4H]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const bool whh_is_pretransposed = whh.size(0) == kH && whh.size(1) == kGateSize;
  const bool whh_is_native = whh.size(0) == kGateSize && whh.size(1) == kH;
  const bool out_shape_ok = write_sequence
      ? (out.dim() == 3 && out.size(0) == batch_size && out.size(1) == seq_len && out.size(2) == kH)
      : (out.dim() == 2 && out.size(0) == batch_size && out.size(1) == kH);
  if (gate.size(2) != kGateSize || !(whh_is_pretransposed || whh_is_native) ||
      b.size(0) != kGateSize || h_state.size(0) != batch_size || h_state.size(1) != kH ||
      c_state.size(0) != batch_size || c_state.size(1) != kH ||
      recur.size(0) != batch_size || recur.size(1) != kGateSize || !out_shape_ok) {
    throw std::invalid_argument("adaptive fixed GEMM scan workspace received incompatible shapes");
  }
  if (!(h_state.is_contiguous() && c_state.is_contiguous() && recur.is_contiguous() &&
        out.is_contiguous())) {
    throw std::invalid_argument("adaptive fixed GEMM scan workspace tensors must be contiguous");
  }

  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4) || kH % rb != 0) {
    throw std::invalid_argument("adaptive fixed GEMM scan read_block must be 1, 2, or 4 and divide hidden_size");
  }

  c_state.zero_();
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int pointwise_items = (batch_size * kH) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
  const bool use_direct_blas = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS", true);
  hipblasHandle_t recurrent_handle = nullptr;
  const auto compute_type =
      recurrent_compute == 1 ? ADAPTIVE_LSTM_HIPBLAS_COMPUTE_16F : ADAPTIVE_LSTM_HIPBLAS_COMPUTE_32F;
  if (use_direct_blas) {
    recurrent_handle = adaptive_lstm_hipblas_handle("adaptive fixed GEMM scan");
    adaptive_lstm_set_hipblas_stream(recurrent_handle, "adaptive fixed GEMM scan");
  }
#endif

  if (write_sequence) {
    if (rb == 4) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 4, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    } else if (rb == 2) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 2, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 1, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    }
  } else {
    if (rb == 4) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 4, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    } else if (rb == 2) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 2, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 1, FastAct>),
          pointwise_blocks,
          256,
          0,
          0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size,
          seq_len,
          0);
    }
  }
  check_last_error_if(check_kernel_errors);

  for (int t = 1; t < seq_len; ++t) {
#if defined(__HIP_PLATFORM_AMD__) && defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS)
    if (use_direct_blas) {
      recurrent_gemm_with_handle(
          recurrent_handle, recur, h_state, whh_t, batch_size, kH, compute_type, "adaptive fixed GEMM scan");
    } else
#endif
    {
    generic_recurrent_gemm(recur, h_state, whh_t, batch_size, kH);
    }
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, true, 4, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, true, 2, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, true, 1, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, false, 4, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, false, 2, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gemm_scan_pointwise_kernel<kH, false, 1, FastAct>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

torch::Tensor adaptive_lstm_h256_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t recurrent_compute) {
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_FAST_ACT", false)) {
    return adaptive_lstm_fixed_gemm_scan_update_forward_workspace_impl<256, true>(
        gate_proj, weight_hh, bias, h_state, c_state, recur, out, write_sequence, read_block, recurrent_compute);
  }
  return adaptive_lstm_fixed_gemm_scan_update_forward_workspace_impl<256, false>(
      gate_proj, weight_hh, bias, h_state, c_state, recur, out, write_sequence, read_block, recurrent_compute);
}

torch::Tensor adaptive_lstm_h512_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t recurrent_compute) {
  if (env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_FAST_ACT", false)) {
    return adaptive_lstm_fixed_gemm_scan_update_forward_workspace_impl<512, true>(
        gate_proj, weight_hh, bias, h_state, c_state, recur, out, write_sequence, read_block, recurrent_compute);
  }
  return adaptive_lstm_fixed_gemm_scan_update_forward_workspace_impl<512, false>(
      gate_proj, weight_hh, bias, h_state, c_state, recur, out, write_sequence, read_block, recurrent_compute);
}

template <int HiddenSize>
torch::Tensor adaptive_lstm_fixed_gate_accum_update_forward_workspace_impl(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block) {
  c10::InferenceMode inference_mode;
  constexpr int kH = HiddenSize;
  constexpr int kGateSize = 4 * HiddenSize;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda() &&
        h_state.is_cuda() && c_state.is_cuda() && out.is_cuda())) {
    throw std::invalid_argument("adaptive fixed gate accum workspace expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16 &&
        h_state.scalar_type() == torch::kFloat16 &&
        out.scalar_type() == torch::kFloat16 &&
        c_state.scalar_type() == torch::kFloat32)) {
    throw std::invalid_argument("adaptive fixed gate accum workspace expects FP16 data and FP32 cell state");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1 &&
        h_state.dim() == 2 && c_state.dim() == 2)) {
    throw std::invalid_argument("expected gate [B,T,4H], weight_hh [H,4H] or [4H,H], state [B,H]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const bool whh_is_pretransposed = whh.size(0) == kH && whh.size(1) == kGateSize;
  const bool whh_is_native = whh.size(0) == kGateSize && whh.size(1) == kH;
  const bool out_shape_ok = write_sequence
      ? (out.dim() == 3 && out.size(0) == batch_size && out.size(1) == seq_len && out.size(2) == kH)
      : (out.dim() == 2 && out.size(0) == batch_size && out.size(1) == kH);
  if (gate.size(2) != kGateSize || !(whh_is_pretransposed || whh_is_native) ||
      b.size(0) != kGateSize || h_state.size(0) != batch_size || h_state.size(1) != kH ||
      c_state.size(0) != batch_size || c_state.size(1) != kH || !out_shape_ok) {
    throw std::invalid_argument("adaptive fixed gate accum workspace received incompatible shapes");
  }
  if (!(h_state.is_contiguous() && c_state.is_contiguous() && out.is_contiguous())) {
    throw std::invalid_argument("adaptive fixed gate accum workspace tensors must be contiguous");
  }

  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4) || kH % rb != 0) {
    throw std::invalid_argument("adaptive fixed gate accum read_block must be 1, 2, or 4 and divide hidden_size");
  }

  h_state.zero_();
  c_state.zero_();
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int pointwise_items = (batch_size * kH) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);

  for (int t = 0; t < seq_len; ++t) {
    batchmajor_recurrent_gemm_accumulate_gate(gate, h_state, whh_t, batch_size, kH, seq_len, t);
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, true, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_fixed_gate_accum_pointwise_kernel<kH, false, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

torch::Tensor adaptive_lstm_h256_gate_accum_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block) {
  return adaptive_lstm_fixed_gate_accum_update_forward_workspace_impl<256>(
      gate_proj, weight_hh, bias, h_state, c_state, out, write_sequence, read_block);
}

torch::Tensor adaptive_lstm_h512_gate_accum_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block) {
  return adaptive_lstm_fixed_gate_accum_update_forward_workspace_impl<512>(
      gate_proj, weight_hh, bias, h_state, c_state, out, write_sequence, read_block);
}

torch::Tensor adaptive_lstm_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block) {
  c10::InferenceMode inference_mode;
  if (!(gate_proj.is_cuda() && weight_hh.is_cuda() && bias.is_cuda() &&
        h_state.is_cuda() && c_state.is_cuda() && recur.is_cuda() && out.is_cuda())) {
    throw std::invalid_argument("adaptive generic GEMM scan workspace expects CUDA/HIP tensors");
  }
  if (!(gate_proj.scalar_type() == torch::kFloat16 &&
        weight_hh.scalar_type() == torch::kFloat16 &&
        bias.scalar_type() == torch::kFloat16 &&
        h_state.scalar_type() == torch::kFloat16 &&
        recur.scalar_type() == torch::kFloat16 &&
        out.scalar_type() == torch::kFloat16 &&
        c_state.scalar_type() == torch::kFloat32)) {
    throw std::invalid_argument("adaptive generic GEMM scan workspace expects FP16 data and FP32 cell state");
  }
  if (!(gate_proj.dim() == 3 && weight_hh.dim() == 2 && bias.dim() == 1 &&
        h_state.dim() == 2 && c_state.dim() == 2 && recur.dim() == 2)) {
    throw std::invalid_argument(
        "expected gate [B,T,4H], weight_hh [H,4H] or [4H,H], state [B,H], recur [B,4H]");
  }

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));
  const int gate_size = static_cast<int>(gate.size(2));
  if (gate_size % 4 != 0) {
    throw std::invalid_argument("adaptive generic GEMM scan gate size must be 4H");
  }
  const int hidden_size = gate_size / 4;
  const bool whh_is_pretransposed = whh.size(0) == hidden_size && whh.size(1) == gate_size;
  const bool whh_is_native = whh.size(0) == gate_size && whh.size(1) == hidden_size;
  const bool out_shape_ok = write_sequence
      ? (out.dim() == 3 && out.size(0) == batch_size && out.size(1) == seq_len &&
         out.size(2) == hidden_size)
      : (out.dim() == 2 && out.size(0) == batch_size && out.size(1) == hidden_size);
  if (!(whh_is_pretransposed || whh_is_native) || b.size(0) != gate_size ||
      h_state.size(0) != batch_size || h_state.size(1) != hidden_size ||
      c_state.size(0) != batch_size || c_state.size(1) != hidden_size ||
      recur.size(0) != batch_size || recur.size(1) != gate_size || !out_shape_ok) {
    throw std::invalid_argument("adaptive generic GEMM scan workspace received incompatible shapes");
  }
  if (!(h_state.is_contiguous() && c_state.is_contiguous() && recur.is_contiguous() &&
        out.is_contiguous())) {
    throw std::invalid_argument("adaptive generic GEMM scan workspace tensors must be contiguous");
  }

  const int rb = static_cast<int>(read_block);
  if (!(rb == 1 || rb == 2 || rb == 4) || hidden_size % rb != 0) {
    throw std::invalid_argument("adaptive generic GEMM scan read_block must be 1, 2, or 4 and divide hidden_size");
  }

  h_state.zero_();
  c_state.zero_();
  auto whh_t = whh_is_pretransposed ? whh : whh.transpose(0, 1).contiguous();
  const int pointwise_items = (batch_size * hidden_size) / rb;
  const int pointwise_blocks = (pointwise_items + 255) / 256;
  const bool check_kernel_errors = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_KERNEL_CHECKS", false);

  for (int t = 0; t < seq_len; ++t) {
    generic_recurrent_gemm(recur, h_state, whh_t, batch_size, hidden_size);
    if (write_sequence) {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<true, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<true, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<true, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      }
    } else {
      if (rb == 4) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<false, 4>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      } else if (rb == 2) {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<false, 2>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      } else {
        GPU_LAUNCH_KERNEL(
            (adaptive_lstm_generic_gemm_scan_pointwise_kernel<false, 1>),
            pointwise_blocks,
            256,
            0,
            0,
            reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(recur.data_ptr<at::Half>()),
            reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
            static_cast<float*>(c_state.data_ptr<float>()),
            reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
            batch_size,
            seq_len,
            hidden_size,
            t);
      }
    }
    check_last_error_if(check_kernel_errors);
  }

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

// ============================================================================
// Persistent scalar recurrent kernel (H128)
// One kernel launch per layer — eliminates per‑timestep dispatch overhead.
// Uses P4 partitioned scalar dot‑product (same as cached_b4).
// ============================================================================

template <int Partitions, bool WriteSequence, bool CheckTail>
__global__ void __launch_bounds__(512)
adaptive_lstm_h128_persistent_kernel(
    const gpu_half* __restrict__ gate_proj,  // [B, T, 512]
    const gpu_half* __restrict__ weight_hh,  // [512, 128] native layout
    const gpu_half* __restrict__ bias,       // [512]
    gpu_half* __restrict__ h_state_io,       // [B, 128] workspace
    float* __restrict__ c_state_io,          // [B, 128] workspace
    gpu_half* __restrict__ out,              // [B, T, 128] or [B, 128]
    int batch_size,
    int seq_len) {

  constexpr int kH = 128;
  constexpr int kGateSize = 512;
  constexpr int kBatchTile = 4;
  constexpr int kPerPartition = kH / Partitions;

  const int b_base = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int h = tid / Partitions;
  const int partition = tid - h * Partitions;

  __shared__ float h_prev[kBatchTile * kH];
  __shared__ float h_next[kBatchTile * kH];

  gpu_half i_w[kPerPartition];
  gpu_half f_w[kPerPartition];
  gpu_half g_w[kPerPartition];
  gpu_half o_w[kPerPartition];

  const gpu_half i_bias = bias[0 * kH + h];
  const gpu_half f_bias = bias[1 * kH + h];
  const gpu_half g_bias = bias[2 * kH + h];
  const gpu_half o_bias = bias[3 * kH + h];

#pragma unroll
  for (int idx = 0; idx < kPerPartition; ++idx) {
    const int k = partition + idx * Partitions;
    i_w[idx] = weight_hh[(0 * kH + h) * kH + k];
    f_w[idx] = weight_hh[(1 * kH + h) * kH + k];
    g_w[idx] = weight_hh[(2 * kH + h) * kH + k];
    o_w[idx] = weight_hh[(3 * kH + h) * kH + k];
  }

  float c_reg[kBatchTile];
#pragma unroll
  for (int bt = 0; bt < kBatchTile; ++bt) {
    c_reg[bt] = 0.0f;
    if (partition == 0) {
      h_prev[bt * kH + h] = 0.0f;
      h_next[bt * kH + h] = 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    float i_recur[kBatchTile];
    float f_recur[kBatchTile];
    float g_recur[kBatchTile];
    float o_recur[kBatchTile];
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = f_recur[bt] = g_recur[bt] = o_recur[bt] = 0.0f;
    }

#pragma unroll
    for (int idx = 0; idx < kPerPartition; ++idx) {
      const int k = partition + idx * Partitions;
      const float iw = half_to_float(i_w[idx]);
      const float fw = half_to_float(f_w[idx]);
      const float gw = half_to_float(g_w[idx]);
      const float ow = half_to_float(o_w[idx]);
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const float hv = h_prev[bt * kH + k];
          i_recur[bt] += hv * iw;
          f_recur[bt] += hv * fw;
          g_recur[bt] += hv * gw;
          o_recur[bt] += hv * ow;
        }
      }
    }

#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      i_recur[bt] = reduce_partition_sum<Partitions>(i_recur[bt]);
      f_recur[bt] = reduce_partition_sum<Partitions>(f_recur[bt]);
      g_recur[bt] = reduce_partition_sum<Partitions>(g_recur[bt]);
      o_recur[bt] = reduce_partition_sum<Partitions>(o_recur[bt]);
    }

    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          const int gate_base = (b * seq_len + t) * kGateSize;
          const float i_acc = half_to_float(gate_proj[gate_base + 0 * kH + h]) + half_to_float(i_bias) + i_recur[bt];
          const float f_acc = half_to_float(gate_proj[gate_base + 1 * kH + h]) + half_to_float(f_bias) + f_recur[bt];
          const float g_acc = half_to_float(gate_proj[gate_base + 2 * kH + h]) + half_to_float(g_bias) + g_recur[bt];
          const float o_acc = half_to_float(gate_proj[gate_base + 3 * kH + h]) + half_to_float(o_bias) + o_recur[bt];

          const float i_gate = sigmoidf_fast(i_acc);
          const float f_gate = sigmoidf_fast(f_acc);
          const float g_gate = tanhf(g_acc);
          const float o_gate = sigmoidf_fast(o_acc);

          c_reg[bt] = f_gate * c_reg[bt] + i_gate * g_gate;
          h_next[bt * kH + h] = o_gate * tanhf(c_reg[bt]);
          if (WriteSequence) {
            out[(b * seq_len + t) * kH + h] = float_to_half(h_next[bt * kH + h]);
          }
        }
      }
    }
    __syncthreads();

    if (partition == 0) {
#pragma unroll
      for (int bt = 0; bt < kBatchTile; ++bt) {
        const int b = b_base + bt;
        if (!CheckTail || b < batch_size) {
          h_prev[bt * kH + h] = h_next[bt * kH + h];
        }
      }
    }
    __syncthreads();
  }

  if (!WriteSequence && partition == 0) {
#pragma unroll
    for (int bt = 0; bt < kBatchTile; ++bt) {
      const int b = b_base + bt;
      if (!CheckTail || b < batch_size) {
        h_state_io[b * kH + h] = float_to_half(h_prev[bt * kH + h]);
        c_state_io[b * kH + h] = c_reg[bt];
      }
    }
  }
}

// ============================================================================
// MFMA-based persistent recurrent kernel (H128)
// Uses the DTK MFMA builtin for the recurrent matmul.
// Only compiled on architectures with mai-insts support (gfx942+).
// gfx906/gfx926/gfx928/gfx936/gfx938 do NOT support MFMA.
// ============================================================================

// MFMA enabled only when setup.py explicitly passes -DMIOPEN_ADAPTIVE_LSTM_ENABLE_MMAC_BUILTIN=1
#if defined(MIOPEN_ADAPTIVE_LSTM_ENABLE_MMAC_BUILTIN)
#define ADAPTIVE_LSTM_HAS_MMAC 1
#endif

#ifdef ADAPTIVE_LSTM_HAS_MMAC
// HCU (Hygon Compute Unit) MMAC builtins — K100_AI / gfx928 native matrix instructions.
// Use __builtin_hcu_mmac_f32_16x16x16_f16 instead of AMD's amdgcn_mfma variant.
typedef _Float16 __attribute__((vector_size(8)))  mmac_f16x4;
typedef float    __attribute__((vector_size(16))) mmac_f32x4;

__device__ __forceinline__
mmac_f32x4 mmac_f32_16x16x16f16_call(mmac_f16x4 a, mmac_f16x4 b, mmac_f32x4 c) {
  return __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);
}
#endif

#ifdef ADAPTIVE_LSTM_HAS_MMAC
template <bool WriteSequence>
__global__ void __launch_bounds__(256)
adaptive_lstm_h128_mmac_persistent_kernel(
    const gpu_half* __restrict__ gate_proj,
    const gpu_half* __restrict__ weight_hh,
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state_io,
    float* __restrict__ c_state_io,
    gpu_half* __restrict__ out,
    int batch_size,
    int seq_len) {

  constexpr int kH = 128;
  constexpr int kGateSize = 512;
  constexpr int kBatchTile = 16;
  constexpr int kMmacK = 16;
  constexpr int kMmacN = 16;
  constexpr int kKTiles = kH / kMmacK;
  constexpr int kHtiles = kH / kMmacK;
  constexpr int kGateRegions = 4;
  constexpr int kWavesPerBlock = 4;
  constexpr int kWaveSize = 64;
  constexpr int kColsPerHTile = kMmacN * kGateRegions;
  constexpr int kHStride = kH + 2;                  // pad to eliminate 16-way LDS bank conflict
  constexpr int kRecurStride = kColsPerHTile + 2;   // pad for same reason

  const int b0 = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int wave_id = tid / kWaveSize;              // 0..3: each wave owns one H tile in a group
  const int lane_id = tid & (kWaveSize - 1);
  const int mmac_row = lane_id & 15;
  const int mmac_col0 = (lane_id >> 4) * 4;

  __shared__ gpu_half h_lds[kBatchTile * kHStride];
  __shared__ gpu_half h_next_lds[kBatchTile * kHStride];
  __shared__ float cell_lds[kBatchTile * kHStride];
  // One recur scratch per wave. The old version used one scratch tile for all 4
  // waves, so the 4 wavefronts redundantly computed and overwrote the same tile.
  __shared__ float recur_h[kWavesPerBlock * kBatchTile * kRecurStride];

  for (int i = tid; i < kBatchTile * kHStride; i += 256) {
    h_lds[i] = __float2half(0.0f);
    h_next_lds[i] = __float2half(0.0f);
    cell_lds[i] = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    // Process 8 hidden tiles as two groups. Within each group, the 4 wavefronts
    // compute 4 different 16-column H tiles concurrently instead of repeating
    // the same MMAC work four times.
    for (int ht_base = 0; ht_base < kHtiles; ht_base += kWavesPerBlock) {
      const int ht = ht_base + wave_id;
      const int h0 = ht * kMmacK;
      float* recur_wave = recur_h + wave_id * kBatchTile * kRecurStride;

      // 4 accumulators per gate, kept in registers across K-tiles.
      mmac_f32x4 acc_i = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_f = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_o = {0.0f, 0.0f, 0.0f, 0.0f};

      if (ht < kHtiles) {
        for (int kt = 0; kt < kKTiles; ++kt) {
          const int k0 = kt * kMmacK;

          // Load A (h_lds tile) once per K-tile — serves all 4 gates.
          mmac_f16x4 a_regs;
          {
            const gpu_half* sa = h_lds + mmac_row * kHStride + k0 + mmac_col0;
            a_regs[0]=(_Float16)half_to_float(sa[0]); a_regs[1]=(_Float16)half_to_float(sa[1]);
            a_regs[2]=(_Float16)half_to_float(sa[2]); a_regs[3]=(_Float16)half_to_float(sa[3]);
          }
          const int w_row = k0 + mmac_row;

          // Gate i
          {
            mmac_f16x4 b_regs;
            const int gc = 0*kH + h0;
            b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
            b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
            b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
            b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
            acc_i = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_i);
          }
          // Gate f
          {
            mmac_f16x4 b_regs;
            const int gc = 1*kH + h0;
            b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
            b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
            b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
            b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
            acc_f = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_f);
          }
          // Gate g
          {
            mmac_f16x4 b_regs;
            const int gc = 2*kH + h0;
            b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
            b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
            b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
            b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
            acc_g = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_g);
          }
          // Gate o
          {
            mmac_f16x4 b_regs;
            const int gc = 3*kH + h0;
            b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
            b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
            b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
            b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
            acc_o = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_o);
          }
        }  // k-tile

        // Store this wave's accumulators to its own LDS scratch.
        const int bl = lane_id & 15;
        const int gb = (lane_id >> 4) * 4;
        recur_wave[bl*kRecurStride+ 0*kMmacN+gb+0] = acc_i[0];
        recur_wave[bl*kRecurStride+ 0*kMmacN+gb+1] = acc_i[1];
        recur_wave[bl*kRecurStride+ 0*kMmacN+gb+2] = acc_i[2];
        recur_wave[bl*kRecurStride+ 0*kMmacN+gb+3] = acc_i[3];
        recur_wave[bl*kRecurStride+ 1*kMmacN+gb+0] = acc_f[0];
        recur_wave[bl*kRecurStride+ 1*kMmacN+gb+1] = acc_f[1];
        recur_wave[bl*kRecurStride+ 1*kMmacN+gb+2] = acc_f[2];
        recur_wave[bl*kRecurStride+ 1*kMmacN+gb+3] = acc_f[3];
        recur_wave[bl*kRecurStride+ 2*kMmacN+gb+0] = acc_g[0];
        recur_wave[bl*kRecurStride+ 2*kMmacN+gb+1] = acc_g[1];
        recur_wave[bl*kRecurStride+ 2*kMmacN+gb+2] = acc_g[2];
        recur_wave[bl*kRecurStride+ 2*kMmacN+gb+3] = acc_g[3];
        recur_wave[bl*kRecurStride+ 3*kMmacN+gb+0] = acc_o[0];
        recur_wave[bl*kRecurStride+ 3*kMmacN+gb+1] = acc_o[1];
        recur_wave[bl*kRecurStride+ 3*kMmacN+gb+2] = acc_o[2];
        recur_wave[bl*kRecurStride+ 3*kMmacN+gb+3] = acc_o[3];
      }
      __syncthreads();

      // Gate math for this wave's H tile. A single wave has 64 lanes and covers
      // 16(batch) * 16(hidden) elements with a stride-64 loop.
      if (ht < kHtiles) {
        for (int i = lane_id; i < kBatchTile * kMmacK; i += kWaveSize) {
          const int b_local = i / kMmacK;
          const int h_off = i - b_local * kMmacK;
          const int b = b0 + b_local;
          if (b >= batch_size) continue;
          const int h = h0 + h_off;
          const int gb2 = (b * seq_len + t) * kGateSize;
          const float gi=half_to_float(gate_proj[gb2+h]), gf=half_to_float(gate_proj[gb2+h+kH]);
          const float gg=half_to_float(gate_proj[gb2+h+2*kH]), go=half_to_float(gate_proj[gb2+h+3*kH]);
          const float ri=recur_wave[b_local*kRecurStride+h_off], rf=recur_wave[b_local*kRecurStride+kMmacK+h_off];
          const float rg=recur_wave[b_local*kRecurStride+2*kMmacK+h_off], ro=recur_wave[b_local*kRecurStride+3*kMmacK+h_off];
          const float bi=half_to_float(bias[h]), bf=half_to_float(bias[h+kH]);
          const float bg=half_to_float(bias[h+2*kH]), bo=half_to_float(bias[h+3*kH]);
          const float ia=gi+ri+bi, fa=gf+rf+bf, ga=gg+rg+bg, oa=go+ro+bo;
          const float ig=sigmoidf_fast(ia), fg=sigmoidf_fast(fa);
          const float gg2=tanhf(ga), og=sigmoidf_fast(oa);
          const int ci = b_local * kHStride + h;
          const float cp = cell_lds[ci];
          const float cn = fg * cp + ig * gg2;
          cell_lds[ci] = cn;
          h_next_lds[b_local*kHStride+h] = float_to_half(og * tanhf(cn));
          if (WriteSequence) out[(b*seq_len+t)*kH+h] = h_next_lds[b_local*kHStride+h];
        }
      }
      __syncthreads();
    }

    for (int i = tid; i < kBatchTile * kH; i += 256) {
      const int b_local = i / kH, h = i - b_local * kH, b = b0 + b_local;
      if (b < batch_size) {
        h_lds[b_local*kHStride+h] = h_next_lds[b_local*kHStride+h];
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < kBatchTile * kH; i += 256) {
    const int b_local = i / kH, h = i - b_local * kH, b = b0 + b_local;
    if (b < batch_size) {
      h_state_io[b*kH+h] = h_lds[b_local*kHStride+h];
      c_state_io[b*kH+h] = cell_lds[b_local*kHStride+h];
    }
  }
}
// ============================================================================
// Profile variant kernel — isolates cost of MMAC / bias / activation / state update
// Variant 0: MMAC only (K-tile loop + MMAC instructions, no bias/act/state)
// Variant 1: MMAC + bias add
// Variant 2: MMAC + bias + sigmoid/tanh activation
// Variant 3: Full LSTM (equivalent to adaptive_lstm_h128_mmac_persistent_kernel)
// ============================================================================
template <bool WriteSequence, int Variant>
__global__ void __launch_bounds__(256)
adaptive_lstm_h128_mmac_profile_variant_kernel(
    const gpu_half* __restrict__ gate_proj,  // [B, T, 512]
    const gpu_half* __restrict__ weight_hh,  // [512, 128]
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state_io,
    float* __restrict__ c_state_io,
    gpu_half* __restrict__ out,
    gpu_half* __restrict__ profile_out,       // [B, 512] for variants 0-2
    int batch_size,
    int seq_len) {

  constexpr int kH = 128;
  constexpr int kGateSize = 512;
  constexpr int kBatchTile = 16;
  constexpr int kMmacK = 16;
  constexpr int kMmacN = 16;
  constexpr int kKTiles = kH / kMmacK;
  constexpr int kHtiles = kH / kMmacK;
  constexpr int kGateRegions = 4;
  constexpr int kColsPerHTile = kMmacN * kGateRegions;
  constexpr int kHStride = kH + 2;                  // pad to eliminate 16-way LDS bank conflict
  constexpr int kRecurStride = kColsPerHTile + 2;   // pad for same reason

  const int b0 = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int lane_id = tid & 63;
  const int mmac_row = lane_id & 15;
  const int mmac_col0 = (lane_id >> 4) * 4;

  __shared__ gpu_half h_lds[kBatchTile * kHStride];
  __shared__ gpu_half h_next_lds[kBatchTile * kHStride];
  __shared__ float cell_lds[kBatchTile * kHStride];
  __shared__ float recur_h[kBatchTile * kRecurStride];

  for (int i = tid; i < kBatchTile * kHStride; i += 256) {
    h_lds[i] = __float2half(0.0f);
    h_next_lds[i] = __float2half(0.0f);
    cell_lds[i] = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    for (int ht = 0; ht < kHtiles; ++ht) {
      const int h0 = ht * kMmacK;

      mmac_f32x4 acc_i = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_f = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_o = {0.0f, 0.0f, 0.0f, 0.0f};

      // ---- K-tile loop (identical for all variants) ----
      for (int kt = 0; kt < kKTiles; ++kt) {
        const int k0 = kt * kMmacK;

        mmac_f16x4 a_regs;
        {
          const gpu_half* sa = h_lds + mmac_row * kHStride + k0 + mmac_col0;
          a_regs[0]=(_Float16)half_to_float(sa[0]); a_regs[1]=(_Float16)half_to_float(sa[1]);
          a_regs[2]=(_Float16)half_to_float(sa[2]); a_regs[3]=(_Float16)half_to_float(sa[3]);
        }
        const int w_row = k0 + mmac_row;

        // Gate i
        {
          mmac_f16x4 b_regs;
          const int gc = 0*kH + h0;
          b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
          b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
          b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
          b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
          acc_i = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_i);
        }
        // Gate f
        {
          mmac_f16x4 b_regs;
          const int gc = 1*kH + h0;
          b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
          b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
          b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
          b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
          acc_f = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_f);
        }
        // Gate g
        {
          mmac_f16x4 b_regs;
          const int gc = 2*kH + h0;
          b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
          b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
          b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
          b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
          acc_g = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_g);
        }
        // Gate o
        {
          mmac_f16x4 b_regs;
          const int gc = 3*kH + h0;
          b_regs[0]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+0)*kH+w_row]);
          b_regs[1]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+1)*kH+w_row]);
          b_regs[2]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+2)*kH+w_row]);
          b_regs[3]=(_Float16)half_to_float(weight_hh[(gc+mmac_col0+3)*kH+w_row]);
          acc_o = mmac_f32_16x16x16f16_call(a_regs, b_regs, acc_o);
        }
      }  // end K-tile

      // Store accumulators to recur_h LDS (identical for all variants)
      {
        const int bl = lane_id & 15;
        const int gb = (lane_id >> 4) * 4;
        recur_h[bl*kRecurStride+ 0*kMmacN+gb+0] = acc_i[0];
        recur_h[bl*kRecurStride+ 0*kMmacN+gb+1] = acc_i[1];
        recur_h[bl*kRecurStride+ 0*kMmacN+gb+2] = acc_i[2];
        recur_h[bl*kRecurStride+ 0*kMmacN+gb+3] = acc_i[3];
        recur_h[bl*kRecurStride+ 1*kMmacN+gb+0] = acc_f[0];
        recur_h[bl*kRecurStride+ 1*kMmacN+gb+1] = acc_f[1];
        recur_h[bl*kRecurStride+ 1*kMmacN+gb+2] = acc_f[2];
        recur_h[bl*kRecurStride+ 1*kMmacN+gb+3] = acc_f[3];
        recur_h[bl*kRecurStride+ 2*kMmacN+gb+0] = acc_g[0];
        recur_h[bl*kRecurStride+ 2*kMmacN+gb+1] = acc_g[1];
        recur_h[bl*kRecurStride+ 2*kMmacN+gb+2] = acc_g[2];
        recur_h[bl*kRecurStride+ 2*kMmacN+gb+3] = acc_g[3];
        recur_h[bl*kRecurStride+ 3*kMmacN+gb+0] = acc_o[0];
        recur_h[bl*kRecurStride+ 3*kMmacN+gb+1] = acc_o[1];
        recur_h[bl*kRecurStride+ 3*kMmacN+gb+2] = acc_o[2];
        recur_h[bl*kRecurStride+ 3*kMmacN+gb+3] = acc_o[3];
      }

      // ---- Pointwise section (varies by variant) ----
      for (int i = tid; i < kBatchTile * kMmacK; i += 256) {
        const int b_local = i / kMmacK, h_off = i - b_local * kMmacK;
        const int b = b0 + b_local;
        if (b >= batch_size) continue;
        const int h = h0 + h_off;
        const int gb2 = (b * seq_len + t) * kGateSize;

        const float gi=half_to_float(gate_proj[gb2+h]), gf=half_to_float(gate_proj[gb2+h+kH]);
        const float gg=half_to_float(gate_proj[gb2+h+2*kH]), go=half_to_float(gate_proj[gb2+h+3*kH]);
        const float ri=recur_h[b_local*kRecurStride+h_off], rf=recur_h[b_local*kRecurStride+kMmacK+h_off];
        const float rg=recur_h[b_local*kRecurStride+2*kMmacK+h_off], ro=recur_h[b_local*kRecurStride+3*kMmacK+h_off];

        float ia = gi + ri, fa = gf + rf, ga = gg + rg, oa = go + ro;

        if constexpr (Variant >= 1) {
          const float bi=half_to_float(bias[h]), bf=half_to_float(bias[h+kH]);
          const float bg=half_to_float(bias[h+2*kH]), bo=half_to_float(bias[h+3*kH]);
          ia += bi; fa += bf; ga += bg; oa += bo;
        }

        float ig = 0, fg = 0, gg2 = 0, og = 0;
        if constexpr (Variant >= 2) {
          ig = sigmoidf_fast(ia);
          fg = sigmoidf_fast(fa);
          gg2 = tanhf(ga);
          og = sigmoidf_fast(oa);
        }

        if constexpr (Variant >= 3) {
          const int ci = b_local * kHStride + h;
          const float cp = cell_lds[ci];
          const float cn = fg * cp + ig * gg2;
          cell_lds[ci] = cn;
          h_next_lds[b_local*kHStride+h] = float_to_half(og * tanhf(cn));
          if (WriteSequence) out[(b*seq_len+t)*kH+h] = h_next_lds[b_local*kHStride+h];
        } else {
          // Variants 0-2: use sigmoid(gate_i) as fake next hidden state
          // to maintain non-zero, evolving input for MMAC across timesteps
          h_next_lds[b_local*kHStride+h] = float_to_half(sigmoidf_fast(ia));

          // Write intermediate results to profile_out at last timestep
          if (t == seq_len - 1) {
            float v0, v1, v2, v3;
            if constexpr (Variant == 0) {
              v0 = gi + ri; v1 = gf + rf; v2 = gg + rg; v3 = go + ro;
            } else if constexpr (Variant == 1) {
              v0 = ia; v1 = fa; v2 = ga; v3 = oa;
            } else {
              v0 = ig; v1 = fg; v2 = gg2; v3 = og;
            }
            profile_out[b*kGateSize + 0*kH + h] = float_to_half(v0);
            profile_out[b*kGateSize + 1*kH + h] = float_to_half(v1);
            profile_out[b*kGateSize + 2*kH + h] = float_to_half(v2);
            profile_out[b*kGateSize + 3*kH + h] = float_to_half(v3);
          }
        }
      }  // pointwise loop
      __syncthreads();
    }  // H-tile

    for (int i = tid; i < kBatchTile * kH; i += 256) {
      const int b_local = i / kH, h = i - b_local * kH, b = b0 + b_local;
      if (b < batch_size) {
        h_lds[b_local*kHStride+h] = h_next_lds[b_local*kHStride+h];
      }
    }
    __syncthreads();
  }  // timestep

  if constexpr (Variant >= 3) {
    for (int i = tid; i < kBatchTile * kH; i += 256) {
      const int b_local = i / kH, h = i - b_local * kH, b = b0 + b_local;
      if (b < batch_size) {
        h_state_io[b*kH+h] = h_lds[b_local*kHStride+h];
        c_state_io[b*kH+h] = cell_lds[b_local*kHStride+h];
      }
    }
  }
}




// Packed-weight variant kernel — B-load from pre-packed [65536] layout
// Layout: [htile=8][ktile=8][K=16][gate=4][N_group=4][frag=4]
// Per K_row: 4 gates × 4 N_groups × 4 frag = 64 halfs, contiguous
template <bool WriteSequence, int Variant>
__global__ void __launch_bounds__(256)
adaptive_lstm_h128_mmac_packed_variant_kernel(
    const gpu_half* __restrict__ gate_proj,  // [B, T, 512]
    const gpu_half* __restrict__ packed_w,   // [65536] packed layout
    const gpu_half* __restrict__ bias,
    gpu_half* __restrict__ h_state_io,
    float* __restrict__ c_state_io,
    gpu_half* __restrict__ out,
    gpu_half* __restrict__ profile_out,
    int batch_size,
    int seq_len) {

  constexpr int kH = 128;
  constexpr int kGateSize = 512;
  constexpr int kMmacM = 16;               // MMAC M dimension (hardware)
  constexpr int kBatchTile = 4;            // logical batch per block (split-B)
  constexpr int kMmacK = 16;
  constexpr int kMmacN = 16;
  constexpr int kKTiles = kH / kMmacK;
  constexpr int kHtiles = kH / kMmacK;
  constexpr int kGateRegions = 4;
  constexpr int kColsPerHTile = kMmacN * kGateRegions;
  constexpr int kHStride = kH + 2;
  constexpr int kRecurStride = kColsPerHTile + 2;
  constexpr int kWavesPerBlock = 4;
  constexpr int kWaveSize = 64;

  constexpr int kRowStride = 64;           // 4 gates × 16 N
  constexpr int kKTileStride = kMmacK * kRowStride;   // 1024
  constexpr int kHTileStride = kKTiles * kKTileStride; // 8192

  const int b0 = blockIdx.x * kBatchTile;
  const int tid = threadIdx.x;
  const int wave_id = tid / kWaveSize;
  const int lane_id = tid & (kWaveSize - 1);
  const int mmac_row = lane_id & 15;
  const int mmac_col0 = (lane_id >> 4) * 4;
  const bool row_active = mmac_row < kBatchTile;

  // LDS rows sized for MMAC M=16; only first kBatchTile rows are valid
  // Double-buffered h_state: h_cur for A-load, h_next for pointwise write
  __shared__ gpu_half h_buf[2][kMmacM * kHStride];
  __shared__ float cell_lds[kMmacM * kHStride];
  __shared__ float recur_h[kWavesPerBlock * kBatchTile * kRecurStride];

  for (int i = tid; i < kMmacM * kHStride; i += 256) {
    h_buf[0][i] = __float2half(0.0f);
    h_buf[1][i] = __float2half(0.0f);
    cell_lds[i] = 0.0f;
  }
  __syncthreads();

  gpu_half* h_cur = h_buf[0];
  gpu_half* h_next = h_buf[1];

  for (int t = 0; t < seq_len; ++t) {
    for (int ht_base = 0; ht_base < kHtiles; ht_base += kWavesPerBlock) {
      const int ht = ht_base + wave_id;
      const int h0 = ht * kMmacK;
      const int ht_row_base = ht * kHTileStride + mmac_row * kRowStride;
      const int ng = mmac_col0 >> 2;
      const int ng_offset = ng * 16;
      float* recur_wave = recur_h + wave_id * kBatchTile * kRecurStride;

      mmac_f32x4 acc_i = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_f = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_g = {0.0f, 0.0f, 0.0f, 0.0f};
      mmac_f32x4 acc_o = {0.0f, 0.0f, 0.0f, 0.0f};

      if (ht < kHtiles) {
        for (int kt = 0; kt < kKTiles; ++kt) {
          const int frag_base = ht_row_base + kt * kKTileStride + ng_offset;

          // Load A from current h_state (double-buffered)
          mmac_f16x4 a_regs;
          {
            const int k0 = kt * kMmacK;
            const gpu_half* sa = h_cur + mmac_row * kHStride + k0 + mmac_col0;
            a_regs[0]=(_Float16)half_to_float(sa[0]); a_regs[1]=(_Float16)half_to_float(sa[1]);
            a_regs[2]=(_Float16)half_to_float(sa[2]); a_regs[3]=(_Float16)half_to_float(sa[3]);
          }

          // Load B from packed layout — all 4 gates contiguous per K_row
          mmac_f16x4 b_i, b_f, b_g, b_o;
          b_i[0]=(_Float16)half_to_float(packed_w[frag_base + 0*4 + 0]);
          b_i[1]=(_Float16)half_to_float(packed_w[frag_base + 0*4 + 1]);
          b_i[2]=(_Float16)half_to_float(packed_w[frag_base + 0*4 + 2]);
          b_i[3]=(_Float16)half_to_float(packed_w[frag_base + 0*4 + 3]);
          b_f[0]=(_Float16)half_to_float(packed_w[frag_base + 1*4 + 0]);
          b_f[1]=(_Float16)half_to_float(packed_w[frag_base + 1*4 + 1]);
          b_f[2]=(_Float16)half_to_float(packed_w[frag_base + 1*4 + 2]);
          b_f[3]=(_Float16)half_to_float(packed_w[frag_base + 1*4 + 3]);
          b_g[0]=(_Float16)half_to_float(packed_w[frag_base + 2*4 + 0]);
          b_g[1]=(_Float16)half_to_float(packed_w[frag_base + 2*4 + 1]);
          b_g[2]=(_Float16)half_to_float(packed_w[frag_base + 2*4 + 2]);
          b_g[3]=(_Float16)half_to_float(packed_w[frag_base + 2*4 + 3]);
          b_o[0]=(_Float16)half_to_float(packed_w[frag_base + 3*4 + 0]);
          b_o[1]=(_Float16)half_to_float(packed_w[frag_base + 3*4 + 1]);
          b_o[2]=(_Float16)half_to_float(packed_w[frag_base + 3*4 + 2]);
          b_o[3]=(_Float16)half_to_float(packed_w[frag_base + 3*4 + 3]);

          acc_i = mmac_f32_16x16x16f16_call(a_regs, b_i, acc_i);
          acc_f = mmac_f32_16x16x16f16_call(a_regs, b_f, acc_f);
          acc_g = mmac_f32_16x16x16f16_call(a_regs, b_g, acc_g);
          acc_o = mmac_f32_16x16x16f16_call(a_regs, b_o, acc_o);
        }  // K-tile

        // Store accumulators for active rows only (mmac_row < kBatchTile)
        if (row_active) {
          const int bl = lane_id & 15;
          const int gb = (lane_id >> 4) * 4;
          recur_wave[bl*kRecurStride+ 0*kMmacN+gb+0] = acc_i[0];
          recur_wave[bl*kRecurStride+ 0*kMmacN+gb+1] = acc_i[1];
          recur_wave[bl*kRecurStride+ 0*kMmacN+gb+2] = acc_i[2];
          recur_wave[bl*kRecurStride+ 0*kMmacN+gb+3] = acc_i[3];
          recur_wave[bl*kRecurStride+ 1*kMmacN+gb+0] = acc_f[0];
          recur_wave[bl*kRecurStride+ 1*kMmacN+gb+1] = acc_f[1];
          recur_wave[bl*kRecurStride+ 1*kMmacN+gb+2] = acc_f[2];
          recur_wave[bl*kRecurStride+ 1*kMmacN+gb+3] = acc_f[3];
          recur_wave[bl*kRecurStride+ 2*kMmacN+gb+0] = acc_g[0];
          recur_wave[bl*kRecurStride+ 2*kMmacN+gb+1] = acc_g[1];
          recur_wave[bl*kRecurStride+ 2*kMmacN+gb+2] = acc_g[2];
          recur_wave[bl*kRecurStride+ 2*kMmacN+gb+3] = acc_g[3];
          recur_wave[bl*kRecurStride+ 3*kMmacN+gb+0] = acc_o[0];
          recur_wave[bl*kRecurStride+ 3*kMmacN+gb+1] = acc_o[1];
          recur_wave[bl*kRecurStride+ 3*kMmacN+gb+2] = acc_o[2];
          recur_wave[bl*kRecurStride+ 3*kMmacN+gb+3] = acc_o[3];
        }
      }
      __syncthreads();

      // Pointwise for this wave's H tile
      if (ht < kHtiles) {
        for (int i = lane_id; i < kBatchTile * kMmacK; i += kWaveSize) {
          const int b_local = i / kMmacK;
          const int h_off = i - b_local * kMmacK;
          const int b = b0 + b_local;
          if (b >= batch_size) continue;
          const int h = h0 + h_off;
          const int gb2 = (b * seq_len + t) * kGateSize;

          const float gi=half_to_float(gate_proj[gb2+h]), gf=half_to_float(gate_proj[gb2+h+kH]);
          const float gg=half_to_float(gate_proj[gb2+h+2*kH]), go=half_to_float(gate_proj[gb2+h+3*kH]);
          const float ri=recur_wave[b_local*kRecurStride+h_off], rf=recur_wave[b_local*kRecurStride+kMmacK+h_off];
          const float rg=recur_wave[b_local*kRecurStride+2*kMmacK+h_off], ro=recur_wave[b_local*kRecurStride+3*kMmacK+h_off];

          float ia = gi + ri, fa = gf + rf, ga = gg + rg, oa = go + ro;

          if constexpr (Variant >= 1) {
            const float bi=half_to_float(bias[h]), bf=half_to_float(bias[h+kH]);
            const float bg=half_to_float(bias[h+2*kH]), bo=half_to_float(bias[h+3*kH]);
            ia += bi; fa += bf; ga += bg; oa += bo;
          }

          float ig = 0, fg = 0, gg2 = 0, og = 0;
          if constexpr (Variant >= 2) {
            ig = sigmoidf_fast(ia);
            fg = sigmoidf_fast(fa);
            gg2 = tanhf(ga);
            og = sigmoidf_fast(oa);
          }

          if constexpr (Variant >= 3) {
            const int ci = b_local * kHStride + h;
            const float cp = cell_lds[ci];
            const float cn = fg * cp + ig * gg2;
            cell_lds[ci] = cn;
            h_next[b_local*kHStride+h] = float_to_half(og * tanhf(cn));
            if (WriteSequence) out[(b*seq_len+t)*kH+h] = h_next[b_local*kHStride+h];
          } else {
            h_next[b_local*kHStride+h] = float_to_half(sigmoidf_fast(ia));
            if (t == seq_len - 1) {
              float v0, v1, v2, v3;
              if constexpr (Variant == 0) {
                v0 = gi + ri; v1 = gf + rf; v2 = gg + rg; v3 = go + ro;
              } else if constexpr (Variant == 1) {
                v0 = ia; v1 = fa; v2 = ga; v3 = oa;
              } else {
                v0 = ig; v1 = fg; v2 = gg2; v3 = og;
              }
              profile_out[b*kGateSize + 0*kH + h] = float_to_half(v0);
              profile_out[b*kGateSize + 1*kH + h] = float_to_half(v1);
              profile_out[b*kGateSize + 2*kH + h] = float_to_half(v2);
              profile_out[b*kGateSize + 3*kH + h] = float_to_half(v3);
            }
          }
        }  // pointwise
      }
      __syncthreads();
    }  // ht_group

    // Swap buffers: next timestep reads what we just wrote
    gpu_half* tmp = h_cur;
    h_cur = h_next;
    h_next = tmp;
  }  // timestep

  if constexpr (Variant >= 3) {
    for (int i = tid; i < kBatchTile * kH; i += 256) {
      const int b_local = i / kH, h = i - b_local * kH, b = b0 + b_local;
      if (b < batch_size) {
        h_state_io[b*kH+h] = h_cur[b_local*kHStride+h];
        c_state_io[b*kH+h] = cell_lds[b_local*kHStride+h];
      }
    }
  }
}

#endif  // ADAPTIVE_LSTM_HAS_MMAC


// Wrapper function — selects between scalar and MFMA persistent kernels
torch::Tensor adaptive_lstm_h128_persistent_mmac_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t fast_act) {
  c10::InferenceMode inference_mode;

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();

  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));

  if (gate.size(2) != 512 || whh.size(0) != 512 || whh.size(1) != 128 ||
      b.size(0) != 512 || h_state.size(0) != batch_size || h_state.size(1) != 128 ||
      c_state.size(0) != batch_size || c_state.size(1) != 128) {
    throw std::invalid_argument("adaptive H128 persistent kernel received incompatible shapes");
  }

  (void)read_block;
  (void)fast_act;

  c_state.zero_();

  // Use MFMA kernel when requested AND compiled for HIP (MFMA types defined)
  const bool use_mmac = env_flag_enabled("MIOPEN_ADAPTIVE_LSTM_USE_MMAC", false);

#ifdef ADAPTIVE_LSTM_HAS_MMAC
  if (use_mmac) {
    const int grid = (batch_size + 15) / 16;
    if (write_sequence) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_mmac_persistent_kernel<true>),
          grid, 256, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_mmac_persistent_kernel<false>),
          grid, 256, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    }
    check_last_error();
    if (!write_sequence) return h_state;
    return out;
  }
#endif

  // Default: scalar persistent kernel
  const bool check_tail = (batch_size & 3) != 0;
  const int grid = (batch_size + 3) / 4;

  if (write_sequence) {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_persistent_kernel<4, true, true>),
          grid, 512, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_persistent_kernel<4, true, false>),
          grid, 512, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    }
  } else {
    if (check_tail) {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_persistent_kernel<4, false, true>),
          grid, 512, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    } else {
      GPU_LAUNCH_KERNEL(
          (adaptive_lstm_h128_persistent_kernel<4, false, false>),
          grid, 512, 0, 0,
          reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()),
          reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()),
          static_cast<float*>(c_state.data_ptr<float>()),
          reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()),
          batch_size, seq_len);
    }
  }

  check_last_error();

  if (!write_sequence) {
    return h_state;
  }
  return out;
}


// Wrapper for profile variants A/B/C/D — isolates cost of MMAC, bias, activation, state update
torch::Tensor adaptive_lstm_h128_mmac_profile_variant_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    torch::Tensor profile_out,
    bool write_sequence,
    int64_t read_block,
    int64_t variant) {
  c10::InferenceMode inference_mode;

  auto gate = gate_proj.contiguous();
  auto whh = weight_hh.contiguous();
  auto b = bias.contiguous();
  auto pout = profile_out.contiguous();

  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));

  if (gate.size(2) != 512 || whh.size(0) != 512 || whh.size(1) != 128 ||
      b.size(0) != 512 || h_state.size(0) != batch_size || h_state.size(1) != 128 ||
      c_state.size(0) != batch_size || c_state.size(1) != 128 ||
      pout.size(0) != batch_size || pout.size(1) != 512) {
    throw std::invalid_argument("adaptive H128 profile variant kernel received incompatible shapes");
  }

  (void)read_block;

  c_state.zero_();

#ifdef ADAPTIVE_LSTM_HAS_MMAC
  const int grid = (batch_size + 15) / 16;

#define LAUNCH_VARIANT(V, WS) \
  GPU_LAUNCH_KERNEL( \
      (adaptive_lstm_h128_mmac_profile_variant_kernel<WS, V>), \
      grid, 256, 0, 0, \
      reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()), \
      reinterpret_cast<const gpu_half*>(whh.data_ptr<at::Half>()), \
      reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()), \
      reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()), \
      static_cast<float*>(c_state.data_ptr<float>()), \
      reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()), \
      reinterpret_cast<gpu_half*>(pout.data_ptr<at::Half>()), \
      batch_size, seq_len);

  if (variant == 0) {
    if (write_sequence) { LAUNCH_VARIANT(0, true) } else { LAUNCH_VARIANT(0, false) }
  } else if (variant == 1) {
    if (write_sequence) { LAUNCH_VARIANT(1, true) } else { LAUNCH_VARIANT(1, false) }
  } else if (variant == 2) {
    if (write_sequence) { LAUNCH_VARIANT(2, true) } else { LAUNCH_VARIANT(2, false) }
  } else {
    if (write_sequence) { LAUNCH_VARIANT(3, true) } else { LAUNCH_VARIANT(3, false) }
  }

#undef LAUNCH_VARIANT

  check_last_error();
#else
  throw std::runtime_error("MMAC profile variants require MIOPEN_ADAPTIVE_LSTM_ENABLE_MMAC_BUILTIN");
#endif

  if (!write_sequence) {
    return h_state;
  }
  return out;
}


// Wrapper for packed-weight profile variants — B-load from pre-packed [65536] layout
torch::Tensor adaptive_lstm_h128_mmac_packed_variant_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& packed_weight,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    torch::Tensor profile_out,
    bool write_sequence,
    int64_t read_block,
    int64_t variant) {
  c10::InferenceMode inference_mode;

  auto gate = gate_proj.contiguous();
  auto pkw = packed_weight.contiguous();
  auto b = bias.contiguous();
  auto pout = profile_out.contiguous();

  const int batch_size = static_cast<int>(gate.size(0));
  const int seq_len = static_cast<int>(gate.size(1));

  if (gate.size(2) != 512 || pkw.size(0) != 65536 ||
      b.size(0) != 512 || h_state.size(0) != batch_size || h_state.size(1) != 128 ||
      c_state.size(0) != batch_size || c_state.size(1) != 128 ||
      pout.size(0) != batch_size || pout.size(1) != 512) {
    throw std::invalid_argument("adaptive H128 packed variant kernel received incompatible shapes");
  }

  (void)read_block;

  c_state.zero_();

#ifdef ADAPTIVE_LSTM_HAS_MMAC
  const int grid = (batch_size + 3) / 4;  // split-B: kBatchTile=4

#define LAUNCH_PACKED_VARIANT(V, WS) \
  GPU_LAUNCH_KERNEL( \
      (adaptive_lstm_h128_mmac_packed_variant_kernel<WS, V>), \
      grid, 256, 0, 0, \
      reinterpret_cast<const gpu_half*>(gate.data_ptr<at::Half>()), \
      reinterpret_cast<const gpu_half*>(pkw.data_ptr<at::Half>()), \
      reinterpret_cast<const gpu_half*>(b.data_ptr<at::Half>()), \
      reinterpret_cast<gpu_half*>(h_state.data_ptr<at::Half>()), \
      static_cast<float*>(c_state.data_ptr<float>()), \
      reinterpret_cast<gpu_half*>(out.data_ptr<at::Half>()), \
      reinterpret_cast<gpu_half*>(pout.data_ptr<at::Half>()), \
      batch_size, seq_len);

  if (variant == 0) {
    if (write_sequence) { LAUNCH_PACKED_VARIANT(0, true) } else { LAUNCH_PACKED_VARIANT(0, false) }
  } else if (variant == 1) {
    if (write_sequence) { LAUNCH_PACKED_VARIANT(1, true) } else { LAUNCH_PACKED_VARIANT(1, false) }
  } else if (variant == 2) {
    if (write_sequence) { LAUNCH_PACKED_VARIANT(2, true) } else { LAUNCH_PACKED_VARIANT(2, false) }
  } else {
    if (write_sequence) { LAUNCH_PACKED_VARIANT(3, true) } else { LAUNCH_PACKED_VARIANT(3, false) }
  }

#undef LAUNCH_PACKED_VARIANT

  check_last_error();
#else
  throw std::runtime_error("MMAC packed variants require MIOPEN_ADAPTIVE_LSTM_ENABLE_MMAC_BUILTIN");
#endif

  if (!write_sequence) {
    return h_state;
  }
  return out;
}

