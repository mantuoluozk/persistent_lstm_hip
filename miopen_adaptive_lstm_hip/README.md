# MIOpen-Inspired Adaptive LSTM HIP

面向海光 DCU（K100_AI / BW150）的通用 LSTM 推理优化项目。核心设计借鉴 MIOpen RNN 动态路径的调度思想，不依赖上游 MIOpen 运行时。

目标：对任意 `hidden_size` 的 `nn.LSTM + nn.Linear` 回归结构提供持续可扩展的 HIP 后端加速。

当前重点：FP16 推理。训练、反向传播、FP32/BF16 路径暂未纳入优化范围。

## 架构

```python
from miopen_adaptive_lstm_hip import convert_regressor_module

model = LSTMRegressor().to("cuda:0").half().eval()
model = convert_regressor_module(model)
out = model(x)
```

dispatcher 根据模型结构选择后端，未命中 HIP 路径时自动回到原生 PyTorch。

内部数据流（参考 MIOpen 官方设计）：

```
workspace gate block = X @ Wx           # PropX: 输入投影 GEMM
workspace gate block += bias

for each timestep:
    workspace recur[t] += H[t-1] @ Wh   # PropHiddenHt: 循环投影 GEMM
    hidden update: gate[t] + recur[t]   # LSTMForwardHiddenStateUpdate
    write Ct, Ht
```

## 后端对比

| 后端 | 激活方式 | 适用 hidden_size | 说明 |
|------|---------|-----------------|------|
| `gemm_scan` | H>128 默认 | 全部 | rocBLAS GEMM + HIP pointwise |
| `persistent_mmac` | **H≤128 默认** | 全部 | 持久化 kernel + packed MMAC，H128 上 4.48s 最优 |
| `persistent_mmac` | `_BACKEND=persistent_mmac` | 全部 | 持久化 kernel + packed MMAC（默认自动选择） |
| `seqmajor_accum` | `_BACKEND=seqmajor_accum` | H128 | MIOpen 风格 seq-major 门控累加 |
| `cached` | `_BACKEND=cached` | H128 | 权重缓存标量 kernel（B2/B4/B8 自适应） |
| `partitioned` | `_BACKEND=partitioned` | 全部 | 分区标量点积（P4/P8 自适应） |
| `scalar` | `_BACKEND=scalar` | 全部 | 通用标量回退（最终兜底） |

### gemm_scan（默认基线）

- 输入投影：`hipblasGemmEx` 一次性计算整个序列的 gate
- 循环传播：每个 timestep `hipblasGemmEx(h_state, w_hh)`
- 隐藏更新：HIP pointwise kernel 做门控激活和细胞更新

优点：rocBLAS GEMM tensor core 吞吐最优。缺点：每层 (seq_len) 次 GEMM + pointwise 启动。

### persistent_mmac（持久化 kernel）

每层 1 次 kernel 启动。分两个子路径：

- **Packed MMAC**（默认，`MIOPEN_ADAPTIVE_LSTM_MMAC_PACKED=1`）：HCU MMAC + packed weight + wave_id + B=4 split，H128 上 4.48s
- **标量回退**（`MIOPEN_ADAPTIVE_LSTM_USE_MMAC=0`）：P4 分区标量，H128 上 ~11.2s

HCU MMAC：`__builtin_hcu_mmac_f32_16x16x16_f16`，64 线程 lane mapping `row=lane&15, col0=(lane>>4)*4`。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND` | `auto` | 后端选择（auto = H≤128→MMAC, H>128→gemm_scan） |
| `MIOPEN_ADAPTIVE_LSTM_USE_MMAC` | `1` | persistent_mmac 是否启用 MMAC（0=标量） |
| `MIOPEN_ADAPTIVE_LSTM_MMAC_PACKED` | `1` | MMAC 是否使用 packed weight |
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE` | `fp16` | 循环 GEMM 精度：`fp32`/`fp16`/`auto_fast`/`auto_aggressive` |
| `MIOPEN_ADAPTIVE_LSTM_DEBUG` | `0` | 打印后端、kernel 等诊断信息 |
| `MIOPEN_ADAPTIVE_LSTM_PROFILE` | `0` | 打印每层耗时分析 |
| `MIOPEN_ADAPTIVE_HIDDEN` | `128` | hidden_size 覆盖 |
| `MIOPEN_ADAPTIVE_BATCH` | `512` | batch_size 覆盖 |
| `MIOPEN_ADAPTIVE_ITERS` | `100` | 计时迭代次数 |

## 使用

```bash
# 编译
python setup.py build_ext --inplace

# 默认 auto（自动选最优：H128→MMAC 4.48s, H>128→gemm_scan）
python run_adaptive_lstm.py

# H256/H512（自动走 gemm_scan）
MIOPEN_ADAPTIVE_HIDDEN=256 python run_adaptive_lstm.py

# 手动覆盖后端
MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mmac python run_adaptive_lstm.py

# 调试输出
MIOPEN_ADAPTIVE_LSTM_DEBUG=1 python run_adaptive_lstm.py

# 多 shape 回归测试
python run_shape_sweep.py
MIOPEN_ADAPTIVE_SWEEP=128:512,256:512,512:128 python run_shape_sweep.py

# 对比原生基线
python run_native_lstm_sweep.py
python run_shape_sweep.py
python compare_lstm_sweeps.py --native native.log --adaptive adaptive.log
```

## 性能记录（K100_AI / gfx928，2026-05-15）

batch=512，100 次迭代。默认 `auto` 后端会自动选择最优路径。

| | H128 | H256 | H512 |
|------|------|------|------|
| **shape** | `5/128/4/24, b512, s1000` | `5/256/4/24, b512, s1000` | `5/512/4/24, b512, s1000` |
| 原生 PyTorch (fp16) | 7.63s | 11.03s | 21.70s |
| **当前最优** | **4.39s** | **7.25s** | **10.06s** |
| 最优后端 | persistent_mmac packed | gemm_scan fp16 | gate_accum fp16 |
| 提升 vs 原生 | **-42%** | **-34%** | **-54%** |
| 精度 max_abs | ~0.020 | ~0.010 | ~0.008 |

### 优化历程

### H128 MMAC 优化

| 阶段 | H128 耗时 | 关键突破 |
|------|----------|---------|
| HCU MMAC 初版 | 144s | 打通 `__builtin_hcu_mmac_*`，但 weight 读反、OOB 等 bugs |
| 关键修复 + wave_id | ~11s | weight 修正、A-reuse、wave_id 消除 4x wavefront 冗余 |
| Packed weight + Split-B | **4.79s** | packed layout、B=4 grid=128，反超 gemm_scan |
| 双缓冲 + 减少 sync | **4.40s** | h_state 指针交换、syncthreads 2→1/timestep |

### H256/H512 优化

| 阶段 | H256 | H512 | 关键突破 |
|------|------|------|---------|
| Multi-size 参数化 | 15.95s | 50.60s | HiddenSize 模板，MMAC 通了但慢于 gemm_scan |
| gemm_scan fp16 | 7.25s | 32.51s | 循环 GEMM 从 fp32→fp16 |
| gate_accum fp16 | — | **10.06s** | H512 专属，累加式 GEMM + fp16 |

已回退：Weight 驻留 (8.78s)、Register-direct (7.95s)、LDS B-tile staging (8.36s)、8-wavefront、MMAC B=8、MinBlocksPerCU

## 优化参考：可借鉴技术

### 官方 MIOpen 架构要点

数据流的核心设计：循环 GEMM 直接累加到 `gate[t]`（beta=1），hidden update kernel 只读 `gate[t]`，不需要独立的 `recur` tensor。Gate workspace 是 seq-major `[T, B, 4H]`。

官方 CK（Composable Kernel）的 GEMM pipeline 设计：
- **LDS 驻留权重 tile**：权重 tile 加载到 LDS 后跨多次 GEMM 复用
- **双缓冲（Pipeline v2）**：global read 和 GEMM 计算重叠
- **XDLops/MFMA**：使用 AMD 矩阵指令做 tile 级 GEMM
- **波前级驻留控制**：通过 `__launch_bounds__` 控制寄存器压力

### MMAC 技术备忘

DTK `dcc 25.10.0-0`（clang 17）上已穿透的正确 intrinsic 调用：

```cpp
using f16x4 = _Float16 __attribute__((vector_size(8)));
using f32x4 = float    __attribute__((vector_size(16)));

__device__ __forceinline__
f32x4 mmac(f16x4 a, f16x4 b, f32x4 c) {
    return __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);
}
```

约束：
- 不能用 `__fp16` 作为函数参数/返回值 → 用 `_Float16` vector
- 不能用 `float[4]` 作为 MFMA 参数 → 用 native vector type
- 返回值是 `f32x4`，不是 void
- HCU MMAC 使用 64 线程，lane mapping：`row=lane&15, col0=(lane>>4)*4`

其他 MFMA 变体（32x32x8f16、16x16x4f16 等）在 DTK 上均不可用。

## 开发规则

实现新功能时遵循以下规则（源自 MIOpen 官方架构）：

1. **Gate accumulation**：循环 GEMM 应累加到 gate workspace（beta=1），避免独立的 `recur` tensor
2. **Hidden update 只读 gate**：pointwise kernel 不负责 `recur + bias`，只消费已累加的 gate block
3. **GEMM 优先**：用 rocBLAS GEMM 做传播，不手写标量循环 matmul 作为默认路径
4. **不要**：大量 hidden-size 专用 kernel、改数学布局不做 A/B 对比、依赖 CUDA graph 不做验证、优化非热点路径

## 目录结构

```
miopen_adaptive_lstm_hip/
├── README.md
├── setup.py
├── run_adaptive_lstm.py           # 单 shape 测试
├── run_shape_sweep.py             # 多 shape 回归
├── run_profile_sweep.py           # 性能分析
├── run_compute_sweep.py           # 精度模式对比
├── csrc/
│   ├── adaptive_lstm_hip.cu       # 核心 kernel（含 MMAC persistent kernel）
│   ├── adaptive_lstm_hip.h
│   ├── adaptive_lstm_pipeline.h
│   └── bindings.cpp
├── miopen_adaptive_lstm_hip/
│   ├── model.py                   # 模型转换和 dispatch
│   ├── api.py / extension.py
│   ├── pipeline.py / scheduler.py / selector.py / descriptors.py / modular.py
│   └── official_refs.py
└── tests/
```

## 路线图

已完成：
- ✅ HCU MMAC 打通、wavefront 并行、LDS bank conflict 修复
- ✅ Weight pre-packing + Split-B (B=4) + 双缓冲 + 减少 syncthreads
- ✅ Multi-size 参数化（H128/H256/H512）
- ✅ Shape-aware 自动后端 + fp16 recurrent 默认
- ✅ H128/H256/H512 全线超越原生 PyTorch

待探索：
- ☐ 线性头融合
- ☐ 4 层融合 persistent kernel
- ☐ H256/H512 大 batch 场景进一步优化
- ☐ BF16/FP32、训练模式
