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

| 后端 | 环境变量 | 说明 |
|------|---------|------|
| `gemm_scan` | 默认 | rocBLAS GEMM + HIP pointwise，稳定基线 |
| `persistent_mmac` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mmac` | 持久化 kernel：每层 1 次启动，MFMA 子路径已超越 gemm_scan |
| `seqmajor_accum` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=seqmajor_accum` | 官方 MIOpen 风格 seq-major 门控累加（实验性） |
| `cached` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=cached` | H128 权重缓存 kernel，标量（实验性） |
| `scalar` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=scalar` | 通用标量回退 |

### gemm_scan（默认基线）

- **输入投影**：`hipblasGemmEx` 一次性计算整个序列的 gate
- **循环传播**：每个 timestep `hipblasGemmEx(h_state, w_hh)`
- **隐藏更新**：HIP pointwise kernel 做门控激活和细胞更新

优点：rocBLAS GEMM（tensor core）吞吐最优。缺点：每层 1000 次 GEMM + 1000 次 pointwise 启动。

### persistent_mmac（持久化 kernel，当前最优）

每层仅 1 次 kernel 启动，消除 per-timestep 启动开销。分两个子路径：

- **标量子路径**：P4 分区标量点积（与 cached_b4 同款运算），~8.3s
- **MFMA 子路径**：HCU MMAC 矩阵指令加速，**~7.3s，超越 gemm_scan**

HCU MMAC 关键发现：
- K100_AI 使用海光自有 `__builtin_hcu_mmac_f32_16x16x16_f16`（非 AMD amdgcn_mfma）
- 签名：`f32x4 mmac(f16x4 a, f16x4 b, f32x4 c)` — 返回值是 `f32x4`
- HCU MMAC 使用 64 线程（AMD MFMA 用 32），每线程 4 个 fp16 值
- 不需要 `+mai-insts` 编译 flag，gfx928 原生支持
- 通过 `MIOPEN_ADAPTIVE_LSTM_USE_MMAC=1` 启用

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND` | `gemm_scan` | 后端选择 |
| `MIOPEN_ADAPTIVE_LSTM_USE_MMAC` | `0` | persistent_mmac 是否启用 MFMA |
| `MIOPEN_ADAPTIVE_LSTM_FAST_ACT` | `0` | 启用 `exp2f` 快速激活 |
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE` | `fp32` | 循环 GEMM 精度：`fp32`/`fp16`/`auto_fast`/`auto_aggressive` |
| `MIOPEN_ADAPTIVE_LSTM_DEBUG` | `0` | 打印后端、kernel 等诊断信息 |
| `MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS` | `1` | 直接 hipBLAS GEMM（绕过 PyTorch 调度） |
| `MIOPEN_ADAPTIVE_LSTM_PROFILE` | `0` | 打印每层耗时分析 |

## 使用

```bash
# 编译
cd miopen_adaptive_lstm_hip
python setup.py build_ext --inplace

# 默认 gemm_scan
python run_adaptive_lstm.py

# persistent_mmac + MFMA（最优）
MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mmac \
MIOPEN_ADAPTIVE_LSTM_USE_MMAC=1 \
python run_adaptive_lstm.py

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

## 性能记录（K100_AI / gfx928，2026-05-14）

默认 shape：`input=5, hidden=128, layers=4, output=24, seq_len=1000, batch=512`（100 次迭代）

| 路径 | 耗时 | 吞吐 | 说明 |
|------|------|------|------|
| **persistent_mmac packed (split-B)** | **5.63s** | **9094 samples/s** | **batch_tile=8, grid=64, 反超 gemm_scan 24%** |
| gemm_scan（默认） | ~7.4s | ~6900 samples/s | rocBLAS GEMM tensor core |
| persistent_mmac packed (batch_tile=16) | 7.54s | 6793 samples/s | wave_id + packed weight |
| persistent_mmac 标量 | 11.16s | 4588 samples/s | batch_tile=4, grid=128 |
| 原生 PyTorch LSTM | ~8.0s | ~6400 samples/s | 基线（extension 加载失败时回退） |

### MMAC 优化迭代

| 迭代 | MMAC | gemm_scan | 关键改动 |
|------|------|-----------|---------|
| 初版 | 144s | 5.6s | batch_tile=4 OOB + 读反 weight + K-loop 未累加 |
| 四修复 | 7.55s | 8.05s | weight 布局修正、K-loop 寄存器累加、batch_tile=16、按需加载 |
| A-reuse | 7.34s | 7.65s | h_lds tile 每 K-tile 只加载 1 次，4 gate 共享 |
| Weight 驻留 | 8.78s | 7.72s | K-tile 外移导致同步开销增大（已回退） |
| LDS bank conflict | 26.05s | 6.40s | h_lds/recur stride 加 padding，消除 16-way conflict (+4.1%) |
| Wavefront 并行 | 11.43s | 6.40s | wave_id 分配 4 波前到 4 个 H-tile，消除 4x 重复计算 (-56.1%) |
| Weight pre-packing | 7.54s | 6.40s | packed [htile][ktile][krow][ngroup][gate][frag] + wave_id (-34.2%) |
| **Split-B (batch_tile=8)** | **5.63s** | **7.44s** | **grid 32→64，MMAC 反超 gemm_scan 24%** |

## 优化参考：可借鉴技术

### 官方 MIOpen 架构要点

数据流的核心设计：循环 GEMM 直接累加到 `gate[t]`（beta=1），hidden update kernel 只读 `gate[t]`，不需要独立的 `recur` tensor。Gate workspace 是 seq-major `[T, B, 4H]`。

官方 CK（Composable Kernel）的 GEMM pipeline 设计：
- **LDS 驻留权重 tile**：权重 tile 加载到 LDS 后跨多次 GEMM 复用
- **双缓冲（Pipeline v2）**：global read 和 GEMM 计算重叠
- **XDLops/MFMA**：使用 AMD 矩阵指令做 tile 级 GEMM
- **波前级驻留控制**：通过 `__launch_bounds__` 控制寄存器压力

### persistent_lstm_hip 可借鉴技术

| 优先级 | 技术 | 预期收益 | 说明 |
|--------|------|---------|------|
| P0 | Uniform batch fast path | 极大 | 检测全批次相同输入，只算 1 行 + 广播 |
| P1 | 线性头融合 | 小 | `torch.mm + add_` 融合进最后 LSTM kernel |
| P1 | MFMA（已实现） | 大 | HCU MMAC 矩阵指令 |
| P2 | 权重 pre-packing | 中 | 从 `[4H, H]` 重排为 `[H/2, 4H, 2]`（k-pair 布局） |
| P2 | 4 层融合 kernel | 中 | 所有层融合到单个 kernel |
| P3 | FP32 LDS 隐藏状态 | 小 | 避免每步 `half_to_float` 转换 |
| P3 | 形状感知调度 | 小 | 根据输入 shape 自动选后端 |

### MFMA 技术备忘

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
├── hcu builtin汇总 - 260425.xlsx   # HCU MMAC 指令表
├── csrc/
│   ├── adaptive_lstm_hip.cu       # 核心 kernel（含 MFMA persistent kernel）
│   ├── adaptive_lstm_hip.h
│   ├── adaptive_lstm_pipeline.h
│   └── bindings.cpp
├── miopen_adaptive_lstm_hip/
│   ├── model.py                   # 模型转换和 dispatch
│   ├── api.py / extension.py
│   ├── pipeline.py / scheduler.py / selector.py / descriptors.py / modular.py
│   └── official_refs.py
├── third_party_refs/              # MIOpen/CK 官方参考源码
│   ├── ck/
│   └── *.cpp
└── tests/
```

## 路线图

1. ✅ HCU MMAC 打通：`__builtin_hcu_mmac_f32_16x16x16_f16` 编译运行，超越 gemm_scan
2. ✅ Wavefront 重复计算修复：wave_id 分配 4 波前并行处理 4 个 H-tile，2.3x 加速
3. ✅ LDS bank conflict 修复：padding stride 消除 16-way conflict
2. MFMA kernel 进一步优化：
   - 权重跨 timestep 驻留（需解决 K-tile 外移同步开销）
   - 隐藏单元并行
   - Fast sigmoid/tanh
3. 扩展到 H256/H512
4. Uniform batch fast path
5. 线性头融合
6. 后续考虑 BF16/FP32、训练模式
