# MIOpen-Inspired Adaptive LSTM HIP

面向海光 DCU（K100_AI / BW150）的通用 LSTM 推理优化项目。核心设计借鉴 MIOpen RNN 动态路径的调度思想，不依赖上游 MIOpen 运行时。

目标：对任意 `hidden_size` 的 `nn.LSTM + nn.Linear` 回归结构提供持续可扩展的 HIP 后端加速。

当前重点：FP16 推理。训练、反向传播、FP32/BF16 路径暂未纳入优化范围。

## 架构

对上层代码入口简单：

```python
from miopen_adaptive_lstm_hip import convert_regressor_module

model = LSTMRegressor().to("cuda:0").half().eval()
model = convert_regressor_module(model)
out = model(x)
```

dispatcher 根据模型结构选择后端。没有命中 HIP 专用路径时，模型自动回到原生 PyTorch，不会因 shape 不匹配而不可用。

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
| `persistent_mfma` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mfma` | 持久化 kernel：每层 1 次启动。MFMA 子路径已超越 gemm_scan |
| `seqmajor_accum` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=seqmajor_accum` | 官方 MIOpen 风格的 seq-major 门控累加路径（实验性） |
| `cached` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=cached` | H128 权重缓存 kernel（标量，实验性） |
| `scalar` | `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=scalar` | 通用标量回退路径 |

### gemm_scan（默认，当前最快）

每层分三步：
1. **输入投影**：`hipblasGemmEx` 一次性计算整个序列的 gate
2. **循环传播**：每个 timestep 调用 `hipblasGemmEx(h_state, w_hh)` 
3. **隐藏状态更新**：小型 HIP pointwise kernel 做门控激活和细胞更新

优点：利用 rocBLAS GEMM（内部使用 tensor core），矩阵乘吞吐最优。
缺点：每层 1000 次 GEMM + 1000 次 pointwise 启动（seq_len=1000 时），启动开销大。

支持通过 `MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE` 控制循环 GEMM 精度：
- `fp32`（默认）：FP32 累加，精度最高
- `auto_aggressive`：除最后一层外全部 FP16 累加，速度最快
- `auto_fast`：仅第一层 FP16 累加，较安全的速度模式
- `fp16_layers:0,1`：指定层 FP16 累加

支持 `MIOPEN_ADAPTIVE_LSTM_FAST_ACT=1` 启用 `exp2f` 快速激活函数（以精度换速度）。

### persistent_mfma（持久化 kernel）

**设计目标**：每层仅 1 次 kernel 启动，消除 per-timestep 的 GEMM + pointwise 启动开销。

**标量子路径**（当前可用）：
- 使用 P4 分区标量点积（与已验证的 cached_b4 kernel 相同运算方式）
- LDS 驻留 hidden state 和 cell state
- 精度正确：max_abs ≈ 6.1e-05
- 性能：约 11.2s（gemm_scan 基线约 5.7s）— 标量 matmul 无法与 rocBLAS GEMM 竞争

**MFMA 子路径**（实验性）：
- 使用 `__builtin_amdgcn_mfma_f32_16x16x16f16` 矩阵指令加速循环点积
- DTK 上的正确签名：`f32x4 mfma(f16x4 a, f16x4 b, f32x4 c, 0, 0, 0)`
- 通过 `-Xclang=-target-feature -Xclang=+mai-insts` 启用编译
- 代码已置于 `#ifdef MIOPEN_ADAPTIVE_LSTM_ENABLE_MFMA_BUILTIN` 保护下
- 通过 `MIOPEN_ADAPTIVE_LSTM_USE_MFMA=1` 运行时启用

**MFMA 在 K100_AI (gfx928) 上的突破**（2026-05-13）：
- K100_AI 使用海光自有 `__builtin_hcu_mmac_f32_16x16x16_f16`（非 AMD amdgcn_mfma）
- 签名：`f32x4 __builtin_hcu_mmac_f32_16x16x16_f16(f16x4 a, f16x4 b, f32x4 c)`
- HCU MMAC 使用 **64 线程**（非 AMD 的 32），每线程 4 个 fp16 值
- 不需要 `+mai-insts` 编译 flag，gfx928 原生支持
- 已实现完整的 persistent MFMA kernel，7.55s 超越 gemm_scan 8.05s
- 通过 `MIOPEN_ADAPTIVE_LSTM_USE_MFMA=1` 运行时启用

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND` | `gemm_scan` | 后端选择：`gemm_scan`、`persistent_mfma`、`seqmajor_accum`、`cached`、`scalar` |
| `MIOPEN_ADAPTIVE_LSTM_USE_MFMA` | `0` | persistent_mfma 后端是否使用 MFMA 子路径 |
| `MIOPEN_ADAPTIVE_LSTM_FAST_ACT` | `0` | 是否使用 `exp2f` 快速激活函数 |
| `MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE` | `fp32` | 循环 GEMM 精度：`fp32`、`fp16`、`auto_fast`、`auto_aggressive` 等 |
| `MIOPEN_ADAPTIVE_LSTM_DEBUG` | `0` | 打印命中的后端、bucket、kernel 等信息 |
| `MIOPEN_ADAPTIVE_LSTM_GEMM_SCAN_READ_BLOCK` | `auto` | gemm_scan pointwise kernel 的 READ_BLOCK 参数 |
| `MIOPEN_ADAPTIVE_LSTM_DIRECT_BLAS` | `1` | 是否使用直接 hipBLAS GEMM（绕过 PyTorch 调度） |
| `MIOPEN_ADAPTIVE_LSTM_PROFILE` | `0` | 打印每层输入投影和循环扫描耗时 |

## 使用

编译：

```bash
cd miopen_adaptive_lstm_hip
python setup.py build_ext --inplace
```

运行默认 gemm_scan：

```bash
python run_adaptive_lstm.py
```

运行 persistent_mfma 标量版：

```bash
MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mfma \
MIOPEN_ADAPTIVE_LSTM_DEBUG=1 \
python run_adaptive_lstm.py
```

打印 debug 信息：

```bash
MIOPEN_ADAPTIVE_LSTM_DEBUG=1 python run_adaptive_lstm.py
```

多 shape 回归测试：

```bash
python run_shape_sweep.py
```

自定义 sweep shape：

```bash
MIOPEN_ADAPTIVE_SWEEP=128:512,256:512,512:128 python run_shape_sweep.py
```

对比原生 PyTorch 基线：

```bash
python run_native_lstm_sweep.py
python run_shape_sweep.py
python compare_lstm_sweeps.py --native native.log --adaptive adaptive.log
```

## 性能记录（K100_AI / gfx928，2026-05-13）

默认 shape：`input=5, hidden=128, layers=4, output=24, seq_len=1000, iterations=100, batch=512`

| 路径 | 耗时 | 吞吐 | 说明 |
|------|------|------|------|
| 原生 PyTorch LSTM | ~8.56s | ~5984 samples/s | 基线 |
| gemm_scan（默认） | 7.65s | 6697 samples/s | rocBLAS GEMM |
| persistent_mfma（标量） | 8.31s | 6167 samples/s | P4 标量 |
| **persistent_mfma（MFMA）** | **7.34s** | **6982 samples/s** | **HCU MMAC 最优** |

### MFMA 优化迭代记录

| 迭代 | MFMA 耗时 | gemm_scan | 关键改动 |
|------|----------|-----------|---------|
| 初版 | 144s (bug) | 5.6s | batch_tile=4 OOB + 读反 weight + K-loop 未累加 |
| 四修复 | 7.55s | 8.05s | weight 布局修正、K-loop 寄存器累加、batch_tile=16、按需加载 weight |
| A fragment 复用 | 7.34s | 7.65s | h_lds tile 每 K-tile 只加载 1 次，4 gate 共享 |
| Weight 跨 timestep | 8.78s (回退) | 7.72s | K-tile 外移导致同步开销增大，实验性保留 |

多 shape 对比（gemm_scan vs 原生 LSTM）：

| Shape | 原生耗时 | gemm_scan | 加速比 |
|-------|---------|-----------|--------|
| H128 B512 | 8.56s | 5.66s | 1.51x |
| H256 B512 | 11.00s | — | — |
| H512 B128 | 12.69s | — | — |

注：H256/H512 数据待补充。

## MFMA 调查记录

DTK 编译器版本：`dcc 25.10.0-0`（基于 clang 17.0.0）

**正确 intrinsic 签名**（已通过最小测试验证）：

```cpp
using f16x4 = _Float16 __attribute__((vector_size(8)));   // 4 x fp16
using f32x4 = float    __attribute__((vector_size(16)));  // 4 x fp32

f32x4 __builtin_amdgcn_mfma_f32_16x16x16f16(
    f16x4 a,     // A: 4 个 _Float16（8 字节）
    f16x4 b,     // B: 4 个 _Float16（8 字节）
    f32x4 c,     // C: 4 个 float（16 字节）— fp32 累加器
    int cbsz,    // 控制参数（传 0）
    int abid,    // 控制参数（传 0）
    int blgp     // 控制参数（传 0）
);
```

关键点：
- 返回值是 `f32x4`（不是 void），不需要传 dest 指针
- A/B 用 `_Float16` 向量（不能用 `float[4]` 或 `__fp16` 标量）
- C 用 `float` 向量（fp32 累加）
- gfx928 需要 `-Xclang=-target-feature -Xclang=+mai-insts` 编译 flag

**其他 MFMA 变体测试结果**：
- `32x32x8f16`：编译失败（DTK 要求 `__fp16` 向量作为累加器，不支持 fp32）
- `16x16x4f16`：同上
- `4x4x4f16`：同上
- `32x32x2f16`：DTK 不支持此 intrinsic
- inline asm `v_mfma_f32_16x16x16f16`：汇编器直接报 `instruction not supported on this GPU`
- `gfx928+mai-insts` target ID：DTK 不支持此格式

## 目录结构

```
miopen_adaptive_lstm_hip/
├── setup.py
├── README.md
├── BENCHMARK_BASELINE.md
├── OFFICIAL_ARCHITECTURE_MAP.md
├── OFFICIAL_IMPLEMENTATION_GUIDE.md
├── run_adaptive_lstm.py          # 单 shape 测试
├── run_shape_sweep.py            # 多 shape 回归测试
├── run_profile_sweep.py          # 性能分析
├── run_compute_sweep.py          # 精度模式对比
├── csrc/
│   ├── adaptive_lstm_hip.cu      # 核心 kernel 实现
│   ├── adaptive_lstm_hip.h      # kernel 声明
│   ├── adaptive_lstm_pipeline.h  # Pipeline trait 定义
│   └── bindings.cpp              # PyBind11 绑定
├── miopen_adaptive_lstm_hip/
│   ├── __init__.py
│   ├── api.py                    # 公开 API
│   ├── model.py                  # 模型转换和 dispatch
│   ├── extension.py              # C++ 扩展加载
│   ├── pipeline.py               # Pipeline / kernel 计划
│   ├── scheduler.py              # 动态算法选择
│   ├── selector.py               # MIOpen 风格 selector
│   ├── descriptors.py            # 描述符定义
│   ├── modular.py                # 模块化 forward 规划
│   └── official_refs.py          # 官方参考映射
└── third_party_refs/             # MIOpen/CK 官方参考代码
    ├── ck/                       # Composable Kernel 参考
    └── *.cpp                     # MIOpen RNN 参考
```

## 路线图

1. ✅ HCU MMAC 打通：`__builtin_hcu_mmac_f32_16x16x16_f16` 编译运行正常，超越 gemm_scan
2. MFMA kernel 进一步优化：
   - 权重跨 timestep 驻留 LDS（消除每 K-tile 的全局内存重读）
   - 隐藏单元并行（不同 h-tile 在不同 block 执行）
   - 更大 batch_tile 或虚拟 batch 优化 GPU 占用率
3. 扩展到 H256/H512（调整 tile size 和 LDS 分配）
4. Uniform batch fast path（全批次相同输入场景，`torch.ones()` 测试）
5. 线性头融合（省一次 GEMM 启动）
6. 后续考虑 BF16/FP32、训练模式、更多硬件上的自动选择
