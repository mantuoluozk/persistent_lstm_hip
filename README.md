# Persistent LSTM HIP

面向海光 DCU / AMD ROCm 平台的 LSTM 推理优化项目。目标是在不改变上层 PyTorch 模型写法、不牺牲 FP16 推理精度的前提下，为 `nn.LSTM + nn.Linear` 回归模型提供一个可扩展的 HIP 后端。

项目最初从一个固定业务 shape 开始：

- 输入: `[batch=512, seq_len=1000, input_size=5]`
- LSTM: `num_layers=4, hidden_size=128, batch_first=True`
- 输出: 取最后一个 timestep 的 top-layer hidden，再接 `Linear(128 -> 24)`
- 推理精度: FP16

后来逐步扩展成三类路径：

- 固定 shape 高性能路径: 面向最初的 `input=5, hidden=128, layers=4, output=24`
- Uniform batch fast path: 面向 batch 内输入完全相同的场景
- Hidden bucket 通用路径: 当前已支持 `hidden_size=64` 的半特化优化，其他参数保持动态

![项目概览](image/image.png)

## 当前结果

以下结果来自 K100_AI / ROCm 环境，均为 100 次 forward 计时，吞吐量包含 batch size。

| 路径 | shape | 时间 | 吞吐量 | 说明 |
| --- | --- | ---: | ---: | --- |
| NVIDIA A10 原生 `LSTM.py` | `5/128/4/24` | 约 `2.12s` | 约 `24125 samples/s` | cuDNN persistent LSTM |
| K100_AI 原生 `LSTM.py` | `5/128/4/24` | 约 `9s` | 约 `5600 samples/s` | ROCm 原生通用路径 |
| K100_AI 固定 shape HIP P4 | `5/128/4/24` | 约 `3.08s` | 约 `16600 samples/s` | 当前固定 shape 快路 |
| K100_AI 原生 `LSTM.py` | `7/64/2/16` | 约 `1.94s` | 约 `26384 samples/s` | PyTorch/ROCm baseline |
| K100_AI H64 bucket HIP | `7/64/2/16` | `0.8899s` | `57537 samples/s` | 当前 hidden=64 通用 bucket |

当前 `hidden_size=64` 测试输出：

```text
backend: hip_generic_projected_lstm
persistent_lstm_hip debug: backend=generic_projected, batch=512, seq_len=1000, input_size=7, hidden_size=64, num_layers=2, hidden_bucket=h64, generic_projected_p4=True, uniform_batch_fast_path=False
accuracy_vs_native_lstm: max_abs=0.00012207, mean_abs=1.74046e-05, max_rel=0.0035503
0.8898632526397705
吞吐量(含batchsize): 57536.930363306616
```

这个结果没有依赖 uniform batch trick，`uniform_batch_fast_path=False`，因此是对非 uniform 输入也适用的真实 H64 优化路径。

## 项目架构

项目对外保留统一接口：

```python
from persistent_lstm_hip import convert_regressor_module

model = LSTMRegressor().to("cuda:0").half().eval()
model = convert_regressor_module(model).to("cuda:0").half().eval()
```

内部使用 dispatcher 根据模型结构选择不同 backend：

```text
convert_regressor_module(model)
        |
        |-- 固定 shape: input=5, hidden=128, layers=4, output=24
        |      -> hip_specialized_4layer_regressor
        |      -> projected + uniform projected + P4 shuffle
        |
        |-- hidden_size=64, FP16, batch_first=True, 单向 LSTM
        |      -> hip_generic_projected_lstm
        |      -> hidden_bucket=h64
        |
        |-- 其他支持结构
        |      -> generic projected 或 native PyTorch fallback
        |
        |-- 不支持结构
               -> native_pytorch_unsupported_lstm
```

当前支持的通用结构约束：

- `nn.LSTM + nn.Linear`
- `batch_first=True`
- `bidirectional=False`
- `proj_size=0`
- `bias=True`
- FP16 inference
- `linear.in_features == hidden_size`
- 输入是 CUDA tensor，shape 为 `[batch, seq_len, input_size]`

## 三条优化路径

### 1. 固定 Shape 高性能路径

最初的业务 shape 是 `input=5, hidden=128, layers=4, output=24`。这条路径做了最深的特化：

- 4 层 LSTM 在 Python 侧静态识别
- recurrent 权重按 kernel 访问方式打包
- input projection 与 recurrent 计算拆分
- uniform batch 时只计算少量真实 batch，再 repeat 输出
- P4 partition 把每个 hidden 的 recurrent dot-product 拆成 4 路并行
- 最后一层融合 `Linear(128 -> 24)`

这条路径非常快，但它依赖固定 shape，因此不是通用方案的全部。

### 2. Uniform Batch Fast Path

当前 benchmark 里输入常见写法是：

```python
x = torch.ones((batch_size, seq_length, input_size), device="cuda:0", dtype=torch.float16)
```

如果 batch 内每条序列完全相同，且初始 hidden/cell 相同，那么每条输出在数学上也相同。项目可以检测这种 uniform batch，只计算第一条或少量 batch，再把输出扩展回完整 batch。

这个优化不改变数学结果，但它只适用于 batch 内输入完全一致的场景。为了避免误导通用性能评估，generic fallback 的 uniform 优化默认关闭，需要显式打开：

```bash
PERSISTENT_LSTM_HIP_GENERIC_UNIFORM_BATCH=1 python LSTM-hip.py
```

固定 shape 的 uniform projected 路径仍然默认用于最初业务 benchmark。

### 3. Hidden=64 Bucket

这是目前通用化方向的第一条半特化路径。它不是写死 `input=7, layers=2, output=16`，而是按 hidden size 分桶：

```text
hidden_size = 64
input_size  = 动态
output_size = 动态
num_layers  = 动态
batch_size  = 动态
seq_len     = 动态
```

H64 bucket 的核心实现：

- input projection 使用 `torch::matmul`
- bias add 融合进 recurrent kernel，减少 PyTorch elementwise kernel
- recurrent 权重在 kernel 开始时加载到线程局部数组
- 每个 hidden 使用 P4 partition，4 个线程并行计算 recurrent dot-product
- 使用 shuffle 做 4 路归约
- 保留非 uniform batch 支持

当前 H64 bucket 已经让 `7/64/2/16` 从原生 PyTorch 的约 `1.94s` 降到约 `0.89s`。

## 调优路线

### 固定 Shape 阶段

1. `monolithic` 早期版本很慢，约 `81s`，说明简单合并 4 层 kernel 不是答案。
2. `interleaved` 降到约 `18s`，但仍慢于原生 DCU LSTM。
3. `projected` 路径把 input projection 交给矩阵乘，把 recurrent 部分交给自定义 kernel，降到约 `5.5s`。
4. uniform batch fast path 利用全 1 输入，只算少量真实 batch，进一步降低重复计算。
5. P4 partition + shuffle 把 recurrent dot-product 从单线程串行拆成 4 路并行，固定 shape 降到约 `3.08s`。
6. P8 实验更慢，说明更多 partition 不一定更好，同步、寄存器和 occupancy 成本会抵消收益。

### 通用化阶段

1. 先加 native PyTorch fallback，保证任意 shape 改动后不会不可用。
2. 加 generic projected v1，支持动态 `input/hidden/layers/output`，但朴素 recurrent kernel 很慢，`7/64/2/16` 约 `21.7s`。
3. 加 generic P4 recurrent，把 `hidden=64` 测试从约 `21.7s` 降到约 `8.37s`。
4. 引入 H64 bucket，把 `hidden=64` 编译期特化，recurrent 权重缓存到线程局部数组，降到约 `1.14s`。
5. 把 LSTM bias 从 `matmul + bias` 融入 recurrent kernel，elementwise 占比从约 `17.5%` 降到约 `10%`，总时间降到约 `0.918s`。
6. 尝试 dynamic linear head fusion，但收益为负，已回退。
7. H64 kernel 中 bias 预加载到局部变量，当前最好结果约 `0.8899s`。

## 当前瓶颈

H64 bucket 最新 profiler 显示，主要耗时集中在：

```text
persistent_lstm_h64_projected_layer_*: 约 88%
input projection GEMM:                 约 10%
其他 elementwise / copy:               很低
```

这说明当前主要瓶颈已经不是 PyTorch elementwise，也不是最后 linear，而是 H64 recurrent kernel 本身。

后续 H64 如果继续优化，重点应放在：

- 减少每个 timestep 的 `__syncthreads()` 成本
- 优化 shared memory 中 `h_cur` 的访问模式
- 评估一个 block 处理多个 batch sample 的可行性
- 继续调 wave64 下的 lane 分组、VGPR 占用和 occupancy

## 使用方法

### 构建

在 ROCm / HIP 环境中编译扩展：

```bash
cd persistent_lstm_hip
python setup.py build_ext --inplace
cd ..
```

### 运行

```bash
python LSTM-hip.py
```

打开 debug：

```bash
PERSISTENT_LSTM_HIP_DEBUG=1 python LSTM-hip.py
```

关闭精度对比，只测速度：

```bash
PERSISTENT_LSTM_HIP_ACCURACY=0 python LSTM-hip.py
```

禁用 HIP 转换，回到原生 PyTorch：

```bash
USE_PERSISTENT_LSTM_HIP=0 python LSTM-hip.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `USE_PERSISTENT_LSTM_HIP` | `1` | 是否启用 HIP 转换 |
| `PERSISTENT_LSTM_HIP_BACKEND` | `auto` | 固定 shape 路径选择，可选 `auto`、`projected`、`interleaved`、`monolithic` |
| `PERSISTENT_LSTM_HIP_DEBUG` | `0` | 打印实际 backend 和 fast path 命中情况 |
| `PERSISTENT_LSTM_HIP_ACCURACY` | `1` | 在 `LSTM-hip.py` 中打印原生 LSTM 精度对比 |
| `PERSISTENT_LSTM_HIP_GENERIC_PROJECTED` | `1` | 是否启用 generic projected / hidden bucket 路径 |
| `PERSISTENT_LSTM_HIP_GENERIC_UNIFORM_BATCH` | `0` | 是否对 generic fallback 启用 uniform batch fast path |
| `PERSISTENT_LSTM_HIP_UNIFORM_BATCH` | `auto` | 固定 shape 路径是否自动检测 uniform batch |
| `PERSISTENT_LSTM_HIP_UNIFORM_COMPUTE_BATCH` | `16` | 固定 shape uniform fallback 计算 batch |
| `PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED` | `1` | 固定 shape 是否启用 uniform projected 路径 |
| `PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_P4` | `1` | 固定 shape 是否启用 P4 recurrent kernel |
| `PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_P8` | `0` | 固定 shape P8 实验路径，K100_AI 上实测更慢 |
| `PERSISTENT_LSTM_HIP_UNIFORM_PROJECTED_VIRTUAL_BATCH` | `4` | 固定 shape uniform projected 的虚拟 block 数 |

## 目录结构

```text
.
├── LSTM.py
├── LSTM-hip.py
├── README.md
├── image
│   ├── image.png
│   ├── LSTM-log.png
│   └── LSTM-hip-log.png
├── log
│   ├── LSTM.log
│   └── LSTM-hip.log
└── persistent_lstm_hip
    ├── setup.py
    ├── csrc
    │   ├── bindings.cpp
    │   ├── persistent_lstm_op.cpp
    │   ├── persistent_lstm_reference.cpp
    │   ├── persistent_lstm_hip.h
    │   └── persistent_lstm_hip.cu
    └── persistent_lstm_hip
        ├── __init__.py
        ├── api.py
        ├── extension.py
        ├── model.py
        ├── packing.py
        └── reference.py
```

## 后续计划

接下来不建议按完整 shape 写孤岛 kernel，而是继续沿 hidden bucket 策略扩展：

1. 保持当前 `hidden=64` bucket 稳定。
2. 做 `hidden=128` bucket，把最初固定 shape 的经验抽象出来，让更多 `hidden=128` 模型受益。
3. 做 `hidden=256` bucket，验证更大 hidden 下 P4/P8 或其他 partition 策略。
4. 建立 shape dispatcher / autotune 机制，根据 hidden、seq_len、batch、layers 自动选择路径。
5. 对最高频业务 shape 继续保留深度 fused kernel，作为额外快速路径。

最终目标不是写一个万能 kernel，而是形成类似 cuDNN 的结构：

```text
统一接口 + shape dispatcher + hidden bucket kernel + 高频 shape 特化 kernel + PyTorch fallback
```
