# Persistent LSTM HIP Skeleton

这个目录不是又一版 Triton 试验，而是一套面向 `ROCm / HIP / PyTorch extension` 的自定义算子骨架，目标是把当前

- `4-layer LSTM`
- `hidden_size = 128`
- `input_size = 5`
- `output_size = 24`
- 只需要 `top-layer last hidden`

这条推理路径逐步迁到真正的底层 persistent kernel 上。

## 设计目标

参考 `baidu-research/persistent-rnn` 的核心思想：

1. recurrent weights 在时间循环外加载
2. hidden / cell state 在 kernel 内跨 timestep 复用
3. 尽量减少中间 `[B, T, H]` 的写回
4. 尽量把层间融合做在底层 kernel 里，而不是 Python 图里

## 当前目录结构

- `persistent_lstm_hip/`
  - `__init__.py`
  - `extension.py`
  - `model.py`
  - `packing.py`
  - `reference.py`
- `csrc/`
  - `bindings.cpp`
  - `persistent_lstm_op.cpp`
  - `persistent_lstm_reference.cpp`
  - `persistent_lstm_hip.h`
  - `persistent_lstm_hip.cu`
- `benchmark_persistent_lstm.py`
- `setup.py`
- `pyproject.toml`

## 当前状态

这套代码的重点是把接口、打包格式和 HIP kernel 入口搭好。

其中：

- Python 侧可以直接从标准 `nn.LSTM + Linear` 模型导出权重
- C++ 侧已经有一个可工作的 reference forward 路径
- HIP 文件已经预留了真正的 persistent kernel 入口和 launch 参数
- 目前 `persistent_lstm_hip.cu` 默认仍回退到 reference forward

也就是说，这是一套“可继续往底层替换”的骨架，而不是已经打赢 vendor kernel 的成品。

## 如何接到原来的 LSTM.py

如果你的模型还是这种结构：

```python
class LSTMRegressor(nn.Module):
    def __init__(self, ...):
        self.lstm = nn.LSTM(...)
        self.linear = nn.Linear(...)
```

那么最简单的接法是：

```python
from persistent_lstm_hip import convert_regressor_module

model = LSTMRegressor(...).to("cuda:0").half().eval()
model = convert_regressor_module(model).to("cuda:0").half().eval()
```

如果你想尽量保留原模型，只替换内部 `nn.LSTM`：

```python
from persistent_lstm_hip import replace_lstm_inplace

model = LSTMRegressor(...).to("cuda:0").half().eval()
replace_lstm_inplace(model)
```

这样原来的 `forward` 里如果写的是：

```python
lstm_out, (hidden, cell) = self.lstm(x)
```

仍然可以工作，因为包装器会保持和 `nn.LSTM` 一样的返回接口。

## 如何构建扩展

在 ROCm / HIP 环境里进入这个目录：

```bash
cd persistent_lstm_hip
python setup.py build_ext --inplace
```

构建成功后，Python 会尝试导入 `persistent_lstm_hip_ext`。

如果扩展没有成功加载：

- 包装器仍然可以工作
- 但会自动回退到 Python reference 路径
- 不会走专门的 HIP kernel

## 当你修改模型结构时

这套接口层的目标是“结构变了，调用层不需要重写”：

- 改 `n_layers`
- 改 `hidden_size`
- 改 `input_size`
- 继续保留 `self.lstm + self.linear` 写法

这些都仍然可以工作。

需要区分的是后端路径：

- 如果命中特化条件，比如当前的 `4-layer / hidden=128 / input=5 / batch_first=True`，后续可以走专门的 HIP kernel。
- 如果没有命中特化条件，接口层会自动回退到通用 reference 路径。

也就是说：

- “接口通用性”已经开始按通用结构设计
- “HIP 特化性能”仍然需要针对具体 shape 继续补底层 kernel

## 建议推进顺序

1. 先用 reference 路径对齐输出和权重打包格式
2. 在 `persistent_lstm_hip.cu` 中先实现 `2-layer` persistent kernel
3. 再扩展到 `4-layer` 或 `2 + 2` 双 kernel 结构
4. 最后再做寄存器占用、LDS/shared memory、wave 数量和 tile 参数调优

## 重点文件

- Python 包装入口：`persistent_lstm_hip/model.py`
- 权重打包逻辑：`persistent_lstm_hip/packing.py`
- C++ 算子入口：`csrc/persistent_lstm_op.cpp`
- HIP kernel 入口：`csrc/persistent_lstm_hip.cu`
