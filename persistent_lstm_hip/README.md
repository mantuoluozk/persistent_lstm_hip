# Persistent LSTM HIP

面向海光 DCU 的 LSTM 推理优化项目。针对固定 shape 做最深特化，FP16 推理。

泛化工作由 [miopen_adaptive_lstm_hip](../miopen_adaptive_lstm_hip) 接棒。

## 性能记录（K100_AI / gfx928）

| 路径 | Shape | 时间 | 说明 |
| --- | --- | ---: | --- |
| 原生 PyTorch LSTM | `5/128/4/24, b512, s1000` | ~9s | baseline |
| **固定形状 HIP** | `5/128/4/24, b512, s1000` | **~3.08s** | 4 层融合 + uniform batch fast path |
| 原生 PyTorch LSTM | `7/64/2/16` | ~1.94s | baseline |
| H64 HIP bucket | `7/64/2/16` | ~0.89s | P4 partition |
| 原生 PyTorch LSTM | `7/128/2/16` | ~4.0s | baseline |
| H128 recur_gemm | `7/128/2/16` | ~2.9-4.1s | GEMM scan + HIP pointwise |
| H128 persistent_scalar | `7/128/2/16` | ~5.86s | 少 kernel 研究路径 |

## 核心技术

- **固定形状特化**：权重预打包、P4 分区标量点积、4 层融合 kernel
- **Uniform batch fast path**：全批次输入相同时只算 1 行 + 广播
- **H64/H128 bucket**：固定 hidden_size，其余参数动态

## 使用

```bash
cd persistent_lstm_hip
python setup.py build_ext --inplace

# 固定形状 HIP
python ../LSTM-hip.py

# H128 最快路径
PERSISTENT_LSTM_HIP_H128_MODE=best python ../LSTM-hip.py

# 调试
PERSISTENT_LSTM_HIP_DEBUG=1 python ../LSTM-hip.py
```
