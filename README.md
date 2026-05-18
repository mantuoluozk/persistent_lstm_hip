# LSTM 推理优化项目

面向海光 DCU（K100_AI / gfx928）的 LSTM 推理优化。包含两个 HIP 后端项目，从不同角度优化 `nn.LSTM + nn.Linear` 推理。

## 项目对比

| | 原生 PyTorch (fp16) | [miopen_adaptive_lstm_hip](miopen_adaptive_lstm_hip/) | [persistent_lstm_hip](persistent_lstm_hip/) |
|------|------|------|------|
| 思路 | MIOpen 内置 | 多后端调度，auto 自适应 | 固定形状特化 |
| H128 `5/128/4/24, b512` | 8.21s | **4.39s (-47%)** | **3.08s (-62%)** |
| H256 `5/256/4/24, b512` | 11.77s | **7.16s (-39%)** | — |
| H512 `5/512/4/24, b512` | 22.73s | **9.83s (-57%)** | — |

## miopen_adaptive_lstm_hip（当前主力）

- 泛化到 H128/H256/H512，auto 后端自动选择最优路径
- 三条路径全线超越原生 PyTorch（47%/39%/57%）
- 详细文档：[miopen_adaptive_lstm_hip/README.md](miopen_adaptive_lstm_hip/README.md)

## persistent_lstm_hip（参考项目）

- 针对固定形状 `5/128/4/24, b512, s1000` 做最深特化
- 3.08s 是已知 H128 最优结果，但不泛化
- 技术参考价值：4 层融合、uniform batch fast path
- 详细文档：[persistent_lstm_hip/README.md](persistent_lstm_hip/README.md)

## 快速开始

```bash
# miopen_adaptive_lstm_hip（推荐）
cd miopen_adaptive_lstm_hip
python setup.py build_ext --inplace
python run_adaptive_lstm.py

# persistent_lstm_hip（固定 shape）
cd persistent_lstm_hip
python setup.py build_ext --inplace
python benchmark_persistent_lstm.py
```
