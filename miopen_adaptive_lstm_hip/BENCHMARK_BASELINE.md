# Benchmark Baseline

K100_AI / gfx928 上的参考性能记录。所有优化决策以此基线为回归标准。

Date: 2026-05-13

Device: Hygon C86-4G (K100_AI), gfx928

Common settings:
- `seq_len=1000`, `input_size=5`, `num_layers=4`, `output_size=24`
- `iterations=100`, inference/eval mode, FP16

## 原生 PyTorch LSTM

```bash
python run_native_lstm_sweep.py
```

| hidden | batch | elapsed_s | throughput |
|--------|-------|-----------|------------|
| 128 | 512 | 8.556 | 5984 |
| 256 | 512 | 10.995 | 4657 |
| 512 | 128 | 12.689 | 1009 |

## gemm_scan（当前默认，最快）

```bash
cd miopen_adaptive_lstm_hip
python run_shape_sweep.py
```

| hidden | batch | elapsed_s | throughput | max_abs | kernel |
|--------|-------|-----------|------------|---------|--------|
| 128 | 512 | 5.655 | 9052 | 3.05e-05 | h128_gemm_scan |
| 256 | 512 | — | — | — | — |
| 512 | 128 | — | — | — | — |

## persistent_mfma（标量，实验性）

```bash
MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=persistent_mfma python run_shape_sweep.py
```

| hidden | batch | elapsed_s | throughput | max_abs | kernel |
|--------|-------|-----------|------------|---------|--------|
| 128 | 512 | 11.23 | 4560 | 6.10e-05 | h128_persistent_mfma |

## 加速比

| shape | gemm_scan vs 原生 | persistent vs 原生 |
|-------|-------------------|---------------------|
| H128 B512 | 1.51x | 0.71x（回退） |

## MFMA 状态（2026-05-13）

- 编译器可生成 `v_mfma_f32_16x16x16f16`（需 `-Xclang=-target-feature -Xclang=+mai-insts`）
- GPU 硬���执行时报告 `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`
- 等待海光研发确认 gfx928 的 MFMA 支持情况
- MFMA kernel 代码已保留在 `csrc/adaptive_lstm_hip.cu` 中，受 `#ifdef MIOPEN_ADAPTIVE_LSTM_ENABLE_MFMA_BUILTIN` 保护
