# Official Architecture Map

This experimental backend is intentionally organized around the same layers as
MIOpen RNN and CK, while remaining small enough to iterate inside this project.

For implementation decisions, use `OFFICIAL_IMPLEMENTATION_GUIDE.md` as the
primary guardrail. This file maps source locations; the guide defines the
development rules that prevent the local backend from drifting away from the
official architecture.

## Source References

MIOpen:

- `projects/miopen/src/rnn/selector.cpp`
- `projects/miopen/src/rnn.cpp`
- `projects/miopen/src/rnn/rnn_util.cpp`
- `projects/miopen/src/rnn/Solutions/Base/fw_data_modular.cpp`
- `projects/miopen/src/rnn/Solutions/Base/bw_data_modular.cpp`
- `projects/miopen/src/kernels/MIOpenRNNHiddenStateUpdate.cpp`

Composable Kernel:

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp`
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp`
- current CK uses `blockwise_gemm_pipeline_xdlops*.hpp` rather than the older
  `blockwise_gemm_pipeline.hpp` name
- current CK no longer has the exact old `blockwise_copy.hpp` path in the
  develop tree queried here; the local implementation keeps copy/prefetch as
  traits until we wire a concrete CK header set
- `include/ck/utility/amd_xdlops.hpp`
- `include/ck_tile/ops/fmha/pipeline/`
- `example/ck_tile/01_fmha/`

## Local Mapping

| Official layer | Local file |
| --- | --- |
| RNN descriptor fields | `miopen_adaptive_lstm_hip/descriptors.py` |
| `CheckDynamicAlgoSelection()` | `miopen_adaptive_lstm_hip/selector.py` |
| dynamic pow2 sequence splitting | `miopen_adaptive_lstm_hip/modular.py` |
| hidden update launch sizing | `miopen_adaptive_lstm_hip/pipeline.py` |
| CK-style tile/pipeline traits | `miopen_adaptive_lstm_hip/pipeline.py`, `csrc/adaptive_lstm_pipeline.h` |
| modular forward dispatch | `miopen_adaptive_lstm_hip/model.py` |
| hidden update kernels | `csrc/adaptive_lstm_hip.cu` |

## Implementation Direction

The current recurrent path follows this progression:

1. MIOpen-style descriptor and selector choose the adaptive path.
2. Modular forward planning creates `prop_x`, `prop_hidden_y`, and
   `hidden_update` steps with power-of-two sequence segmentation.
3. H128 measured default uses `gemm_scan` for now because it is faster on the
   tested K100_AI/BW150 stack.
4. `seqmajor_accum` is the official-style architecture path: seq-major gate
   workspace, GEMM accumulation into `gate[t]`, then a focused hidden-update
   kernel. It remains experimental until it beats `gemm_scan`.
5. H256+ is represented as an XDLops candidate plan; the next implementation
   step is to replace the partitioned fallback with a tiled/MFMA recurrent
   microkernel.

## DCU / Older ROCm Strategy

The target deployment may not have upstream AMD MIOpen roundedDynamic available.
Therefore the practical optimization path is:

1. Use upstream MIOpen's architecture as the blueprint.
2. Keep runtime dependencies limited to PyTorch/ROCm GEMM plus small HIP kernels.
3. Prefer `gemm_scan` as the H128 mainline because it matches MIOpen's modular
   `PropHiddenY + LSTMForwardHiddenStateUpdate` shape without requiring new
   CK template support in the installed ROCm stack.
4. Keep cached/persistent kernels as benchmark alternatives, not as the default
   when GEMM propagation is faster.
