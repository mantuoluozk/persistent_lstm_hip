# Official Implementation Guide

This file is the guardrail for future optimization work in this experimental
backend. When implementation choices conflict, follow this guide before adding
new kernels, flags, or shortcuts.

## What MIOpen Actually Does

The upstream MIOpen RNN dynamic path is modular. It is not a single handwritten
persistent LSTM kernel for every hidden size.

The relevant path is:

```text
selector.cpp
  CheckDynamicAlgoSelection()

fw_data_modular.cpp
  PropX()
  PropHiddenY()
  PropHiddenHt()
  UpdateHStatePerTimeSeq()

rnn_util.cpp
  LSTMForwardHiddenStateUpdate()

kernels/MIOpenRNNHiddenStateUpdate.cpp
  LSTMFwdHiddenUpdate
```

The dataflow is:

```text
workspace gate block = X @ Wx
workspace gate block += input/recurrent bias

for each timestep:
    workspace gate block[t] += H[t-1] @ Wh
    hidden update reads workspace gate block[t]
    hidden update writes Ct and Ht back to workspace
```

The important detail is that `LSTMFwdHiddenUpdate` only consumes an already
accumulated gate block. It does not perform the recurrent matrix multiply, and
it does not carry a separate `recur` tensor through the pointwise update.

## Dynamic Algo Selection

MIOpen enables the rounded dynamic path only for a narrow configuration:

- training mode in upstream MIOpen
- unidirectional RNN
- linear input mode
- LSTM
- no dropout
- `algoMode == miopenRNNroundedDynamic`, or forced by environment

Our inference backend may deliberately enable the same shape of algorithm for
inference because the target DCU stack lacks the upstream implementation. That
is a local adaptation, not a signal to ignore the rest of the design.

## Workspace Layout Rule

Official MIOpen is built around workspace/reserve-space offsets and descriptors.
For our H128 fast path, this means the preferred local layout should be:

```text
gate workspace: [T, B, 4H] or equivalent seq-major contiguous timestep blocks
hidden state:   [B, H]
cell state:     [B, H]
output seq:     [T, B, H] or equivalent seq-major contiguous timestep blocks
```

Avoid making `[B, T, 4H]` the internal fast-path layout, because `gate[:, t, :]`
is not contiguous in batch-major layout. That prevents the official-style
operation:

```text
gate[t] = X[t] @ Wx
gate[t] += H[t-1] @ Wh
```

The next major implementation should therefore be a new path, not another tweak
to the current batch-major `h128_gemm_scan`:

```text
h128_seqmajor_accum
```

## Hidden Update Rule

Follow `MIOpenRNNHiddenStateUpdate.cpp`:

- read the four gates from the accumulated gate workspace
- apply sigmoid/tanh
- update cell state
- write hidden state
- use MIOpen-style `READ_BLOCK` specialization
- use `items_per_group` from the MIOpen launch heuristic

Do not keep adding work to the hidden update kernel. In particular, do not make
the pointwise kernel responsible for a separate recurrent tensor plus bias if a
workspace accumulation path can avoid it.

## GEMM Rule

MIOpen delegates propagation to GEMM:

- `PropX`: input projection into gate workspace
- `PropHiddenY`: inter-layer input projection into gate workspace
- `PropHiddenHt`: recurrent projection into the same gate workspace

For our local backend:

- use PyTorch/ROCm GEMM first for portability
- use direct hipBLAS only as a measured optional dispatch optimization
- do not replace GEMM with hand-written scalar recurrent kernels as the default
- keep cached/persistent kernels as comparison paths unless they beat GEMM scan

## Dynamic Segmentation Rule

Official dynamic algo uses power-of-two segmentation:

```text
PropX:       MaskedPow2Range(total_seq_cnt)
PropHiddenY: getLowerBoundPow2(total_seq_cnt), then descending bit chunks
```

For fixed batch and fixed sequence length, segmentation may collapse to a small
number of large GEMM regions. Still, the planner should keep this concept
explicit because dynamic batch/packed sequences depend on it.

## CK Lessons To Apply

Composable Kernel is a guide for GEMM microkernel design, not something to copy
blindly into RNN code.

Relevant CK ideas:

- LDS-backed tiles
- double buffering / prefetch stages
- wave-level residency
- XDLops/MFMA blockwise GEMM
- pipeline selector based on loop count and tile support

Use these ideas only when building a real recurrent GEMM microkernel. Do not
turn every optimization into a new environment flag or unrelated kernel variant.

## What Not To Do

Avoid these unless there is a measured reason and a rollback path:

- adding many hidden-size-specific kernels without a selector rule
- changing the math layout without an accuracy A/B
- relying on CUDA/HIP graph replay on DTK/BW150 without validation
- optimizing around benchmark noise from first-run library initialization
- adding workspace optimizations that do not touch the hot recurrent path
- making direct hipBLAS the only path
- replacing official-style GEMM accumulation with scalar recurrent loops

Current evidence:

- CUDA/HIP graph replay is unsafe on the tested BW150/DTK stack for this backend.
  It produced large accuracy drift and must remain optional with validation.
- Direct hipBLAS can improve the current batch-major path, but it is not the
  official architecture. Treat it as a dispatch optimization, not the main
  design direction.
- Recurrent GEMM compute type defaults to FP32 accumulation. FP16 accumulation
  is allowed only as an explicit `MIOPEN_ADAPTIVE_LSTM_RECURRENT_COMPUTE=fp16`
  experiment and must pass accuracy on all target shapes before becoming a
  default.
- Cached B4/B8 handwritten recurrent kernels are slower than GEMM scan for the
  current H128 workload.
- For non-H128 shapes, do not fall back to Python timestep loops. Use the same
  modular `gemm_scan` structure: workspace gate block, GEMM recurrent
  propagation, and a small HIP hidden-update kernel selected by read block.
- In the local `gemm_scan` path, keep the measured `READ_BLOCK=1` default.
  K100_AI measurements showed forced `READ_BLOCK=4` regressed H256/H512, so
  vectorized read blocks should remain an explicit A/B override instead of an
  auto default.

## Current Correct Implementation Target

Keep `gemm_scan` as the measured default and extend it across shapes. The H128
`seqmajor_accum` path remains the architecture reference, but it should not
become the default until it beats the stable measured path.

Stable measured path:

```text
generic_gemm_scan
```

Expected internal flow:

```text
1. Allocate/reuse batch-major gate workspace [B*T, 4H].
2. PropX writes X @ Wx into gate workspace, preferably through the extension
   GEMM path rather than Python-level `torch.mm` dispatch.
3. For each timestep:
       GEMM writes H @ Wh into recurrent workspace [B, 4H].
       hidden update reads gate[t] + recurrent[t] + bias.
       hidden update writes H and C workspace with the selected READ_BLOCK.
4. For non-last layers, write batch-major hidden output [B, T, H].
5. Reuse all H/C/recurrent/output workspaces across forwards for the same shape.
```

For the measured target shapes, H256 and H512 may use fixed-hidden pointwise
specializations inside this same flow. These are acceptable because the
architecture remains modular GEMM scan; the specialization only removes dynamic
hidden-size arithmetic from the hidden-update kernel.

H256/H512 may also use a batch-major gate-accumulation variant. This mirrors the
official idea of accumulating recurrent GEMM into the gate workspace, while
preserving the local batch-major layout to avoid a full seq-major rewrite. It
regressed K100_AI measurements because the recurrent GEMM writes into a strided
batch-major slice, so keep it disabled by default and guarded by
`MIOPEN_ADAPTIVE_LSTM_GATE_ACCUM`.

Architecture reference path:

```text
h128_seqmajor_accum
```

Expected internal flow:

```text
1. Allocate/reuse seq-major gate workspace [T, B, 512].
2. PropX writes X @ Wx into gate workspace.
3. Add bias into gate workspace, preferably once per layer.
4. For each timestep:
       GEMM accumulates H @ Wh into gate[t] with beta=1.
       hidden update reads gate[t] only.
       hidden update writes H and C workspace.
5. For non-last layers, write seq-major hidden output for the next layer.
6. Convert only at boundaries if the public API expects batch-major.
```

Validation requirements:

- compare against native PyTorch LSTM
- keep `max_abs` in the established FP16 tolerance range
- compare against current `h128_gemm_scan`
- profile input projection and recurrent sections separately
- keep a clean environment switch for A/B:

```text
MIOPEN_ADAPTIVE_LSTM_RECURRENT_BACKEND=seqmajor_accum
```

## Decision Rule For Future Changes

Before changing code, answer these questions:

1. Does this move us closer to workspace gate accumulation?
2. Does it reduce recurrent-loop GEMM/hidden-update cost, not just allocation
   overhead?
3. Is it compatible with MIOpen's modular split between GEMM propagation and
   hidden update?
4. Can it be A/B tested against the current stable path?
5. Does it preserve accuracy on BW150/DTK?

If the answer is not clearly yes, pause and inspect the official references
again before implementing.
