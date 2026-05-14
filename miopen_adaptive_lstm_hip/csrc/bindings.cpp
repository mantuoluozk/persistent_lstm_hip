#include <torch/extension.h>

#include "adaptive_lstm_hip.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "adaptive_lstm_hidden_update_forward",
      &adaptive_lstm_hidden_update_forward,
      "MIOpen-inspired adaptive LSTM hidden update");
  m.def(
      "adaptive_lstm_hidden_update_partitioned_forward",
      &adaptive_lstm_hidden_update_partitioned_forward,
      "MIOpen-inspired partitioned adaptive LSTM hidden update");
  m.def(
      "adaptive_lstm_h128_cached_update_forward",
      &adaptive_lstm_h128_cached_update_forward,
      "MIOpen-inspired cached H128 adaptive LSTM hidden update");
  m.def(
      "adaptive_lstm_h128_cached_b4_update_forward",
      &adaptive_lstm_h128_cached_b4_update_forward,
      "MIOpen-inspired cached H128 B4 adaptive LSTM hidden update");
  m.def(
      "adaptive_lstm_h128_cached_b8_update_forward",
      &adaptive_lstm_h128_cached_b8_update_forward,
      "MIOpen-inspired cached H128 B8 adaptive LSTM hidden update");
  m.def(
      "adaptive_lstm_h128_gemm_scan_update_forward",
      &adaptive_lstm_h128_gemm_scan_update_forward,
      "MIOpen-inspired H128 GEMM scan LSTM hidden update");
  m.def(
      "adaptive_lstm_h128_gemm_scan_update_forward_workspace",
      &adaptive_lstm_h128_gemm_scan_update_forward_workspace,
      "MIOpen-inspired H128 GEMM scan LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h128_seqmajor_accum_update_forward",
      &adaptive_lstm_h128_seqmajor_accum_update_forward,
      "MIOpen-style H128 seq-major accumulated gate LSTM hidden update");
  m.def(
      "adaptive_lstm_input_gemm_forward_workspace",
      &adaptive_lstm_input_gemm_forward_workspace,
      "MIOpen-inspired input projection GEMM into caller gate workspace");
  m.def(
      "adaptive_lstm_h256_gemm_scan_update_forward_workspace",
      &adaptive_lstm_h256_gemm_scan_update_forward_workspace,
      "MIOpen-inspired H256 GEMM scan LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h512_gemm_scan_update_forward_workspace",
      &adaptive_lstm_h512_gemm_scan_update_forward_workspace,
      "MIOpen-inspired H512 GEMM scan LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h256_gate_accum_update_forward_workspace",
      &adaptive_lstm_h256_gate_accum_update_forward_workspace,
      "MIOpen-inspired H256 gate-accumulated GEMM scan LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h512_gate_accum_update_forward_workspace",
      &adaptive_lstm_h512_gate_accum_update_forward_workspace,
      "MIOpen-inspired H512 gate-accumulated GEMM scan LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h128_persistent_mmac_update_forward_workspace",
      &adaptive_lstm_h128_persistent_mmac_update_forward_workspace,
      "MIOpen-inspired H128 persistent MFMA LSTM hidden update with caller workspace");
  m.def(
      "adaptive_lstm_h128_mmac_profile_variant_forward_workspace",
      &adaptive_lstm_h128_mmac_profile_variant_forward_workspace,
      "MIOpen-inspired H128 MMAC profile variant LSTM hidden update (0=MMAC,1=+bias,2=+act,3=full)");
  m.def(
      "adaptive_lstm_h128_mmac_packed_variant_forward_workspace",
      &adaptive_lstm_h128_mmac_packed_variant_forward_workspace,
      "MIOpen-inspired H128 MMAC packed-weight profile variant LSTM hidden update (0=MMAC,1=+bias,2=+act,3=full)");
  m.def(
      "adaptive_lstm_gemm_scan_update_forward_workspace",
      &adaptive_lstm_gemm_scan_update_forward_workspace,
      "MIOpen-inspired generic GEMM scan LSTM hidden update with caller workspace");
}
