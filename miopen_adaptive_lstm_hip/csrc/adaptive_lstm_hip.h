#pragma once

#include <torch/extension.h>

torch::Tensor adaptive_lstm_hidden_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t read_block,
    int64_t items_per_group);

torch::Tensor adaptive_lstm_hidden_update_partitioned_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t partitions);

torch::Tensor adaptive_lstm_h128_cached_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence);

torch::Tensor adaptive_lstm_h128_cached_b4_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence);

torch::Tensor adaptive_lstm_h128_cached_b8_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence);

torch::Tensor adaptive_lstm_h128_gemm_scan_update_forward(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    bool write_sequence,
    int64_t read_block);

torch::Tensor adaptive_lstm_h128_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block);

torch::Tensor adaptive_lstm_h128_seqmajor_accum_update_forward(
    const torch::Tensor& gate_seq,
    const torch::Tensor& weight_hh,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor seq_out,
    bool write_sequence,
    int64_t read_block);

torch::Tensor adaptive_lstm_input_gemm_forward_workspace(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight_ih_t,
    torch::Tensor gate);

torch::Tensor adaptive_lstm_h256_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t recurrent_compute);

torch::Tensor adaptive_lstm_h512_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t recurrent_compute);

torch::Tensor adaptive_lstm_h256_gate_accum_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block);

torch::Tensor adaptive_lstm_h512_gate_accum_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block);

torch::Tensor adaptive_lstm_h128_persistent_mmac_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block,
    int64_t fast_act);

torch::Tensor adaptive_lstm_h128_mmac_profile_variant_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor out,
    torch::Tensor profile_out,
    bool write_sequence,
    int64_t read_block,
    int64_t variant);

torch::Tensor adaptive_lstm_gemm_scan_update_forward_workspace(
    const torch::Tensor& gate_proj,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias,
    torch::Tensor h_state,
    torch::Tensor c_state,
    torch::Tensor recur,
    torch::Tensor out,
    bool write_sequence,
    int64_t read_block);
