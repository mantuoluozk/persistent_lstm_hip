#pragma once

#include <torch/extension.h>
#include <vector>

torch::Tensor persistent_lstm4_forward_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm4_forward_packed_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_t,
    const torch::Tensor& weight_hh_l0_t,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_t,
    const torch::Tensor& weight_hh_l1_t,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_t,
    const torch::Tensor& weight_hh_l2_t,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_t,
    const torch::Tensor& weight_hh_l3_t,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm4_forward_interleaved_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_packed,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_packed,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_packed,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_packed,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm4_forward_projected_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm4_forward_monolithic_hip(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0_packed,
    const torch::Tensor& weight_hh_l0_packed,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1_packed,
    const torch::Tensor& weight_hh_l1_packed,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2_packed,
    const torch::Tensor& weight_hh_l2_packed,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3_packed,
    const torch::Tensor& weight_hh_l3_packed,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm4_forward_reference(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih_l0,
    const torch::Tensor& weight_hh_l0,
    const torch::Tensor& bias_l0,
    const torch::Tensor& weight_ih_l1,
    const torch::Tensor& weight_hh_l1,
    const torch::Tensor& bias_l1,
    const torch::Tensor& weight_ih_l2,
    const torch::Tensor& weight_hh_l2,
    const torch::Tensor& bias_l2,
    const torch::Tensor& weight_ih_l3,
    const torch::Tensor& weight_hh_l3,
    const torch::Tensor& bias_l3,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);

torch::Tensor persistent_lstm_regressor_forward_generic_projected_hip(
    const torch::Tensor& x,
    const std::vector<torch::Tensor>& weight_ih,
    const std::vector<torch::Tensor>& weight_hh,
    const std::vector<torch::Tensor>& bias,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias);
