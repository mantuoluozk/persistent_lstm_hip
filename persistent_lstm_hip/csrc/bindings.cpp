#include <torch/extension.h>

torch::Tensor persistent_lstm4_forward(
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

torch::Tensor persistent_lstm4_forward_packed(
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

torch::Tensor persistent_lstm4_forward_interleaved(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("persistent_lstm4_forward", &persistent_lstm4_forward, "Persistent LSTM 4-layer forward");
  m.def("persistent_lstm4_forward_packed", &persistent_lstm4_forward_packed, "Persistent LSTM 4-layer forward packed");
  m.def("persistent_lstm4_forward_interleaved", &persistent_lstm4_forward_interleaved, "Persistent LSTM 4-layer forward interleaved");
}
