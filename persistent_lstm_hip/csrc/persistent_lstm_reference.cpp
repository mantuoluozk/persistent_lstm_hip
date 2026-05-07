#include "persistent_lstm_hip.h"

#include <vector>

namespace {

torch::Tensor run_layer(
    const torch::Tensor& x,
    const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const torch::Tensor& bias) {
  auto batch_size = x.size(0);
  auto seq_len = x.size(1);
  auto hidden_dim = weight_hh.size(1);

  auto h = torch::zeros({batch_size, hidden_dim}, x.options());
  auto c = torch::zeros({batch_size, hidden_dim}, x.options());
  std::vector<torch::Tensor> outputs;
  outputs.reserve(seq_len);

  for (int64_t t = 0; t < seq_len; ++t) {
    auto x_t = x.select(1, t);
    auto gates = torch::matmul(x_t, weight_ih.transpose(0, 1)) +
                 torch::matmul(h, weight_hh.transpose(0, 1)) + bias;

    auto chunks = gates.chunk(4, 1);
    auto i_gate = torch::sigmoid(chunks[0]);
    auto f_gate = torch::sigmoid(chunks[1]);
    auto g_gate = torch::tanh(chunks[2]);
    auto o_gate = torch::sigmoid(chunks[3]);

    c = f_gate * c + i_gate * g_gate;
    h = o_gate * torch::tanh(c);
    outputs.push_back(h);
  }

  return torch::stack(outputs, 1);
}

}  // namespace

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
    const torch::Tensor& linear_bias) {
  auto out0 = run_layer(x, weight_ih_l0, weight_hh_l0, bias_l0);
  auto out1 = run_layer(out0, weight_ih_l1, weight_hh_l1, bias_l1);
  auto out2 = run_layer(out1, weight_ih_l2, weight_hh_l2, bias_l2);
  auto out3 = run_layer(out2, weight_ih_l3, weight_hh_l3, bias_l3);
  auto last = out3.select(1, out3.size(1) - 1);
  return torch::matmul(last, linear_weight.transpose(0, 1)) + linear_bias;
}

