#include "persistent_lstm_hip.h"

#include <stdexcept>

namespace {

void check_shape(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

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
    const torch::Tensor& linear_bias) {
  check_shape(x.dim() == 3, "x must be [B, T, I]");
  check_shape(weight_ih_l0.dim() == 2, "weight_ih_l0 must be rank-2");
  check_shape(weight_hh_l0.dim() == 2, "weight_hh_l0 must be rank-2");
  check_shape(linear_weight.dim() == 2, "linear_weight must be rank-2");

  if (x.is_cuda()) {
    return persistent_lstm4_forward_hip(
        x,
        weight_ih_l0,
        weight_hh_l0,
        bias_l0,
        weight_ih_l1,
        weight_hh_l1,
        bias_l1,
        weight_ih_l2,
        weight_hh_l2,
        bias_l2,
        weight_ih_l3,
        weight_hh_l3,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  return persistent_lstm4_forward_reference(
      x,
      weight_ih_l0,
      weight_hh_l0,
      bias_l0,
      weight_ih_l1,
      weight_hh_l1,
      bias_l1,
      weight_ih_l2,
      weight_hh_l2,
      bias_l2,
      weight_ih_l3,
      weight_hh_l3,
      bias_l3,
      linear_weight,
      linear_bias);
}

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
    const torch::Tensor& linear_bias) {
  if (x.is_cuda()) {
    return persistent_lstm4_forward_packed_hip(
        x,
        weight_ih_l0_t,
        weight_hh_l0_t,
        bias_l0,
        weight_ih_l1_t,
        weight_hh_l1_t,
        bias_l1,
        weight_ih_l2_t,
        weight_hh_l2_t,
        bias_l2,
        weight_ih_l3_t,
        weight_hh_l3_t,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  return persistent_lstm4_forward_reference(
      x,
      weight_ih_l0_t.transpose(0, 1).contiguous(),
      weight_hh_l0_t.transpose(0, 1).contiguous(),
      bias_l0,
      weight_ih_l1_t.transpose(0, 1).contiguous(),
      weight_hh_l1_t.transpose(0, 1).contiguous(),
      bias_l1,
      weight_ih_l2_t.transpose(0, 1).contiguous(),
      weight_hh_l2_t.transpose(0, 1).contiguous(),
      bias_l2,
      weight_ih_l3_t.transpose(0, 1).contiguous(),
      weight_hh_l3_t.transpose(0, 1).contiguous(),
      bias_l3,
      linear_weight,
      linear_bias);
}

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
    const torch::Tensor& linear_bias) {
  if (x.is_cuda()) {
    return persistent_lstm4_forward_interleaved_hip(
        x,
        weight_ih_l0_packed,
        weight_hh_l0_packed,
        bias_l0,
        weight_ih_l1_packed,
        weight_hh_l1_packed,
        bias_l1,
        weight_ih_l2_packed,
        weight_hh_l2_packed,
        bias_l2,
        weight_ih_l3_packed,
        weight_hh_l3_packed,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto unpack = [](const torch::Tensor& packed, int64_t original_k) {
    check_shape(packed.dim() == 3, "interleaved packed weight must be rank-3");
    auto contiguous = packed.contiguous();
    const auto pairs = contiguous.size(0);
    const auto out = contiguous.size(1);
    const auto restored = contiguous.permute({0, 2, 1}).contiguous().view({pairs * 2, out});
    return restored.narrow(0, 0, original_k).transpose(0, 1).contiguous();
  };

  return persistent_lstm4_forward_reference(
      x,
      unpack(weight_ih_l0_packed, 5),
      unpack(weight_hh_l0_packed, 128),
      bias_l0,
      unpack(weight_ih_l1_packed, 128),
      unpack(weight_hh_l1_packed, 128),
      bias_l1,
      unpack(weight_ih_l2_packed, 128),
      unpack(weight_hh_l2_packed, 128),
      bias_l2,
      unpack(weight_ih_l3_packed, 128),
      unpack(weight_hh_l3_packed, 128),
      bias_l3,
      linear_weight,
      linear_bias);
}

torch::Tensor persistent_lstm4_forward_projected(
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
    const torch::Tensor& linear_bias) {
  if (x.is_cuda()) {
    return persistent_lstm4_forward_projected_hip(
        x,
        weight_ih_l0,
        weight_hh_l0_packed,
        bias_l0,
        weight_ih_l1,
        weight_hh_l1_packed,
        bias_l1,
        weight_ih_l2,
        weight_hh_l2_packed,
        bias_l2,
        weight_ih_l3,
        weight_hh_l3_packed,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto unpack = [](const torch::Tensor& packed, int64_t original_k) {
    check_shape(packed.dim() == 3, "projected packed recurrent weight must be rank-3");
    auto contiguous = packed.contiguous();
    const auto pairs = contiguous.size(0);
    const auto out = contiguous.size(1);
    const auto restored = contiguous.permute({0, 2, 1}).contiguous().view({pairs * 2, out});
    return restored.narrow(0, 0, original_k).transpose(0, 1).contiguous();
  };

  return persistent_lstm4_forward_reference(
      x,
      weight_ih_l0.contiguous(),
      unpack(weight_hh_l0_packed, 128),
      bias_l0,
      weight_ih_l1.contiguous(),
      unpack(weight_hh_l1_packed, 128),
      bias_l1,
      weight_ih_l2.contiguous(),
      unpack(weight_hh_l2_packed, 128),
      bias_l2,
      weight_ih_l3.contiguous(),
      unpack(weight_hh_l3_packed, 128),
      bias_l3,
      linear_weight,
      linear_bias);
}

torch::Tensor persistent_lstm4_forward_monolithic(
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
    const torch::Tensor& linear_bias) {
  if (x.is_cuda()) {
    return persistent_lstm4_forward_monolithic_hip(
        x,
        weight_ih_l0_packed,
        weight_hh_l0_packed,
        bias_l0,
        weight_ih_l1_packed,
        weight_hh_l1_packed,
        bias_l1,
        weight_ih_l2_packed,
        weight_hh_l2_packed,
        bias_l2,
        weight_ih_l3_packed,
        weight_hh_l3_packed,
        bias_l3,
        linear_weight,
        linear_bias);
  }

  auto unpack = [](const torch::Tensor& packed, int64_t original_k) {
    check_shape(packed.dim() == 3, "monolithic packed weight must be rank-3");
    auto contiguous = packed.contiguous();
    const auto pairs = contiguous.size(0);
    const auto out = contiguous.size(1);
    const auto restored = contiguous.permute({0, 2, 1}).contiguous().view({pairs * 2, out});
    return restored.narrow(0, 0, original_k).transpose(0, 1).contiguous();
  };

  return persistent_lstm4_forward_reference(
      x,
      unpack(weight_ih_l0_packed, 5),
      unpack(weight_hh_l0_packed, 128),
      bias_l0,
      unpack(weight_ih_l1_packed, 128),
      unpack(weight_hh_l1_packed, 128),
      bias_l1,
      unpack(weight_ih_l2_packed, 128),
      unpack(weight_hh_l2_packed, 128),
      bias_l2,
      unpack(weight_ih_l3_packed, 128),
      unpack(weight_hh_l3_packed, 128),
      bias_l3,
      linear_weight,
      linear_bias);
}
