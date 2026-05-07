import os
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _persistent_lstm_seq_kernel_cached_wh(
    wx_ptr,
    wh_i_ptr,
    wh_f_ptr,
    wh_g_ptr,
    wh_o_ptr,
    out_ptr,
    seq_len,
    batch_size,
    hidden_dim,
    stride_wx_b,
    stride_wx_s,
    stride_wx_d,
    stride_out_b,
    stride_out_s,
    stride_out_d,
    BLOCK_M: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_HID)

    mask_m = offs_m < batch_size
    mask_h = offs_h < hidden_dim
    mask_bh = mask_m[:, None] & mask_h[None, :]
    mask_wh = mask_h[:, None] & mask_h[None, :]

    dtype = wx_ptr.dtype.element_ty
    h = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    ptrs_w = offs_h[:, None] * hidden_dim + offs_h[None, :]
    gate_stride = hidden_dim * stride_wx_d

    # 参考 persistent-rnn 的核心做法：recurrent weights 在时间循环外加载一次，
    # 然后在多个 timestep 中重复使用，而不是每步都重新从全局显存读取。
    w_hi = tl.load(wh_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_hf = tl.load(wh_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_hg = tl.load(wh_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_ho = tl.load(wh_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    for t in range(seq_len):
        base_wx = wx_ptr + offs_m[:, None] * stride_wx_b + t * stride_wx_s + offs_h[None, :] * stride_wx_d
        wx_i = tl.load(base_wx + 0 * gate_stride, mask=mask_bh, other=0.0)
        wx_f = tl.load(base_wx + 1 * gate_stride, mask=mask_bh, other=0.0)
        wx_g = tl.load(base_wx + 2 * gate_stride, mask=mask_bh, other=0.0)
        wx_o = tl.load(base_wx + 3 * gate_stride, mask=mask_bh, other=0.0)

        i_raw = (wx_i + tl.dot(h, w_hi)).to(tl.float32)
        f_raw = (wx_f + tl.dot(h, w_hf)).to(tl.float32)
        g_raw = (wx_g + tl.dot(h, w_hg)).to(tl.float32)
        o_raw = (wx_o + tl.dot(h, w_ho)).to(tl.float32)

        i_gate = tl.sigmoid(i_raw).to(dtype)
        f_gate = tl.sigmoid(f_raw).to(dtype)
        g_gate = (2.0 * tl.sigmoid(2.0 * g_raw) - 1.0).to(dtype)
        o_gate = tl.sigmoid(o_raw).to(dtype)

        c = (f_gate * c + i_gate * g_gate).to(dtype)
        c_fp32 = c.to(tl.float32)
        h = (o_gate * (2.0 * tl.sigmoid(2.0 * c_fp32) - 1.0).to(dtype)).to(dtype)

        out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + t * stride_out_s + offs_h[None, :] * stride_out_d
        tl.store(out_ptrs, h, mask=mask_bh)


@triton.jit
def _persistent_lstm_last_kernel_cached_wh(
    wx_ptr,
    wh_i_ptr,
    wh_f_ptr,
    wh_g_ptr,
    wh_o_ptr,
    out_ptr,
    seq_len,
    batch_size,
    hidden_dim,
    stride_wx_b,
    stride_wx_s,
    stride_wx_d,
    stride_out_b,
    stride_out_d,
    BLOCK_M: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_HID)

    mask_m = offs_m < batch_size
    mask_h = offs_h < hidden_dim
    mask_bh = mask_m[:, None] & mask_h[None, :]
    mask_wh = mask_h[:, None] & mask_h[None, :]

    dtype = wx_ptr.dtype.element_ty
    h = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    ptrs_w = offs_h[:, None] * hidden_dim + offs_h[None, :]
    gate_stride = hidden_dim * stride_wx_d

    w_hi = tl.load(wh_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_hf = tl.load(wh_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_hg = tl.load(wh_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    w_ho = tl.load(wh_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    for t in range(seq_len):
        base_wx = wx_ptr + offs_m[:, None] * stride_wx_b + t * stride_wx_s + offs_h[None, :] * stride_wx_d
        wx_i = tl.load(base_wx + 0 * gate_stride, mask=mask_bh, other=0.0)
        wx_f = tl.load(base_wx + 1 * gate_stride, mask=mask_bh, other=0.0)
        wx_g = tl.load(base_wx + 2 * gate_stride, mask=mask_bh, other=0.0)
        wx_o = tl.load(base_wx + 3 * gate_stride, mask=mask_bh, other=0.0)

        i_raw = (wx_i + tl.dot(h, w_hi)).to(tl.float32)
        f_raw = (wx_f + tl.dot(h, w_hf)).to(tl.float32)
        g_raw = (wx_g + tl.dot(h, w_hg)).to(tl.float32)
        o_raw = (wx_o + tl.dot(h, w_ho)).to(tl.float32)

        i_gate = tl.sigmoid(i_raw).to(dtype)
        f_gate = tl.sigmoid(f_raw).to(dtype)
        g_gate = (2.0 * tl.sigmoid(2.0 * g_raw) - 1.0).to(dtype)
        o_gate = tl.sigmoid(o_raw).to(dtype)

        c = (f_gate * c + i_gate * g_gate).to(dtype)
        c_fp32 = c.to(tl.float32)
        h = (o_gate * (2.0 * tl.sigmoid(2.0 * c_fp32) - 1.0).to(dtype)).to(dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_h[None, :] * stride_out_d
    tl.store(out_ptrs, h, mask=mask_bh)


@dataclass
class PackedLayerWeights:
    weight_ih: torch.Tensor
    bias: torch.Tensor
    wh_i: torch.Tensor
    wh_f: torch.Tensor
    wh_g: torch.Tensor
    wh_o: torch.Tensor


def _pack_layer_weights(native_lstm: nn.LSTM, layer_idx: int) -> PackedLayerWeights:
    weight_ih = getattr(native_lstm, f"weight_ih_l{layer_idx}").contiguous()
    weight_hh = getattr(native_lstm, f"weight_hh_l{layer_idx}")
    bias_ih = getattr(native_lstm, f"bias_ih_l{layer_idx}")
    bias_hh = getattr(native_lstm, f"bias_hh_l{layer_idx}")

    hidden_dim = native_lstm.hidden_size
    wh_i = weight_hh[0:hidden_dim, :].t().contiguous()
    wh_f = weight_hh[hidden_dim : 2 * hidden_dim, :].t().contiguous()
    wh_g = weight_hh[2 * hidden_dim : 3 * hidden_dim, :].t().contiguous()
    wh_o = weight_hh[3 * hidden_dim : 4 * hidden_dim, :].t().contiguous()

    # 把两路 bias 在输入投影阶段合并，kernel 内只保留 recurrent matmul + gate update。
    bias = (bias_ih + bias_hh).contiguous()

    return PackedLayerWeights(
        weight_ih=weight_ih,
        bias=bias,
        wh_i=wh_i,
        wh_f=wh_f,
        wh_g=wh_g,
        wh_o=wh_o,
    )


def _project_inputs(x: torch.Tensor, packed: PackedLayerWeights) -> torch.Tensor:
    return F.linear(x, packed.weight_ih, packed.bias).contiguous()


def persistent_lstm_layer_sequence(
    wx: torch.Tensor,
    packed: PackedLayerWeights,
    block_m: int = 16,
) -> torch.Tensor:
    batch_size, seq_len, gates_dim = wx.shape
    hidden_dim = gates_dim // 4

    if block_m < 16:
        raise ValueError("Triton tl.dot 要求 BLOCK_M >= 16。")

    out = torch.empty((batch_size, seq_len, hidden_dim), device=wx.device, dtype=wx.dtype)
    grid = (triton.cdiv(batch_size, block_m),)

    _persistent_lstm_seq_kernel_cached_wh[grid](
        wx,
        packed.wh_i,
        packed.wh_f,
        packed.wh_g,
        packed.wh_o,
        out,
        seq_len,
        batch_size,
        hidden_dim,
        wx.stride(0),
        wx.stride(1),
        wx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_HID=hidden_dim,
        num_warps=4,
        num_stages=1,
    )
    return out


def persistent_lstm_layer_last(
    wx: torch.Tensor,
    packed: PackedLayerWeights,
    block_m: int = 16,
) -> torch.Tensor:
    batch_size, seq_len, gates_dim = wx.shape
    hidden_dim = gates_dim // 4

    if block_m < 16:
        raise ValueError("Triton tl.dot 要求 BLOCK_M >= 16。")

    out = torch.empty((batch_size, hidden_dim), device=wx.device, dtype=wx.dtype)
    grid = (triton.cdiv(batch_size, block_m),)

    _persistent_lstm_last_kernel_cached_wh[grid](
        wx,
        packed.wh_i,
        packed.wh_f,
        packed.wh_g,
        packed.wh_o,
        out,
        seq_len,
        batch_size,
        hidden_dim,
        wx.stride(0),
        wx.stride(1),
        wx.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_HID=hidden_dim,
        num_warps=4,
        num_stages=1,
    )
    return out


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output)


def persistent_forward_4layer(
    x: torch.Tensor,
    packed_layers: List[PackedLayerWeights],
    linear: nn.Linear,
    block_m: int = 16,
) -> torch.Tensor:
    layer_input = x

    for layer_idx, packed in enumerate(packed_layers):
        wx = _project_inputs(layer_input, packed)
        if layer_idx < len(packed_layers) - 1:
            layer_input = persistent_lstm_layer_sequence(wx, packed, block_m=block_m)
        else:
            last_output = persistent_lstm_layer_last(wx, packed, block_m=block_m)

    return F.linear(last_output, linear.weight, linear.bias)


def benchmark_demo() -> None:
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    device = "cuda:0"
    seq_length = 1000
    batch_size = 512
    input_size = 5
    hidden_size = 128
    output_size = 24
    n_layers = 4
    iterations = 100
    block_m = int(os.environ.get("LSTM_BLOCK_M", "16"))

    native_model = LSTMRegressor(
        input_dim=input_size,
        hidden_dim=hidden_size,
        output_dim=output_size,
        n_layers=n_layers,
    ).to(device).half()
    native_model.eval()

    packed_layers = [_pack_layer_weights(native_model.lstm, layer_idx) for layer_idx in range(n_layers)]
    x = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        native_out = native_model(x)
        persistent_out = persistent_forward_4layer(x, packed_layers, native_model.linear, block_m=block_m)
        diff = torch.max(torch.abs(native_out - persistent_out))
        print("当前对比口径: 4 层 LSTM + Linear，每层 1 个 persistent recurrent kernel")
        print("设计参考: recurrent weights 在时间循环外加载一次，再跨 timestep 复用")
        print(f"FP16 max diff: {diff.item():.6f}")
        print(f"BLOCK_M: {block_m}")

    for _ in range(10):
        _ = native_model(x)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = native_model(x)
    torch.cuda.synchronize()
    native_time = time.time() - start_time

    for _ in range(10):
        _ = persistent_forward_4layer(x, packed_layers, native_model.linear, block_m=block_m)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = persistent_forward_4layer(x, packed_layers, native_model.linear, block_m=block_m)
    torch.cuda.synchronize()
    persistent_time = time.time() - start_time

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Per-layer persistent Triton time: {persistent_time:.4f} s")
    print(f"Speedup: {native_time / persistent_time:.2f}x")


if __name__ == "__main__":
    benchmark_demo()
