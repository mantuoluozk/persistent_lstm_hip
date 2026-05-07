import math
import time
from typing import List

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _persistent_lstm_sequence_kernel(
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

    for t in range(seq_len):
        base_wx = wx_ptr + offs_m[:, None] * stride_wx_b + t * stride_wx_s + offs_h[None, :] * stride_wx_d
        wx_i = tl.load(base_wx + 0 * gate_stride, mask=mask_bh, other=0.0)
        wx_f = tl.load(base_wx + 1 * gate_stride, mask=mask_bh, other=0.0)
        wx_g = tl.load(base_wx + 2 * gate_stride, mask=mask_bh, other=0.0)
        wx_o = tl.load(base_wx + 3 * gate_stride, mask=mask_bh, other=0.0)

        w_hi = tl.load(wh_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
        i_raw = (wx_i + tl.dot(h, w_hi)).to(tl.float32)
        i_gate = tl.sigmoid(i_raw).to(dtype)

        w_hf = tl.load(wh_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
        f_raw = (wx_f + tl.dot(h, w_hf)).to(tl.float32)
        f_gate = tl.sigmoid(f_raw).to(dtype)

        w_hg = tl.load(wh_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
        g_raw = (wx_g + tl.dot(h, w_hg)).to(tl.float32)
        g_gate = (2.0 * tl.sigmoid(2.0 * g_raw) - 1.0).to(dtype)

        w_ho = tl.load(wh_o_ptr + ptrs_w, mask=mask_wh, other=0.0)
        o_raw = (wx_o + tl.dot(h, w_ho)).to(tl.float32)
        o_gate = tl.sigmoid(o_raw).to(dtype)

        c = (f_gate * c + i_gate * g_gate).to(dtype)
        c_fp32 = c.to(tl.float32)
        h = (o_gate * (2.0 * tl.sigmoid(2.0 * c_fp32) - 1.0).to(dtype)).to(dtype)

        out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + t * stride_out_s + offs_h[None, :] * stride_out_d
        tl.store(out_ptrs, h, mask=mask_bh)


@triton.jit
def _persistent_lstm_last_kernel(
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

    for t in range(seq_len):
        base_wx = wx_ptr + offs_m[:, None] * stride_wx_b + t * stride_wx_s + offs_h[None, :] * stride_wx_d
        wx_i = tl.load(base_wx + 0 * gate_stride, mask=mask_bh, other=0.0)
        wx_f = tl.load(base_wx + 1 * gate_stride, mask=mask_bh, other=0.0)
        wx_g = tl.load(base_wx + 2 * gate_stride, mask=mask_bh, other=0.0)
        wx_o = tl.load(base_wx + 3 * gate_stride, mask=mask_bh, other=0.0)

        w_hi = tl.load(wh_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
        i_raw = (wx_i + tl.dot(h, w_hi)).to(tl.float32)
        i_gate = tl.sigmoid(i_raw).to(dtype)

        w_hf = tl.load(wh_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
        f_raw = (wx_f + tl.dot(h, w_hf)).to(tl.float32)
        f_gate = tl.sigmoid(f_raw).to(dtype)

        w_hg = tl.load(wh_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
        g_raw = (wx_g + tl.dot(h, w_hg)).to(tl.float32)
        g_gate = (2.0 * tl.sigmoid(2.0 * g_raw) - 1.0).to(dtype)

        w_ho = tl.load(wh_o_ptr + ptrs_w, mask=mask_wh, other=0.0)
        o_raw = (wx_o + tl.dot(h, w_ho)).to(tl.float32)
        o_gate = tl.sigmoid(o_raw).to(dtype)

        c = (f_gate * c + i_gate * g_gate).to(dtype)
        c_fp32 = c.to(tl.float32)
        h = (o_gate * (2.0 * tl.sigmoid(2.0 * c_fp32) - 1.0).to(dtype)).to(dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_h[None, :] * stride_out_d
    tl.store(out_ptrs, h, mask=mask_bh)


def _split_recurrent_weights(weight_hh: torch.Tensor):
    hidden_dim = weight_hh.shape[1]
    wh_i = weight_hh[0:hidden_dim, :].t().contiguous()
    wh_f = weight_hh[hidden_dim : 2 * hidden_dim, :].t().contiguous()
    wh_g = weight_hh[2 * hidden_dim : 3 * hidden_dim, :].t().contiguous()
    wh_o = weight_hh[3 * hidden_dim : 4 * hidden_dim, :].t().contiguous()
    return wh_i, wh_f, wh_g, wh_o


def _project_inputs(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    wx = torch.matmul(x, weight_ih.t())
    wx = wx + bias_ih.view(1, 1, -1)
    wx = wx + bias_hh.view(1, 1, -1)
    return wx.contiguous()


def persistent_lstm_layer_sequence(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_m: int = 64,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    hidden_dim = weight_hh.shape[1]

    wx = _project_inputs(x, weight_ih, bias_ih, bias_hh)
    wh_i, wh_f, wh_g, wh_o = _split_recurrent_weights(weight_hh)
    out = torch.empty((batch_size, seq_len, hidden_dim), device=x.device, dtype=x.dtype)

    block_hid = triton.next_power_of_2(hidden_dim)
    grid = (triton.cdiv(batch_size, block_m),)

    _persistent_lstm_sequence_kernel[grid](
        wx,
        wh_i,
        wh_f,
        wh_g,
        wh_o,
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
        BLOCK_HID=block_hid,
        num_warps=4,
        num_stages=1,
    )
    return out


def persistent_lstm_layer_last(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    block_m: int = 64,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    hidden_dim = weight_hh.shape[1]

    wx = _project_inputs(x, weight_ih, bias_ih, bias_hh)
    wh_i, wh_f, wh_g, wh_o = _split_recurrent_weights(weight_hh)
    out = torch.empty((batch_size, hidden_dim), device=x.device, dtype=x.dtype)

    block_hid = triton.next_power_of_2(hidden_dim)
    grid = (triton.cdiv(batch_size, block_m),)

    _persistent_lstm_last_kernel[grid](
        wx,
        wh_i,
        wh_f,
        wh_g,
        wh_o,
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
        BLOCK_HID=block_hid,
        num_warps=4,
        num_stages=1,
    )
    return out


class PersistentLSTMLayerWeights(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)


class PersistentTritonLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float = 0.2,
        block_m: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.block_m = block_m

        layers: List[PersistentLSTMLayerWeights] = []
        for layer_idx in range(n_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(PersistentLSTMLayerWeights(layer_input_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_input = x

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.n_layers - 1:
                layer_output = persistent_lstm_layer_sequence(
                    layer_input,
                    layer.weight_ih,
                    layer.weight_hh,
                    layer.bias_ih,
                    layer.bias_hh,
                    block_m=self.block_m,
                )
                if self.training and self.dropout_p > 0.0:
                    layer_input = self.dropout(layer_output)
                else:
                    layer_input = layer_output
            else:
                last_output = persistent_lstm_layer_last(
                    layer_input,
                    layer.weight_ih,
                    layer.weight_hh,
                    layer.bias_ih,
                    layer.bias_hh,
                    block_m=self.block_m,
                )

        if self.training and self.dropout_p > 0.0:
            last_output = self.dropout(last_output)
        return self.linear(last_output)

    @classmethod
    def from_native_module(
        cls,
        native_module: nn.Module,
        block_m: int = 64,
    ) -> "PersistentTritonLSTMRegressor":
        lstm = native_module.lstm
        dropout_module = getattr(native_module, "dropout", None)
        dropout = dropout_module.p if dropout_module is not None else getattr(lstm, "dropout", 0.0)

        triton_module = cls(
            input_dim=lstm.input_size,
            hidden_dim=lstm.hidden_size,
            output_dim=native_module.linear.out_features,
            n_layers=lstm.num_layers,
            dropout=dropout,
            block_m=block_m,
        )
        triton_module.load_from_native_module(native_module)
        return triton_module

    def load_from_native_module(self, native_module: nn.Module) -> None:
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                layer.weight_ih.copy_(getattr(native_module.lstm, f"weight_ih_l{layer_idx}"))
                layer.weight_hh.copy_(getattr(native_module.lstm, f"weight_hh_l{layer_idx}"))
                layer.bias_ih.copy_(getattr(native_module.lstm, f"bias_ih_l{layer_idx}"))
                layer.bias_hh.copy_(getattr(native_module.lstm, f"bias_hh_l{layer_idx}"))

            self.linear.weight.copy_(native_module.linear.weight)
            self.linear.bias.copy_(native_module.linear.bias)


class NativeLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = self.dropout(lstm_out[:, -1, :]) if self.training else lstm_out[:, -1, :]
        return self.linear(last_output)


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

    native_model = NativeLSTMRegressor(
        input_dim=input_size,
        hidden_dim=hidden_size,
        output_dim=output_size,
        n_layers=n_layers,
        dropout=0.2,
    ).to(device).half()

    triton_model = PersistentTritonLSTMRegressor.from_native_module(native_model, block_m=64).to(device).half()

    native_model.eval()
    triton_model.eval()

    input_tensor = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        native_out = native_model(input_tensor)
        triton_out = triton_model(input_tensor)
        diff = torch.max(torch.abs(native_out - triton_out))
        print(f"FP16 max diff: {diff.item():.6f}")

    for _ in range(10):
        _ = native_model(input_tensor)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = native_model(input_tensor)
    torch.cuda.synchronize()
    native_time = time.time() - start_time

    for _ in range(10):
        _ = triton_model(input_tensor)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = triton_model(input_tensor)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Persistent Triton time: {triton_time:.4f} s")
    print(f"Speedup: {native_time / triton_time:.2f}x")


if __name__ == "__main__":
    benchmark_demo()
