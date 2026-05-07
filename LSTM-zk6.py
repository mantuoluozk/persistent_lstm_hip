import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _persistent_lstm_2layer_seq_kernel(
    wx0_ptr,
    wi1_i_ptr,
    wi1_f_ptr,
    wi1_g_ptr,
    wi1_o_ptr,
    wh0_i_ptr,
    wh0_f_ptr,
    wh0_g_ptr,
    wh0_o_ptr,
    wh1_i_ptr,
    wh1_f_ptr,
    wh1_g_ptr,
    wh1_o_ptr,
    b1_ptr,
    out_ptr,
    seq_len,
    batch_size,
    hidden_dim,
    stride_wx0_b,
    stride_wx0_s,
    stride_wx0_d,
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

    dtype = wx0_ptr.dtype.element_ty
    ptrs_w = offs_h[:, None] * hidden_dim + offs_h[None, :]
    gate_stride = hidden_dim * stride_wx0_d

    h0 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c0 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    h1 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c1 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    wh0_i = tl.load(wh0_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh0_f = tl.load(wh0_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh0_g = tl.load(wh0_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh0_o = tl.load(wh0_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    wi1_i = tl.load(wi1_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi1_f = tl.load(wi1_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi1_g = tl.load(wi1_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi1_o = tl.load(wi1_o_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh1_i = tl.load(wh1_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh1_f = tl.load(wh1_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh1_g = tl.load(wh1_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh1_o = tl.load(wh1_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    b1_i = tl.load(b1_ptr + 0 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b1_f = tl.load(b1_ptr + 1 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b1_g = tl.load(b1_ptr + 2 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b1_o = tl.load(b1_ptr + 3 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]

    for t in range(seq_len):
        base = wx0_ptr + offs_m[:, None] * stride_wx0_b + t * stride_wx0_s + offs_h[None, :] * stride_wx0_d
        x0_i = tl.load(base + 0 * gate_stride, mask=mask_bh, other=0.0)
        x0_f = tl.load(base + 1 * gate_stride, mask=mask_bh, other=0.0)
        x0_g = tl.load(base + 2 * gate_stride, mask=mask_bh, other=0.0)
        x0_o = tl.load(base + 3 * gate_stride, mask=mask_bh, other=0.0)

        i0_raw = (x0_i + tl.dot(h0, wh0_i)).to(tl.float32)
        f0_raw = (x0_f + tl.dot(h0, wh0_f)).to(tl.float32)
        g0_raw = (x0_g + tl.dot(h0, wh0_g)).to(tl.float32)
        o0_raw = (x0_o + tl.dot(h0, wh0_o)).to(tl.float32)

        i0 = tl.sigmoid(i0_raw).to(dtype)
        f0 = tl.sigmoid(f0_raw).to(dtype)
        g0 = (2.0 * tl.sigmoid(2.0 * g0_raw) - 1.0).to(dtype)
        o0 = tl.sigmoid(o0_raw).to(dtype)

        c0 = (f0 * c0 + i0 * g0).to(dtype)
        h0 = (o0 * (2.0 * tl.sigmoid(2.0 * c0.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        i1_raw = (tl.dot(h0, wi1_i) + tl.dot(h1, wh1_i) + b1_i).to(tl.float32)
        f1_raw = (tl.dot(h0, wi1_f) + tl.dot(h1, wh1_f) + b1_f).to(tl.float32)
        g1_raw = (tl.dot(h0, wi1_g) + tl.dot(h1, wh1_g) + b1_g).to(tl.float32)
        o1_raw = (tl.dot(h0, wi1_o) + tl.dot(h1, wh1_o) + b1_o).to(tl.float32)

        i1 = tl.sigmoid(i1_raw).to(dtype)
        f1 = tl.sigmoid(f1_raw).to(dtype)
        g1 = (2.0 * tl.sigmoid(2.0 * g1_raw) - 1.0).to(dtype)
        o1 = tl.sigmoid(o1_raw).to(dtype)

        c1 = (f1 * c1 + i1 * g1).to(dtype)
        h1 = (o1 * (2.0 * tl.sigmoid(2.0 * c1.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + t * stride_out_s + offs_h[None, :] * stride_out_d
        tl.store(out_ptrs, h1, mask=mask_bh)


@triton.jit
def _persistent_lstm_2layer_last_kernel(
    wx2_ptr,
    wi3_i_ptr,
    wi3_f_ptr,
    wi3_g_ptr,
    wi3_o_ptr,
    wh2_i_ptr,
    wh2_f_ptr,
    wh2_g_ptr,
    wh2_o_ptr,
    wh3_i_ptr,
    wh3_f_ptr,
    wh3_g_ptr,
    wh3_o_ptr,
    b3_ptr,
    out_ptr,
    seq_len,
    batch_size,
    hidden_dim,
    stride_wx2_b,
    stride_wx2_s,
    stride_wx2_d,
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

    dtype = wx2_ptr.dtype.element_ty
    ptrs_w = offs_h[:, None] * hidden_dim + offs_h[None, :]
    gate_stride = hidden_dim * stride_wx2_d

    h2 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c2 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    h3 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c3 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    wh2_i = tl.load(wh2_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh2_f = tl.load(wh2_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh2_g = tl.load(wh2_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh2_o = tl.load(wh2_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    wi3_i = tl.load(wi3_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi3_f = tl.load(wi3_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi3_g = tl.load(wi3_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wi3_o = tl.load(wi3_o_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh3_i = tl.load(wh3_i_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh3_f = tl.load(wh3_f_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh3_g = tl.load(wh3_g_ptr + ptrs_w, mask=mask_wh, other=0.0)
    wh3_o = tl.load(wh3_o_ptr + ptrs_w, mask=mask_wh, other=0.0)

    b3_i = tl.load(b3_ptr + 0 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b3_f = tl.load(b3_ptr + 1 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b3_g = tl.load(b3_ptr + 2 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]
    b3_o = tl.load(b3_ptr + 3 * hidden_dim + offs_h, mask=mask_h, other=0.0)[None, :]

    for t in range(seq_len):
        base = wx2_ptr + offs_m[:, None] * stride_wx2_b + t * stride_wx2_s + offs_h[None, :] * stride_wx2_d
        x2_i = tl.load(base + 0 * gate_stride, mask=mask_bh, other=0.0)
        x2_f = tl.load(base + 1 * gate_stride, mask=mask_bh, other=0.0)
        x2_g = tl.load(base + 2 * gate_stride, mask=mask_bh, other=0.0)
        x2_o = tl.load(base + 3 * gate_stride, mask=mask_bh, other=0.0)

        i2_raw = (x2_i + tl.dot(h2, wh2_i)).to(tl.float32)
        f2_raw = (x2_f + tl.dot(h2, wh2_f)).to(tl.float32)
        g2_raw = (x2_g + tl.dot(h2, wh2_g)).to(tl.float32)
        o2_raw = (x2_o + tl.dot(h2, wh2_o)).to(tl.float32)

        i2 = tl.sigmoid(i2_raw).to(dtype)
        f2 = tl.sigmoid(f2_raw).to(dtype)
        g2 = (2.0 * tl.sigmoid(2.0 * g2_raw) - 1.0).to(dtype)
        o2 = tl.sigmoid(o2_raw).to(dtype)

        c2 = (f2 * c2 + i2 * g2).to(dtype)
        h2 = (o2 * (2.0 * tl.sigmoid(2.0 * c2.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        i3_raw = (tl.dot(h2, wi3_i) + tl.dot(h3, wh3_i) + b3_i).to(tl.float32)
        f3_raw = (tl.dot(h2, wi3_f) + tl.dot(h3, wh3_f) + b3_f).to(tl.float32)
        g3_raw = (tl.dot(h2, wi3_g) + tl.dot(h3, wh3_g) + b3_g).to(tl.float32)
        o3_raw = (tl.dot(h2, wi3_o) + tl.dot(h3, wh3_o) + b3_o).to(tl.float32)

        i3 = tl.sigmoid(i3_raw).to(dtype)
        f3 = tl.sigmoid(f3_raw).to(dtype)
        g3 = (2.0 * tl.sigmoid(2.0 * g3_raw) - 1.0).to(dtype)
        o3 = tl.sigmoid(o3_raw).to(dtype)

        c3 = (f3 * c3 + i3 * g3).to(dtype)
        h3 = (o3 * (2.0 * tl.sigmoid(2.0 * c3.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_h[None, :] * stride_out_d
    tl.store(out_ptrs, h3, mask=mask_bh)


@dataclass
class LayerPack:
    wi_i: torch.Tensor
    wi_f: torch.Tensor
    wi_g: torch.Tensor
    wi_o: torch.Tensor
    wh_i: torch.Tensor
    wh_f: torch.Tensor
    wh_g: torch.Tensor
    wh_o: torch.Tensor
    bias: torch.Tensor
    weight_ih_full: torch.Tensor


def _pack_layer(lstm: nn.LSTM, layer_idx: int) -> LayerPack:
    hidden_dim = lstm.hidden_size
    weight_ih = getattr(lstm, f"weight_ih_l{layer_idx}")
    weight_hh = getattr(lstm, f"weight_hh_l{layer_idx}")
    bias = (getattr(lstm, f"bias_ih_l{layer_idx}") + getattr(lstm, f"bias_hh_l{layer_idx}")).contiguous()

    wi_i = weight_ih[0:hidden_dim, :].t().contiguous()
    wi_f = weight_ih[hidden_dim : 2 * hidden_dim, :].t().contiguous()
    wi_g = weight_ih[2 * hidden_dim : 3 * hidden_dim, :].t().contiguous()
    wi_o = weight_ih[3 * hidden_dim : 4 * hidden_dim, :].t().contiguous()

    wh_i = weight_hh[0:hidden_dim, :].t().contiguous()
    wh_f = weight_hh[hidden_dim : 2 * hidden_dim, :].t().contiguous()
    wh_g = weight_hh[2 * hidden_dim : 3 * hidden_dim, :].t().contiguous()
    wh_o = weight_hh[3 * hidden_dim : 4 * hidden_dim, :].t().contiguous()

    return LayerPack(
        wi_i=wi_i,
        wi_f=wi_f,
        wi_g=wi_g,
        wi_o=wi_o,
        wh_i=wh_i,
        wh_f=wh_f,
        wh_g=wh_g,
        wh_o=wh_o,
        bias=bias,
        weight_ih_full=weight_ih.contiguous(),
    )


def _preproject(x: torch.Tensor, pack: LayerPack) -> torch.Tensor:
    return F.linear(x, pack.weight_ih_full, pack.bias).contiguous()


def persistent_pair_01(
    x: torch.Tensor,
    layer0: LayerPack,
    layer1: LayerPack,
    block_m: int = 16,
) -> torch.Tensor:
    wx0 = _preproject(x, layer0)
    batch_size, seq_len, _ = wx0.shape
    hidden_dim = layer0.wh_i.shape[0]
    out = torch.empty((batch_size, seq_len, hidden_dim), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(batch_size, block_m),)
    _persistent_lstm_2layer_seq_kernel[grid](
        wx0,
        layer1.wi_i,
        layer1.wi_f,
        layer1.wi_g,
        layer1.wi_o,
        layer0.wh_i,
        layer0.wh_f,
        layer0.wh_g,
        layer0.wh_o,
        layer1.wh_i,
        layer1.wh_f,
        layer1.wh_g,
        layer1.wh_o,
        layer1.bias,
        out,
        seq_len,
        batch_size,
        hidden_dim,
        wx0.stride(0),
        wx0.stride(1),
        wx0.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_HID=hidden_dim,
        num_warps=4,
        num_stages=1,
    )
    return out


def persistent_pair_23_last(
    x: torch.Tensor,
    layer2: LayerPack,
    layer3: LayerPack,
    block_m: int = 16,
) -> torch.Tensor:
    wx2 = _preproject(x, layer2)
    batch_size, seq_len, _ = wx2.shape
    hidden_dim = layer2.wh_i.shape[0]
    out = torch.empty((batch_size, hidden_dim), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(batch_size, block_m),)
    _persistent_lstm_2layer_last_kernel[grid](
        wx2,
        layer3.wi_i,
        layer3.wi_f,
        layer3.wi_g,
        layer3.wi_o,
        layer2.wh_i,
        layer2.wh_f,
        layer2.wh_g,
        layer2.wh_o,
        layer3.wh_i,
        layer3.wh_f,
        layer3.wh_g,
        layer3.wh_o,
        layer3.bias,
        out,
        seq_len,
        batch_size,
        hidden_dim,
        wx2.stride(0),
        wx2.stride(1),
        wx2.stride(2),
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = self.dropout(lstm_out[:, -1, :])
        return self.linear(last_output)


def fused_forward_4layer_2kernels(
    x: torch.Tensor,
    packs: list[LayerPack],
    linear: nn.Linear,
    block_m: int = 16,
) -> torch.Tensor:
    seq_after_l1 = persistent_pair_01(x, packs[0], packs[1], block_m=block_m)
    last_after_l3 = persistent_pair_23_last(seq_after_l1, packs[2], packs[3], block_m=block_m)
    return F.linear(last_after_l3, linear.weight, linear.bias)


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

    packs = [_pack_layer(native_model.lstm, i) for i in range(n_layers)]
    x = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        native_out = native_model(x)
        fused_out = fused_forward_4layer_2kernels(x, packs, native_model.linear, block_m=block_m)
        diff = torch.max(torch.abs(native_out - fused_out))
        print("当前对比口径: 4 层 LSTM + Linear，2 个双层 persistent kernel")
        print("设计参考: recurrent weights 在时间循环外加载一次，中间整段序列只写回 1 次")
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
        _ = fused_forward_4layer_2kernels(x, packs, native_model.linear, block_m=block_m)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = fused_forward_4layer_2kernels(x, packs, native_model.linear, block_m=block_m)
    torch.cuda.synchronize()
    fused_time = time.time() - start_time

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Two-kernel persistent Triton time: {fused_time:.4f} s")
    print(f"Speedup: {native_time / fused_time:.2f}x")


if __name__ == "__main__":
    benchmark_demo()
