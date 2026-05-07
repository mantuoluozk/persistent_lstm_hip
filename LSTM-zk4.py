import os
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _persistent_lstm_4layer_last_kernel(
    x_ptr,
    w_ih_ptr,
    w_hh_ptr,
    b_ptr,
    w_out_ptr,
    b_out_ptr,
    out_ptr,
    seq_len,
    batch_size,
    input_dim,
    output_dim,
    stride_x_b,
    stride_x_s,
    stride_x_d,
    stride_out_b,
    stride_out_d,
    BLOCK_M: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    OUTPUT_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_HID)
    offs_o = tl.arange(0, OUTPUT_PAD)

    mask_m = offs_m < batch_size
    mask_h = offs_h < BLOCK_HID
    mask_bh = mask_m[:, None] & mask_h[None, :]
    mask_wh = mask_h[:, None] & mask_h[None, :]
    mask_out = mask_m[:, None] & (offs_o[None, :] < output_dim)
    mask_wout = mask_h[:, None] & (offs_o[None, :] < OUTPUT_PAD)

    dtype = x_ptr.dtype.element_ty

    h0 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    h1 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    h2 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    h3 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c0 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c1 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c2 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c3 = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    ptrs_hh = offs_h[:, None] * BLOCK_HID + offs_h[None, :]
    ptrs_out = offs_h[:, None] * OUTPUT_PAD + offs_o[None, :]
    gate_matrix_stride = BLOCK_HID * BLOCK_HID
    layer_matrix_stride = 4 * gate_matrix_stride
    layer_bias_stride = 4 * BLOCK_HID

    for t in range(seq_len):
        base_x = x_ptr + offs_m[:, None] * stride_x_b + t * stride_x_s + offs_h[None, :] * stride_x_d
        layer_input = tl.load(
            base_x,
            mask=mask_m[:, None] & (offs_h[None, :] < input_dim),
            other=0.0,
        )

        layer0_matrix_base = 0 * layer_matrix_stride
        layer0_bias_base = 0 * layer_bias_stride
        w0_ii = tl.load(w_ih_ptr + layer0_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_if = tl.load(w_ih_ptr + layer0_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_ig = tl.load(w_ih_ptr + layer0_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_io = tl.load(w_ih_ptr + layer0_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_hi = tl.load(w_hh_ptr + layer0_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_hf = tl.load(w_hh_ptr + layer0_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_hg = tl.load(w_hh_ptr + layer0_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w0_ho = tl.load(w_hh_ptr + layer0_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        b0_i = tl.load(b_ptr + layer0_bias_base + 0 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b0_f = tl.load(b_ptr + layer0_bias_base + 1 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b0_g = tl.load(b_ptr + layer0_bias_base + 2 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b0_o = tl.load(b_ptr + layer0_bias_base + 3 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        i0_raw = (tl.dot(layer_input, w0_ii) + tl.dot(h0, w0_hi) + b0_i).to(tl.float32)
        f0_raw = (tl.dot(layer_input, w0_if) + tl.dot(h0, w0_hf) + b0_f).to(tl.float32)
        g0_raw = (tl.dot(layer_input, w0_ig) + tl.dot(h0, w0_hg) + b0_g).to(tl.float32)
        o0_raw = (tl.dot(layer_input, w0_io) + tl.dot(h0, w0_ho) + b0_o).to(tl.float32)
        i0 = tl.sigmoid(i0_raw).to(dtype)
        f0 = tl.sigmoid(f0_raw).to(dtype)
        g0 = (2.0 * tl.sigmoid(2.0 * g0_raw) - 1.0).to(dtype)
        o0 = tl.sigmoid(o0_raw).to(dtype)
        c0 = (f0 * c0 + i0 * g0).to(dtype)
        h0 = (o0 * (2.0 * tl.sigmoid(2.0 * c0.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        layer1_input = h0
        layer1_matrix_base = 1 * layer_matrix_stride
        layer1_bias_base = 1 * layer_bias_stride
        w1_ii = tl.load(w_ih_ptr + layer1_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_if = tl.load(w_ih_ptr + layer1_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_ig = tl.load(w_ih_ptr + layer1_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_io = tl.load(w_ih_ptr + layer1_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_hi = tl.load(w_hh_ptr + layer1_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_hf = tl.load(w_hh_ptr + layer1_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_hg = tl.load(w_hh_ptr + layer1_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w1_ho = tl.load(w_hh_ptr + layer1_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        b1_i = tl.load(b_ptr + layer1_bias_base + 0 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b1_f = tl.load(b_ptr + layer1_bias_base + 1 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b1_g = tl.load(b_ptr + layer1_bias_base + 2 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b1_o = tl.load(b_ptr + layer1_bias_base + 3 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        i1_raw = (tl.dot(layer1_input, w1_ii) + tl.dot(h1, w1_hi) + b1_i).to(tl.float32)
        f1_raw = (tl.dot(layer1_input, w1_if) + tl.dot(h1, w1_hf) + b1_f).to(tl.float32)
        g1_raw = (tl.dot(layer1_input, w1_ig) + tl.dot(h1, w1_hg) + b1_g).to(tl.float32)
        o1_raw = (tl.dot(layer1_input, w1_io) + tl.dot(h1, w1_ho) + b1_o).to(tl.float32)
        i1 = tl.sigmoid(i1_raw).to(dtype)
        f1 = tl.sigmoid(f1_raw).to(dtype)
        g1 = (2.0 * tl.sigmoid(2.0 * g1_raw) - 1.0).to(dtype)
        o1 = tl.sigmoid(o1_raw).to(dtype)
        c1 = (f1 * c1 + i1 * g1).to(dtype)
        h1 = (o1 * (2.0 * tl.sigmoid(2.0 * c1.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        layer2_input = h1
        layer2_matrix_base = 2 * layer_matrix_stride
        layer2_bias_base = 2 * layer_bias_stride
        w2_ii = tl.load(w_ih_ptr + layer2_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_if = tl.load(w_ih_ptr + layer2_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_ig = tl.load(w_ih_ptr + layer2_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_io = tl.load(w_ih_ptr + layer2_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_hi = tl.load(w_hh_ptr + layer2_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_hf = tl.load(w_hh_ptr + layer2_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_hg = tl.load(w_hh_ptr + layer2_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w2_ho = tl.load(w_hh_ptr + layer2_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        b2_i = tl.load(b_ptr + layer2_bias_base + 0 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b2_f = tl.load(b_ptr + layer2_bias_base + 1 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b2_g = tl.load(b_ptr + layer2_bias_base + 2 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b2_o = tl.load(b_ptr + layer2_bias_base + 3 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        i2_raw = (tl.dot(layer2_input, w2_ii) + tl.dot(h2, w2_hi) + b2_i).to(tl.float32)
        f2_raw = (tl.dot(layer2_input, w2_if) + tl.dot(h2, w2_hf) + b2_f).to(tl.float32)
        g2_raw = (tl.dot(layer2_input, w2_ig) + tl.dot(h2, w2_hg) + b2_g).to(tl.float32)
        o2_raw = (tl.dot(layer2_input, w2_io) + tl.dot(h2, w2_ho) + b2_o).to(tl.float32)
        i2 = tl.sigmoid(i2_raw).to(dtype)
        f2 = tl.sigmoid(f2_raw).to(dtype)
        g2 = (2.0 * tl.sigmoid(2.0 * g2_raw) - 1.0).to(dtype)
        o2 = tl.sigmoid(o2_raw).to(dtype)
        c2 = (f2 * c2 + i2 * g2).to(dtype)
        h2 = (o2 * (2.0 * tl.sigmoid(2.0 * c2.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

        layer3_input = h2
        layer3_matrix_base = 3 * layer_matrix_stride
        layer3_bias_base = 3 * layer_bias_stride
        w3_ii = tl.load(w_ih_ptr + layer3_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_if = tl.load(w_ih_ptr + layer3_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_ig = tl.load(w_ih_ptr + layer3_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_io = tl.load(w_ih_ptr + layer3_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_hi = tl.load(w_hh_ptr + layer3_matrix_base + 0 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_hf = tl.load(w_hh_ptr + layer3_matrix_base + 1 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_hg = tl.load(w_hh_ptr + layer3_matrix_base + 2 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        w3_ho = tl.load(w_hh_ptr + layer3_matrix_base + 3 * gate_matrix_stride + ptrs_hh, mask=mask_wh, other=0.0)
        b3_i = tl.load(b_ptr + layer3_bias_base + 0 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b3_f = tl.load(b_ptr + layer3_bias_base + 1 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b3_g = tl.load(b_ptr + layer3_bias_base + 2 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        b3_o = tl.load(b_ptr + layer3_bias_base + 3 * BLOCK_HID + offs_h, mask=mask_h, other=0.0)[None, :]
        i3_raw = (tl.dot(layer3_input, w3_ii) + tl.dot(h3, w3_hi) + b3_i).to(tl.float32)
        f3_raw = (tl.dot(layer3_input, w3_if) + tl.dot(h3, w3_hf) + b3_f).to(tl.float32)
        g3_raw = (tl.dot(layer3_input, w3_ig) + tl.dot(h3, w3_hg) + b3_g).to(tl.float32)
        o3_raw = (tl.dot(layer3_input, w3_io) + tl.dot(h3, w3_ho) + b3_o).to(tl.float32)
        i3 = tl.sigmoid(i3_raw).to(dtype)
        f3 = tl.sigmoid(f3_raw).to(dtype)
        g3 = (2.0 * tl.sigmoid(2.0 * g3_raw) - 1.0).to(dtype)
        o3 = tl.sigmoid(o3_raw).to(dtype)
        c3 = (f3 * c3 + i3 * g3).to(dtype)
        h3 = (o3 * (2.0 * tl.sigmoid(2.0 * c3.to(tl.float32)) - 1.0).to(dtype)).to(dtype)

    w_out = tl.load(w_out_ptr + ptrs_out, mask=mask_wout, other=0.0)
    b_out = tl.load(b_out_ptr + offs_o, mask=offs_o < OUTPUT_PAD, other=0.0)[None, :]
    out = (tl.dot(h3, w_out) + b_out).to(dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_o[None, :] * stride_out_d
    tl.store(out_ptrs, out, mask=mask_out)


def _pack_lstm_weights(native_model: nn.Module):
    lstm = native_model.lstm
    linear = native_model.linear

    hidden_dim = lstm.hidden_size
    n_layers = lstm.num_layers
    output_dim = linear.out_features
    output_pad = triton.next_power_of_2(output_dim)

    device = linear.weight.device
    dtype = linear.weight.dtype

    w_ih = torch.zeros((n_layers, 4, hidden_dim, hidden_dim), device=device, dtype=dtype)
    w_hh = torch.zeros((n_layers, 4, hidden_dim, hidden_dim), device=device, dtype=dtype)
    bias = torch.zeros((n_layers, 4, hidden_dim), device=device, dtype=dtype)

    for layer in range(n_layers):
        raw_w_ih = getattr(lstm, f"weight_ih_l{layer}").view(4, hidden_dim, -1)
        raw_w_hh = getattr(lstm, f"weight_hh_l{layer}").view(4, hidden_dim, hidden_dim)
        raw_b_ih = getattr(lstm, f"bias_ih_l{layer}").view(4, hidden_dim)
        raw_b_hh = getattr(lstm, f"bias_hh_l{layer}").view(4, hidden_dim)

        layer_input_dim = raw_w_ih.shape[-1]
        for gate in range(4):
            w_ih[layer, gate, :layer_input_dim, :].copy_(raw_w_ih[gate].t())
            w_hh[layer, gate].copy_(raw_w_hh[gate].t())
            bias[layer, gate].copy_(raw_b_ih[gate] + raw_b_hh[gate])

    w_out = torch.zeros((hidden_dim, output_pad), device=device, dtype=dtype)
    b_out = torch.zeros((output_pad,), device=device, dtype=dtype)
    w_out[:, :output_dim].copy_(linear.weight.t())
    b_out[:output_dim].copy_(linear.bias)

    return (
        w_ih.contiguous(),
        w_hh.contiguous(),
        bias.contiguous(),
        w_out.contiguous(),
        b_out.contiguous(),
        output_dim,
        output_pad,
    )


def persistent_lstm_4layer_last(
    x: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    bias: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
    output_dim: int,
    output_pad: int,
    block_m: int = 16,
) -> torch.Tensor:
    batch_size, seq_len, input_dim = x.shape
    hidden_dim = w_hh.shape[-1]

    if hidden_dim != 128 or w_ih.shape[0] != 4:
        raise ValueError("当前版本是针对 4 层、hidden_size=128 的专用 persistent kernel。")
    if block_m < 16:
        raise ValueError("Triton tl.dot 要求 BLOCK_M >= 16，请使用 16、32 等更大的 tile。")

    out = torch.empty((batch_size, output_dim), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(batch_size, block_m),)

    _persistent_lstm_4layer_last_kernel[grid](
        x,
        w_ih,
        w_hh,
        bias,
        w_out,
        b_out,
        out,
        seq_len,
        batch_size,
        input_dim,
        output_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_HID=hidden_dim,
        OUTPUT_PAD=output_pad,
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


def benchmark_demo():
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

    packed = _pack_lstm_weights(native_model)
    w_ih, w_hh, bias, w_out, b_out, output_dim, output_pad = packed

    x = torch.randn((batch_size, seq_length, input_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        native_out = native_model(x)
        fused_out = persistent_lstm_4layer_last(
            x,
            w_ih,
            w_hh,
            bias,
            w_out,
            b_out,
            output_dim=output_dim,
            output_pad=output_pad,
            block_m=block_m,
        )
        diff = torch.max(torch.abs(native_out - fused_out))
        print("当前对比口径: 4 层 LSTM + Linear，单个 Triton fused recurrent kernel")
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
        _ = persistent_lstm_4layer_last(
            x,
            w_ih,
            w_hh,
            bias,
            w_out,
            b_out,
            output_dim=output_dim,
            output_pad=output_pad,
            block_m=block_m,
        )
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = persistent_lstm_4layer_last(
            x,
            w_ih,
            w_hh,
            bias,
            w_out,
            b_out,
            output_dim=output_dim,
            output_pad=output_pad,
            block_m=block_m,
        )
    torch.cuda.synchronize()
    fused_time = time.time() - start_time

    print(f"Native PyTorch time: {native_time:.4f} s")
    print(f"Fused persistent Triton time: {fused_time:.4f} s")
    print(f"Speedup: {native_time / fused_time:.2f}x")


if __name__ == "__main__":
    benchmark_demo()
