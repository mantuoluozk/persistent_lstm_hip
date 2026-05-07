import torch
import torch.nn as nn
import triton
import triton.language as tl
import time

# ==============================================================================
# V2 终极内存优化版: 斩断寄存器溢出
# ==============================================================================
@triton.jit
def _persistent_lstm_fwd_kernel_v2(
    wx_ptr, wh_i_ptr, wh_f_ptr, wh_g_ptr, wh_o_ptr, out_ptr,
    seq_len, batch_size,
    stride_wx_b, stride_wx_s, stride_wx_d,
    stride_out_b, stride_out_d,
    BLOCK_M: tl.constexpr,
    BLOCK_HID: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_HID)
    mask_m = offs_m < batch_size

    # 初始化隐藏状态 h 和 c
    dtype = wx_ptr.dtype.element_ty
    h = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)
    c = tl.zeros([BLOCK_M, BLOCK_HID], dtype=dtype)

    # 预先计算权重的二维偏移量
    ptrs_w = offs_h[:, None] * BLOCK_HID + offs_h[None, :]

    for t in range(seq_len):
        # 1. 加载输入 (包含了 wx 和 所有的 bias，外部已融合完毕)
        # wx 现在的维度是 512 (4 * 128)
        base_wx = wx_ptr + offs_m[:, None] * stride_wx_b + t * stride_wx_s
        wx_i = tl.load(base_wx + (0 * BLOCK_HID + offs_h)[None, :], mask=mask_m[:, None], other=0.0)
        wx_f = tl.load(base_wx + (1 * BLOCK_HID + offs_h)[None, :], mask=mask_m[:, None], other=0.0)
        wx_g = tl.load(base_wx + (2 * BLOCK_HID + offs_h)[None, :], mask=mask_m[:, None], other=0.0)
        wx_o = tl.load(base_wx + (3 * BLOCK_HID + offs_h)[None, :], mask=mask_m[:, None], other=0.0)

        # 2. 按需加载权重并立即计算 (绝不贪心同时持有 4 个矩阵！)
        # --- i gate ---
        w_hi = tl.load(wh_i_ptr + ptrs_w)
        i_raw = (wx_i + tl.dot(h, w_hi)).to(tl.float32)
        i = tl.sigmoid(i_raw).to(dtype)

        # --- f gate ---
        w_hf = tl.load(wh_f_ptr + ptrs_w)
        f_raw = (wx_f + tl.dot(h, w_hf)).to(tl.float32)
        f = tl.sigmoid(f_raw).to(dtype)

        # --- g gate ---
        w_hg = tl.load(wh_g_ptr + ptrs_w)
        g_raw = (wx_g + tl.dot(h, w_hg)).to(tl.float32)
        g = (2.0 * tl.sigmoid(2.0 * g_raw) - 1.0).to(dtype)

        # --- o gate ---
        w_ho = tl.load(wh_o_ptr + ptrs_w)
        o_raw = (wx_o + tl.dot(h, w_ho)).to(tl.float32)
        o = tl.sigmoid(o_raw).to(dtype)

        # 3. 状态更新
        c = (f * c + i * g).to(dtype)
        c_fp32 = c.to(tl.float32)
        h = (o * (2.0 * tl.sigmoid(2.0 * c_fp32) - 1.0).to(dtype)).to(dtype)

    # 写入结果
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_h[None, :] * stride_out_d
    tl.store(out_ptrs, h, mask=mask_m[:, None])


def triton_lstm_layer_v2(x, weight_ih, weight_hh, bias_ih, bias_hh):
    batch_size, seq_len, input_dim = x.shape
    hidden_dim = weight_hh.shape[1]

    # 【神级优化】: 在外部把所有的线性投影和 Bias 全部算完加好！
    # 内核里只剩最纯粹的 h * w 矩阵乘法
    wx = torch.matmul(x, weight_ih.t()) + bias_ih + bias_hh
    wx = wx.contiguous()

    # 将 4 个权重矩阵彻底切开，方便内核按需取用
    wh_i = weight_hh[0:hidden_dim, :].t().contiguous()
    wh_f = weight_hh[hidden_dim:2*hidden_dim, :].t().contiguous()
    wh_g = weight_hh[2*hidden_dim:3*hidden_dim, :].t().contiguous()
    wh_o = weight_hh[3*hidden_dim:4*hidden_dim, :].t().contiguous()
    
    out = torch.empty((batch_size, hidden_dim), device=x.device, dtype=x.dtype)

    BLOCK_M = 64 # 现在我们可以放心地用 64 了！
    BLOCK_HID = triton.next_power_of_2(hidden_dim)
    grid = (triton.cdiv(batch_size, BLOCK_M), )

    _persistent_lstm_fwd_kernel_v2[grid](
        wx, wh_i, wh_f, wh_g, wh_o, out,
        seq_len, batch_size,
        wx.stride(0), wx.stride(1), wx.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_HID=BLOCK_HID,
        num_warps=4,
        num_stages=1 # 因为权重在循环内加载，关掉流水线避免内存浪费
    )
    return out

if __name__ == "__main__":
    seq_length = 1000
    batch_size = 512
    input_size = 5
    hidden_size = 128
    n_layers = 1
    
    model_native = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True).to("cuda:0").half()
    
    w_ih = model_native.weight_ih_l0
    w_hh = model_native.weight_hh_l0
    b_ih = model_native.bias_ih_l0
    b_hh = model_native.bias_hh_l0
    
    input_tensor = torch.randn((batch_size, seq_length, input_size), device="cuda:0", dtype=torch.float16)

    # 预热验证
    with torch.no_grad():
        out_native, _ = model_native(input_tensor)
        out_native_last = out_native[:, -1, :] 
        out_triton = triton_lstm_layer_v2(input_tensor, w_ih, w_hh, b_ih, b_hh)
        diff = torch.max(torch.abs(out_native_last - out_triton))
        print(f"当前对比口径: 单层 LSTM (n_layers={n_layers})")
        print(f"FP16 最大误差 (Max diff): {diff.item():.4f}")

    # 测速
    iterations = 100
    for _ in range(10): _ = model_native(input_tensor) 
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = model_native(input_tensor)
    torch.cuda.synchronize()
    native_time = time.time() - start_time
    
    for _ in range(10): _ = triton_lstm_layer_v2(input_tensor, w_ih, w_hh, b_ih, b_hh)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        _ = triton_lstm_layer_v2(input_tensor, w_ih, w_hh, b_ih, b_hh)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    print(f"原生 PyTorch 耗时: {native_time:.4f} 秒")
    print(f"纯血 Triton 耗时: {triton_time:.4f} 秒")
    print(f"提速倍数: {native_time / triton_time:.2f}x 倍！")
