# CLAUDE.md — 项目开发规范

面向海光 DCU 的 LSTM 推理优化项目。所有代码修改、实验、文档更新遵循以下规则。

## 1. 官方架构核心思想

优化方向必须对齐 MIOpen RNN 官方设计，不凭空造轮子。

### MIOpen 数据流（核心参考）

```
workspace gate block = X @ Wx           # PropX
workspace gate block += bias            # AddBias

for each timestep:
    workspace gate[t] += H[t-1] @ Wh   # PropHiddenHt（GEMM 累加到 gate）
    hidden update kernel 只读 gate[t]   # LSTMForwardHiddenStateUpdate
    write Ct, Ht
```

### 关键设计原则

1. **Gate accumulation**：循环 GEMM 直接累加到 gate workspace（beta=1），避免独立的 `recur` tensor
2. **GEMM 优先**：用 rocBLAS GEMM（tensor core）做矩阵乘，标量 matmul 只作为回退
3. **Hidden update 职责单一**：只消费已累加的 gate block，不做 recurrent matmul
4. **权重跨 timestep 复用**：权重不变，应驻留 LDS 避免重复从 global memory 加载
5. **Persistent kernel**：每层 1 次 kernel 启动，消除 per-timestep 调度开销

### CK（Composable Kernel）参考

- LDS 驻留权重 tile + 双缓冲（Pipeline v2）
- XDLops/MMAC 矩阵指令做 tile 级 GEMM
- 波前级驻留控制（`__launch_bounds__`）

### 不做什么

- 不为每个 hidden_size 手写专用 kernel（走参数化模板）
- 不改变数学布局不做 A/B 精度对比
- 不依赖 CUDA/HIP graph 不做独立验证
- 不优化非热点路径（allocation、workspace 非循环部分）
- 不把直接 hipBLAS 作为唯一路径（保留 PyTorch GEMM 回退）

## 2. Git 工作流

**每轮代码修改后必须 commit**，方便回退和追溯。

```bash
# 修改完代码后
git add <changed_files>
git commit -m "简明描述做了什么改动"
```

commit message 规范：
- 动词开头：Add / Fix / Optimize / Refactor / Revert
- 包含关键数据（如性能变化、精度变化）
- 实验性改动标注 "(experimental)"，回退标注 "(reverted)"

示例：
```
Optimize MMAC kernel: A-fragment reuse, saves 75% h_lds loads
Revert weight cross-timestep residency (8.78s vs 7.34s baseline)
Fix MMAC lane mapping for 64-thread HCU builtin
```

**回退方法**：
```bash
git checkout <commit-hash> -- <file>    # 回退单个文件
git revert <commit-hash>                # 回退整个 commit
```

## 3. README 维护规则

每次有以下进展必须更新 README.md：

### 性能数据更新
- 每次优化后，更新性能记录表格（耗时、吞吐、精度）
- 新增路径或变体，补充到后端对比表
- MMAC 优化迭代记录表保持最新

### 问题解决记录
- 解决的关键技术问题（如 DTK intrinsic 签名、lane mapping）
- 失败的尝试也记录（标注"已回退"），避免重复踩坑

### 优化路线更新
- 完成的项目打 ✅
- 新增的优化方向追加到路线图
- 调整优先级

### 新增内容
- 新环境变量、新后端、新使用方式
- 新的参考文档、架构图
- 目录结构变化

## 4. 代码修改注意事项

### C++/HIP kernel
- 新 kernel 用 `#ifdef` 保护，setup.py 传 `-D` 宏控制
- LDS 总量不超过 64KB，编译时检查 `local memory exceeds limit`
- MMAC tile size 固定 16×16×16，K100_AI 用 64 线程 lane mapping
- 变量命名统一：`mmac_*`（非 `mfma_*`），HCU builtin 用 `__builtin_hcu_mmac_*`

### Python 端
- 新后端添加到 `_recurrent_backend()` 白名单
- 环境变量命名 `MIOPEN_ADAPTIVE_LSTM_*`
- 后端名作为 `persistent_mmac`、`gemm_scan` 等简洁标识

### 测试
- 单 shape 测试：`MIOPEN_ADAPTIVE_LSTM_DEBUG=1 python run_adaptive_lstm.py`
- 多 shape：`python run_shape_sweep.py`
- 对比基线：`python run_native_lstm_sweep.py`
- 默认 shape: `input=5, hidden=128, layers=4, output=24, seq_len=1000, batch=512`

## 5. 硬件/编译器信息

- GPU：Hygon C86-4G (K100_AI)，gfx928
- 编译器：DTK dcc 25.10.0-0 (clang 17)
- HCU MMAC builtin：`__builtin_hcu_mmac_f32_16x16x16_f16(f16x4 a, f16x4 b, f32x4 c)`
- LDS：64KB / CU，wavefront 64 线程
- `f16x4` = `_Float16 __attribute__((vector_size(8)))`
- `f32x4` = `float __attribute__((vector_size(16)))`
