import torch
import torch.nn as nn
#from thop import profile
import time
#from fvcore.nn import FlopCountAnalysis

torch.backends.cudnn.benchmark = True  # 对 ROCm 的 MIOpen 也有效

class LSTMRegressor(nn.Module):
    """LSTM回归模型 - 用于时间序列预测"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        last_output = self.dropout(last_output)

        return self.linear(last_output)  # [batch_size, output_dim]

model = LSTMRegressor(
    input_dim=5,
    hidden_dim=128,
    output_dim=24,
    n_layers=4
).to("cuda:0").half()
# model = torch.compile(model, mode="reduce-overhead")

seq_length = 1000
batch_size = 512
input_size = 5
iterations = 100

# 准备输入数据
input = torch.ones((batch_size, seq_length, input_size)).to("cuda:0").half()

# flop_count = FlopCountAnalysis(model, input)
# Macs, params = profile(model, torch.randn(1, 1, seq_length, input_size).to("cuda:0"))
# print(f"Macs: {Macs}")
# print(f"Params: {params}")
# print(f"Model Flops: {flop_count.total()} Flops")

# Warmup
for _ in range(10):
    _ = model(input)
torch.cuda.synchronize()
# 测试模型的运行时间
start_time = time.time()
for _ in range(iterations):
    output_new = model(input)
    #print(output_new.shape)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(f"吞吐量(含batchsize):", iterations / elapsed_time * batch_size)
