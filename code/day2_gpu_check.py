"""W01D2：GPU 计时实验——证明不加 synchronize 会得到假数字。"""
import time
import torch

print("GPU:", torch.cuda.get_device_name(0))

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")

# 预热：第一次算要初始化 CUDA context、编译 kernel，必须丢掉
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()

# ---- 错误示范：不加 synchronize，量到的是"排队时间" ----
t0 = time.perf_counter()
for _ in range(20):
    _ = a @ b
wrong = (time.perf_counter() - t0) / 20 * 1000      # ms

# ---- 正确示范：首尾各 synchronize 一次，等 GPU 干完 ----
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _ = a @ b
torch.cuda.synchronize()
right = (time.perf_counter() - t0) / 20 * 1000      # ms

flops = 2 * 4096**3                                  # 4096³ 矩阵乘的浮点运算量
print(f"不 sync : {wrong:8.3f} ms/次   ← 假数字（只量到了内核派发时间）")
print(f"加 sync : {right:8.3f} ms/次   ← 真数字")
print(f"倍数差  : {right / wrong:8.1f}×")
print(f"反推算力: {flops / (right / 1000) / 1e12:5.1f} TFLOPS (fp32)")
print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
