"""W01D6 · NumPy 热身：为 W02 线性代数铺路。运行：uv run python code\w01_numpy_warmup.py"""

import numpy as np                                    # 数值计算库，约定别名 np

print("=== 1) 创建与形状 ===")
a = np.arange(6).reshape(2, 3)                        # arange(6)=[0..5]；reshape(2,3) 排成 2 行 3 列
print("a =\n", a)
print("shape:", a.shape, "| dtype:", a.dtype, "| ndim:", a.ndim)   # 形状 / 元素类型 / 维度数

print("\n=== 2) 广播：形状不同也能运算 ===")
print("a + 100 =\n", a + 100)                          # 标量广播：每个元素都 +100
print("a * [1,10,100] =\n", a * np.array([1, 10, 100])) # (2,3) 与 (3,) 广播：每列乘不同系数

print("\n=== 3) 矩阵乘 ===")
b = np.ones((3, 4))                                    # 3×4 的全 1 矩阵
print("a@b shape:", (a @ b).shape)                     # @ 是矩阵乘，(2,3)@(3,4) → (2,4)
print("a@b == np.dot(a,b):", np.allclose(a @ b, np.dot(a, b)))   # 两种写法等价

print("\n=== 4) 轴向统计（最容易搞混的地方）===")
m = np.arange(12).reshape(3, 4)
print("m =\n", m)
print("axis=0 每列平均:", m.mean(axis=0), " shape:", m.mean(axis=0).shape)
print("axis=1 每行平均:", m.mean(axis=1), " shape:", m.mean(axis=1).shape)

print("\n=== 5) 音频里最常用的操作：分帧 ===")
wav = np.arange(16000)                                 # 假装是 1 秒 16kHz 音频
frames = wav[:16000].reshape(-1, 400)                  # -1 = 自动算行数；400 点 = 25ms
print("frames shape:", frames.shape)                   # 应该是 (40, 400) = 40 帧 × 每帧 400 点
print("每帧能量(前5帧):", (frames**2).sum(axis=1)[:5])

print("\n=== 6) SVD（W02 正课内容，今天只求知道怎么调）===")
x = np.random.default_rng(42).normal(size=(5, 3))      # 固定随机种子，结果可复现
u, s, vt = np.linalg.svd(x, full_matrices=False)
print("形状:", u.shape, s.shape, vt.shape)
print("奇异值:", s.round(3))
