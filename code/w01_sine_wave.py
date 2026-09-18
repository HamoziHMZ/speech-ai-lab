"""W01D4 · 生成 440Hz 正弦波 → 存 wav → 画波形图。运行：uv run python code\w01_sine_wave.py"""

from pathlib import Path                            # 标准库，处理文件路径，不用装
import numpy as np                                  # 数值计算库，约定别名 np
import soundfile as sf                              # 读写音频文件（wav 等）
import matplotlib                                   # 画图库
matplotlib.use("Agg")                               # ★ 必须在 import pyplot 之前：只存文件、不弹窗
import matplotlib.pyplot as plt                     # 画图接口，约定别名 plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 指定中文字体，否则画出来是方块
plt.rcParams["axes.unicode_minus"] = False            # 用中文字体后负号会变方块，这行修它

SR, FREQ, DUR, AMP = 16000, 440, 0.5, 0.3         # 采样率/频率Hz/时长秒/幅度，大写=常量不改

out = Path(r"D:\speech-ai\lab\out")                 # r"..." 原始字符串：反斜杠不当转义符
out.mkdir(parents=True, exist_ok=True)              # 建目录：父目录一起建、已存在不报错（幂等）

n = int(SR * DUR)                                   # 采样点数 = 16000 × 2 = 32000
t = np.arange(n) / SR                               # 时间轴数组 [0, 1/16000, ...]，向量化无循环
wave = AMP * np.sin(2 * np.pi * FREQ * t)           # 正弦波 A·sin(2πft)，整批点一次算完

tag = f"f{int(FREQ)}_a{str(AMP).replace('.', 'p')}"  # 把频率和幅度拼进文件名，各次实验不互相覆盖
wav_path = out / f"sine_{tag}.wav"                    # ★ / 不是除法，是路径拼接；f"..." 里能嵌变量
sf.write(wav_path, wave, SR, subtype="PCM_16")      # 写 wav：浮点映射为16位整数，体积=点数×2+44

plt.figure(figsize=(10, 3))                          # 新建画布，10×3 英寸的横向长条
plt.plot(t[:1600], wave[:1600])                      # 只画前1600点=0.1秒=44个周期，画全看不清
plt.ylim(-1.1, 1.1)                                 # 固定纵轴范围，才看得出波峰被削平
print("峰值:", np.abs(wave).max())                   # AMP=1.0 时峰值=1.0，正好贴上限
plt.xlabel("time (s)")                               # 横轴标题
plt.ylabel("amplitude")                              # 纵轴标题
plt.title("440Hz 正弦波")          # 图标题，中文需先设字体（见实验5）
plt.grid(alpha=0.3)                                  # 网格线，alpha=透明度
plt.tight_layout()                                   # 自动收紧边距，防标签被裁掉
plt.savefig(out / f"sine_{tag}.png", dpi=150)        # 图名也带参数，dpi 越大越清晰
size = wav_path.stat().st_size                       # 读文件字节数
print(f"采样点: {n}")                                 # f-string：花括号里填变量值
print(f"wav 大小: {size} 字节（理论值 {n * 2 + 44}）")  # 打印理论值 vs 实际值，方便自查
print(f"png 已保存: {out / f'sine_{tag}.png'}")         # 内层用单引号，避免和外层双引号撞车
