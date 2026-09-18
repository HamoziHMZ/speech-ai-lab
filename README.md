# speech-ai-lab

语音 AI 应用工程师学习路线（36 周计划）的实验区。

## 环境

- Python 3.11（由 uv 管理，见 `.python-version`）
- PyTorch 2.11.0+cu128（CUDA 12.8，见 `pyproject.toml`）
- 数据与模型权重放在 `D:\speech-ai\data`，**不进本仓库**

## 复现

```powershell
uv sync                                  # 按 uv.lock 还原依赖
uv run python code\w01_sine_wave.py      # 跑一个脚本
```

## 目录

- `code/`  实验脚本，一周一个文件（`wXX_主题.py`）
- `out/`   运行产物（图、音频），不进 Git

## 已完成

- W01D1 uv 与 Python 环境
- W01D2 CUDA 版 PyTorch 与 GPU 自检
- W01D3 工具链（ffmpeg / HF 镜像 / Obsidian）
- W01D4 第一个音频脚本（正弦波 → wav → 波形图）
