# AI 科普视频工厂 · GitHub / Streamlit 单文件版

这是从本地 `D:\AI-Video-Factory` 整理出的自包含 Streamlit 应用。

## 文件

```text
app.py                 单文件应用
requirements.txt       Python 依赖
packages.txt           Streamlit Cloud 系统包（FFmpeg + Noto CJK）
runtime.txt            Python 3.12
.streamlit/config.toml 主题
.github/workflows/ci.yml 编译检查
deploy_to_github.ps1   有 Git/GitHub CLI 时的推送辅助脚本
```

## 本地运行

```powershell
pip install -r requirements.txt
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="qwenlocal"
streamlit run app.py
```

## Streamlit Cloud

1. 新建仓库并上传本目录文件；
2. 在 Streamlit Cloud 选择 `app.py`；
3. 在 Secrets 中配置：

```toml
OLLAMA_BASE_URL = "https://your-ollama-host.example.com"
OLLAMA_MODEL = "qwenlocal"
```

云端 `127.0.0.1` 不是你的电脑；没有远程 Ollama 时会使用内置已核查脚本。

## 功能

- 主题 → 结构化 JSON 科普脚本；
- 真实调用 Ollama `/api/tags`、`/api/chat`；
- `claim_ids` 校验与事实核查页；
- 11 张 1080×1920 程序化分镜；
- SRT 字幕；
- 可选本地 Piper 配音；
- 可选 FFmpeg 合成 MP4；
- 失败自动回退，不中断页面。

## 限制

- 只内置《为什么 AI 会产生幻觉？》的已核查事实库；
- 云端默认没有 Piper 模型，视频可能静音；
- 不自动发布到任何平台；
- 事实核查仍需人工阅读原文确认。
