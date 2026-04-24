# VoxCPM2 中文配音脚本

这个目录里已经准备了一个本地 **语音生成 + 语音转文本** 工作台：

- `tts_voxcpm2.py`：语音生成 CLI。
- `voxcpm_gui.py`：桌面版 UI 入口，包含“语音生成 / 语音转文本”两个页面。
- `voxcpm_service.py`：TTS 核心服务层。
- `asr_service.py`：ASR 核心服务层。
- `sample_text.txt`：示例文案，可以直接替换成你的配音文本。
- `requirements-voxcpm2.txt`：脚本需要的最小 Python 包。
- `requirements-voxcpm2-gui.txt`：桌面版额外依赖。

## 1. 安装依赖

```powershell
python -m pip install -r requirements-voxcpm2.txt
python -m pip install voxcpm --no-deps
```

如果要运行桌面版和本地转写，再补一条：

```powershell
python -m pip install -r requirements-voxcpm2-gui.txt
```

如果 Hugging Face 下载很慢，可以在同一个 PowerShell 窗口里先设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 2. 生成示例配音

先检查依赖和 CUDA：

```powershell
python .\tts_voxcpm2.py --check-env
```

确认没问题后生成示例：

```powershell
python .\tts_voxcpm2.py
```

如果程序目录存在 `VoxCPM2`，或者同级 `model/VoxCPM2` 已经放好，脚本会默认直接使用本地模型，不再联网下载。

默认会读取 `sample_text.txt`，输出到：

```text
outputs/voxcpm2_output.wav
```

## 3. 直接传入一句话

```powershell
python .\tts_voxcpm2.py --text "大家好，欢迎来到我的频道。今天我们来讲一个非常实用的模型。" --output outputs/demo.wav
```

## 4. 调整声音风格

```powershell
python .\tts_voxcpm2.py --voice "成熟男性，沉稳自然，普通话标准，适合纪录片旁白"
```

也可以指定情绪和语速：

```powershell
python .\tts_voxcpm2.py --voice "年轻女性，温柔亲切，语速稍慢，情绪积极"
```

## 5. 使用授权参考音频做音色克隆

```powershell
python .\tts_voxcpm2.py --reference-wav .\speaker.wav --voice "自然、清晰、适合视频讲解"
```

如果有参考音频的逐字稿，可以使用更高相似度的方式：

```powershell
python .\tts_voxcpm2.py --reference-wav .\speaker.wav --prompt-wav .\speaker.wav --prompt-text "这里填写 speaker.wav 里实际说的话。"
```

只克隆你自己或已授权的声音，不要用于冒充他人。

## 6. 12GB 显存建议

- 先用默认 `--chunk-max-chars 120` 生成，稳定后再调大。
- 如果显存紧张，关闭其他占用 GPU 的软件，保持默认不加载 denoiser。
- 长文建议分段，脚本会自动拼接并在段落之间加入短静音。
- `--steps` 默认是 `10`，调高可能音质略有变化，但会更慢。
- 脚本默认关闭 `torch.compile`，首次运行更稳；如果后续想提速，可以加 `--optimize`。

## 7. 桌面版 UI

启动桌面应用：

```powershell
python .\voxcpm_gui.py
```

桌面版支持：

- 语音生成页面：文案输入、音色描述、输出目录选择
- 高级参数：`cfg`、`steps`、`chunk_max_chars`、`silence_ms`
- 可选音色克隆：参考音频、Prompt 音频、Prompt 文本
- Prompt 音频一键自动转写
- 语音转文本页面：支持 `Whisper-large-v3-turbo` 和 `faster-whisper-small`
- TTS / STT 各自独立队列、独立取消、独立结果列表
- TTS 内置试听、打开文件、打开输出目录
- 记住最近一次页签和各页面设置

如果你只是想验证 GUI 依赖是否完整，不真正打开界面：

```powershell
python .\voxcpm_gui.py --smoke-test
```

无界面自检：

```powershell
python .\voxcpm_gui.py --self-test-tts
python .\voxcpm_gui.py --self-test-stt
```

## 8. 打包 exe

当前环境里已经有 `PyInstaller`，可以直接运行：

```powershell
.\build_voxcpm_gui.ps1
```

这个打包脚本会自动：

- 打包 `dist\VoxCPM2Studio`
- 同步根目录运行版 `VoxCPM2Studio.exe` 和 `_internal`
- 校验 `..\model` 下的 `VoxCPM2`、`Whisper-large-v3-turbo`、`faster-whisper-small`
- 运行 `--smoke-test`、`--self-test-tts`、`--self-test-stt`

同时它已经额外保留了 `voxcpm` 关键模块源码，避免打包版在 TorchScript 初始化时出现 `Can't get source for <function snake ...>` 这类报错。

如果想让 `dist` 目录直接可离线使用，把本地模型一起复制到打包结果旁边：

```powershell
.\build_voxcpm_gui.ps1 -CopyModel
```

如果你已经把源码和 exe 整理进 `语音生成` 文件夹，推荐把模型单独放在同级的 `model\VoxCPM2` 目录下，程序会自动识别。

先看打包命令但不真正执行：

```powershell
.\build_voxcpm_gui.ps1 -DryRun
```

## 9. 当前机器上的说明

这台机器默认 Python 是 3.13。`voxcpm` 的完整依赖里有一个 `funasr -> editdistance` 链路，在 Windows + Python 3.13 下会触发本地编译失败。配音推理默认不加载 denoiser，所以这里采用更稳的安装方式：先安装核心推理依赖，再用 `--no-deps` 安装 `voxcpm` 本体。
