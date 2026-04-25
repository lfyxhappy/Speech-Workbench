# Speech Workbench

一个面向本地部署场景的中文语音工作台，提供三类核心能力：

- `语音生成（TTS）`：基于本地 `VoxCPM2` 模型生成中文配音
- `语音转文本（STT/ASR）`：基于本地 `Whisper-large-v3-turbo` 或 `faster-whisper-small` 模型转写音频
- `音效素材库（AudioFX）`：基于本地 `AudioLDM2` 模型生成雨声、铃声、脚步声、环境音等非语音音效

项目以 **Windows 桌面使用体验** 为优先目标，提供：

- 可直接运行的 `PyQt6` 图形界面
- 保留源码方式运行
- 可打包为离线使用的 `exe`
- 本地模型目录识别
- TTS / STT / AudioFX 三页面工作流
- 各自独立的任务队列与后台线程

这个仓库适合：

- 做中文视频讲解配音
- 本地批量测试音色和参数
- 把微信语音、录音文件、音频素材转成文字
- 批量生成游戏、视频或桌面应用可用的短音效素材
- 做离线工具，不依赖网页服务

## 功能概览

### 1. 语音生成

- 文案输入
- 音色描述与预设音色选择
- 高级参数调整：`CFG`、`推理步数`、`分段长度`、`段间静音`
- 可选参考音频 / Prompt 音频 / Prompt 文本
- 支持 Prompt 音频一键自动转写
- 支持“首段锁定后续音色”，增强长文前后音色一致性
- 生成完成后内置试听
- 任务队列串行执行

### 2. 语音转文本

- 支持 `Whisper-large-v3-turbo`
- 支持 `faster-whisper-small`
- 支持单文件或多文件加入队列
- 支持多种输出格式：
  - `纯文本`
  - `智能分段`
  - `时间戳文本`
  - `SRT 字幕`
- 可调“智能分段停顿阈值”
- 页面内同时提供：
  - 阅读预览
  - 时间轴预览

### 3. 音效素材库

- 支持 `AudioLDM2`
- 一行一个提示词批量生成 `.wav`
- 支持每条提示词生成多个版本
- 可调音频时长、推理步数、guidance scale、随机种子
- 支持 CUDA 与 CPU offload 选项
- 生成结果列表内可直接试听

### 4. 打包与发布

- 提供 `PyInstaller` one-folder 打包方案
- 自动同步根目录运行版
- 自动执行打包后自检
- 保持模型目录在 `exe` 外部，不把大模型权重塞进发布包

## 目录结构

```text
语音生成/
├─ voxcpm_gui.py                    # 图形界面入口
├─ tts_voxcpm2.py                   # 命令行 TTS 入口
├─ voxcpm_service.py                # TTS 服务层
├─ asr_service.py                   # STT/ASR 服务层
├─ audiofx_service.py               # AudioLDM2 音效生成服务层
├─ app_shared.py                    # 公共数据结构与工具
├─ build_voxcpm_gui.ps1             # 打包脚本
├─ requirements-voxcpm2.txt         # TTS 最小依赖
├─ requirements-voxcpm2-gui.txt     # GUI + STT + AudioFX 依赖
├─ sample_text.txt                  # TTS 示例文案
├─ pyinstaller_hooks/               # PyInstaller hooks
├─ tests/                           # 单元测试
├─ tools/aria2/                     # 附带 aria2 工具文件
└─ README.md
```

推荐与模型目录搭配的布局：

```text
各种各样的模型/
├─ model/
│  ├─ VoxCPM2/
│  ├─ Whisper-large-v3-turbo/
│  ├─ faster-whisper-small/
│  └─ AudioLDM2/
└─ 语音生成/
   └─ ...
```

程序会优先识别同级 `../model` 下的模型目录。

GitHub Release 发布包只包含应用和运行依赖，不包含上述模型权重。请自行下载模型并按这个目录结构放到本地。

## 依赖环境

建议环境：

- Windows 10 / 11
- Python 3.13
- NVIDIA 显卡，12GB 显存或更高更适合长文配音

安装基础依赖：

```powershell
python -m pip install -r requirements-voxcpm2.txt
python -m pip install voxcpm --no-deps
python -m pip install -r requirements-voxcpm2-gui.txt
```

`requirements-voxcpm2-gui.txt` 当前内容包含：

- `PyQt6`
- `faster-whisper`
- `openai-whisper`（通过 Git 安装）
- `tiktoken`
- `diffusers`
- `accelerate`
- `sentencepiece`

如果 Hugging Face 下载慢，可以先设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 模型准备

本项目默认使用以下四套本地模型，模型权重不会上传到 GitHub Release：

### TTS

- `model/VoxCPM2`

### STT

- `model/Whisper-large-v3-turbo`
- `model/faster-whisper-small`

### AudioFX

- `model/AudioLDM2`

其中 `Whisper-large-v3-turbo` 当前按本地权重文件方式加载：

- `large-v3-turbo.pt`

不依赖 `modelscope` 作为主运行链路。

`AudioLDM2` 使用 `diffusers.AudioLDM2Pipeline` 通过本地路径加载，并设置 `local_files_only=True`，运行时不会主动联网下载模型。

## 快速开始

### 1. 启动图形界面

```powershell
python .\voxcpm_gui.py
```

启动后会看到三个页面：

- `语音生成`
- `语音转文本`
- `音效素材库`

### 2. 命令行方式生成示例配音

```powershell
python .\tts_voxcpm2.py
```

默认会读取：

- [sample_text.txt](C:/算法/小应用/各种各样的模型/语音生成/sample_text.txt)

默认输出到：

- `outputs/voxcpm2_output.wav`

### 3. 直接生成一句话

```powershell
python .\tts_voxcpm2.py --text "大家好，欢迎来到我的频道。今天我们来讲一个非常实用的模型。" --output outputs/demo.wav
```

### 4. 检查环境

```powershell
python .\tts_voxcpm2.py --check-env
python .\voxcpm_gui.py --smoke-test
python .\voxcpm_gui.py --self-test-tts
python .\voxcpm_gui.py --self-test-stt
python .\voxcpm_gui.py --self-test-audiofx
```

## 语音生成说明

### 典型用法

在 `语音生成` 页面里，你可以：

- 输入待配音文案
- 选择预设音色，再手动补充描述
- 调整 TTS 参数
- 使用参考音频提升音色稳定性
- 使用 Prompt 音频 + Prompt 文本做更高相似度克隆

### 常用参数

- `CFG`
  - 控制模型对音色描述的遵从程度
  - 越高越强调提示词，但过高可能发飘或生硬
- `推理步数`
  - 越高通常越慢，但细节可能更稳
- `分段长度`
  - 长文案会自动拆段生成
  - 值越大，上下文更完整，但更慢、更吃显存
- `段间静音`
  - 控制拼接时的停顿感
- `加载 denoiser`
  - 额外降噪，可能更干净，但也更吃资源
- `开启 torch.compile 优化`
  - 首次更慢，后续连续生成可能更快
- `首段锁定后续音色`
  - 把第 1 段音频作为后续段落参考，增强长文一致性

### 12GB 显存建议

对于 12GB 显存机器，建议从下面这组开始：

- `CFG = 2.0`
- `steps = 10`
- `chunk_max_chars = 120`
- `silence_ms = 250`

如果更关注前后连贯性，可以尝试：

- 开启 `首段锁定后续音色`
- 把 `分段长度` 调到 `140 ~ 180`
- 把 `段间静音` 调到 `120 ~ 180 ms`

## 语音转文本说明

### 支持的模型

- `Whisper-large-v3-turbo`
  - 更偏精度
  - 适合正式转写
- `faster-whisper-small`
  - 更偏速度
  - 适合快速草稿或短音频

### 支持的输出格式

#### 1. 纯文本

只保留识别后的文本内容，适合：

- 复制到文档
- 再交给 AI 做总结
- 手动整理纪要

#### 2. 智能分段

按停顿时长自动分段，适合：

- 微信语音
- 口语聊天
- 长录音整理

这是当前最推荐的阅读格式。

#### 3. 时间戳文本

每段都带开始和结束时间，适合：

- 回听定位
- 校对
- 后期剪辑参考

#### 4. SRT 字幕

适合直接导入：

- 剪映
- Premiere Pro
- CapCut
- 其他字幕工具

### 关于 AAC / 微信音频兼容

项目已经对 `Whisper-large-v3-turbo` 这条路径做了专门兼容：

- 修复了打包版缺少 `whisper/assets` 资源的问题
- 修复了窗口版 `stdout/stderr` 为空导致 `tqdm` 报错的问题
- 改用了更稳的 `PyAV` 解码路径处理 `.aac`

因此像微信导出的 `.aac` 文件现在也能正常转写。

## 打包说明

执行打包：

```powershell
.\build_voxcpm_gui.ps1
```

脚本会自动完成：

- 打包 `dist/VoxCPM2Studio`
- 同步根目录 `VoxCPM2Studio.exe`
- 同步根目录 `_internal`
- 检查四套模型目录是否存在
- 运行以下自检：
  - `--smoke-test`
  - `--self-test-tts`
  - `--self-test-stt`
  - `--self-test-audiofx`

如果只想预览动作而不执行：

```powershell
.\build_voxcpm_gui.ps1 -DryRun
```

如果只在本机自用，也可以把 `VoxCPM2` TTS 模型复制到打包目录：

```powershell
.\build_voxcpm_gui.ps1 -CopyModel
```

正式发布到 GitHub 时不建议使用 `-CopyModel`，本项目的 Release 包默认不包含任何模型权重。

## 测试

运行单元测试：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

当前测试覆盖的重点包括：

- TTS 服务层参数传递
- STT 输出路径与格式渲染
- AAC 解码链路
- 无控制台窗口环境下的 Whisper 转写
- 智能分段与 SRT 输出

## 已知限制

- 当前仅面向 Windows 设计
- TTS 首版以单模型 `VoxCPM2` 为核心，不做多模型管理
- STT 当前只做转写，不包含翻译、VTT、SRT 编辑器等更完整字幕工作流
- TTS / STT 允许并行运行，但会共享 GPU，速度可能下降
- `Whisper-large-v3-turbo` 加载较重，首次启动或首次转写会比小模型慢

## 安全与使用提醒

- 只克隆你自己或已经获得授权的声音
- 不要将本项目用于冒充、欺骗或侵犯他人权益的用途
- 模型与推理结果都在本地运行，但你仍应妥善保管音频素材和输出文件

## 适合继续演进的方向

后续可以继续扩展：

- `VTT` / `JSON` / `Markdown` 导出
- 更自然的 TTS 分段规则
- 响度归一化与淡入淡出拼接
- 批量项目管理
- 更完善的字幕编辑与导出
- 任务历史与结果检索

---

如果你想把它当成一个真正长期维护的桌面工具，这个仓库现在已经具备一个比较完整的起点：本地模型、三页面 UI、可打包、可离线运行、并且围绕中文使用场景做了不少细节优化。
