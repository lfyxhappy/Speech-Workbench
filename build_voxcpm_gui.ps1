param(
    [string]$Python = "python",
    [switch]$CopyModel,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$GuiEntry = Join-Path $Root "voxcpm_gui.py"
$DistDir = Join-Path $Root "dist\VoxCPM2Studio"
$HookDir = Join-Path $Root "pyinstaller_hooks"
$RuntimeExe = Join-Path $Root "VoxCPM2Studio.exe"
$RuntimeInternal = Join-Path $Root "_internal"
$ModelRootCandidate = Join-Path $Root "..\model"
$ModelRoot = $null
if (Test-Path -LiteralPath $ModelRootCandidate) {
    $ModelRoot = (Resolve-Path $ModelRootCandidate).Path
}
$ModelCandidates = @(
    (Join-Path $Root "..\model\VoxCPM2"),
    (Join-Path $Root "model\VoxCPM2"),
    (Join-Path $Root "VoxCPM2")
)
$ModelDir = $null
foreach ($Candidate in $ModelCandidates) {
    if (Test-Path $Candidate) {
        $ModelDir = (Resolve-Path $Candidate).Path
        break
    }
}

$Arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--name", "VoxCPM2Studio",
    "--paths", $Root,
    "--additional-hooks-dir", $HookDir,
    "--hidden-import", "PyQt6.QtMultimedia",
    "--hidden-import", "whisper",
    "--hidden-import", "faster_whisper",
    "--hidden-import", "ctranslate2",
    "--hidden-import", "tiktoken",
    "--hidden-import", "diffusers",
    "--hidden-import", "accelerate",
    "--hidden-import", "sentencepiece",
    "--hidden-import", "audiofx_service",
    "--hidden-import", "diffusers.pipelines.audioldm2",
    "--hidden-import", "diffusers.pipelines.audioldm2.pipeline_audioldm2",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "onnxruntime-gpu",
    "--exclude-module", "onnxruntime.training",
    "--copy-metadata", "accelerate",
    "--copy-metadata", "av",
    "--copy-metadata", "certifi",
    "--copy-metadata", "charset-normalizer",
    "--copy-metadata", "ctranslate2",
    "--copy-metadata", "diffusers",
    "--copy-metadata", "faster-whisper",
    "--copy-metadata", "filelock",
    "--copy-metadata", "huggingface-hub",
    "--copy-metadata", "httpcore",
    "--copy-metadata", "httpx",
    "--copy-metadata", "idna",
    "--copy-metadata", "importlib-metadata",
    "--copy-metadata", "librosa",
    "--copy-metadata", "numpy",
    "--copy-metadata", "openai-whisper",
    "--copy-metadata", "packaging",
    "--copy-metadata", "Pillow",
    "--copy-metadata", "protobuf",
    "--copy-metadata", "PyYAML",
    "--copy-metadata", "regex",
    "--copy-metadata", "requests",
    "--copy-metadata", "safetensors",
    "--copy-metadata", "scipy",
    "--copy-metadata", "sentencepiece",
    "--copy-metadata", "soundfile",
    "--copy-metadata", "tokenizers",
    "--copy-metadata", "torch",
    "--copy-metadata", "torchcodec",
    "--copy-metadata", "tqdm",
    "--copy-metadata", "transformers",
    "--copy-metadata", "urllib3",
    "--copy-metadata", "zipp",
    "--collect-data", "PyQt6.QtMultimedia",
    "--collect-binaries", "PyQt6.QtMultimedia",
    $GuiEntry
)

function Assert-RequiredPath {
    param(
        [string]$PathValue,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label 不存在：$PathValue"
    }
}

function Assert-AppNotRunning {
    $Running = Get-Process -Name "VoxCPM2Studio" -ErrorAction SilentlyContinue
    if ($Running) {
        throw "检测到 VoxCPM2Studio 正在运行，请先关闭后再发布。"
    }
}

function Invoke-AndAssertSuccess {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$ErrorMessage
    )

    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -PassThru -Wait -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        throw $ErrorMessage
    }
}

if ($DryRun) {
    Write-Host "Dry run:"
    Write-Host $Python ($Arguments -join " ")
    if ($CopyModel -and $ModelDir) {
        Write-Host "Will copy model directory to: $DistDir\VoxCPM2"
    } elseif ($CopyModel) {
        Write-Host "Model directory not found. Expected one of:"
        $ModelCandidates | ForEach-Object { Write-Host "  $_" }
    }
    Write-Host "Will validate models under: $ModelRootCandidate"
    Write-Host "Will sync runtime exe to: $RuntimeExe"
    Write-Host "Will sync runtime internal dir to: $RuntimeInternal"
    Write-Host "Will run: --smoke-test, --self-test-tts, --self-test-stt, --self-test-audiofx"
    exit 0
}

if (-not $ModelRoot) {
    throw "模型根目录不存在：$ModelRootCandidate"
}
Assert-RequiredPath -PathValue (Join-Path $ModelRoot "VoxCPM2") -Label "VoxCPM2 模型目录"
Assert-RequiredPath -PathValue (Join-Path $ModelRoot "Whisper-large-v3-turbo") -Label "Whisper-large-v3-turbo 模型目录"
Assert-RequiredPath -PathValue (Join-Path $ModelRoot "faster-whisper-small") -Label "faster-whisper-small 模型目录"
Assert-RequiredPath -PathValue (Join-Path $ModelRoot "AudioLDM2") -Label "AudioLDM2 模型目录"
Assert-AppNotRunning

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller 打包失败。"
}

if ($CopyModel -and $ModelDir) {
    $ResolvedModel = $ModelDir
    $TargetModelDir = Join-Path $DistDir "VoxCPM2"
    if (Test-Path $TargetModelDir) {
        Remove-Item -LiteralPath $TargetModelDir -Recurse -Force
    }
    Copy-Item -LiteralPath $ResolvedModel -Destination $TargetModelDir -Recurse
} elseif ($CopyModel) {
    throw "没有找到 VoxCPM2 模型目录，无法复制。"
}

Assert-RequiredPath -PathValue (Join-Path $DistDir "VoxCPM2Studio.exe") -Label "打包输出 exe"
Assert-RequiredPath -PathValue (Join-Path $DistDir "_internal") -Label "打包输出 _internal"

if (Test-Path -LiteralPath $RuntimeInternal) {
    Remove-Item -LiteralPath $RuntimeInternal -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $DistDir "_internal") -Destination $RuntimeInternal -Recurse
Copy-Item -LiteralPath (Join-Path $DistDir "VoxCPM2Studio.exe") -Destination $RuntimeExe -Force

Invoke-AndAssertSuccess -Executable $RuntimeExe -Arguments @("--smoke-test") -ErrorMessage "打包版 smoke test 失败。"
Invoke-AndAssertSuccess -Executable $RuntimeExe -Arguments @("--self-test-tts") -ErrorMessage "打包版 TTS 自检失败。"
Invoke-AndAssertSuccess -Executable $RuntimeExe -Arguments @("--self-test-stt") -ErrorMessage "打包版 STT 自检失败。"
Invoke-AndAssertSuccess -Executable $RuntimeExe -Arguments @("--self-test-audiofx") -ErrorMessage "打包版 AudioFX 自检失败。"

Write-Host "打包完成：$DistDir"
if (-not $CopyModel) {
    Write-Host "提示：如果你希望 dist 目录可直接离线生成，请把 VoxCPM2 文件夹放到 exe 同级，或使用 -CopyModel。"
}
