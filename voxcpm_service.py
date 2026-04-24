from __future__ import annotations

import importlib.util
import tempfile
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app_shared import CancelToken, TaskCancelledError, TaskProgress

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

DEFAULT_VOICE = "年轻女性，温柔自然，普通话标准，语速适中，适合视频讲解"
DEFAULT_CFG = 2.0
DEFAULT_STEPS = 10
DEFAULT_CHUNK_MAX_CHARS = 120
DEFAULT_SILENCE_MS = 250
REQUIRED_PACKAGES = ("numpy", "soundfile", "torch", "voxcpm")

ProgressCallback = Callable[[TaskProgress], None]

_MODEL_CACHE: dict[tuple[str, bool, bool, str | None, bool], Any] = {}
_MODEL_LOCK = threading.Lock()


@dataclass(slots=True)
class ModelLoadOptions:
    load_denoiser: bool = False
    optimize: bool = False
    cache_dir: str | None = None
    local_files_only: bool = False


@dataclass(slots=True)
class GenerateRequest:
    text: str
    voice: str = DEFAULT_VOICE
    reference_wav: str | None = None
    prompt_wav: str | None = None
    prompt_text: str | None = None
    cfg: float = DEFAULT_CFG
    steps: int = DEFAULT_STEPS
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    silence_ms: int = DEFAULT_SILENCE_MS
    reuse_first_chunk_as_reference: bool = False
    load_denoiser: bool = False
    optimize: bool = False
    model_path: str = field(default_factory=lambda: str(get_default_model_dir()))
    cache_dir: str | None = None
    local_files_only: bool = False
    output_path: str | None = None


@dataclass(slots=True)
class GenerateResult:
    output_path: str
    sample_rate: int
    chunks_count: int
    duration_seconds: float


@dataclass(slots=True)
class EnvironmentStatus:
    ready: bool
    python_version: str
    missing_packages: list[str]
    cuda_available: bool
    gpu_name: str | None
    total_vram_gb: float | None
    default_model_path: str
    model_exists: bool


def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_default_model_dir() -> Path:
    app_base_dir = get_app_base_dir().resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        app_base_dir / "VoxCPM2",
        app_base_dir.parent / "model" / "VoxCPM2",
        app_base_dir / "model" / "VoxCPM2",
        cwd / "VoxCPM2",
        cwd / "model" / "VoxCPM2",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    if app_base_dir.name == "语音生成":
        return (app_base_dir.parent / "model" / "VoxCPM2").resolve()
    return (app_base_dir / "VoxCPM2").resolve()


def get_default_output_dir() -> Path:
    return (get_app_base_dir() / "outputs").resolve()


def get_user_state_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata).resolve() / "VoxCPM2Studio"
    return (get_app_base_dir() / ".app_state").resolve()


def get_settings_path() -> Path:
    return get_user_state_dir() / "settings.json"


def build_timestamped_output_path(output_dir: str | Path | None = None) -> Path:
    resolved_dir = Path(output_dir) if output_dir else get_default_output_dir()
    resolved_dir = resolved_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolved_dir / f"{stamp}.wav"


def normalize_voice_prompt(voice: str) -> str:
    voice = voice.strip()
    if not voice:
        return ""
    if voice.startswith(("(", "（")):
        return voice
    return f"({voice})"


def split_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []

    chunks: list[str] = []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    for paragraph in paragraphs:
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？!?；;])", paragraph)
            if item.strip()
        ]
        if not sentences:
            sentences = [paragraph]

        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(split_long_sentence(sentence, max_chars))
                continue

            candidate = f"{current}{sentence}" if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

    return chunks


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces = [
        item.strip()
        for item in re.split(r"(?<=[，,、：:])", sentence)
        if item.strip()
    ]
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(piece) > max_chars:
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        current = piece

    if current:
        chunks.append(current)
    return chunks


def check_environment(model_path: str | None = None) -> EnvironmentStatus:
    missing_packages = [
        package for package in REQUIRED_PACKAGES if importlib.util.find_spec(package) is None
    ]
    cuda_available = False
    gpu_name = None
    total_vram_gb = None

    if "torch" not in missing_packages:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                total_vram_gb = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
        except Exception:
            pass

    default_model_path = str(Path(model_path).expanduser().resolve()) if model_path else str(get_default_model_dir())
    model_exists = Path(default_model_path).exists()
    ready = not missing_packages

    return EnvironmentStatus(
        ready=ready,
        python_version=sys.version.split()[0],
        missing_packages=missing_packages,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        default_model_path=default_model_path,
        model_exists=model_exists,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    current: int = 0,
    total: int = 0,
    percent: int | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        TaskProgress(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )
    )


def load_model_once(
    model_path: str,
    options: ModelLoadOptions,
    progress_callback: ProgressCallback | None = None,
):
    try:
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise RuntimeError(
            "缺少 voxcpm 依赖，请先运行：\n"
            "python -m pip install -r requirements-voxcpm2.txt\n"
            "python -m pip install voxcpm --no-deps"
        ) from exc

    model_id_or_path = str(model_path).strip()
    if not model_id_or_path:
        raise ValueError("模型路径不能为空。")

    path_obj = Path(model_id_or_path)
    cache_key_path = str(path_obj.expanduser().resolve()) if path_obj.exists() else model_id_or_path
    cache_key = (
        cache_key_path,
        options.load_denoiser,
        options.optimize,
        options.cache_dir,
        options.local_files_only,
    )

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            _emit_progress(progress_callback, "loading_model", "复用已加载模型。", percent=15)
            return _MODEL_CACHE[cache_key]

        _emit_progress(
            progress_callback,
            "loading_model",
            "首次加载模型到内存，这一步会比较慢，请稍候。",
            percent=10,
        )
        model = VoxCPM.from_pretrained(
            model_id_or_path,
            load_denoiser=options.load_denoiser,
            cache_dir=options.cache_dir,
            local_files_only=options.local_files_only,
            optimize=options.optimize,
        )
        _MODEL_CACHE[cache_key] = model
        return model


def _validate_request(request: GenerateRequest) -> None:
    if not isinstance(request.text, str) or not request.text.strip():
        raise ValueError("请输入要配音的文案。")
    if request.chunk_max_chars <= 0:
        raise ValueError("分段长度必须大于 0。")
    if request.steps <= 0:
        raise ValueError("推理步数必须大于 0。")
    if request.silence_ms < 0:
        raise ValueError("段间静音不能为负数。")

    if bool(request.prompt_wav) != bool(request.prompt_text and request.prompt_text.strip()):
        raise ValueError("Prompt 音频和 Prompt 文本需要同时填写。")

    if request.reference_wav and not Path(request.reference_wav).expanduser().exists():
        raise FileNotFoundError(f"参考音频不存在：{request.reference_wav}")
    if request.prompt_wav and not Path(request.prompt_wav).expanduser().exists():
        raise FileNotFoundError(f"Prompt 音频不存在：{request.prompt_wav}")


def generate_tts(
    request: GenerateRequest,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
) -> GenerateResult:
    global np, sf
    if np is None or sf is None:
        try:
            import numpy as numpy_module
            import soundfile as soundfile_module
        except ImportError as exc:
            raise RuntimeError(
                "缺少音频依赖，请先运行：\n"
                "python -m pip install -r requirements-voxcpm2.txt"
            ) from exc
        np = numpy_module
        sf = soundfile_module

    if np is None or sf is None:
        raise RuntimeError(
            "缺少音频依赖，请先运行：\n"
            "python -m pip install -r requirements-voxcpm2.txt"
        )

    _emit_progress(progress_callback, "checking_environment", "正在检查运行环境。", percent=3)
    status = check_environment(request.model_path)
    if status.missing_packages:
        raise RuntimeError(
            "缺少依赖："
            + ", ".join(status.missing_packages)
            + "\n请先运行：\npython -m pip install -r requirements-voxcpm2.txt\npython -m pip install voxcpm --no-deps"
        )

    _validate_request(request)
    _emit_progress(progress_callback, "segmenting", "正在拆分文案。", percent=8)
    chunks = split_text(request.text, request.chunk_max_chars)
    if not chunks:
        raise ValueError("输入文本为空。")

    output_path = (
        Path(request.output_path).expanduser()
        if request.output_path
        else build_timestamped_output_path(get_default_output_dir())
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model_once(
        request.model_path,
        ModelLoadOptions(
            load_denoiser=request.load_denoiser,
            optimize=request.optimize,
            cache_dir=request.cache_dir,
            local_files_only=request.local_files_only,
        ),
        progress_callback=progress_callback,
    )

    sample_rate = model.tts_model.sample_rate
    silence = np.zeros(int(sample_rate * request.silence_ms / 1000), dtype=np.float32)
    voice_prompt = normalize_voice_prompt(request.voice)
    wav_parts: list[np.ndarray] = []
    temp_reference_path: Path | None = None

    generate_kwargs = {
        "cfg_value": request.cfg,
        "inference_timesteps": request.steps,
    }
    if request.reference_wav:
        generate_kwargs["reference_wav_path"] = str(Path(request.reference_wav).expanduser())
    if request.prompt_wav:
        generate_kwargs["prompt_wav_path"] = str(Path(request.prompt_wav).expanduser())
    if request.prompt_text:
        generate_kwargs["prompt_text"] = request.prompt_text

    total_chunks = len(chunks)
    try:
        for index, chunk in enumerate(chunks, start=1):
            if cancel_token and cancel_token.is_cancelled():
                raise TaskCancelledError()

            current_kwargs = dict(generate_kwargs)
            if request.reuse_first_chunk_as_reference and temp_reference_path and temp_reference_path.exists():
                current_kwargs["reference_wav_path"] = str(temp_reference_path)

            preview = chunk[:45] + ("..." if len(chunk) > 45 else "")
            percent = 20 + int(70 * index / total_chunks)
            progress_message = f"正在生成第 {index}/{total_chunks} 段：{preview}"
            if request.reuse_first_chunk_as_reference and index >= 2 and temp_reference_path:
                progress_message += "（已复用首段音色）"
            _emit_progress(
                progress_callback,
                "generating",
                progress_message,
                current=index,
                total=total_chunks,
                percent=percent,
            )
            prompt_text = f"{voice_prompt}{chunk}" if voice_prompt else chunk
            wav = np.asarray(model.generate(text=prompt_text, **current_kwargs), dtype=np.float32)
            wav_parts.append(wav)

            if request.reuse_first_chunk_as_reference and index == 1 and total_chunks > 1:
                first_chunk_output = Path(tempfile.NamedTemporaryFile(prefix="voxcpm_first_chunk_", suffix=".wav", delete=False).name)
                sf.write(first_chunk_output, wav, sample_rate, subtype="PCM_16")
                temp_reference_path = first_chunk_output

            if index != total_chunks and request.silence_ms > 0:
                wav_parts.append(silence)

        _emit_progress(progress_callback, "writing", "正在写入音频文件。", percent=95)
        final_wav = np.concatenate(wav_parts) if wav_parts else np.zeros(0, dtype=np.float32)
        sf.write(output_path, final_wav, sample_rate, subtype="PCM_16")
    finally:
        if temp_reference_path and temp_reference_path.exists():
            try:
                temp_reference_path.unlink()
            except OSError:
                pass

    duration_seconds = float(len(final_wav) / sample_rate) if sample_rate else 0.0
    result = GenerateResult(
        output_path=str(output_path),
        sample_rate=sample_rate,
        chunks_count=total_chunks,
        duration_seconds=duration_seconds,
    )
    _emit_progress(
        progress_callback,
        "completed",
        f"生成完成：{output_path}",
        current=total_chunks,
        total=total_chunks,
        percent=100,
    )
    return result
