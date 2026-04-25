from __future__ import annotations

import importlib.util
import random
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app_shared import CancelToken, TaskCancelledError, TaskProgress
from voxcpm_service import get_app_base_dir

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

DEFAULT_AUDIOFX_DURATION_SECONDS = 10.0
DEFAULT_AUDIOFX_STEPS = 50
DEFAULT_AUDIOFX_GUIDANCE_SCALE = 3.5
DEFAULT_AUDIOFX_VARIANTS = 1
REQUIRED_AUDIOFX_PACKAGES = (
    "numpy",
    "soundfile",
    "torch",
    "transformers",
    "diffusers",
    "accelerate",
    "sentencepiece",
)

AudioFxProgressCallback = Callable[[TaskProgress], None]

_PIPELINE_CACHE: dict[tuple[str, str, bool], Any] = {}
_PIPELINE_LOCK = threading.Lock()


@dataclass(slots=True)
class AudioFxRequest:
    prompt: str
    model_path: str = field(default_factory=lambda: str(get_default_audiofx_model_dir()))
    output_path: str | None = None
    duration_seconds: float = DEFAULT_AUDIOFX_DURATION_SECONDS
    steps: int = DEFAULT_AUDIOFX_STEPS
    guidance_scale: float = DEFAULT_AUDIOFX_GUIDANCE_SCALE
    seed: int | None = None
    use_gpu: bool = True
    cpu_offload: bool = False
    sequence_index: int = 1


@dataclass(slots=True)
class AudioFxResult:
    output_path: str
    prompt: str
    sample_rate: int
    duration_seconds: float
    seed: int


@dataclass(slots=True)
class AudioFxEnvironmentStatus:
    ready: bool
    python_version: str
    missing_packages: list[str]
    cuda_available: bool
    gpu_name: str | None
    total_vram_gb: float | None
    default_model_path: str
    model_exists: bool
    model_index_exists: bool


def get_default_audiofx_model_dir() -> Path:
    app_base_dir = get_app_base_dir().resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        app_base_dir.parent / "model" / "AudioLDM2",
        app_base_dir / "model" / "AudioLDM2",
        cwd / "model" / "AudioLDM2",
        cwd / "AudioLDM2",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def get_default_audiofx_output_dir() -> Path:
    return (get_app_base_dir() / "outputs" / "sfx").resolve()


def summarize_prompt_for_filename(prompt: str, max_chars: int = 28) -> str:
    normalized = re.sub(r"\s+", "_", prompt.strip())
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        return "sound_effect"
    return normalized[:max_chars].strip("_") or "sound_effect"


def build_audiofx_output_path(
    output_dir: str | Path | None,
    prompt: str,
    sequence_index: int = 1,
) -> Path:
    resolved_dir = Path(output_dir).expanduser().resolve() if output_dir else get_default_audiofx_output_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = summarize_prompt_for_filename(prompt)
    candidate = resolved_dir / f"{stamp}_{sequence_index:03d}_{safe_prompt}.wav"
    return ensure_unique_path(candidate)


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"无法生成不冲突的输出文件名：{path}")


def check_audiofx_environment(model_path: str | None = None) -> AudioFxEnvironmentStatus:
    missing_packages = [
        package for package in REQUIRED_AUDIOFX_PACKAGES if importlib.util.find_spec(package) is None
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
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    default_model_path = (
        str(Path(model_path).expanduser().resolve())
        if model_path
        else str(get_default_audiofx_model_dir())
    )
    model_dir = Path(default_model_path)
    model_exists = model_dir.exists()
    model_index_exists = (model_dir / "model_index.json").exists()
    ready = not missing_packages and model_exists and model_index_exists

    return AudioFxEnvironmentStatus(
        ready=ready,
        python_version=sys.version.split()[0],
        missing_packages=missing_packages,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        default_model_path=default_model_path,
        model_exists=model_exists,
        model_index_exists=model_index_exists,
    )


def _emit_progress(
    progress_callback: AudioFxProgressCallback | None,
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


def _validate_request(request: AudioFxRequest) -> None:
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("请输入音效提示词。")
    if request.duration_seconds <= 0:
        raise ValueError("音频时长必须大于 0。")
    if request.steps <= 0:
        raise ValueError("推理步数必须大于 0。")
    if request.guidance_scale <= 0:
        raise ValueError("Guidance Scale 必须大于 0。")

    model_dir = Path(request.model_path).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"AudioLDM2 模型目录不存在：{model_dir}")
    if not (model_dir / "model_index.json").exists():
        raise FileNotFoundError(f"AudioLDM2 模型目录缺少 model_index.json：{model_dir}")


def _get_sample_rate(pipeline: Any) -> int:
    vocoder = getattr(pipeline, "vocoder", None)
    config = getattr(vocoder, "config", None)
    sample_rate = getattr(config, "sampling_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return 16_000


def _normalize_audio(audio: Any) -> Any:
    if np is None:
        return audio
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim > 1 and audio_array.shape[0] == 1:
        audio_array = audio_array[0]
    return np.clip(audio_array, -1.0, 1.0)


def _pipeline_cache_key(model_path: str, device_mode: str, cpu_offload: bool) -> tuple[str, str, bool]:
    path_obj = Path(model_path).expanduser()
    cache_key_path = str(path_obj.resolve()) if path_obj.exists() else str(model_path)
    return cache_key_path, device_mode, cpu_offload


def _disable_optional_onnx_runtime_for_diffusers() -> None:
    try:
        import diffusers.utils.import_utils as diffusers_import_utils

        # AudioLDM2 uses PyTorch weights here. A broken optional onnxruntime install
        # can otherwise fail during diffusers' generic component type checks.
        diffusers_import_utils._onnx_available = False
    except Exception:
        pass


def _patch_legacy_language_model(pipeline: Any, model_path: str, torch_dtype: Any) -> None:
    language_model = getattr(pipeline, "language_model", None)
    if language_model is None or hasattr(language_model, "_update_model_kwargs_for_generation"):
        return

    try:
        from transformers import GPT2LMHeadModel

        patched_language_model = GPT2LMHeadModel.from_pretrained(
            Path(model_path).expanduser() / "language_model",
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if hasattr(pipeline, "register_modules"):
            pipeline.register_modules(language_model=patched_language_model)
        else:
            pipeline.language_model = patched_language_model
    except Exception as exc:
        raise RuntimeError(
            "当前本地 AudioLDM2 模型使用旧版 GPT2Model 结构，和新版 diffusers 不兼容；"
            f"尝试自动兼容修复失败：{exc}"
        ) from exc


def load_audiofx_pipeline_once(
    model_path: str,
    use_gpu: bool = True,
    cpu_offload: bool = False,
    progress_callback: AudioFxProgressCallback | None = None,
):
    try:
        import torch

        _disable_optional_onnx_runtime_for_diffusers()
        from diffusers import AudioLDM2Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "缺少 AudioLDM2 依赖，请先运行：\n"
            "python -m pip install diffusers accelerate sentencepiece"
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    if use_gpu and not cuda_available:
        raise RuntimeError("已勾选使用 GPU，但当前 CUDA 不可用。请取消 GPU 选项，或检查显卡驱动和 PyTorch CUDA。")

    device_mode = "cuda" if use_gpu and cuda_available else "cpu"
    cache_key = _pipeline_cache_key(model_path, device_mode, cpu_offload)

    with _PIPELINE_LOCK:
        if cache_key in _PIPELINE_CACHE:
            _emit_progress(progress_callback, "loading_model", "复用已加载的 AudioLDM2 模型。", percent=20)
            return _PIPELINE_CACHE[cache_key]

        torch_dtype = torch.float16 if device_mode == "cuda" else torch.float32
        _emit_progress(
            progress_callback,
            "loading_model",
            "首次加载 AudioLDM2 到内存，这一步会比较慢，请稍候。",
            percent=12,
        )
        try:
            pipeline = AudioLDM2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            _patch_legacy_language_model(pipeline, model_path, torch_dtype)
            if device_mode == "cuda":
                if cpu_offload:
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline = pipeline.to("cuda")
            else:
                pipeline = pipeline.to("cpu")
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
        except RuntimeError as exc:
            _raise_friendly_runtime_error(exc)
        except Exception as exc:
            raise RuntimeError(f"加载 AudioLDM2 失败：{exc}") from exc

        _PIPELINE_CACHE[cache_key] = pipeline
        return pipeline


def _raise_friendly_runtime_error(exc: RuntimeError) -> None:
    message = str(exc)
    lower_message = message.lower()
    if "out of memory" in lower_message or "cuda error" in lower_message and "memory" in lower_message:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        raise RuntimeError(
            "显存不足：AudioLDM2 加载或生成失败。建议启用 CPU offload，或降低音频时长/推理步数后重试。"
        ) from exc
    raise RuntimeError(f"AudioLDM2 运行失败：{message}") from exc


def generate_audiofx(
    request: AudioFxRequest,
    progress_callback: AudioFxProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
) -> AudioFxResult:
    global np, sf
    if np is None or sf is None:
        try:
            import numpy as numpy_module
            import soundfile as soundfile_module
        except ImportError as exc:
            raise RuntimeError("缺少音频依赖，请先运行：python -m pip install numpy soundfile") from exc
        np = numpy_module
        sf = soundfile_module

    _emit_progress(progress_callback, "checking_environment", "正在检查 AudioLDM2 运行环境。", percent=3)
    status = check_audiofx_environment(request.model_path)
    if status.missing_packages:
        raise RuntimeError(
            "缺少 AudioLDM2 依赖："
            + ", ".join(status.missing_packages)
            + "\n请先运行：\npython -m pip install diffusers accelerate sentencepiece"
        )

    _validate_request(request)
    if cancel_token and cancel_token.is_cancelled():
        raise TaskCancelledError()

    output_path = (
        Path(request.output_path).expanduser()
        if request.output_path
        else build_audiofx_output_path(get_default_audiofx_output_dir(), request.prompt, request.sequence_index)
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = ensure_unique_path(output_path)

    seed = request.seed
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)

    pipeline = load_audiofx_pipeline_once(
        request.model_path,
        use_gpu=request.use_gpu,
        cpu_offload=request.cpu_offload,
        progress_callback=progress_callback,
    )
    sample_rate = _get_sample_rate(pipeline)

    if cancel_token and cancel_token.is_cancelled():
        raise TaskCancelledError()

    _emit_progress(
        progress_callback,
        "generating",
        "正在生成音效；单次模型调用无法立即中断，如果取消会在本轮结束后停止。",
        percent=35,
    )
    try:
        import torch

        generator_device = "cuda" if request.use_gpu and not request.cpu_offload and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        output = pipeline(
            prompt=request.prompt.strip(),
            audio_length_in_s=float(request.duration_seconds),
            num_inference_steps=int(request.steps),
            guidance_scale=float(request.guidance_scale),
            generator=generator,
        )
    except RuntimeError as exc:
        _raise_friendly_runtime_error(exc)
    except Exception as exc:
        raise RuntimeError(f"AudioLDM2 生成失败：{exc}") from exc

    if cancel_token and cancel_token.is_cancelled():
        raise TaskCancelledError("已在当前生成结束后取消，未写入本次音频。")

    _emit_progress(progress_callback, "writing", "正在写入 WAV 文件。", percent=92)
    audio = _normalize_audio(output.audios[0])
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    duration_seconds = float(len(audio) / sample_rate) if sample_rate else 0.0

    result = AudioFxResult(
        output_path=str(output_path),
        prompt=request.prompt,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        seed=seed,
    )
    _emit_progress(
        progress_callback,
        "completed",
        f"音效生成完成：{output_path}",
        current=1,
        total=1,
        percent=100,
    )
    return result
