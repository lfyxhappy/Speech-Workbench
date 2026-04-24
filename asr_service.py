from __future__ import annotations

import contextlib
import importlib.util
import gc
import io
import itertools
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import av
except ImportError:
    av = None

from app_shared import CancelToken, TaskCancelledError, TaskProgress, format_timestamp
from voxcpm_service import get_app_base_dir

ASR_MODEL_OPENAI = "Whisper-large-v3-turbo"
ASR_MODEL_FASTER = "faster-whisper-small"
DEFAULT_ASR_MODEL = ASR_MODEL_OPENAI
SUPPORTED_ASR_MODELS = (ASR_MODEL_OPENAI, ASR_MODEL_FASTER)
TRANSCRIPT_FORMAT_PLAIN = "plain_text"
TRANSCRIPT_FORMAT_SMART = "smart_paragraph"
TRANSCRIPT_FORMAT_TIMESTAMPS = "timestamp_text"
TRANSCRIPT_FORMAT_SRT = "srt"
DEFAULT_TRANSCRIPT_FORMAT = TRANSCRIPT_FORMAT_SMART
LEGACY_TRANSCRIPT_FORMAT = TRANSCRIPT_FORMAT_TIMESTAMPS
SUPPORTED_TRANSCRIPT_FORMATS = (
    TRANSCRIPT_FORMAT_PLAIN,
    TRANSCRIPT_FORMAT_SMART,
    TRANSCRIPT_FORMAT_TIMESTAMPS,
    TRANSCRIPT_FORMAT_SRT,
)
TRANSCRIPT_FORMAT_LABELS = {
    TRANSCRIPT_FORMAT_PLAIN: "纯文本",
    TRANSCRIPT_FORMAT_SMART: "智能分段",
    TRANSCRIPT_FORMAT_TIMESTAMPS: "时间戳文本",
    TRANSCRIPT_FORMAT_SRT: "SRT 字幕",
}
SMART_SEGMENT_DEFAULT_PAUSE_SECONDS = 0.8
SMART_SEGMENT_MAX_CHARS = 140
COMMON_ASR_PACKAGES = ("numpy", "soundfile", "torch")
MODEL_ASR_PACKAGES = {
    ASR_MODEL_OPENAI: ("whisper", "tiktoken", "librosa", "av"),
    ASR_MODEL_FASTER: ("faster_whisper", "ctranslate2"),
}

AsrProgressCallback = Callable[[TaskProgress], None]

_ASR_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_ASR_MODEL_LOCK = threading.Lock()


class _NullWriter:
    def write(self, _value: str) -> int:
        return 0

    def flush(self) -> None:
        return None


@dataclass(slots=True)
class AsrEnvironmentStatus:
    ready: bool
    python_version: str
    missing_packages: list[str]
    cuda_available: bool
    gpu_name: str | None
    total_vram_gb: float | None
    model_paths: dict[str, str]
    model_exists: dict[str, bool]


@dataclass(slots=True)
class AsrModelLoadOptions:
    device: str = "auto"
    compute_type: str | None = None


@dataclass(slots=True)
class TranscribeRequest:
    audio_path: str
    model_kind: str = DEFAULT_ASR_MODEL
    output_path: str | None = None
    language: str | None = None
    with_timestamps: bool = True
    output_format: str | None = None
    smart_segment_pause_seconds: float = SMART_SEGMENT_DEFAULT_PAUSE_SECONDS


@dataclass(slots=True)
class TranscribeSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(slots=True)
class TranscribeResult:
    output_path: str
    text: str
    segments: list[TranscribeSegment]
    duration_seconds: float
    model_kind: str


def get_default_openai_whisper_dir() -> Path:
    app_base_dir = get_app_base_dir().resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        app_base_dir.parent / "model" / ASR_MODEL_OPENAI,
        app_base_dir / "model" / ASR_MODEL_OPENAI,
        cwd / "model" / ASR_MODEL_OPENAI,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def get_default_faster_whisper_dir() -> Path:
    app_base_dir = get_app_base_dir().resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        app_base_dir.parent / "model" / ASR_MODEL_FASTER,
        app_base_dir / "model" / ASR_MODEL_FASTER,
        cwd / "model" / ASR_MODEL_FASTER,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def get_default_stt_output_dir() -> Path:
    return (get_app_base_dir() / "outputs" / "stt").resolve()


def normalize_transcript_format(output_format: str | None, with_timestamps: bool = True) -> str:
    if output_format is None:
        return LEGACY_TRANSCRIPT_FORMAT if with_timestamps else TRANSCRIPT_FORMAT_PLAIN
    if output_format not in SUPPORTED_TRANSCRIPT_FORMATS:
        raise ValueError(f"不支持的转写输出格式：{output_format}")
    return output_format


def get_transcript_format_label(output_format: str) -> str:
    normalized = normalize_transcript_format(output_format)
    return TRANSCRIPT_FORMAT_LABELS[normalized]


def get_transcript_file_suffix(output_format: str | None, with_timestamps: bool = True) -> str:
    normalized = normalize_transcript_format(output_format, with_timestamps=with_timestamps)
    return ".srt" if normalized == TRANSCRIPT_FORMAT_SRT else ".txt"


def build_transcript_output_path(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    output_format: str | None = None,
    with_timestamps: bool = True,
) -> Path:
    source = Path(audio_path).expanduser()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else get_default_stt_output_dir()
    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", source.stem).strip("_") or "audio"
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = get_transcript_file_suffix(output_format, with_timestamps=with_timestamps)
    return target_dir / f"{timestamp}_{safe_name}{suffix}"


def get_asr_model_path(model_kind: str) -> Path:
    if model_kind == ASR_MODEL_OPENAI:
        return get_default_openai_whisper_dir()
    if model_kind == ASR_MODEL_FASTER:
        return get_default_faster_whisper_dir()
    raise ValueError(f"不支持的转写模型：{model_kind}")


def get_required_asr_packages(model_kind: str | None = None) -> tuple[str, ...]:
    packages = list(COMMON_ASR_PACKAGES)
    if model_kind is None:
        for name in SUPPORTED_ASR_MODELS:
            packages.extend(MODEL_ASR_PACKAGES[name])
    else:
        if model_kind not in SUPPORTED_ASR_MODELS:
            raise ValueError(f"不支持的转写模型：{model_kind}")
        packages.extend(MODEL_ASR_PACKAGES[model_kind])

    ordered: list[str] = []
    seen: set[str] = set()
    for package in packages:
        if package in seen:
            continue
        seen.add(package)
        ordered.append(package)
    return tuple(ordered)


def _emit_progress(
    progress_callback: AsrProgressCallback | None,
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


def _check_cancel(cancel_token: CancelToken | None) -> None:
    if cancel_token and cancel_token.is_cancelled():
        raise TaskCancelledError()


def _read_audio_duration_seconds(audio_path: str) -> float:
    global sf
    if sf is None:
        try:
            import soundfile as soundfile_module
        except ImportError:
            return 0.0
        sf = soundfile_module
    try:
        info = sf.info(audio_path)
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames / info.samplerate)


def _resample_audio(samples, source_sr: int, target_sr: int):
    global librosa

    if source_sr == target_sr:
        return samples
    if librosa is None:
        try:
            import librosa as librosa_module
        except ImportError as exc:
            raise RuntimeError(
                "当前音频不是 16k 采样率，且缺少 librosa，无法为 Whisper 自动重采样。"
            ) from exc
        librosa = librosa_module
    return librosa.resample(samples, orig_sr=source_sr, target_sr=target_sr)


def _decode_audio_with_pyav(audio_path: str):
    global np, av

    if np is None:
        try:
            import numpy as numpy_module
        except ImportError as exc:
            raise RuntimeError("缺少 numpy，无法加载音频。") from exc
        np = numpy_module

    if av is None:
        try:
            import av as av_module
        except ImportError as exc:
            raise RuntimeError("缺少 av，无法解码音频。") from exc
        av = av_module

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    raw_buffer = io.BytesIO()
    dtype = None

    def ignore_invalid_frames(frames):
        iterator = iter(frames)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue

    def group_frames(frames, num_samples: int | None = None):
        fifo = av.audio.fifo.AudioFifo()
        for frame in frames:
            frame.pts = None
            fifo.write(frame)
            if num_samples is not None and fifo.samples >= num_samples:
                yield fifo.read()
        if fifo.samples > 0:
            yield fifo.read()

    def resample_frames(frames):
        for frame in itertools.chain(frames, [None]):
            yield from resampler.resample(frame)

    with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = ignore_invalid_frames(frames)
        frames = group_frames(frames, 500000)
        frames = resample_frames(frames)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    gc.collect()

    if dtype is None:
        raise RuntimeError(f"无法从音频中解码出有效数据：{audio_path}")

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return audio.astype(np.float32) / 32768.0


def _load_audio_for_openai_whisper(audio_path: str):
    global np, sf, librosa

    if np is None:
        try:
            import numpy as numpy_module
        except ImportError as exc:
            raise RuntimeError("缺少 numpy，无法加载音频。") from exc
        np = numpy_module

    try:
        return np.asarray(_decode_audio_with_pyav(audio_path), dtype=np.float32)
    except Exception:
        pass

    if sf is not None:
        try:
            samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if getattr(samples, "ndim", 1) > 1:
                samples = np.mean(samples, axis=1)
            if sample_rate != 16000:
                samples = _resample_audio(samples, sample_rate, 16000)
            return np.asarray(samples, dtype=np.float32)
        except Exception:
            pass

    if librosa is None:
        try:
            import librosa as librosa_module
        except ImportError as exc:
            raise RuntimeError(
                "缺少 librosa，无法为 Whisper 读取音频。"
            ) from exc
        librosa = librosa_module

    samples, _ = librosa.load(audio_path, sr=16000, mono=True)
    return np.asarray(samples, dtype=np.float32)


def _resolve_device_and_compute_type(options: AsrModelLoadOptions) -> tuple[str, str]:
    import torch

    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = options.compute_type
    if compute_type:
        return device, compute_type
    if device == "cuda":
        return device, "float16"
    return device, "int8"


def check_asr_environment(model_kind: str | None = None) -> AsrEnvironmentStatus:
    import sys

    missing_packages = [
        package
        for package in get_required_asr_packages(model_kind)
        if importlib.util.find_spec(package) is None
    ]

    cuda_available = False
    gpu_name = None
    total_vram_gb = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    model_paths = {
        ASR_MODEL_OPENAI: str(get_default_openai_whisper_dir()),
        ASR_MODEL_FASTER: str(get_default_faster_whisper_dir()),
    }
    model_exists = {name: Path(path).exists() for name, path in model_paths.items()}
    return AsrEnvironmentStatus(
        ready=not missing_packages,
        python_version=sys.version.split()[0],
        missing_packages=missing_packages,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        model_paths=model_paths,
        model_exists=model_exists,
    )


def load_asr_model_once(
    model_kind: str,
    options: AsrModelLoadOptions,
    progress_callback: AsrProgressCallback | None = None,
):
    model_path = get_asr_model_path(model_kind)
    if not model_path.exists():
        raise FileNotFoundError(f"转写模型不存在：{model_path}")

    device, compute_type = _resolve_device_and_compute_type(options)
    cache_key = (model_kind, str(model_path), f"{device}:{compute_type}")

    with _ASR_MODEL_LOCK:
        if cache_key in _ASR_MODEL_CACHE:
            _emit_progress(progress_callback, "loading_model", f"复用已加载转写模型：{model_kind}", percent=15)
            return _ASR_MODEL_CACHE[cache_key]

        _emit_progress(progress_callback, "loading_model", f"首次加载转写模型：{model_kind}", percent=10)
        if model_kind == ASR_MODEL_FASTER:
            from faster_whisper import WhisperModel

            model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
        elif model_kind == ASR_MODEL_OPENAI:
            import whisper

            weight_path = model_path / "large-v3-turbo.pt"
            if not weight_path.exists():
                raise FileNotFoundError(f"没有找到 Whisper 权重文件：{weight_path}")
            model = whisper.load_model(str(weight_path), device=device)
        else:
            raise ValueError(f"不支持的转写模型：{model_kind}")

        _ASR_MODEL_CACHE[cache_key] = model
        return model


def _run_faster_whisper(
    model,
    request: TranscribeRequest,
    progress_callback: AsrProgressCallback | None,
    cancel_token: CancelToken | None,
) -> tuple[str, list[TranscribeSegment]]:
    _emit_progress(progress_callback, "transcribing", "正在转写音频。", percent=35)
    segments_iter, _ = model.transcribe(
        request.audio_path,
        language=None if not request.language or request.language == "auto" else request.language,
        vad_filter=False,
        beam_size=5,
        word_timestamps=False,
    )

    collected: list[TranscribeSegment] = []
    for index, segment in enumerate(segments_iter, start=1):
        _check_cancel(cancel_token)
        text = segment.text.strip()
        collected.append(
            TranscribeSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=text,
            )
        )
        percent = min(35 + index * 5, 90)
        _emit_progress(progress_callback, "transcribing", f"已转写 {index} 段。", current=index, percent=percent)

    full_text = "".join(segment.text for segment in collected).strip()
    return full_text, collected


def _run_openai_whisper(
    model,
    request: TranscribeRequest,
    progress_callback: AsrProgressCallback | None,
    cancel_token: CancelToken | None,
) -> tuple[str, list[TranscribeSegment]]:
    _emit_progress(progress_callback, "transcribing", "正在转写音频。", percent=35)
    _check_cancel(cancel_token)
    audio_samples = _load_audio_for_openai_whisper(request.audio_path)
    use_fp16 = str(getattr(model, "device", "cpu")) != "cpu"
    safe_stdout = sys.stdout if hasattr(sys.stdout, "write") else _NullWriter()
    safe_stderr = sys.stderr if hasattr(sys.stderr, "write") else _NullWriter()
    with contextlib.redirect_stdout(safe_stdout), contextlib.redirect_stderr(safe_stderr):
        result = model.transcribe(
            audio_samples,
            language=None if not request.language or request.language == "auto" else request.language,
            fp16=use_fp16,
            verbose=False,
        )
    _check_cancel(cancel_token)
    raw_segments = result.get("segments") or []
    segments = [
        TranscribeSegment(
            start_seconds=float(item.get("start", 0.0)),
            end_seconds=float(item.get("end", 0.0)),
            text=str(item.get("text", "")).strip(),
        )
        for item in raw_segments
        if str(item.get("text", "")).strip()
    ]
    full_text = str(result.get("text", "")).strip()
    _emit_progress(progress_callback, "transcribing", f"转写完成，共 {len(segments)} 段。", current=len(segments), total=len(segments), percent=90)
    return full_text, segments


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def build_smart_paragraphs(
    segments: list[TranscribeSegment],
    pause_threshold_seconds: float = SMART_SEGMENT_DEFAULT_PAUSE_SECONDS,
    max_chars_per_paragraph: int = SMART_SEGMENT_MAX_CHARS,
) -> list[str]:
    paragraphs: list[str] = []
    current_parts: list[str] = []
    current_length = 0
    previous_end: float | None = None

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        should_break = False
        if previous_end is not None and segment.start_seconds - previous_end >= pause_threshold_seconds:
            should_break = True
        if current_parts and current_length + len(text) > max_chars_per_paragraph:
            should_break = True

        if should_break and current_parts:
            paragraphs.append(" ".join(current_parts).strip())
            current_parts = []
            current_length = 0

        current_parts.append(text)
        current_length += len(text)
        previous_end = segment.end_seconds

    if current_parts:
        paragraphs.append(" ".join(current_parts).strip())
    return paragraphs


def render_transcript_text(
    text: str,
    segments: list[TranscribeSegment],
    output_format: str | None = None,
    with_timestamps: bool = True,
    smart_segment_pause_seconds: float = SMART_SEGMENT_DEFAULT_PAUSE_SECONDS,
) -> str:
    normalized = normalize_transcript_format(output_format, with_timestamps=with_timestamps)

    if normalized == TRANSCRIPT_FORMAT_PLAIN:
        return text.strip() + ("\n" if text.strip() else "")

    if normalized == TRANSCRIPT_FORMAT_SMART:
        paragraphs = build_smart_paragraphs(segments, pause_threshold_seconds=smart_segment_pause_seconds)
        if paragraphs:
            return "\n\n".join(paragraphs).strip() + "\n"
        return text.strip() + ("\n" if text.strip() else "")

    if normalized == TRANSCRIPT_FORMAT_TIMESTAMPS:
        lines = [
            f"[{format_timestamp(segment.start_seconds)} - {format_timestamp(segment.end_seconds)}] {segment.text}"
            for segment in segments
            if segment.text.strip()
        ]
        return "\n".join(lines).strip() + ("\n" if lines else "")

    if normalized == TRANSCRIPT_FORMAT_SRT:
        blocks: list[str] = []
        for index, segment in enumerate(segments, start=1):
            if not segment.text.strip():
                continue
            blocks.append(
                f"{index}\n{_format_srt_timestamp(segment.start_seconds)} --> {_format_srt_timestamp(segment.end_seconds)}\n{segment.text.strip()}"
            )
        return "\n\n".join(blocks).strip() + ("\n" if blocks else "")

    raise ValueError(f"不支持的转写输出格式：{normalized}")


def _write_transcript(
    output_path: Path,
    text: str,
    segments: list[TranscribeSegment],
    output_format: str | None,
    with_timestamps: bool,
    smart_segment_pause_seconds: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_transcript_text(
        text,
        segments,
        output_format=output_format,
        with_timestamps=with_timestamps,
        smart_segment_pause_seconds=smart_segment_pause_seconds,
    )
    output_path.write_text(rendered, encoding="utf-8")


def transcribe_audio(
    request: TranscribeRequest,
    progress_callback: AsrProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
) -> TranscribeResult:
    if request.model_kind not in SUPPORTED_ASR_MODELS:
        raise ValueError(f"不支持的转写模型：{request.model_kind}")

    _emit_progress(progress_callback, "checking_environment", "正在检查转写环境。", percent=3)
    status = check_asr_environment(request.model_kind)
    if status.missing_packages:
        raise RuntimeError(
            f"{request.model_kind} 缺少转写依赖："
            + ", ".join(status.missing_packages)
            + "\n请先运行：\npython -m pip install -r requirements-voxcpm2-gui.txt"
        )

    audio_path = Path(request.audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    normalized_format = normalize_transcript_format(request.output_format, with_timestamps=request.with_timestamps)
    output_path = (
        Path(request.output_path).expanduser().resolve()
        if request.output_path
        else build_transcript_output_path(
            audio_path,
            output_format=normalized_format,
            with_timestamps=request.with_timestamps,
        )
    )

    model = load_asr_model_once(request.model_kind, AsrModelLoadOptions(), progress_callback=progress_callback)
    _check_cancel(cancel_token)

    if request.model_kind == ASR_MODEL_FASTER:
        text, segments = _run_faster_whisper(model, request, progress_callback, cancel_token)
    else:
        text, segments = _run_openai_whisper(model, request, progress_callback, cancel_token)

    _check_cancel(cancel_token)
    _emit_progress(progress_callback, "writing", "正在写入转写结果。", percent=95)
    _write_transcript(
        output_path,
        text,
        segments,
        output_format=normalized_format,
        with_timestamps=request.with_timestamps,
        smart_segment_pause_seconds=request.smart_segment_pause_seconds,
    )
    duration_seconds = _read_audio_duration_seconds(str(audio_path))

    result = TranscribeResult(
        output_path=str(output_path),
        text=text,
        segments=segments,
        duration_seconds=duration_seconds,
        model_kind=request.model_kind,
    )
    _emit_progress(progress_callback, "completed", f"转写完成：{output_path}", current=len(segments), total=len(segments), percent=100)
    return result
