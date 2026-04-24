from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxcpm_service import (
    DEFAULT_CFG,
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_SILENCE_MS,
    DEFAULT_STEPS,
    DEFAULT_VOICE,
    GenerateRequest,
    check_environment,
    generate_tts,
    get_default_model_dir,
)


def read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    input_path = Path(args.input)
    if input_path.exists():
        return input_path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"没有找到输入文件：{input_path}\n"
        "请使用 --text 传入文字，或把文案写入 sample_text.txt。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use VoxCPM2 to generate Chinese voice-over WAV files."
    )
    parser.add_argument(
        "--text",
        help="直接传入要配音的中文文本；不传时读取 --input 文件。",
    )
    parser.add_argument(
        "--input",
        default="sample_text.txt",
        help="输入文本文件，默认 sample_text.txt。",
    )
    parser.add_argument(
        "--output",
        default="outputs/voxcpm2_output.wav",
        help="输出 WAV 路径，默认 outputs/voxcpm2_output.wav。",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="声音描述，会自动放到每段文本开头。留空则不加声音描述。",
    )
    parser.add_argument(
        "--reference-wav",
        help="可选：授权参考音频路径，用于音色克隆。",
    )
    parser.add_argument(
        "--prompt-wav",
        help="可选：高相似度克隆时的参考音频路径，通常可与 --reference-wav 相同。",
    )
    parser.add_argument(
        "--prompt-text",
        help="可选：--prompt-wav 对应的逐字稿。",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=DEFAULT_CFG,
        help="风格控制强度，默认 2.0。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="推理步数，默认 10；更高可能更慢。",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=DEFAULT_CHUNK_MAX_CHARS,
        help="长文自动分段长度，默认 120 个字符。",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_MS,
        help="分段之间插入的静音毫秒数，默认 250。",
    )
    parser.add_argument(
        "--load-denoiser",
        action="store_true",
        help="加载降噪器；默认关闭以节省显存。",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="开启 torch.compile 优化；首次运行会更慢，12GB 显存建议先保持关闭。",
    )
    parser.add_argument(
        "--model-id",
        default=str(get_default_model_dir()) if get_default_model_dir().exists() else "openbmb/VoxCPM2",
        help="Hugging Face 模型 ID 或本地模型目录；如果当前目录有 VoxCPM2，则默认优先使用本地目录。",
    )
    parser.add_argument(
        "--cache-dir",
        help="可选：指定 Hugging Face 模型缓存目录。",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用本地已下载模型文件，不联网下载。",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="只检查依赖和 CUDA 状态，不下载模型、不生成音频。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    status = check_environment(args.model_id)
    if status.cuda_available and status.gpu_name:
        print(f"GPU: {status.gpu_name} ({status.total_vram_gb:.1f} GB VRAM)")
    else:
        print("警告：PyTorch 没有检测到 CUDA，将会非常慢。", file=sys.stderr)

    if args.check_env:
        if status.missing_packages:
            print("缺少依赖：" + ", ".join(status.missing_packages), file=sys.stderr)
            return 1
        print("依赖检查通过：VoxCPM、PyTorch、soundfile、numpy 均可导入。")
        return 0

    try:
        raw_text = read_text(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def console_progress(progress) -> None:
        if progress.stage == "generating" and progress.total:
            print(f"[{progress.current}/{progress.total}] {progress.message}")
        else:
            print(progress.message)

    try:
        result = generate_tts(
            GenerateRequest(
                text=raw_text,
                voice=args.voice,
                reference_wav=args.reference_wav,
                prompt_wav=args.prompt_wav,
                prompt_text=args.prompt_text,
                cfg=args.cfg,
                steps=args.steps,
                chunk_max_chars=args.chunk_max_chars,
                silence_ms=args.silence_ms,
                load_denoiser=args.load_denoiser,
                optimize=args.optimize,
                model_path=args.model_id,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                output_path=args.output,
            ),
            progress_callback=console_progress,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"完成：{Path(result.output_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
