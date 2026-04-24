from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download VoxCPM2 into a local folder.")
    parser.add_argument("--repo-id", default="openbmb/VoxCPM2")
    parser.add_argument("--local-dir", default="VoxCPM2")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint. Defaults to HF_ENDPOINT or https://huggingface.co.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-sleep", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"local_dir={local_dir}", flush=True)
    print(f"endpoint={args.endpoint}", flush=True)
    print("开始下载，已存在的文件会自动复用。", flush=True)

    from huggingface_hub import snapshot_download

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"尝试 {attempt}/{args.retries}", flush=True)
            downloaded_path = snapshot_download(
                repo_id=args.repo_id,
                local_dir=local_dir,
                endpoint=args.endpoint,
                etag_timeout=60,
                max_workers=args.max_workers,
            )
            print(f"下载完成：{Path(downloaded_path).resolve()}", flush=True)
            return 0
        except Exception as exc:
            last_error = exc
            print(f"本次下载中断：{type(exc).__name__}: {exc}", flush=True)
            if attempt < args.retries:
                print(f"{args.retry_sleep} 秒后自动续传...", flush=True)
                time.sleep(args.retry_sleep)

    print(f"下载失败：{last_error}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
