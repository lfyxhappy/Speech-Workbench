from __future__ import annotations

import threading
from dataclasses import dataclass


class TaskCancelledError(RuntimeError):
    def __init__(self, message: str = "任务已取消。") -> None:
        super().__init__(message)


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(slots=True)
class TaskProgress:
    stage: str
    message: str
    current: int = 0
    total: int = 0
    percent: int | None = None


@dataclass(slots=True)
class QueueTaskRecord:
    task_id: str
    page_kind: str
    status: str
    title: str
    input_path: str
    output_path: str | None = None
    error: str | None = None


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
