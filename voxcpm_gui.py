from __future__ import annotations

import argparse
import json
import os
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque

from PyQt6.QtCore import QObject, QThread, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app_shared import CancelToken, QueueTaskRecord, TaskCancelledError, TaskProgress, format_timestamp
from audiofx_service import (
    DEFAULT_AUDIOFX_DURATION_SECONDS,
    DEFAULT_AUDIOFX_GUIDANCE_SCALE,
    DEFAULT_AUDIOFX_STEPS,
    DEFAULT_AUDIOFX_VARIANTS,
    AudioFxRequest,
    AudioFxResult,
    build_audiofx_output_path,
    check_audiofx_environment,
    generate_audiofx,
    get_default_audiofx_model_dir,
    get_default_audiofx_output_dir,
)
from asr_service import (
    ASR_MODEL_FASTER,
    ASR_MODEL_OPENAI,
    DEFAULT_ASR_MODEL,
    DEFAULT_TRANSCRIPT_FORMAT,
    LEGACY_TRANSCRIPT_FORMAT,
    SMART_SEGMENT_DEFAULT_PAUSE_SECONDS,
    TRANSCRIPT_FORMAT_LABELS,
    TRANSCRIPT_FORMAT_PLAIN,
    TRANSCRIPT_FORMAT_SMART,
    TRANSCRIPT_FORMAT_SRT,
    TRANSCRIPT_FORMAT_TIMESTAMPS,
    TranscribeRequest,
    TranscribeResult,
    build_transcript_output_path,
    check_asr_environment,
    get_default_faster_whisper_dir,
    get_default_openai_whisper_dir,
    get_default_stt_output_dir,
    get_transcript_format_label,
    render_transcript_text,
    transcribe_audio,
)
from voxcpm_service import (
    DEFAULT_CFG,
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_SILENCE_MS,
    DEFAULT_STEPS,
    DEFAULT_VOICE,
    GenerateRequest,
    GenerateResult,
    build_timestamped_output_path,
    check_environment,
    generate_tts,
    get_default_model_dir,
    get_default_output_dir,
    get_settings_path,
)

VOICE_PRESETS = [
    DEFAULT_VOICE,
    "成熟男性，沉稳自然，普通话标准，适合纪录片旁白",
    "年轻男性，清晰阳光，节奏轻快，适合知识分享",
    "温柔女声，亲切自然，语速稍慢，适合情感讲述",
    "专业女声，干净利落，吐字清晰，适合产品介绍",
    "磁性男声，厚实稳重，适合商业解说和专题配音",
]


@dataclass(slots=True)
class QueueJob:
    record: QueueTaskRecord
    payload: object
    cancel_token: CancelToken
    kind: str


@dataclass(slots=True)
class PromptFillResult:
    text: str
    model_kind: str


class PathField(QWidget):
    def __init__(self, button_text: str, pick_directory: bool = False, allow_multi: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.pick_directory = pick_directory
        self.allow_multi = allow_multi
        self.line_edit = QLineEdit()
        self.button = QPushButton(button_text)
        self.button.setFixedWidth(90)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)

        self.button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, value: str) -> None:
        self.line_edit.setText(value)

    def setEnabled(self, enabled: bool) -> None:
        self.line_edit.setEnabled(enabled)
        self.button.setEnabled(enabled)
        super().setEnabled(enabled)

    def setToolTip(self, text: str) -> None:
        self.line_edit.setToolTip(text)
        self.button.setToolTip(text)
        super().setToolTip(text)

    def _browse(self) -> None:
        if self.pick_directory:
            selected = QFileDialog.getExistingDirectory(self, "选择目录", self.text() or str(get_default_output_dir()))
            if selected:
                self.setText(selected)
            return

        if self.allow_multi:
            selected, _ = QFileDialog.getOpenFileNames(
                self,
                "选择音频文件",
                self.text() or str(get_default_output_dir()),
                "音频文件 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;所有文件 (*.*)",
            )
            if selected:
                self.setText(";".join(selected))
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            self.text() or str(get_default_output_dir()),
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;所有文件 (*.*)",
        )
        if selected:
            self.setText(selected)


class CollapsibleSection(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.content_widget = QWidget()
        self.content_widget.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)

        self.toggle_button.toggled.connect(self._on_toggled)

    def setContentLayout(self, layout):
        self.content_widget.setLayout(layout)

    def isExpanded(self) -> bool:
        return self.toggle_button.isChecked()

    def setExpanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.content_widget.setVisible(checked)
        self.toggled.emit(checked)


class TaskWorker(QObject):
    progress = pyqtSignal(str, object)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str, str)

    @pyqtSlot(str, object, object, str)
    def run_job(self, task_id: str, payload: object, cancel_token: CancelToken, job_kind: str) -> None:
        try:
            if job_kind == "tts":
                result = generate_tts(payload, progress_callback=lambda item: self.progress.emit(task_id, item), cancel_token=cancel_token)
            elif job_kind == "stt":
                result = transcribe_audio(payload, progress_callback=lambda item: self.progress.emit(task_id, item), cancel_token=cancel_token)
            elif job_kind == "prompt_fill":
                result = transcribe_audio(payload, progress_callback=lambda item: self.progress.emit(task_id, item), cancel_token=cancel_token)
                result = PromptFillResult(text=result.text, model_kind=result.model_kind)
            elif job_kind == "audiofx":
                result = generate_audiofx(payload, progress_callback=lambda item: self.progress.emit(task_id, item), cancel_token=cancel_token)
            else:
                raise ValueError(f"未知任务类型：{job_kind}")
            self.finished.emit(task_id, result)
        except TaskCancelledError as exc:
            self.cancelled.emit(task_id, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(task_id, str(exc))


class PromptWorkerController(QObject):
    start_job = pyqtSignal(str, object, object, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.thread = QThread()
        self.worker = TaskWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        self.start_job.connect(self.worker.run_job)

    def shutdown(self) -> None:
        self.thread.quit()
        self.thread.wait(3000)


class QueueController(QObject):
    start_job = pyqtSignal(str, object, object, str)
    queue_changed = pyqtSignal()
    current_job_changed = pyqtSignal(object)
    progress = pyqtSignal(object, object)
    result = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)
    cancelled = pyqtSignal(object, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.queue: Deque[QueueJob] = deque()
        self.records: list[QueueTaskRecord] = []
        self.record_map: dict[str, QueueTaskRecord] = {}
        self.current_job: QueueJob | None = None

        self.thread = QThread()
        self.worker = TaskWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

        self.start_job.connect(self.worker.run_job)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)

    def shutdown(self) -> None:
        if self.current_job:
            self.current_job.cancel_token.cancel()
        self.thread.quit()
        self.thread.wait(3000)

    def add_job(self, job: QueueJob) -> None:
        self.records.append(job.record)
        self.record_map[job.record.task_id] = job.record
        self.queue.append(job)
        self.queue_changed.emit()
        self._kick_next()

    def get_record(self, task_id: str) -> QueueTaskRecord | None:
        return self.record_map.get(task_id)

    def remove_record(self, task_id: str) -> None:
        if self.current_job and self.current_job.record.task_id == task_id:
            return
        self.queue = deque(job for job in self.queue if job.record.task_id != task_id)
        self.records = [record for record in self.records if record.task_id != task_id]
        self.record_map.pop(task_id, None)
        self.queue_changed.emit()

    def clear_finished(self) -> None:
        removable_statuses = {"completed", "failed", "cancelled"}
        self.records = [record for record in self.records if record.status not in removable_statuses]
        self.record_map = {record.task_id: record for record in self.records}
        self.queue = deque(job for job in self.queue if job.record.task_id in self.record_map)
        self.queue_changed.emit()

    def cancel_current(self) -> None:
        if self.current_job:
            self.current_job.record.status = "cancelling"
            self.current_job.cancel_token.cancel()
            self.queue_changed.emit()

    def _kick_next(self) -> None:
        if self.current_job or not self.queue:
            return
        self.current_job = self.queue.popleft()
        self.current_job.record.status = "running"
        self.current_job_changed.emit(self.current_job.record)
        self.queue_changed.emit()
        self.start_job.emit(
            self.current_job.record.task_id,
            self.current_job.payload,
            self.current_job.cancel_token,
            self.current_job.kind,
        )

    @pyqtSlot(str, object)
    def _on_progress(self, task_id: str, progress_item: TaskProgress) -> None:
        record = self.record_map.get(task_id)
        if record:
            self.progress.emit(record, progress_item)

    @pyqtSlot(str, object)
    def _on_finished(self, task_id: str, result: object) -> None:
        job = self.current_job
        record = self.record_map.get(task_id)
        if record:
            record.status = "completed"
            if isinstance(result, (GenerateResult, TranscribeResult, AudioFxResult)):
                record.output_path = result.output_path
            self.result.emit(record, result)
        self.current_job = None
        self.current_job_changed.emit(None)
        self.queue_changed.emit()
        self._kick_next()

    @pyqtSlot(str, str)
    def _on_failed(self, task_id: str, message: str) -> None:
        record = self.record_map.get(task_id)
        if record:
            record.status = "failed"
            record.error = message
            self.failed.emit(record, message)
        self.current_job = None
        self.current_job_changed.emit(None)
        self.queue_changed.emit()
        self._kick_next()

    @pyqtSlot(str, str)
    def _on_cancelled(self, task_id: str, message: str) -> None:
        record = self.record_map.get(task_id)
        if record:
            record.status = "cancelled"
            record.error = message
            self.cancelled.emit(record, message)
        self.current_job = None
        self.current_job_changed.emit(None)
        self.queue_changed.emit()
        self._kick_next()


class QueuePanel(QWidget):
    remove_requested = pyqtSignal(str)
    clear_finished_requested = pyqtSignal()
    cancel_current_requested = pyqtSignal()

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        self.list_widget = QListWidget()
        self.current_label = QLabel("当前任务：空闲")
        button_row = QHBoxLayout()
        self.cancel_button = QPushButton("取消当前")
        self.remove_button = QPushButton("移除选中")
        self.clear_button = QPushButton("清空已完成")
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.clear_button)

        group_layout.addWidget(self.current_label)
        group_layout.addWidget(self.list_widget, 1)
        group_layout.addLayout(button_row)
        layout.addWidget(group)

        self.cancel_button.clicked.connect(self.cancel_current_requested.emit)
        self.clear_button.clicked.connect(self.clear_finished_requested.emit)
        self.remove_button.clicked.connect(self._emit_remove_selected)
        self.cancel_button.setToolTip("取消当前正在执行的任务。当前已经开始生成或转写的这一小段通常会先跑完，再停止。")
        self.remove_button.setToolTip("移除当前选中的队列项。正在运行的任务不能直接移除，请先取消。")
        self.clear_button.setToolTip("清理所有已完成、失败或已取消的历史任务，保留仍在排队或运行中的任务。")
        self.list_widget.setToolTip("这里显示当前页面的任务队列和历史结果。点击某一项后可移除选中任务。")

    def refresh(self, records: list[QueueTaskRecord], current_record: QueueTaskRecord | None) -> None:
        self.list_widget.clear()
        for record in records:
            suffix = f" -> {Path(record.output_path).name}" if record.output_path else ""
            item = QListWidgetItem(f"[{record.status}] {record.title}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, record.task_id)
            self.list_widget.addItem(item)
        self.current_label.setText(
            f"当前任务：{current_record.title}" if current_record else "当前任务：空闲"
        )

    def _emit_remove_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(task_id, str):
            self.remove_requested.emit(task_id)


class TtsPage(QWidget):
    def __init__(self, queue_controller: QueueController, app_settings: dict[str, object], parent: QWidget | None = None):
        super().__init__(parent)
        self.queue_controller = queue_controller
        self.app_settings = app_settings
        self.current_output_path: Path | None = None
        self.current_progress_task_id: str | None = None

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)

        self.prompt_worker_controller = PromptWorkerController(self)
        self.prompt_transcribe_worker = self.prompt_worker_controller.worker
        self._prompt_cancel_token: CancelToken | None = None
        self._prompt_task_id: str | None = None

        self._build_ui()
        self.prompt_model_combo.currentTextChanged.connect(self._refresh_environment)
        self._load_settings()
        self._refresh_environment()
        self._prepare_player(None)

        self.queue_controller.queue_changed.connect(self._refresh_queue)
        self.queue_controller.current_job_changed.connect(self._refresh_queue)
        self.queue_controller.progress.connect(self._on_queue_progress)
        self.queue_controller.result.connect(self._on_queue_result)
        self.queue_controller.failed.connect(self._on_queue_failed)
        self.queue_controller.cancelled.connect(self._on_queue_cancelled)

        self.prompt_transcribe_worker.progress.connect(self._on_prompt_progress)
        self.prompt_transcribe_worker.finished.connect(self._on_prompt_finished)
        self.prompt_transcribe_worker.failed.connect(self._on_prompt_failed)
        self.prompt_transcribe_worker.cancelled.connect(self._on_prompt_cancelled)
        self._refresh_queue()

    def shutdown(self) -> None:
        self.media_player.stop()
        if self._prompt_cancel_token:
            self._prompt_cancel_token.cancel()
        self.prompt_worker_controller.shutdown()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        warning = QLabel("提示：语音生成与语音转文本会共享 GPU，同时运行时速度可能下降。")
        warning.setStyleSheet("color: #8a4b08;")
        layout.addWidget(warning)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([540, 450, 520])
        self._install_tooltips()

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        input_group = QGroupBox("文案输入")
        input_layout = QVBoxLayout(input_group)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("把需要配音的中文文案粘贴到这里。")
        input_layout.addWidget(self.text_edit)
        layout.addWidget(input_group, 1)

        basic_group = QGroupBox("基础设置")
        basic_form = QFormLayout(basic_group)
        self.voice_edit = QComboBox()
        self.voice_edit.setEditable(True)
        self.voice_edit.addItems(VOICE_PRESETS)
        self.voice_edit.setCurrentText(DEFAULT_VOICE)
        self.output_dir_field = PathField("选择目录", pick_directory=True)
        self.output_dir_field.setText(str(get_default_output_dir()))
        basic_form.addRow("音色描述", self.voice_edit)
        basic_form.addRow("输出目录", self.output_dir_field)
        layout.addWidget(basic_group)
        return panel

    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        runtime_group = QGroupBox("运行设置")
        runtime_form = QFormLayout(runtime_group)
        self.model_path_field = PathField("选择模型", pick_directory=True)
        self.model_path_field.setText(str(get_default_model_dir()))

        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.1, 20.0)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setValue(DEFAULT_CFG)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(DEFAULT_STEPS)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(20, 1000)
        self.chunk_spin.setValue(DEFAULT_CHUNK_MAX_CHARS)

        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(0, 5000)
        self.silence_spin.setValue(DEFAULT_SILENCE_MS)
        self.silence_spin.setSuffix(" ms")

        self.load_denoiser_check = QCheckBox("加载 denoiser")
        self.optimize_check = QCheckBox("开启 torch.compile 优化")
        self.reuse_first_chunk_check = QCheckBox("首段锁定后续音色")
        self.auto_play_check = QCheckBox("任务完成后自动播放")
        self.auto_play_check.setChecked(True)

        runtime_form.addRow("模型目录", self.model_path_field)
        runtime_form.addRow("CFG", self.cfg_spin)
        runtime_form.addRow("推理步数", self.steps_spin)
        runtime_form.addRow("分段长度", self.chunk_spin)
        runtime_form.addRow("段间静音", self.silence_spin)
        runtime_form.addRow("", self.load_denoiser_check)
        runtime_form.addRow("", self.optimize_check)
        runtime_form.addRow("", self.reuse_first_chunk_check)
        runtime_form.addRow("", self.auto_play_check)
        layout.addWidget(runtime_group)
        layout.addWidget(self._build_settings_help_panel())

        self.clone_section = CollapsibleSection("音色克隆（可选）")
        clone_layout = QFormLayout()
        self.reference_wav_field = PathField("选择音频", pick_directory=False)
        self.prompt_wav_field = PathField("选择音频", pick_directory=False)
        self.prompt_text_edit = QPlainTextEdit()
        self.prompt_text_edit.setPlaceholderText("如果填写了 Prompt 音频，这里要填写对应逐字稿。")
        self.prompt_text_edit.setFixedHeight(110)
        self.prompt_model_combo = QComboBox()
        self.prompt_model_combo.addItems([ASR_MODEL_OPENAI, ASR_MODEL_FASTER])
        self.prompt_transcribe_button = QPushButton("一键转写 Prompt 音频")
        self.prompt_transcribe_button.clicked.connect(self.on_prompt_transcribe_clicked)
        clone_layout.addRow("参考音频", self.reference_wav_field)
        clone_layout.addRow("Prompt 音频", self.prompt_wav_field)
        clone_layout.addRow("转写模型", self.prompt_model_combo)
        clone_layout.addRow("", self.prompt_transcribe_button)
        clone_layout.addRow("Prompt 文本", self.prompt_text_edit)
        self.clone_section.setContentLayout(clone_layout)
        layout.addWidget(self.clone_section)

        self.queue_panel = QueuePanel("生成队列")
        self.queue_panel.cancel_current_requested.connect(self.queue_controller.cancel_current)
        self.queue_panel.clear_finished_requested.connect(self.queue_controller.clear_finished)
        self.queue_panel.remove_requested.connect(self.queue_controller.remove_record)
        layout.addWidget(self.queue_panel)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_settings_help_panel(self) -> QWidget:
        help_group = QGroupBox("参数详解")
        help_layout = QVBoxLayout(help_group)
        help_label = QLabel(
            "<b>音色描述</b>：控制整体说话风格，比如年龄感、性别感、语气、语速、适用场景。先选一个预设，再补充关键词，通常更容易得到稳定结果。<br><br>"
            "<b>输出目录</b>：生成的音频会按时间戳自动命名，避免覆盖旧文件。建议固定到同一个目录，方便统一管理。<br><br>"
            "<b>模型目录</b>：本地 VoxCPM2 模型所在的位置。一般保持默认即可，只有你手动移动过模型文件夹时才需要修改。<br><br>"
            "<b>CFG</b>：控制模型“听不听你的音色描述和风格提示”。数值越高，越强调提示词；数值太低可能风格不明显，太高有时会显得生硬。日常中文配音一般 1.5 到 3.0 比较稳，默认 2.0 适合大多数讲解和旁白。<br><br>"
            "<b>推理步数</b>：控制每段音频生成时的采样迭代次数。通常越高越慢，但细节和稳定性可能更好；太低时偶尔会更飘。12GB 显存机器日常建议先用 8 到 12，默认 10 是速度和效果的折中。<br><br>"
            "<b>分段长度</b>：长文案会按这个长度自动拆成多段，再分别生成后拼接。值越大，单段上下文更完整，但单次生成更慢、更吃显存；值越小，更稳、更省显存，但段落会变多。做中文讲解时 80 到 160 通常比较合适，默认 120 偏稳。<br><br>"
            "<b>段间静音</b>：每一段拼接时中间插入的停顿时长，单位是毫秒。值太小，句子之间可能粘连；值太大，听感会拖。普通讲解、解说、旁白一般 150 到 300 ms 比较自然，默认 250 ms。<br><br>"
            "<b>加载 denoiser</b>：启用额外的后处理降噪。可能让声音更干净一些，但会增加显存占用和生成时间；如果当前声音已经干净，建议保持关闭，优先保证速度和稳定性。<br><br>"
            "<b>开启 torch.compile 优化</b>：让 PyTorch 尝试为当前模型做编译优化。优点是同一会话里连续生成时，后续任务可能更快；缺点是第一次生成前会额外花时间，而且可能多占一点显存。首次使用或只生成一两条时建议关闭，频繁批量生成时再考虑开启。<br><br>"
            "<b>首段锁定后续音色</b>：第 1 段先正常生成，再把第 1 段音频作为后续段落的参考音频，用来增强整篇文案前后音色一致性。通常对长文配音比较有帮助，但如果第 1 段状态不好，后面也会一起偏掉。<br><br>"
            "<b>任务完成后自动播放</b>：生成结束后自动试听当前结果。适合反复调音色时快速试听；如果你经常批量排队生成，可以关闭，避免每次完成都自动播放。<br><br>"
            "<b>Prompt 自动转写</b>：把 Prompt 音频自动识别成文字并填入 Prompt 文本，省去手动听写。适合做音色克隆或高相似度复刻时快速准备逐字稿。"
        )
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setStyleSheet(
            "QLabel {background: #f7f7f8; border: 1px solid #d9d9df; border-radius: 8px; padding: 12px; line-height: 1.45;}"
        )
        help_layout.addWidget(help_label)
        return help_group

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        result_group = QGroupBox("状态与结果")
        result_layout = QVBoxLayout(result_group)

        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.stage_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)

        button_row = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.stop_button = QPushButton("停止")
        self.open_file_button = QPushButton("打开文件")
        self.open_dir_button = QPushButton("打开目录")
        for widget in (self.play_button, self.stop_button, self.open_file_button, self.open_dir_button):
            button_row.addWidget(widget)

        self.enqueue_button = QPushButton("加入队列")
        self.enqueue_button.setStyleSheet("font-size: 16px; padding: 12px 20px;")

        result_layout.addWidget(QLabel("环境状态"))
        result_layout.addWidget(self.environment_label)
        result_layout.addWidget(QLabel("当前阶段"))
        result_layout.addWidget(self.stage_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(QLabel("日志"))
        result_layout.addWidget(self.log_edit, 1)
        result_layout.addWidget(QLabel("输出文件"))
        result_layout.addWidget(self.output_file_edit)
        result_layout.addLayout(button_row)
        result_layout.addWidget(self.enqueue_button)
        layout.addWidget(result_group, 1)

        self.enqueue_button.clicked.connect(self.on_enqueue_clicked)
        self.play_button.clicked.connect(self.play_output)
        self.stop_button.clicked.connect(self.media_player.stop)
        self.open_file_button.clicked.connect(self.open_output_file)
        self.open_dir_button.clicked.connect(self.open_output_directory)
        self.environment_label.setToolTip("显示当前 Python、CUDA、依赖和模型目录状态。红字通常表示缺少依赖或模型路径异常。")
        self.stage_label.setToolTip("显示当前任务所处阶段，例如检查环境、拆分文案、生成中、写文件或完成。")
        self.progress_bar.setToolTip("显示当前生成任务的大致进度。")
        self.log_edit.setToolTip("显示本页任务的过程日志，包括入队、生成进度、报错和完成提示。")
        self.output_file_edit.setToolTip("这里显示当前最新生成结果的完整文件路径。")
        return panel

    def _install_tooltips(self) -> None:
        self.text_edit.setToolTip("在这里输入或粘贴要配音的文案。长文本会自动按分段长度拆开，再逐段生成并拼接。")
        self.voice_edit.setToolTip("控制声音风格。可以描述年龄、性别、语气、语速和适用场景，也可以直接选预设后再微调。")
        self.output_dir_field.setToolTip("生成音频的保存目录。程序会按时间戳自动命名，避免覆盖旧文件。")
        self.model_path_field.setToolTip("本地 VoxCPM2 模型所在目录。一般保持默认即可，只有你移动过模型时才需要改。")
        self.cfg_spin.setToolTip("控制模型对音色描述和风格提示的遵从程度。越高越强调提示词，通常 1.5 到 3.0 比较稳。")
        self.steps_spin.setToolTip("每段音频生成时的推理步数。越高通常越慢，但细节可能更稳；默认 10 是速度和效果的折中。")
        self.chunk_spin.setToolTip("长文按多少字符左右拆成一段。值越大上下文更完整，但更慢、更吃显存；默认 120 偏稳。")
        self.silence_spin.setToolTip("相邻两段拼接时插入的静音时长，单位毫秒。普通讲解和旁白常用 150 到 300 ms。")
        self.load_denoiser_check.setToolTip("启用额外后处理降噪。可能让声音更干净，但会增加显存占用和生成时间。")
        self.optimize_check.setToolTip("开启 PyTorch 编译优化。首次生成会更慢，但同一会话内后续连续生成可能更快。")
        self.reuse_first_chunk_check.setToolTip("第 1 段先正常生成，从第 2 段开始自动复用首段音频作为参考音频，增强长文前后音色一致性。")
        self.auto_play_check.setToolTip("生成完成后自动播放当前结果。适合反复调音试听，批量排队时可关闭。")
        self.reference_wav_field.setToolTip("可选的参考音频，用于音色克隆。建议使用你自己或已获授权的声音样本。")
        self.prompt_wav_field.setToolTip("用于高相似度克隆的 Prompt 音频。通常需要和下方 Prompt 文本配对使用。")
        self.prompt_model_combo.setToolTip("选择把 Prompt 音频转成文字时使用的转写模型。大模型更准，小模型更快。")
        self.prompt_transcribe_button.setToolTip("自动识别 Prompt 音频内容，并把结果填入 Prompt 文本，省去手动听写。")
        self.prompt_text_edit.setToolTip("Prompt 音频对应的逐字稿。做高相似度音色克隆时，Prompt 音频和 Prompt 文本需要同时填写。")
        self.play_button.setToolTip("播放当前生成完成的音频。")
        self.stop_button.setToolTip("停止当前试听播放。")
        self.open_file_button.setToolTip("在系统中直接打开当前输出音频文件。")
        self.open_dir_button.setToolTip("打开当前输出音频所在目录。")
        self.enqueue_button.setToolTip("把当前表单内容加入生成队列。队列会按顺序自动执行。")

    def _settings_payload(self) -> dict[str, object]:
        return {
            "voice": self.voice_edit.currentText().strip(),
            "output_dir": self.output_dir_field.text(),
            "model_path": self.model_path_field.text(),
            "cfg": self.cfg_spin.value(),
            "steps": self.steps_spin.value(),
            "chunk_max_chars": self.chunk_spin.value(),
            "silence_ms": self.silence_spin.value(),
            "load_denoiser": self.load_denoiser_check.isChecked(),
            "optimize": self.optimize_check.isChecked(),
            "reuse_first_chunk_as_reference": self.reuse_first_chunk_check.isChecked(),
            "reference_wav": self.reference_wav_field.text(),
            "prompt_wav": self.prompt_wav_field.text(),
            "prompt_text": self.prompt_text_edit.toPlainText(),
            "prompt_model": self.prompt_model_combo.currentText(),
            "clone_expanded": self.clone_section.isExpanded(),
            "auto_play": self.auto_play_check.isChecked(),
        }

    def _load_settings(self) -> None:
        payload = self.app_settings.get("tts", {})
        if not isinstance(payload, dict):
            return
        self.voice_edit.setCurrentText(str(payload.get("voice", DEFAULT_VOICE)))
        self.output_dir_field.setText(str(payload.get("output_dir", str(get_default_output_dir()))))
        self.model_path_field.setText(str(payload.get("model_path", str(get_default_model_dir()))))
        self.cfg_spin.setValue(float(payload.get("cfg", DEFAULT_CFG)))
        self.steps_spin.setValue(int(payload.get("steps", DEFAULT_STEPS)))
        self.chunk_spin.setValue(int(payload.get("chunk_max_chars", DEFAULT_CHUNK_MAX_CHARS)))
        self.silence_spin.setValue(int(payload.get("silence_ms", DEFAULT_SILENCE_MS)))
        self.load_denoiser_check.setChecked(bool(payload.get("load_denoiser", False)))
        self.optimize_check.setChecked(bool(payload.get("optimize", False)))
        self.reuse_first_chunk_check.setChecked(bool(payload.get("reuse_first_chunk_as_reference", False)))
        self.reference_wav_field.setText(str(payload.get("reference_wav", "")))
        self.prompt_wav_field.setText(str(payload.get("prompt_wav", "")))
        self.prompt_text_edit.setPlainText(str(payload.get("prompt_text", "")))
        self.prompt_model_combo.setCurrentText(str(payload.get("prompt_model", DEFAULT_ASR_MODEL)))
        self.clone_section.setExpanded(bool(payload.get("clone_expanded", False)))
        self.auto_play_check.setChecked(bool(payload.get("auto_play", True)))

    def save_settings(self) -> None:
        self.app_settings["tts"] = self._settings_payload()

    def _refresh_environment(self) -> None:
        status = check_environment(self.model_path_field.text() or str(get_default_model_dir()))
        asr_status = check_asr_environment(self.prompt_model_combo.currentText())
        lines = [f"TTS Python: {status.python_version}"]
        if status.cuda_available and status.gpu_name:
            lines.append(f"CUDA: 可用，GPU 为 {status.gpu_name}（{status.total_vram_gb:.1f} GB）")
        else:
            lines.append("CUDA: 未检测到，可生成但速度会明显变慢。")
        lines.append("TTS 依赖检查通过。" if not status.missing_packages else "缺少 TTS 依赖：" + ", ".join(status.missing_packages))
        lines.append(
            f"Prompt 转写依赖检查通过（{self.prompt_model_combo.currentText()}）。"
            if not asr_status.missing_packages
            else f"Prompt 转写缺少依赖（{self.prompt_model_combo.currentText()}）："
            + ", ".join(asr_status.missing_packages)
        )
        lines.append(
            "模型目录："
            + (
                f"{status.default_model_path}（已找到）"
                if status.model_exists
                else f"{status.default_model_path}（未找到，可手动选择）"
            )
        )
        self.environment_label.setStyleSheet("color: #0b7a0b;" if not status.missing_packages and not asr_status.missing_packages else "color: #b42318;")
        self.environment_label.setText("\n".join(lines))

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _prepare_player(self, output_path: Path | None) -> None:
        if output_path and output_path.exists():
            self.media_player.setSource(QUrl.fromLocalFile(str(output_path)))
            enabled = True
        else:
            enabled = False
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.open_file_button.setEnabled(enabled)
        self.open_dir_button.setEnabled(enabled)

    def _validate_form(self) -> tuple[bool, str]:
        if not self.text_edit.toPlainText().strip():
            return False, "请输入要配音的文案。"
        output_dir = Path(self.output_dir_field.text()).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            return False, "输出目录不存在，请先选择一个有效目录。"
        model_dir = Path(self.model_path_field.text()).expanduser()
        if not model_dir.exists():
            return False, "模型目录不存在，请确认 VoxCPM2 所在位置。"
        reference_wav = self.reference_wav_field.text()
        if reference_wav and not Path(reference_wav).expanduser().exists():
            return False, "参考音频不存在，请重新选择。"
        prompt_wav = self.prompt_wav_field.text()
        prompt_text = self.prompt_text_edit.toPlainText().strip()
        if prompt_wav and not Path(prompt_wav).expanduser().exists():
            return False, "Prompt 音频不存在，请重新选择。"
        if bool(prompt_wav) != bool(prompt_text):
            return False, "Prompt 音频和 Prompt 文本需要同时填写。"
        return True, ""

    def _set_prompt_transcribe_busy(self, busy: bool) -> None:
        self.prompt_transcribe_button.setEnabled(not busy)
        self.prompt_model_combo.setEnabled(not busy)

    def _set_form_busy(self, busy: bool) -> None:
        widgets = [
            self.text_edit,
            self.voice_edit,
            self.output_dir_field,
            self.model_path_field,
            self.cfg_spin,
            self.steps_spin,
            self.chunk_spin,
            self.silence_spin,
            self.load_denoiser_check,
            self.optimize_check,
            self.auto_play_check,
            self.reference_wav_field,
            self.prompt_wav_field,
            self.prompt_text_edit,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)
        self.enqueue_button.setEnabled(not busy)
        self._set_prompt_transcribe_busy(busy or self._prompt_task_id is not None)

    def on_prompt_transcribe_clicked(self) -> None:
        prompt_wav = self.prompt_wav_field.text().strip()
        if not prompt_wav:
            QMessageBox.warning(self, "参数错误", "请先选择 Prompt 音频。")
            return
        if not Path(prompt_wav).expanduser().exists():
            QMessageBox.warning(self, "参数错误", "Prompt 音频不存在，请重新选择。")
            return
        status = check_asr_environment(self.prompt_model_combo.currentText())
        if status.missing_packages:
            QMessageBox.critical(
                self,
                "缺少依赖",
                f"{self.prompt_model_combo.currentText()} 缺少依赖：{', '.join(status.missing_packages)}\n"
                "请先运行：\npython -m pip install -r requirements-voxcpm2-gui.txt",
            )
            self._refresh_environment()
            return

        self._prompt_cancel_token = CancelToken()
        self._prompt_task_id = f"prompt_{uuid.uuid4().hex}"
        request = TranscribeRequest(
            audio_path=prompt_wav,
            model_kind=self.prompt_model_combo.currentText(),
            output_path=str(build_transcript_output_path(prompt_wav)),
            with_timestamps=False,
        )
        self._set_prompt_transcribe_busy(True)
        self._append_log(f"开始转写 Prompt 音频：{Path(prompt_wav).name}")
        self.prompt_worker_controller.start_job.emit(
            self._prompt_task_id or "",
            request,
            self._prompt_cancel_token or CancelToken(),
            "prompt_fill",
        )

    def on_enqueue_clicked(self) -> None:
        valid, error_message = self._validate_form()
        if not valid:
            QMessageBox.warning(self, "参数错误", error_message)
            return

        output_path = build_timestamped_output_path(self.output_dir_field.text())
        request = GenerateRequest(
            text=self.text_edit.toPlainText(),
            voice=self.voice_edit.currentText().strip(),
            reference_wav=self.reference_wav_field.text() or None,
            prompt_wav=self.prompt_wav_field.text() or None,
            prompt_text=self.prompt_text_edit.toPlainText().strip() or None,
            cfg=self.cfg_spin.value(),
            steps=self.steps_spin.value(),
            chunk_max_chars=self.chunk_spin.value(),
            silence_ms=self.silence_spin.value(),
            reuse_first_chunk_as_reference=self.reuse_first_chunk_check.isChecked(),
            load_denoiser=self.load_denoiser_check.isChecked(),
            optimize=self.optimize_check.isChecked(),
            model_path=self.model_path_field.text().strip(),
            output_path=str(output_path),
        )
        title = self.text_edit.toPlainText().strip().splitlines()[0][:20] or "配音任务"
        record = QueueTaskRecord(
            task_id=uuid.uuid4().hex,
            page_kind="tts",
            status="queued",
            title=title,
            input_path=title,
            output_path=str(output_path),
        )
        self.queue_controller.add_job(QueueJob(record=record, payload=request, cancel_token=CancelToken(), kind="tts"))
        self.output_file_edit.setText(str(output_path))
        self._append_log(f"已加入队列：{title}")

    def _on_queue_progress(self, record: QueueTaskRecord, progress_item: TaskProgress) -> None:
        if record.page_kind != "tts":
            return
        self.current_progress_task_id = record.task_id
        self.stage_label.setText(progress_item.message)
        if progress_item.percent is not None:
            self.progress_bar.setValue(progress_item.percent)
        self._append_log(f"[{record.title}] {progress_item.message}")

    def _on_queue_result(self, record: QueueTaskRecord, result: object) -> None:
        if record.page_kind != "tts" or not isinstance(result, GenerateResult):
            return
        self.current_output_path = Path(result.output_path)
        self.output_file_edit.setText(result.output_path)
        self.progress_bar.setValue(100)
        self.stage_label.setText(f"生成完成，共 {result.chunks_count} 段，时长约 {result.duration_seconds:.1f} 秒。")
        self._append_log(f"[{record.title}] {self.stage_label.text()}")
        self._prepare_player(self.current_output_path)
        if self.auto_play_check.isChecked():
            self.play_output()
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_failed(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "tts":
            return
        self.stage_label.setText("生成失败")
        self._append_log(f"[{record.title}] 生成失败：{message}")
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_cancelled(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "tts":
            return
        self.stage_label.setText("任务已取消")
        self._append_log(f"[{record.title}] {message}")
        self._set_form_busy(False)

    def _on_prompt_progress(self, task_id: str, progress_item: TaskProgress) -> None:
        if task_id != self._prompt_task_id:
            return
        self._append_log(f"[Prompt 转写] {progress_item.message}")

    def _on_prompt_finished(self, task_id: str, result: object) -> None:
        if task_id != self._prompt_task_id or not isinstance(result, PromptFillResult):
            return
        self.prompt_text_edit.setPlainText(result.text)
        self._append_log(f"Prompt 转写完成，已填入文本。模型：{result.model_kind}")
        self._prompt_task_id = None
        self._prompt_cancel_token = None
        self._refresh_environment()
        self._set_form_busy(bool(self.queue_controller.current_job))

    def _on_prompt_failed(self, task_id: str, message: str) -> None:
        if task_id != self._prompt_task_id:
            return
        self._append_log(f"Prompt 转写失败：{message}")
        QMessageBox.critical(self, "Prompt 转写失败", message)
        self._prompt_task_id = None
        self._prompt_cancel_token = None
        self._refresh_environment()
        self._set_form_busy(bool(self.queue_controller.current_job))

    def _on_prompt_cancelled(self, task_id: str, message: str) -> None:
        if task_id != self._prompt_task_id:
            return
        self._append_log(message)
        self._prompt_task_id = None
        self._prompt_cancel_token = None
        self._refresh_environment()
        self._set_form_busy(bool(self.queue_controller.current_job))

    def _refresh_queue(self) -> None:
        self.queue_panel.refresh(self.queue_controller.records, self.queue_controller.current_job.record if self.queue_controller.current_job else None)
        is_running = bool(self.queue_controller.current_job)
        self._set_form_busy(is_running)

    def play_output(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            self._prepare_player(self.current_output_path)
            self.media_player.play()

    def open_output_file(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path)

    def open_output_directory(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path.parent)
        else:
            output_dir = Path(self.output_dir_field.text()).expanduser()
            if output_dir.exists():
                os.startfile(output_dir)


class SttPage(QWidget):
    def __init__(self, queue_controller: QueueController, app_settings: dict[str, object], parent: QWidget | None = None):
        super().__init__(parent)
        self.queue_controller = queue_controller
        self.app_settings = app_settings
        self.current_output_path: Path | None = None
        self._build_ui()
        self.model_combo.currentTextChanged.connect(self._refresh_environment)
        self._load_settings()
        self._refresh_environment()

        self.queue_controller.queue_changed.connect(self._refresh_queue)
        self.queue_controller.current_job_changed.connect(self._refresh_queue)
        self.queue_controller.progress.connect(self._on_queue_progress)
        self.queue_controller.result.connect(self._on_queue_result)
        self.queue_controller.failed.connect(self._on_queue_failed)
        self.queue_controller.cancelled.connect(self._on_queue_cancelled)
        self._refresh_queue()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        warning = QLabel("提示：语音生成与语音转文本会共享 GPU，同时运行时速度可能下降。")
        warning.setStyleSheet("color: #8a4b08;")
        layout.addWidget(warning)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([520, 420, 560])
        self._install_tooltips()

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        input_group = QGroupBox("音频输入")
        input_form = QFormLayout(input_group)
        self.audio_files_field = PathField("选择音频", pick_directory=False, allow_multi=True)
        self.output_dir_field = PathField("选择目录", pick_directory=True)
        self.output_dir_field.setText(str(get_default_stt_output_dir()))
        input_form.addRow("音频文件", self.audio_files_field)
        input_form.addRow("输出目录", self.output_dir_field)
        layout.addWidget(input_group)

        queue_hint = QLabel("支持一次选择多个音频文件；每个文件会作为一个转写任务加入队列。")
        queue_hint.setWordWrap(True)
        queue_hint.setStyleSheet("color: #555;")
        layout.addWidget(queue_hint)

        self.queue_panel = QueuePanel("转写队列")
        self.queue_panel.cancel_current_requested.connect(self.queue_controller.cancel_current)
        self.queue_panel.clear_finished_requested.connect(self.queue_controller.clear_finished)
        self.queue_panel.remove_requested.connect(self.queue_controller.remove_record)
        layout.addWidget(self.queue_panel, 1)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        settings_group = QGroupBox("转写设置")
        settings_form = QFormLayout(settings_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems([ASR_MODEL_OPENAI, ASR_MODEL_FASTER])
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "zh", "en"])
        self.output_format_combo = QComboBox()
        for format_key, label in TRANSCRIPT_FORMAT_LABELS.items():
            self.output_format_combo.addItem(label, format_key)
        self.smart_pause_spin = QDoubleSpinBox()
        self.smart_pause_spin.setRange(0.2, 5.0)
        self.smart_pause_spin.setSingleStep(0.1)
        self.smart_pause_spin.setDecimals(1)
        self.smart_pause_spin.setSuffix(" 秒")
        self.smart_pause_spin.setValue(SMART_SEGMENT_DEFAULT_PAUSE_SECONDS)
        settings_form.addRow("转写模型", self.model_combo)
        settings_form.addRow("语言", self.language_combo)
        settings_form.addRow("输出格式", self.output_format_combo)
        settings_form.addRow("分段停顿", self.smart_pause_spin)
        layout.addWidget(settings_group)

        help_group = QGroupBox("模型说明")
        help_layout = QVBoxLayout(help_group)
        help_label = QLabel(
            "<b>Whisper-large-v3-turbo</b>：更偏精度，适合作为默认主力模型。<br><br>"
            "<b>faster-whisper-small</b>：更偏速度，适合短音频和快速草稿。<br><br>"
            "<b>纯文本</b>：适合复制整理和喂给 AI 做总结。<br><br>"
            "<b>智能分段</b>：按停顿自动分段，更适合阅读聊天或讲述类录音。<br><br>"
            "<b>时间戳文本</b>：适合回听定位。<br><br>"
            "<b>SRT 字幕</b>：适合导入剪映、PR 等视频工具。"
        )
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setStyleSheet(
            "QLabel {background: #f7f7f8; border: 1px solid #d9d9df; border-radius: 8px; padding: 12px; line-height: 1.45;}"
        )
        help_layout.addWidget(help_label)
        layout.addWidget(help_group)
        layout.addStretch(1)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        result_group = QGroupBox("状态与结果")
        result_layout = QVBoxLayout(result_group)
        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.stage_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.full_text_edit = QTextEdit()
        self.full_text_edit.setReadOnly(True)
        self.segment_text_edit = QTextEdit()
        self.segment_text_edit.setReadOnly(True)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)
        self.open_output_button = QPushButton("打开结果文件")
        self.open_output_dir_button = QPushButton("打开输出目录")
        open_row = QHBoxLayout()
        open_row.addWidget(self.open_output_button)
        open_row.addWidget(self.open_output_dir_button)
        self.enqueue_button = QPushButton("加入队列")
        self.enqueue_button.setStyleSheet("font-size: 16px; padding: 12px 20px;")

        result_layout.addWidget(QLabel("环境状态"))
        result_layout.addWidget(self.environment_label)
        result_layout.addWidget(QLabel("当前阶段"))
        result_layout.addWidget(self.stage_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(QLabel("日志"))
        result_layout.addWidget(self.log_edit, 1)
        result_layout.addWidget(QLabel("阅读预览"))
        result_layout.addWidget(self.full_text_edit, 1)
        result_layout.addWidget(QLabel("时间轴预览"))
        result_layout.addWidget(self.segment_text_edit, 1)
        result_layout.addWidget(QLabel("输出文件"))
        result_layout.addWidget(self.output_file_edit)
        result_layout.addLayout(open_row)
        result_layout.addWidget(self.enqueue_button)
        layout.addWidget(result_group, 1)

        self.enqueue_button.clicked.connect(self.on_enqueue_clicked)
        self.open_output_button.clicked.connect(self.open_output_file)
        self.open_output_dir_button.clicked.connect(self.open_output_directory)
        self.environment_label.setToolTip("显示当前转写模型依赖、CUDA 状态和本地模型目录检查结果。")
        self.stage_label.setToolTip("显示当前转写任务所处阶段，例如检查环境、加载模型、转写中和写文件。")
        self.progress_bar.setToolTip("显示当前转写任务的大致进度。")
        self.log_edit.setToolTip("显示本页转写日志，包括入队、加载模型、分段进度、失败和完成信息。")
        self.full_text_edit.setToolTip("展示适合阅读的预览文本，优先按智能分段方式显示。")
        self.segment_text_edit.setToolTip("展示带时间戳的逐段结果，方便回听定位、校对和剪辑。")
        self.output_file_edit.setToolTip("这里显示当前最新转写结果文件的完整路径。")
        return panel

    def _install_tooltips(self) -> None:
        self.audio_files_field.setToolTip("选择一个或多个待转写音频文件。多个文件会分别作为独立任务加入转写队列。")
        self.output_dir_field.setToolTip("转写结果文本的保存目录。默认会自动生成带时间戳的文件名。")
        self.model_combo.setToolTip("选择语音转文字模型。Whisper-large-v3-turbo 更偏精度，faster-whisper-small 更偏速度。")
        self.language_combo.setToolTip("指定音频语言。选 auto 会自动识别，中文音频也可以直接选 zh。")
        self.output_format_combo.setToolTip("选择转写结果的导出格式。聊天记录推荐用智能分段，视频字幕推荐用 SRT。")
        self.smart_pause_spin.setToolTip("智能分段模式下，停顿超过这个时长就会另起一段。聊天音频通常 0.6 到 1.2 秒比较自然。")
        self.open_output_button.setToolTip("打开当前选中的最新转写结果文件。")
        self.open_output_dir_button.setToolTip("打开转写结果所在目录。")
        self.enqueue_button.setToolTip("把当前选择的音频文件加入转写队列。队列会逐个文件顺序执行。")

    def _settings_payload(self) -> dict[str, object]:
        return {
            "audio_files": self.audio_files_field.text(),
            "output_dir": self.output_dir_field.text(),
            "model": self.model_combo.currentText(),
            "language": self.language_combo.currentText(),
            "output_format": self.output_format_combo.currentData(),
            "smart_pause_seconds": self.smart_pause_spin.value(),
        }

    def _load_settings(self) -> None:
        payload = self.app_settings.get("stt", {})
        if not isinstance(payload, dict):
            return
        self.audio_files_field.setText(str(payload.get("audio_files", "")))
        self.output_dir_field.setText(str(payload.get("output_dir", str(get_default_stt_output_dir()))))
        self.model_combo.setCurrentText(str(payload.get("model", DEFAULT_ASR_MODEL)))
        self.language_combo.setCurrentText(str(payload.get("language", "auto")))
        format_value = payload.get("output_format")
        if format_value is None:
            format_value = LEGACY_TRANSCRIPT_FORMAT if bool(payload.get("with_timestamps", True)) else TRANSCRIPT_FORMAT_PLAIN
        index = self.output_format_combo.findData(str(format_value))
        self.output_format_combo.setCurrentIndex(index if index >= 0 else self.output_format_combo.findData(DEFAULT_TRANSCRIPT_FORMAT))
        self.smart_pause_spin.setValue(float(payload.get("smart_pause_seconds", SMART_SEGMENT_DEFAULT_PAUSE_SECONDS)))

    def save_settings(self) -> None:
        self.app_settings["stt"] = self._settings_payload()

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _refresh_environment(self) -> None:
        status = check_asr_environment(self.model_combo.currentText())
        lines = [f"Python: {status.python_version}"]
        if status.cuda_available and status.gpu_name:
            lines.append(f"CUDA: 可用，GPU 为 {status.gpu_name}（{status.total_vram_gb:.1f} GB）")
        else:
            lines.append("CUDA: 未检测到，转写速度会明显变慢。")
        lines.append(
            f"依赖检查通过（{self.model_combo.currentText()}）。"
            if not status.missing_packages
            else f"缺少依赖（{self.model_combo.currentText()}）："
            + ", ".join(status.missing_packages)
        )
        for model_name, model_path in status.model_paths.items():
            lines.append(
                f"{model_name}：{model_path}（{'已找到' if status.model_exists.get(model_name) else '未找到'}）"
            )
        self.environment_label.setStyleSheet("color: #0b7a0b;" if not status.missing_packages else "color: #b42318;")
        self.environment_label.setText("\n".join(lines))

    def _set_form_busy(self, busy: bool) -> None:
        widgets = [
            self.audio_files_field,
            self.output_dir_field,
            self.model_combo,
            self.language_combo,
            self.output_format_combo,
            self.smart_pause_spin,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)
        self.enqueue_button.setEnabled(not busy)

    def _refresh_queue(self) -> None:
        self.queue_panel.refresh(self.queue_controller.records, self.queue_controller.current_job.record if self.queue_controller.current_job else None)
        self._set_form_busy(bool(self.queue_controller.current_job))

    def _validate_form(self) -> tuple[bool, str, list[str]]:
        audio_entries = [item.strip() for item in self.audio_files_field.text().split(";") if item.strip()]
        if not audio_entries:
            return False, "请至少选择一个音频文件。", []
        invalid = [item for item in audio_entries if not Path(item).expanduser().exists()]
        if invalid:
            return False, f"以下音频文件不存在：{invalid[0]}", []
        output_dir = Path(self.output_dir_field.text()).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            return False, "输出目录不存在，请先选择一个有效目录。", []
        model_dir = get_default_openai_whisper_dir() if self.model_combo.currentText() == ASR_MODEL_OPENAI else get_default_faster_whisper_dir()
        if not model_dir.exists():
            return False, f"转写模型目录不存在：{model_dir}", []
        return True, "", audio_entries

    def on_enqueue_clicked(self) -> None:
        valid, error_message, audio_entries = self._validate_form()
        if not valid:
            QMessageBox.warning(self, "参数错误", error_message)
            return
        status = check_asr_environment(self.model_combo.currentText())
        if status.missing_packages:
            QMessageBox.critical(
                self,
                "缺少依赖",
                f"{self.model_combo.currentText()} 缺少依赖：{', '.join(status.missing_packages)}\n"
                "请先运行：\npython -m pip install -r requirements-voxcpm2-gui.txt",
            )
            self._refresh_environment()
            return
        for audio_path in audio_entries:
            output_format = str(self.output_format_combo.currentData() or DEFAULT_TRANSCRIPT_FORMAT)
            output_path = build_transcript_output_path(
                audio_path,
                self.output_dir_field.text(),
                output_format=output_format,
                with_timestamps=output_format == TRANSCRIPT_FORMAT_TIMESTAMPS,
            )
            request = TranscribeRequest(
                audio_path=audio_path,
                model_kind=self.model_combo.currentText(),
                output_path=str(output_path),
                language=self.language_combo.currentText(),
                with_timestamps=output_format == TRANSCRIPT_FORMAT_TIMESTAMPS,
                output_format=output_format,
                smart_segment_pause_seconds=self.smart_pause_spin.value(),
            )
            title = Path(audio_path).name
            record = QueueTaskRecord(
                task_id=uuid.uuid4().hex,
                page_kind="stt",
                status="queued",
                title=title,
                input_path=audio_path,
                output_path=str(output_path),
            )
            self.queue_controller.add_job(QueueJob(record=record, payload=request, cancel_token=CancelToken(), kind="stt"))
            self._append_log(f"已加入队列：{title}，格式：{get_transcript_format_label(output_format)}")

    def _on_queue_progress(self, record: QueueTaskRecord, progress_item: TaskProgress) -> None:
        if record.page_kind != "stt":
            return
        self.stage_label.setText(progress_item.message)
        if progress_item.percent is not None:
            self.progress_bar.setValue(progress_item.percent)
        self._append_log(f"[{record.title}] {progress_item.message}")

    def _on_queue_result(self, record: QueueTaskRecord, result: object) -> None:
        if record.page_kind != "stt" or not isinstance(result, TranscribeResult):
            return
        self.current_output_path = Path(result.output_path)
        self.output_file_edit.setText(result.output_path)
        self.progress_bar.setValue(100)
        self.stage_label.setText(f"转写完成，共 {len(result.segments)} 段。")
        reading_preview = render_transcript_text(
            result.text,
            result.segments,
            output_format=TRANSCRIPT_FORMAT_SMART,
            with_timestamps=False,
            smart_segment_pause_seconds=self.smart_pause_spin.value(),
        ).strip()
        timeline_preview = render_transcript_text(
            result.text,
            result.segments,
            output_format=TRANSCRIPT_FORMAT_TIMESTAMPS,
            with_timestamps=True,
            smart_segment_pause_seconds=self.smart_pause_spin.value(),
        ).strip()
        self.full_text_edit.setPlainText(reading_preview or result.text)
        self.segment_text_edit.setPlainText(timeline_preview)
        current_format = str(self.output_format_combo.currentData() or DEFAULT_TRANSCRIPT_FORMAT)
        self._append_log(f"[{record.title}] 转写完成，已导出为{get_transcript_format_label(current_format)}。")
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_failed(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "stt":
            return
        self.stage_label.setText("转写失败")
        self._append_log(f"[{record.title}] 转写失败：{message}")
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_cancelled(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "stt":
            return
        self.stage_label.setText("任务已取消")
        self._append_log(f"[{record.title}] {message}")
        self._set_form_busy(False)

    def open_output_file(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path)

    def open_output_directory(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path.parent)
        else:
            output_dir = Path(self.output_dir_field.text()).expanduser()
            if output_dir.exists():
                os.startfile(output_dir)


class AudioFxPage(QWidget):
    def __init__(self, queue_controller: QueueController, app_settings: dict[str, object], parent: QWidget | None = None):
        super().__init__(parent)
        self.queue_controller = queue_controller
        self.app_settings = app_settings
        self.current_output_path: Path | None = None

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)

        self._build_ui()
        self._load_settings()
        self._refresh_environment()
        self._prepare_player(None)

        self.model_path_field.line_edit.textChanged.connect(self._refresh_environment)
        self.use_gpu_check.stateChanged.connect(self._refresh_environment)
        self.cpu_offload_check.stateChanged.connect(self._refresh_environment)
        self.result_list.currentItemChanged.connect(self._on_result_selected)

        self.queue_controller.queue_changed.connect(self._refresh_queue)
        self.queue_controller.current_job_changed.connect(self._refresh_queue)
        self.queue_controller.progress.connect(self._on_queue_progress)
        self.queue_controller.result.connect(self._on_queue_result)
        self.queue_controller.failed.connect(self._on_queue_failed)
        self.queue_controller.cancelled.connect(self._on_queue_cancelled)
        self._refresh_queue()

    def shutdown(self) -> None:
        self.media_player.stop()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        warning = QLabel("提示：音效生成会占用较多显存。批量生成时建议先用 3 到 5 秒短音频试参，再放大时长。")
        warning.setStyleSheet("color: #8a4b08;")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([620, 760])
        self._install_tooltips()

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        prompt_group = QGroupBox("批量提示词")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "一行一个音效提示词，建议使用英文更稳定。\n"
            "gentle rain on a window, soft ambience\n"
            "small brass bell ringing once in a quiet room\n"
            "footsteps on wet stone, distant cave reverb"
        )
        prompt_layout.addWidget(self.prompt_edit)
        layout.addWidget(prompt_group, 1)

        settings_group = QGroupBox("生成参数")
        settings_form = QFormLayout(settings_group)
        self.model_path_field = PathField("选择模型", pick_directory=True)
        self.model_path_field.setText(str(get_default_audiofx_model_dir()))
        self.output_dir_field = PathField("选择目录", pick_directory=True)
        self.output_dir_field.setText(str(get_default_audiofx_output_dir()))

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 60.0)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" 秒")
        self.duration_spin.setValue(DEFAULT_AUDIOFX_DURATION_SECONDS)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(DEFAULT_AUDIOFX_STEPS)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.1, 20.0)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setValue(DEFAULT_AUDIOFX_GUIDANCE_SCALE)

        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("留空表示随机")

        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(1, 20)
        self.variants_spin.setValue(DEFAULT_AUDIOFX_VARIANTS)

        self.use_gpu_check = QCheckBox("使用 GPU（CUDA）")
        self.use_gpu_check.setChecked(True)
        self.cpu_offload_check = QCheckBox("启用 CPU offload（显存紧张时使用）")
        self.auto_play_check = QCheckBox("生成完成后自动播放")
        self.auto_play_check.setChecked(True)

        settings_form.addRow("模型目录", self.model_path_field)
        settings_form.addRow("输出目录", self.output_dir_field)
        settings_form.addRow("音频时长", self.duration_spin)
        settings_form.addRow("推理步数", self.steps_spin)
        settings_form.addRow("Guidance Scale", self.guidance_spin)
        settings_form.addRow("随机种子", self.seed_edit)
        settings_form.addRow("每条生成数量", self.variants_spin)
        settings_form.addRow("", self.use_gpu_check)
        settings_form.addRow("", self.cpu_offload_check)
        settings_form.addRow("", self.auto_play_check)
        layout.addWidget(settings_group)

        button_row = QHBoxLayout()
        self.enqueue_button = QPushButton("加入队列")
        self.preview_button = QPushButton("生成试听")
        self.open_dir_button = QPushButton("打开输出目录")
        self.enqueue_button.setStyleSheet("font-size: 16px; padding: 12px 20px;")
        self.preview_button.setStyleSheet("font-size: 16px; padding: 12px 20px;")
        button_row.addWidget(self.enqueue_button)
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.open_dir_button)
        layout.addLayout(button_row)

        self.enqueue_button.clicked.connect(self.on_enqueue_clicked)
        self.preview_button.clicked.connect(self.on_preview_clicked)
        self.open_dir_button.clicked.connect(self.open_output_directory)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        result_group = QGroupBox("状态与结果")
        result_layout = QVBoxLayout(result_group)
        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        self.stage_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.result_list = QListWidget()
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)

        play_row = QHBoxLayout()
        self.play_button = QPushButton("播放选中")
        self.stop_button = QPushButton("停止")
        self.open_file_button = QPushButton("打开文件")
        self.open_result_dir_button = QPushButton("打开目录")
        play_row.addWidget(self.play_button)
        play_row.addWidget(self.stop_button)
        play_row.addWidget(self.open_file_button)
        play_row.addWidget(self.open_result_dir_button)

        self.queue_panel = QueuePanel("音效生成队列")
        self.queue_panel.cancel_button.setText("取消当前任务")
        self.queue_panel.clear_button.setText("清理已完成")
        self.queue_panel.cancel_current_requested.connect(self.queue_controller.cancel_current)
        self.queue_panel.clear_finished_requested.connect(self.queue_controller.clear_finished)
        self.queue_panel.remove_requested.connect(self.queue_controller.remove_record)

        result_layout.addWidget(QLabel("环境状态"))
        result_layout.addWidget(self.environment_label)
        result_layout.addWidget(QLabel("当前阶段"))
        result_layout.addWidget(self.stage_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(QLabel("日志"))
        result_layout.addWidget(self.log_edit, 1)
        result_layout.addWidget(QLabel("结果列表"))
        result_layout.addWidget(self.result_list, 1)
        result_layout.addWidget(QLabel("当前音频"))
        result_layout.addWidget(self.output_file_edit)
        result_layout.addLayout(play_row)
        result_layout.addWidget(self.queue_panel, 1)
        layout.addWidget(result_group, 1)

        self.play_button.clicked.connect(self.play_output)
        self.stop_button.clicked.connect(self.media_player.stop)
        self.open_file_button.clicked.connect(self.open_output_file)
        self.open_result_dir_button.clicked.connect(self.open_output_directory)
        return panel

    def _install_tooltips(self) -> None:
        self.prompt_edit.setToolTip("一行一个音效提示词，空行会自动忽略。AudioLDM2 对英文提示词通常更稳定。")
        self.model_path_field.setToolTip("本地 AudioLDM2 模型目录。默认读取项目上级 model\\AudioLDM2，不会运行时联网。")
        self.output_dir_field.setToolTip("生成 WAV 素材的保存目录。默认是当前应用 outputs\\sfx。")
        self.duration_spin.setToolTip("单条音效目标时长。首次试参建议 3 到 5 秒，满意后再增加。")
        self.steps_spin.setToolTip("推理步数。越高越慢，细节可能更稳；默认 50。")
        self.guidance_spin.setToolTip("提示词遵从强度。默认 3.5，太高可能声音更僵硬。")
        self.seed_edit.setToolTip("固定种子可以复现相近结果；留空时每个任务随机。填写种子后，批量任务会自动递增避免完全相同。")
        self.variants_spin.setToolTip("每条提示词生成几个版本。总任务数 = 提示词行数 x 每条生成数量。")
        self.use_gpu_check.setToolTip("使用 CUDA 显卡生成。速度更快，但需要足够显存。")
        self.cpu_offload_check.setToolTip("把部分模型权重在 CPU/GPU 间调度，可降低显存压力，但会变慢。")
        self.auto_play_check.setToolTip("生成完成后自动播放最新结果。批量生成时可关闭。")
        self.enqueue_button.setToolTip("把所有提示词按批量参数展开后加入音效生成队列。")
        self.preview_button.setToolTip("只使用第一条有效提示词生成 1 个试听版本。")
        self.open_dir_button.setToolTip("打开当前设置的输出目录。")
        self.environment_label.setToolTip("显示 Python、CUDA、依赖和本地模型目录状态。")
        self.result_list.setToolTip("展示已完成的音效结果。选中一条后可以直接播放或打开文件。")

    def _settings_payload(self) -> dict[str, object]:
        return {
            "prompts": self.prompt_edit.toPlainText(),
            "model_path": self.model_path_field.text(),
            "output_dir": self.output_dir_field.text(),
            "duration_seconds": self.duration_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance_scale": self.guidance_spin.value(),
            "seed": self.seed_edit.text().strip(),
            "variants": self.variants_spin.value(),
            "use_gpu": self.use_gpu_check.isChecked(),
            "cpu_offload": self.cpu_offload_check.isChecked(),
            "auto_play": self.auto_play_check.isChecked(),
        }

    def _load_settings(self) -> None:
        payload = self.app_settings.get("audiofx", {})
        if not isinstance(payload, dict):
            return
        self.prompt_edit.setPlainText(str(payload.get("prompts", "")))
        self.model_path_field.setText(str(payload.get("model_path", str(get_default_audiofx_model_dir()))))
        self.output_dir_field.setText(str(payload.get("output_dir", str(get_default_audiofx_output_dir()))))
        self.duration_spin.setValue(float(payload.get("duration_seconds", DEFAULT_AUDIOFX_DURATION_SECONDS)))
        self.steps_spin.setValue(int(payload.get("steps", DEFAULT_AUDIOFX_STEPS)))
        self.guidance_spin.setValue(float(payload.get("guidance_scale", DEFAULT_AUDIOFX_GUIDANCE_SCALE)))
        self.seed_edit.setText(str(payload.get("seed", "")))
        self.variants_spin.setValue(int(payload.get("variants", DEFAULT_AUDIOFX_VARIANTS)))
        self.use_gpu_check.setChecked(bool(payload.get("use_gpu", True)))
        self.cpu_offload_check.setChecked(bool(payload.get("cpu_offload", False)))
        self.auto_play_check.setChecked(bool(payload.get("auto_play", True)))

    def save_settings(self) -> None:
        self.app_settings["audiofx"] = self._settings_payload()

    def _refresh_environment(self) -> None:
        status = check_audiofx_environment(self.model_path_field.text() or str(get_default_audiofx_model_dir()))
        lines = [f"AudioLDM2 Python: {status.python_version}"]
        if status.cuda_available and status.gpu_name:
            vram_text = f"{status.total_vram_gb:.1f} GB" if status.total_vram_gb else "未知显存"
            lines.append(f"CUDA: 可用，GPU 为 {status.gpu_name}（{vram_text}）")
        else:
            lines.append("CUDA: 未检测到；如果勾选使用 GPU，生成会被阻止。")
        lines.append("AudioLDM2 依赖检查通过。" if not status.missing_packages else "缺少 AudioLDM2 依赖：" + ", ".join(status.missing_packages))
        lines.append(
            "模型目录："
            + (
                f"{status.default_model_path}（已找到）"
                if status.model_exists and status.model_index_exists
                else f"{status.default_model_path}（未找到或缺少 model_index.json）"
            )
        )
        if self.use_gpu_check.isChecked() and not status.cuda_available:
            lines.append("当前勾选了使用 GPU，但 CUDA 不可用。")

        ready = not status.missing_packages and status.model_exists and status.model_index_exists
        if self.use_gpu_check.isChecked() and not status.cuda_available:
            ready = False
        self.environment_label.setStyleSheet("color: #0b7a0b;" if ready else "color: #b42318;")
        self.environment_label.setText("\n".join(lines))

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _parse_prompts(self) -> list[str]:
        return [line.strip() for line in self.prompt_edit.toPlainText().splitlines() if line.strip()]

    def _parse_seed(self) -> tuple[bool, str, int | None]:
        raw_seed = self.seed_edit.text().strip()
        if not raw_seed:
            return True, "", None
        try:
            seed = int(raw_seed)
        except ValueError:
            return False, "随机种子必须是整数，或留空表示随机。", None
        if seed < 0:
            return False, "随机种子不能为负数。", None
        return True, "", seed

    def _validate_form(self) -> tuple[bool, str, list[str], int | None]:
        prompts = self._parse_prompts()
        if not prompts:
            return False, "请至少输入一条音效提示词。", [], None

        output_dir = Path(self.output_dir_field.text()).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"输出目录无法创建或访问：{exc}", [], None

        model_dir = Path(self.model_path_field.text()).expanduser()
        if not model_dir.exists():
            return False, "模型目录不存在，请确认 AudioLDM2 所在位置。", [], None
        if not (model_dir / "model_index.json").exists():
            return False, "模型目录缺少 model_index.json，请确认选择的是 AudioLDM2 根目录。", [], None

        valid_seed, seed_error, seed = self._parse_seed()
        if not valid_seed:
            return False, seed_error, [], None

        status = check_audiofx_environment(str(model_dir))
        if status.missing_packages:
            return False, "缺少 AudioLDM2 依赖：" + ", ".join(status.missing_packages), [], None
        if self.use_gpu_check.isChecked() and not status.cuda_available:
            return False, "当前 CUDA 不可用，请取消“使用 GPU”后使用 CPU，或检查 CUDA 版 PyTorch。", [], None
        return True, "", prompts, seed

    def _set_form_busy(self, busy: bool) -> None:
        widgets = [
            self.prompt_edit,
            self.model_path_field,
            self.output_dir_field,
            self.duration_spin,
            self.steps_spin,
            self.guidance_spin,
            self.seed_edit,
            self.variants_spin,
            self.use_gpu_check,
            self.cpu_offload_check,
            self.auto_play_check,
            self.enqueue_button,
            self.preview_button,
        ]
        for widget in widgets:
            widget.setEnabled(not busy)

    def _enqueue_prompts(self, prompts: list[str], base_seed: int | None, variants: int, preview: bool = False) -> None:
        sequence_index = 0
        for prompt in prompts:
            for _variant_index in range(variants):
                sequence_index += 1
                seed = base_seed + sequence_index - 1 if base_seed is not None else None
                output_path = build_audiofx_output_path(
                    self.output_dir_field.text(),
                    prompt,
                    sequence_index=sequence_index,
                )
                request = AudioFxRequest(
                    prompt=prompt,
                    model_path=self.model_path_field.text().strip(),
                    output_path=str(output_path),
                    duration_seconds=self.duration_spin.value(),
                    steps=self.steps_spin.value(),
                    guidance_scale=self.guidance_spin.value(),
                    seed=seed,
                    use_gpu=self.use_gpu_check.isChecked(),
                    cpu_offload=self.cpu_offload_check.isChecked(),
                    sequence_index=sequence_index,
                )
                title_prefix = "试听" if preview else f"{sequence_index:03d}"
                title = f"{title_prefix} {prompt[:24]}"
                record = QueueTaskRecord(
                    task_id=uuid.uuid4().hex,
                    page_kind="audiofx",
                    status="queued",
                    title=title,
                    input_path=prompt,
                    output_path=str(output_path),
                )
                self.queue_controller.add_job(QueueJob(record=record, payload=request, cancel_token=CancelToken(), kind="audiofx"))
                self._append_log(f"已加入队列：{title}")

    def on_enqueue_clicked(self) -> None:
        valid, error_message, prompts, seed = self._validate_form()
        if not valid:
            QMessageBox.warning(self, "参数错误", error_message)
            self._refresh_environment()
            return
        self._enqueue_prompts(prompts, seed, self.variants_spin.value(), preview=False)

    def on_preview_clicked(self) -> None:
        valid, error_message, prompts, seed = self._validate_form()
        if not valid:
            QMessageBox.warning(self, "参数错误", error_message)
            self._refresh_environment()
            return
        self._enqueue_prompts(prompts[:1], seed, 1, preview=True)

    def _refresh_queue(self) -> None:
        self.queue_panel.refresh(self.queue_controller.records, self.queue_controller.current_job.record if self.queue_controller.current_job else None)
        self._set_form_busy(bool(self.queue_controller.current_job))

    def _on_queue_progress(self, record: QueueTaskRecord, progress_item: TaskProgress) -> None:
        if record.page_kind != "audiofx":
            return
        self.stage_label.setText(progress_item.message)
        if progress_item.percent is not None:
            self.progress_bar.setValue(progress_item.percent)
        self._append_log(f"[{record.title}] {progress_item.message}")

    def _on_queue_result(self, record: QueueTaskRecord, result: object) -> None:
        if record.page_kind != "audiofx" or not isinstance(result, AudioFxResult):
            return
        self.current_output_path = Path(result.output_path)
        self.output_file_edit.setText(result.output_path)
        self.progress_bar.setValue(100)
        self.stage_label.setText(f"音效生成完成，时长约 {result.duration_seconds:.1f} 秒，采样率 {result.sample_rate} Hz。")
        item_text = (
            f"[完成] {result.prompt} | {result.duration_seconds:.1f}s | "
            f"{result.sample_rate} Hz | seed={result.seed} | {Path(result.output_path).name}"
        )
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, result.output_path)
        self.result_list.addItem(item)
        self.result_list.setCurrentItem(item)
        self._append_log(f"[{record.title}] 已保存：{result.output_path}")
        self._prepare_player(self.current_output_path)
        if self.auto_play_check.isChecked():
            self.play_output()
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_failed(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "audiofx":
            return
        self.stage_label.setText("音效生成失败")
        self._append_log(f"[{record.title}] 生成失败：{message}")
        self._refresh_environment()
        self._set_form_busy(False)

    def _on_queue_cancelled(self, record: QueueTaskRecord, message: str) -> None:
        if record.page_kind != "audiofx":
            return
        self.stage_label.setText("任务已取消")
        self._append_log(f"[{record.title}] {message}")
        self._set_form_busy(False)

    def _prepare_player(self, output_path: Path | None) -> None:
        if output_path and output_path.exists():
            self.media_player.setSource(QUrl.fromLocalFile(str(output_path)))
            enabled = True
        else:
            enabled = False
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.open_file_button.setEnabled(enabled)
        self.open_result_dir_button.setEnabled(enabled)

    def _on_result_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        output_path = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(output_path, str):
            return
        self.current_output_path = Path(output_path)
        self.output_file_edit.setText(output_path)
        self._prepare_player(self.current_output_path)

    def play_output(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            self._prepare_player(self.current_output_path)
            self.media_player.play()

    def open_output_file(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path)

    def open_output_directory(self) -> None:
        if self.current_output_path and self.current_output_path.exists():
            os.startfile(self.current_output_path.parent)
            return
        output_dir = Path(self.output_dir_field.text()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(output_dir)


class VoiceWorkbenchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("语音生成与转写工作台")
        self.resize(1680, 920)
        self.settings_path = get_settings_path()
        self._shutdown_done = False
        self.app_settings = self._load_settings_payload()

        self.tts_queue_controller = QueueController(self)
        self.stt_queue_controller = QueueController(self)
        self.audiofx_queue_controller = QueueController(self)

        self._build_ui()
        self._restore_window_placement()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        title = QLabel("语音生成与转写工作台")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        subtitle = QLabel("本地模型、本地生成、本地转写；适合中文讲解、旁白和音色克隆测试。")
        subtitle.setStyleSheet("color: #555;")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        self.tab_widget = QTabWidget()
        self.tts_page = TtsPage(self.tts_queue_controller, self.app_settings, self)
        self.stt_page = SttPage(self.stt_queue_controller, self.app_settings, self)
        self.audiofx_page = AudioFxPage(self.audiofx_queue_controller, self.app_settings, self)
        self.tab_widget.addTab(self.tts_page, "语音生成")
        self.tab_widget.addTab(self.stt_page, "语音转文本")
        self.tab_widget.addTab(self.audiofx_page, "音效素材库")
        root_layout.addWidget(self.tab_widget, 1)

        app_settings = self.app_settings.get("app", {})
        if isinstance(app_settings, dict):
            self.tab_widget.setCurrentIndex(int(app_settings.get("last_tab", 0)))

    def _load_settings_payload(self) -> dict[str, object]:
        if not self.settings_path.exists():
            return {"app": {}, "tts": {}, "stt": {}, "audiofx": {}}
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {"app": {}, "tts": {}, "stt": {}, "audiofx": {}}

        if "app" in payload or "tts" in payload or "stt" in payload or "audiofx" in payload:
            payload.setdefault("app", {})
            payload.setdefault("tts", {})
            payload.setdefault("stt", {})
            payload.setdefault("audiofx", {})
            return payload

        migrated = {
            "app": {"last_tab": 0},
            "tts": payload,
            "stt": {},
            "audiofx": {},
        }
        return migrated

    def _restore_window_placement(self) -> None:
        app_settings = self.app_settings.get("app", {})
        if not isinstance(app_settings, dict):
            self._center_on_screen()
            return

        window_rect = app_settings.get("window_rect")
        if not isinstance(window_rect, dict):
            self._center_on_screen()
            return

        try:
            width = int(window_rect.get("width", self.width()))
            height = int(window_rect.get("height", self.height()))
            x = int(window_rect["x"])
            y = int(window_rect["y"])
        except (KeyError, TypeError, ValueError):
            self._center_on_screen()
            return

        self.resize(width, height)
        self.move(x, y)

    def _center_on_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def save_settings(self) -> None:
        self.app_settings.setdefault("app", {})
        if isinstance(self.app_settings["app"], dict):
            self.app_settings["app"]["last_tab"] = self.tab_widget.currentIndex()
            self.app_settings["app"]["window_rect"] = {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
            }
        self.tts_page.save_settings()
        self.stt_page.save_settings()
        self.audiofx_page.save_settings()
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(self.app_settings, ensure_ascii=False, indent=2), encoding="utf-8")

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.save_settings()
        self.tts_page.shutdown()
        self.audiofx_page.shutdown()
        self.tts_queue_controller.shutdown()
        self.stt_queue_controller.shutdown()
        self.audiofx_queue_controller.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def run_tts_self_test() -> None:
    output_path = build_timestamped_output_path(get_default_output_dir())
    result = generate_tts(
        GenerateRequest(
            text="这是一次打包版语音生成自检。",
            output_path=str(output_path),
        )
    )
    print(f"TTS 自检通过：{result.output_path}")


def run_stt_self_test() -> None:
    sample_audio = get_default_openai_whisper_dir() / "example" / "asr_example.wav"
    result = transcribe_audio(
        TranscribeRequest(
            audio_path=str(sample_audio),
            model_kind=ASR_MODEL_OPENAI,
            output_path=str(build_transcript_output_path(sample_audio, get_default_stt_output_dir())),
            with_timestamps=True,
        )
    )
    print(f"STT 自检通过：{result.output_path}")


def run_audiofx_self_test() -> None:
    from audiofx_service import (
        AudioFxRequest,
        generate_audiofx,
        get_default_audiofx_model_dir,
        get_default_audiofx_output_dir,
    )

    output_path = get_default_audiofx_output_dir() / "selftest_audiofx.wav"
    result = generate_audiofx(
        AudioFxRequest(
            prompt="gentle rain on a window, soft ambience",
            model_path=str(get_default_audiofx_model_dir()),
            output_path=str(output_path),
            duration_seconds=2.0,
            steps=4,
            guidance_scale=3.5,
            seed=20260425,
            use_gpu=True,
            cpu_offload=False,
        )
    )
    print(f"AudioFX 自检通过：{result.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="语音生成与转写桌面 UI")
    parser.add_argument("--smoke-test", action="store_true", help="创建窗口并立即退出，用于验证 GUI 依赖是否完整。")
    parser.add_argument("--self-test-tts", action="store_true", help="执行一次无界面的 TTS 自检。")
    parser.add_argument("--self-test-stt", action="store_true", help="执行一次无界面的 STT 自检。")
    parser.add_argument("--self-test-audiofx", action="store_true", help="执行一次无界面的 AudioLDM2 音效生成自检。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test_tts:
        run_tts_self_test()
        return 0
    if args.self_test_stt:
        run_stt_self_test()
        return 0
    if args.self_test_audiofx:
        run_audiofx_self_test()
        return 0

    app = QApplication([])
    window = VoiceWorkbenchWindow()
    app.aboutToQuit.connect(window.shutdown)
    if args.smoke_test:
        QTimer.singleShot(100, app.quit)
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
