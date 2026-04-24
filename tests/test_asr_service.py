from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asr_service


class DummySegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class DummyFasterModel:
    def transcribe(self, *_args, **_kwargs):
        return iter(
            [
                DummySegment(0.0, 0.8, "你好"),
                DummySegment(0.8, 1.6, "世界"),
            ]
        ), {"language": "zh"}


class DummyOpenAIModel:
    def transcribe(self, *_args, **_kwargs):
        return {
            "text": "你好世界",
            "segments": [
                {"start": 0.0, "end": 0.8, "text": "你好"},
                {"start": 0.8, "end": 1.6, "text": "世界"},
            ],
        }


class AsrServiceTests(unittest.TestCase):
    def test_build_transcript_output_path_uses_txt(self) -> None:
        path = asr_service.build_transcript_output_path("demo.wav")
        self.assertEqual(path.suffix, ".txt")

    def test_build_transcript_output_path_uses_srt_for_srt_format(self) -> None:
        path = asr_service.build_transcript_output_path("demo.wav", output_format=asr_service.TRANSCRIPT_FORMAT_SRT)
        self.assertEqual(path.suffix, ".srt")

    def test_transcribe_audio_writes_faster_whisper_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            audio_path = temp_dir / "demo.wav"
            audio_path.write_bytes(b"fake")
            output_path = temp_dir / "demo.txt"

            status = asr_service.AsrEnvironmentStatus(
                ready=True,
                python_version="3.13.5",
                missing_packages=[],
                cuda_available=True,
                gpu_name="Fake GPU",
                total_vram_gb=12.0,
                model_paths={
                    asr_service.ASR_MODEL_OPENAI: str(asr_service.get_default_openai_whisper_dir()),
                    asr_service.ASR_MODEL_FASTER: str(asr_service.get_default_faster_whisper_dir()),
                },
                model_exists={
                    asr_service.ASR_MODEL_OPENAI: True,
                    asr_service.ASR_MODEL_FASTER: True,
                },
            )

            with (
                mock.patch("asr_service.check_asr_environment", return_value=status),
                mock.patch("asr_service.load_asr_model_once", return_value=DummyFasterModel()),
                mock.patch("asr_service._read_audio_duration_seconds", return_value=1.6),
            ):
                result = asr_service.transcribe_audio(
                    asr_service.TranscribeRequest(
                        audio_path=str(audio_path),
                        model_kind=asr_service.ASR_MODEL_FASTER,
                        output_path=str(output_path),
                        with_timestamps=True,
                    )
                )

            self.assertEqual(result.text, "你好世界")
            self.assertEqual(len(result.segments), 2)
            self.assertTrue(output_path.exists())
            self.assertIn("你好", output_path.read_text(encoding="utf-8"))

    def test_transcribe_audio_writes_openai_whisper_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            audio_path = temp_dir / "demo.wav"
            audio_path.write_bytes(b"fake")
            output_path = temp_dir / "demo.txt"

            status = asr_service.AsrEnvironmentStatus(
                ready=True,
                python_version="3.13.5",
                missing_packages=[],
                cuda_available=True,
                gpu_name="Fake GPU",
                total_vram_gb=12.0,
                model_paths={
                    asr_service.ASR_MODEL_OPENAI: str(asr_service.get_default_openai_whisper_dir()),
                    asr_service.ASR_MODEL_FASTER: str(asr_service.get_default_faster_whisper_dir()),
                },
                model_exists={
                    asr_service.ASR_MODEL_OPENAI: True,
                    asr_service.ASR_MODEL_FASTER: True,
                },
            )

            with (
                mock.patch("asr_service.check_asr_environment", return_value=status),
                mock.patch("asr_service.load_asr_model_once", return_value=DummyOpenAIModel()),
                mock.patch("asr_service._load_audio_for_openai_whisper", return_value=[0.0, 0.0, 0.0]),
                mock.patch("asr_service._read_audio_duration_seconds", return_value=1.6),
            ):
                result = asr_service.transcribe_audio(
                    asr_service.TranscribeRequest(
                        audio_path=str(audio_path),
                        model_kind=asr_service.ASR_MODEL_OPENAI,
                        output_path=str(output_path),
                        with_timestamps=True,
                    )
                )

            self.assertEqual(result.text, "你好世界")
            self.assertEqual(len(result.segments), 2)
            self.assertIn("[00:00:00.000 - 00:00:00.800]", output_path.read_text(encoding="utf-8"))

    def test_load_audio_for_openai_whisper_prefers_pyav_decode(self) -> None:
        fake_audio = [0, 1, 2]
        with mock.patch("asr_service._decode_audio_with_pyav", return_value=fake_audio):
            result = asr_service._load_audio_for_openai_whisper("demo.aac")
        self.assertEqual(result.dtype.name, "float32")
        self.assertEqual(result.tolist(), [0.0, 1.0, 2.0])

    def test_render_transcript_text_smart_paragraph_breaks_on_pause(self) -> None:
        segments = [
            asr_service.TranscribeSegment(0.0, 0.5, "你好"),
            asr_service.TranscribeSegment(0.6, 1.0, "我们开始"),
            asr_service.TranscribeSegment(2.3, 2.8, "这里换一段"),
        ]
        rendered = asr_service.render_transcript_text(
            "",
            segments,
            output_format=asr_service.TRANSCRIPT_FORMAT_SMART,
            with_timestamps=False,
            smart_segment_pause_seconds=0.8,
        )
        self.assertIn("你好 我们开始\n\n这里换一段", rendered)

    def test_render_transcript_text_srt_uses_subtitle_format(self) -> None:
        segments = [
            asr_service.TranscribeSegment(1.23, 3.54, "第一句"),
            asr_service.TranscribeSegment(4.0, 5.0, "第二句"),
        ]
        rendered = asr_service.render_transcript_text(
            "",
            segments,
            output_format=asr_service.TRANSCRIPT_FORMAT_SRT,
            with_timestamps=True,
        )
        self.assertIn("1\n00:00:01,230 --> 00:00:03,540\n第一句", rendered)
        self.assertIn("\n\n2\n00:00:04,000 --> 00:00:05,000\n第二句", rendered)

    def test_openai_whisper_works_without_console_streams(self) -> None:
        captured = {}

        class ConsoleSensitiveModel:
            device = "cuda"

            def transcribe(self, *_args, **_kwargs):
                captured["stdout_has_write"] = hasattr(sys.stdout, "write")
                captured["stderr_has_write"] = hasattr(sys.stderr, "write")
                return {
                    "text": "你好世界",
                    "segments": [
                        {"start": 0.0, "end": 0.8, "text": "你好"},
                        {"start": 0.8, "end": 1.6, "text": "世界"},
                    ],
                }

        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            with mock.patch("asr_service._load_audio_for_openai_whisper", return_value=[0.0, 0.0, 0.0]):
                text, segments = asr_service._run_openai_whisper(
                    ConsoleSensitiveModel(),
                    asr_service.TranscribeRequest(audio_path="demo.aac", model_kind=asr_service.ASR_MODEL_OPENAI),
                    progress_callback=None,
                    cancel_token=None,
                )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertEqual(text, "你好世界")
        self.assertEqual(len(segments), 2)
        self.assertTrue(captured["stdout_has_write"])
        self.assertTrue(captured["stderr_has_write"])


if __name__ == "__main__":
    unittest.main()
