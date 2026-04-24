from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import voxcpm_service as service


class DummyTTSModel:
    sample_rate = 24000


class DummyModel:
    def __init__(self) -> None:
        self.tts_model = DummyTTSModel()
        self.calls: list[tuple[str, dict]] = []

    def generate(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return np.ones(24000, dtype=np.float32)


class VoxCPMServiceTests(unittest.TestCase):
    def test_split_text_splits_long_paragraphs(self) -> None:
        text = "第一句很短。第二句也很短。第三句有点长，但是依然应该被拆分。"
        chunks = service.split_text(text, 12)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_check_environment_returns_status(self) -> None:
        status = service.check_environment()
        self.assertIsInstance(status.ready, bool)
        self.assertIsInstance(status.missing_packages, list)
        self.assertTrue(status.default_model_path)

    def test_generate_tts_passes_reference_and_prompt_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            reference_wav = temp_dir / "reference.wav"
            prompt_wav = temp_dir / "prompt.wav"
            reference_wav.write_bytes(b"ref")
            prompt_wav.write_bytes(b"prompt")
            output_path = temp_dir / "result.wav"

            request = service.GenerateRequest(
                text="你好，欢迎使用桌面版配音工具。",
                voice="成熟男性，沉稳自然",
                reference_wav=str(reference_wav),
                prompt_wav=str(prompt_wav),
                prompt_text="大家好，这是一段 prompt。",
                output_path=str(output_path),
            )

            dummy_model = DummyModel()
            status = service.EnvironmentStatus(
                ready=True,
                python_version="3.13.5",
                missing_packages=[],
                cuda_available=True,
                gpu_name="Fake GPU",
                total_vram_gb=12.0,
                default_model_path=str(service.get_default_model_dir()),
                model_exists=True,
            )

            with (
                mock.patch("voxcpm_service.check_environment", return_value=status),
                mock.patch("voxcpm_service.load_model_once", return_value=dummy_model),
                mock.patch("voxcpm_service.sf.write") as mocked_write,
            ):
                result = service.generate_tts(request)

            self.assertEqual(result.output_path, str(output_path.resolve()))
            self.assertEqual(result.chunks_count, 1)
            self.assertEqual(dummy_model.calls[0][1]["reference_wav_path"], str(reference_wav))
            self.assertEqual(dummy_model.calls[0][1]["prompt_wav_path"], str(prompt_wav))
            self.assertEqual(dummy_model.calls[0][1]["prompt_text"], "大家好，这是一段 prompt。")
            mocked_write.assert_called_once()

    def test_generate_tts_requires_prompt_pair(self) -> None:
        request = service.GenerateRequest(
            text="测试文本",
            prompt_wav="only_prompt.wav",
        )
        with self.assertRaisesRegex(ValueError, "同时填写"):
            service.generate_tts(request)

    def test_generate_tts_reuses_first_chunk_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            output_path = temp_dir / "result.wav"

            request = service.GenerateRequest(
                text="第一句很短。第二句也很短。第三句也很短。",
                chunk_max_chars=6,
                reuse_first_chunk_as_reference=True,
                output_path=str(output_path),
            )

            dummy_model = DummyModel()
            status = service.EnvironmentStatus(
                ready=True,
                python_version="3.13.5",
                missing_packages=[],
                cuda_available=True,
                gpu_name="Fake GPU",
                total_vram_gb=12.0,
                default_model_path=str(service.get_default_model_dir()),
                model_exists=True,
            )

            writes: list[str] = []

            def fake_write(path, *_args, **_kwargs):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"wav")
                writes.append(str(target))

            with (
                mock.patch("voxcpm_service.check_environment", return_value=status),
                mock.patch("voxcpm_service.load_model_once", return_value=dummy_model),
                mock.patch("voxcpm_service.sf.write", side_effect=fake_write),
            ):
                result = service.generate_tts(request)

            self.assertGreaterEqual(result.chunks_count, 2)
            self.assertNotIn("reference_wav_path", dummy_model.calls[0][1])
            for _, kwargs in dummy_model.calls[1:]:
                self.assertIn("reference_wav_path", kwargs)
            self.assertEqual(len({kwargs["reference_wav_path"] for _, kwargs in dummy_model.calls[1:]}), 1)
            self.assertEqual(len(writes), 2)
            self.assertEqual(Path(writes[-1]).resolve(), output_path.resolve())


if __name__ == "__main__":
    unittest.main()
