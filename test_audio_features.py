import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_features import extract_audio_modalities


class AudioFeatureTests(unittest.TestCase):
    def test_extracts_stereo_audio_on_the_requested_window_grid_and_reports_tail(self) -> None:
        sample_rate = 8_000
        duration_seconds = 2.25
        timeline = np.arange(round(sample_rate * duration_seconds)) / sample_rate
        mono = (0.1 * np.sin(2 * np.pi * 180 * timeline)).astype(np.float32)
        stereo = np.column_stack((mono, mono * 0.5))

        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "S001" / "processed" / "audio" / "S001.wav"
            audio_path.parent.mkdir(parents=True)
            sf.write(audio_path, stereo, sample_rate)

            features, vocalizations = extract_audio_modalities(
                audio_path,
                window_seconds=1.0,
            )

        self.assertEqual(features["session_id"], "S001")
        self.assertEqual(features["complete_windows"], 2)
        self.assertEqual(features["tail_samples_excluded"], sample_rate // 4)
        self.assertEqual([row["timestamp_seconds"] for row in features["windows"]], [0.5, 1.5])
        self.assertEqual(len(features["windows"][0]["mfcc"]), 13)
        self.assertFalse(vocalizations["training_authorized"])
        self.assertEqual(vocalizations["speaker_attribution"], "not_performed")

    def test_rejects_empty_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "S001" / "processed" / "audio" / "empty.wav"
            audio_path.parent.mkdir(parents=True)
            sf.write(audio_path, np.array([], dtype=np.float32), 8_000)

            with self.assertRaisesRegex(ValueError, "audio_must_be_nonempty"):
                extract_audio_modalities(audio_path)

    def test_limits_acoustic_windows_to_the_stable_multimodal_grid(self) -> None:
        sample_rate = 8_000
        audio = np.zeros(sample_rate * 10, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "S001" / "processed" / "audio" / "S001.wav"
            audio_path.parent.mkdir(parents=True)
            sf.write(audio_path, audio, sample_rate)

            features, _ = extract_audio_modalities(
                audio_path,
                window_seconds=5.0,
                expected_complete_windows=1,
            )

        self.assertEqual(features["complete_windows"], 1)
        self.assertEqual(features["tail_samples_excluded"], sample_rate * 5)
        self.assertEqual(len(features["windows"]), 1)


if __name__ == "__main__":
    unittest.main()
