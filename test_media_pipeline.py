import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import media_pipeline
from media import synchronize_pose_with_segments


class MediaPipelineTests(unittest.TestCase):
    def test_audio_artifact_write_is_atomic_when_replacement_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "features" / "acoustic_features.json"
            output_path.parent.mkdir(parents=True)
            output_path.write_text("preserved\n", encoding="utf-8")

            with patch.object(media_pipeline.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    media_pipeline.persist_or_reuse_audio_artifact(
                        output_path,
                        {"source_audio_sha256": "new-hash", "windows": []},
                        source_audio_sha256="new-hash",
                        overwrite=True,
                    )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserved\n")
            self.assertEqual(list(output_path.parent.iterdir()), [output_path])

    def test_rejects_reusing_acoustic_features_from_a_different_window_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            session_dir = project_root / "dataset" / "dataset" / "sessions" / "S001"
            (session_dir / "processed" / "segments").mkdir(parents=True)
            (session_dir / "processed" / "features").mkdir(parents=True)
            audio_path = session_dir / "processed" / "audio" / "S001_16khz_mono.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            (session_dir / "processed" / "features" / "acoustic_features.json").write_text(
                json.dumps({"source_audio_sha256": "audio-hash", "window_seconds": 1.0}),
                encoding="utf-8",
            )

            segments = {
                "segments": [], "effective_duration_seconds": 0.0,
                "max_segment_drift_seconds": 0.0, "source_drift_seconds": 0.0,
                "trimmed_stream": None, "tail_trim_seconds": 0.0,
            }
            acoustic = {
                "source_audio_sha256": "audio-hash", "window_seconds": 5.0, "windows": [],
            }
            vocalizations = {
                "source_audio_sha256": "audio-hash", "candidate_count": 0, "candidates": [],
            }

            with (
                patch.object(media_pipeline, "run_existing_extraction", return_value={}),
                patch.object(media_pipeline, "segment_and_synchronize", return_value=segments),
                patch.object(media_pipeline, "load_or_create_window_index", return_value={"windows": []}),
                patch.object(media_pipeline, "extract_audio_modalities", return_value=(acoustic, vocalizations)),
                patch.object(media_pipeline, "sha256_audio_file", return_value="audio-hash"),
            ):
                with self.assertRaisesRegex(FileExistsError, "stale audio feature output"):
                    media_pipeline.process_session(
                        "S001", project_root, 5.0, 0.2, None, True, False
                    )

    def test_manifest_aligns_pose_acoustics_and_vocalization_candidates(self) -> None:
        segments = {
            "session_id": "S001",
            "clock": "session seconds",
            "window_seconds": 5.0,
            "source_drift_seconds": 0.0,
            "max_segment_drift_seconds": 0.0,
            "segments": [
                {"index": 0, "start_seconds": 0.0, "end_seconds": 5.0},
                {"index": 1, "start_seconds": 5.0, "end_seconds": 10.0},
            ],
        }
        pose = {"sampling_fps": 1.0, "frames": [{"timestamp_seconds": 1.0}]}
        acoustics = {
            "window_seconds": 5.0,
            "windows": [
                {"timestamp_seconds": 2.5},
                {"timestamp_seconds": 7.5},
            ],
        }
        vocalizations = {
            "candidates": [
                {"start_seconds": 4.8, "end_seconds": 5.2},
                {"start_seconds": 8.0, "end_seconds": 8.5},
            ],
        }
        movement = {
            "windows": [
                {"window_id": "S001_W0000"},
                {"window_id": "S001_W0001"},
            ],
        }

        result = synchronize_pose_with_segments(
            segments,
            pose,
            "../features/pose_landmarks.json",
            acoustic_payload=acoustics,
            acoustic_features_path="../features/acoustic_features.json",
            vocalization_payload=vocalizations,
            vocalization_features_path="../features/vocalization_candidates.json",
            movement_payload=movement,
            movement_features_path="../features/movement_interaction.json",
        )

        self.assertEqual(result["segments"][0]["acoustic_window_indices"], [0])
        self.assertEqual(result["segments"][1]["acoustic_window_indices"], [1])
        self.assertEqual(result["segments"][0]["vocalization_candidate_indices"], [0])
        self.assertEqual(result["segments"][1]["vocalization_candidate_indices"], [0, 1])
        self.assertEqual(result["acoustic_features_path"], "../features/acoustic_features.json")
        self.assertEqual(result["segments"][0]["movement_window_index"], 0)
        self.assertEqual(result["segments"][1]["movement_window_index"], 1)

    def test_audio_features_use_the_shared_multimodal_window_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            session_dir = project_root / "dataset" / "dataset" / "sessions" / "S001"
            (session_dir / "processed" / "segments").mkdir(parents=True)
            audio_path = session_dir / "processed" / "audio" / "S001_16khz_mono.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")

            segments = {
                "segments": [],
                "effective_duration_seconds": 0.0,
                "max_segment_drift_seconds": 0.0,
                "source_drift_seconds": 0.0,
                "trimmed_stream": None,
                "tail_trim_seconds": 0.0,
            }
            acoustic = {
                "source_audio_sha256": "audio-hash",
                "windows": [],
            }
            vocalizations = {
                "source_audio_sha256": "audio-hash",
                "candidate_count": 0,
            }

            with (
                patch.object(media_pipeline, "run_existing_extraction", return_value={}),
                patch.object(media_pipeline, "segment_and_synchronize", return_value=segments),
                patch.object(
                    media_pipeline,
                    "load_or_create_window_index",
                    return_value={"session_id": "S001", "windows": []},
                ) as load_window_index,
                patch.object(
                    media_pipeline,
                    "extract_audio_modalities",
                    return_value=(acoustic, vocalizations),
                ) as extract_audio,
                patch.object(media_pipeline, "sha256_audio_file", return_value="audio-hash"),
                patch.object(media_pipeline, "sha256_feature_file", return_value="artifact-hash"),
                patch.object(media_pipeline, "validate_multimodal_artifacts", return_value=[]),
                patch.object(media_pipeline, "synchronize_pose_with_segments", return_value={}),
            ):
                summary = media_pipeline.process_session(
                    "S001",
                    project_root,
                    window_seconds=5.0,
                    max_tail_trim_seconds=0.2,
                    pose_model=None,
                    skip_pose=True,
                    overwrite=True,
                )

            extract_audio.assert_called_once_with(
                audio_path,
                window_seconds=5.0,
                expected_complete_windows=0,
            )
            load_window_index.assert_called_once_with(
                session_dir,
                overwrite=True,
            )
            self.assertEqual(
                summary["derived_artifacts"]["acoustic_features"]["path"],
                "processed/features/acoustic_features.json",
            )
            self.assertIn("sha256", summary["derived_artifacts"]["vocalization_candidates"])


if __name__ == "__main__":
    unittest.main()
