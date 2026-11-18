import tempfile
import unittest
from pathlib import Path

from dataset_source import resolve_sessions_dir, source_metadata


class DatasetSourceTests(unittest.TestCase):
    def test_resolves_canonical_checkout_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "dataset" / "sessions"
            expected.mkdir(parents=True)
            self.assertEqual(resolve_sessions_dir(root), expected.resolve())

    def test_supports_direct_sessions_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "sessions"
            expected.mkdir(parents=True)
            self.assertEqual(resolve_sessions_dir(root), expected.resolve())

    def test_supports_legacy_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "dataset" / "dataset" / "sessions"
            expected.mkdir(parents=True)
            self.assertEqual(resolve_sessions_dir(root), expected.resolve())

    def test_returns_canonical_missing_path_for_empty_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_sessions_dir(root), (root / "dataset" / "sessions").resolve())

    def test_exposes_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = root / "dataset" / "sessions"
            sessions.mkdir(parents=True)
            metadata = source_metadata(sessions)
            self.assertEqual(metadata["repository"], "https://github.com/alejav0240/dataset-ml")
            self.assertEqual(metadata["sessions_dir"], str(sessions.resolve()))


if __name__ == "__main__":
    unittest.main()
