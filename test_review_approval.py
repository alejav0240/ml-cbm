import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from review_approval import (
    ANNOTATION_FILES,
    APPROVAL_RELATIVE_PATH,
    GUIDE_RELATIVE_PATH,
    load_annotation_approval,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewApprovalTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        session = root / "S001"
        for relative_path in ANNOTATION_FILES:
            path = session / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"session_id":"S001","events":[]}', encoding="utf-8")
        guide_path = session / GUIDE_RELATIVE_PATH
        guide_path.parent.mkdir(parents=True, exist_ok=True)
        guide_path.write_text("Approved synthetic guide", encoding="utf-8")
        approval = {
            "schema_version": "1.0",
            "session_id": "S001",
            "annotations": {
                "status": "approved",
                "approved_by": "Reviewer",
                "approved_at": "2026-10-14",
                "annotator_primary": "Primary",
                "reviewer": "Reviewer",
            },
            "guide": {
                "status": "approved",
                "version": "v1.0",
                "approved_by": "Clinician",
                "approved_at": "2026-10-18",
                "path": GUIDE_RELATIVE_PATH.as_posix(),
                "sha256": digest(guide_path),
            },
            "annotation_files_sha256": {
                relative_path.as_posix(): digest(session / relative_path)
                for relative_path in ANNOTATION_FILES
            },
        }
        approval_path = session / APPROVAL_RELATIVE_PATH
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        return session

    def test_validates_approved_record_and_exact_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._fixture(Path(temp_dir))

            report = load_annotation_approval(session)

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(len(report["approved_files"]), 3)

    def test_rejects_changed_event_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._fixture(Path(temp_dir))
            (session / ANNOTATION_FILES[0]).write_text("tampered", encoding="utf-8")

            report = load_annotation_approval(session)

        self.assertFalse(report["valid"])
        self.assertIn(
            f"annotation_approval_source_hash_mismatch:{ANNOTATION_FILES[0].as_posix()}",
            report["issues"],
        )

    def test_rejects_paths_not_in_approved_annotation_set_without_reading_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._fixture(Path(temp_dir))
            approval_path = session / APPROVAL_RELATIVE_PATH
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
            approval["annotation_files_sha256"]["../../outside.json"] = "0" * 64
            approval_path.write_text(json.dumps(approval), encoding="utf-8")

            report = load_annotation_approval(session)

        self.assertFalse(report["valid"])
        self.assertIn("annotation_approval_file_set_mismatch", report["issues"])


if __name__ == "__main__":
    unittest.main()
