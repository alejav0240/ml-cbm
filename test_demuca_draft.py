import json
import tempfile
import unittest
from pathlib import Path

from demuca_draft import DEMUCA_DRAFT, build_demuca_draft, write_demuca_draft


class DemucaDraftTests(unittest.TestCase):
    def test_builds_requested_editable_shape_with_zero_defaults(self):
        draft = build_demuca_draft("S002")
        self.assertEqual(draft["sesion_id"], "S002")
        self.assertEqual(set(draft) - {"sesion_id"}, set(DEMUCA_DRAFT))
        self.assertTrue(all(value == 0 for domain in DEMUCA_DRAFT for value in draft[domain].values()))

    def test_writes_atomic_draft_and_reuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "S002"
            (session / "metadata").mkdir(parents=True)
            (session / "metadata/session.json").write_text(
                json.dumps({"session_id": "S002"}), encoding="utf-8"
            )
            first = write_demuca_draft(session)
            second = write_demuca_draft(session)
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads((session / "processed/features/demuca_draft.json").read_text()),
                first,
            )


if __name__ == "__main__":
    unittest.main()
