import copy
import json
import unittest
from pathlib import Path

from label_contract import validate_task_contract


TASK = json.loads((Path(__file__).parent / "task.json").read_text(encoding="utf-8"))


class LabelContractTests(unittest.TestCase):
    def test_current_task_has_complete_observable_definitions(self):
        validate_task_contract(TASK)

    def test_rejects_missing_definition(self):
        task = copy.deepcopy(TASK)
        del task["label_definitions"]["audio"]["HABLA"]
        with self.assertRaisesRegex(ValueError, "label_definition_set_mismatch:audio"):
            validate_task_contract(task)

    def test_rejects_non_patient_target(self):
        task = copy.deepcopy(TASK)
        task["label_definitions"]["visual"]["MOVIMIENTO"]["target"] = "therapist"
        with self.assertRaisesRegex(ValueError, "label_target_must_be_patient:visual:MOVIMIENTO"):
            validate_task_contract(task)


if __name__ == "__main__":
    unittest.main()
