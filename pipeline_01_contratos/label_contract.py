"""Validation for the observable-event label contract."""

from __future__ import annotations

from typing import Any


REQUIRED_DEFINITION_FIELDS = {
    "description",
    "positive_examples",
    "negative_examples",
    "target",
    "observable",
}


def validate_task_contract(task: dict[str, Any]) -> None:
    """Reject incomplete or ambiguous label definitions before dataset work."""
    label_sources = task.get("label_sources")
    definitions = task.get("label_definitions")
    if not isinstance(label_sources, dict) or not label_sources:
        raise ValueError("label_sources_must_be_nonempty_mapping")
    if not isinstance(definitions, dict):
        raise ValueError("label_definitions_missing")

    for modality, labels in label_sources.items():
        if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
            raise ValueError(f"invalid_label_list:{modality}")
        modality_definitions = definitions.get(modality)
        if not isinstance(modality_definitions, dict):
            raise ValueError(f"label_definitions_missing_modality:{modality}")
        if set(modality_definitions) != set(labels):
            raise ValueError(f"label_definition_set_mismatch:{modality}")
        for label in labels:
            definition = modality_definitions[label]
            if not isinstance(definition, dict):
                raise ValueError(f"invalid_label_definition:{modality}:{label}")
            missing = REQUIRED_DEFINITION_FIELDS - set(definition)
            if missing:
                raise ValueError(
                    f"label_definition_missing_fields:{modality}:{label}:{','.join(sorted(missing))}"
                )
            if not isinstance(definition["description"], str) or not definition["description"].strip():
                raise ValueError(f"label_description_required:{modality}:{label}")
            for field in ("positive_examples", "negative_examples"):
                examples = definition[field]
                if not isinstance(examples, list) or not examples or not all(
                    isinstance(example, str) and example.strip() for example in examples
                ):
                    raise ValueError(f"label_examples_required:{modality}:{label}:{field}")
            if definition["target"] != "patient":
                raise ValueError(f"label_target_must_be_patient:{modality}:{label}")
            if definition["observable"] is not True:
                raise ValueError(f"label_must_be_observable:{modality}:{label}")

    extra_modalities = set(definitions) - set(label_sources)
    if extra_modalities:
        raise ValueError("label_definitions_extra_modalities:" + ",".join(sorted(extra_modalities)))

    if any("demuca" in label.lower() for labels in label_sources.values() for label in labels):
        raise ValueError("demuca_fields_are_not_observable_event_targets")
