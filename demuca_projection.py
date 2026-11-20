"""Project multimodal observable-event predictions into a therapist-reviewable DEMUCA shape.

This module does not claim to be a clinically calibrated DEMUCA model. It provides a
deterministic, auditable MVP projection so every analyzed session returns the DEMUCA
schema while the therapist remains the final reviewer.
"""

from __future__ import annotations

from typing import Any


DEMUCA_SCALE = {
    "min": 0,
    "max": 2,
    "labels": {"0": "nada", "1": "medio", "2": "alto"},
}

# Each entry is (model event label, invert). Inverted signals represent the
# observable absence of an event (for example, low movement as a passivity proxy).
# Empty entries mean that the current event model has no direct signal for that
# construct; those items are returned as 0 with evidence_status=unsupported.
SIGNAL_MAP: dict[str, tuple[tuple[str, bool], ...]] = {
    "comportamientos_restrictivos.estereotipias": (("visual__MOVIMIENTO_REPETITIVO", False),),
    "comportamientos_restrictivos.agresividad": (
        ("audio__GRITO", False), ("interaction__TACTO", False),
    ),
    "comportamientos_restrictivos.desinteres": (
        ("visual__CONTACTO_VISUAL", True), ("interaction__MIRADA", True),
    ),
    "comportamientos_restrictivos.pasividad": (("visual__MOVIMIENTO", True),),
    "comportamientos_restrictivos.aislamiento": (
        ("interaction__ACTIVIDAD_CONJUNTA", True),
        ("interaction__INTERACCION_MUSICAL", True),
    ),
    "comportamientos_restrictivos.resistencia": (
        ("audio__LLANTO", False), ("audio__GRITO", False),
        ("interaction__RESPUESTA", True),
    ),
    "comportamientos_restrictivos.berrinche_crisis": (
        ("audio__LLANTO", False), ("audio__GRITO", False),
        ("visual__MOVIMIENTO_REPETITIVO", False),
    ),
    "exploracion_vocal.vocalizaciones": (("audio__VOCALIZACION", False),),
    "exploracion_vocal.balbuceos": (("audio__VOCALIZACION", False),),
    "exploracion_vocal.silabas_canonicas": (("audio__HABLA", False),),
    "exploracion_vocal.imitacion_de_canciones": (
        ("interaction__IMITACION", False), ("audio__ACTIVIDAD_MUSICAL", False),
    ),
    "exploracion_vocal.creacion_vocal": (
        ("audio__VOCALIZACION", False), ("audio__ACTIVIDAD_MUSICAL", False),
    ),
    "exploracion_vocal.comunicacion_verbal": (
        ("audio__HABLA", False), ("interaction__INTERACCION_VERBAL", False),
    ),
    "interaccion_social_cognitiva.contacto_visual": (
        ("visual__CONTACTO_VISUAL", False), ("interaction__MIRADA", False),
    ),
    "interaccion_social_cognitiva.interaccion_con_otros_objetos": (
        ("visual__INTERACCION_OBJETO", False), ("interaction__TACTO", False),
    ),
    "interaccion_social_cognitiva.interaccion_con_instrumentos_musicales": (
        ("audio__ACTIVIDAD_MUSICAL", False),
        ("interaction__INTERACCION_MUSICAL", False),
    ),
    "interaccion_social_cognitiva.interaccion_con_el_educador_musicoterapeuta": (
        ("interaction__ACTIVIDAD_CONJUNTA", False),
        ("interaction__RESPUESTA", False),
    ),
    "interaccion_social_cognitiva.interaccion_con_sus_padres": (),
    "interaccion_social_cognitiva.interaccion_con_sus_pares": (),
    "interaccion_social_cognitiva.atencion": (
        ("visual__CONTACTO_VISUAL", False), ("interaction__MIRADA", False),
        ("interaction__RESPUESTA", False),
    ),
    "interaccion_social_cognitiva.imitacion": (("interaction__IMITACION", False),),
    "movimiento_corporal.andar": (("visual__MOVIMIENTO", False),),
    "movimiento_corporal.correr": (("visual__MOVIMIENTO", False),),
    "movimiento_corporal.parar": (("visual__MOVIMIENTO", True),),
    "movimiento_corporal.gesticular": (("interaction__GESTO", False),),
    "movimiento_corporal.danzar": (
        ("visual__MOVIMIENTO", False), ("audio__ACTIVIDAD_MUSICAL", False),
    ),
    "movimiento_corporal.movimiento_en_su_lugar": (
        ("visual__MOVIMIENTO_REPETITIVO", False), ("visual__MOVIMIENTO", False),
    ),
    "movimiento_corporal.saltar": (("visual__MOVIMIENTO", False),),
    "exploracion_sonora.sonido_silencio": (("audio__SILENCIO", False),),
    "exploracion_sonora.timbre": (),
    "exploracion_sonora.planos_de_altura": (),
    "exploracion_sonora.movimiento_sonoro": (),
    "exploracion_sonora.contrastes_de_intensidad": (),
    "exploracion_ritmica.repeticion_de_ideas_ritmicas_melodicas": (
        ("interaction__IMITACION", False),
        ("interaction__INTERACCION_MUSICAL", False),
    ),
    "exploracion_ritmica.sensacion_de_conclusion": (),
    "exploracion_ritmica.pulso_interno": (),
    "exploracion_ritmica.regulacion_temporal": (),
    "exploracion_ritmica.ritmo_real": (("interaction__INTERACCION_MUSICAL", False),),
    "exploracion_ritmica.apoyo": (),
    "exploracion_ritmica.contrastes_de_tempo": (),
}


def _score_from_probability(value: float) -> int:
    """Map evidence probability to the unified DEMUCA 0–2 scale."""
    if value < 0.33:
        return 0
    if value < 0.66:
        return 1
    return 2


def _probability(item: dict[str, Any], label: str) -> float | None:
    probabilities = item.get("probabilities")
    if isinstance(probabilities, dict) and isinstance(probabilities.get(label), (int, float)):
        return max(0.0, min(1.0, float(probabilities[label])))
    predictions = item.get("predictions")
    if isinstance(predictions, dict) and isinstance(predictions.get(label), (int, float)):
        return float(bool(predictions[label]))
    return None


def _project_item(
    windows: list[dict[str, Any]],
    signals: tuple[tuple[str, bool], ...],
) -> tuple[int, dict[str, Any]]:
    if not signals:
        return 0, {
            "evidence_status": "unsupported",
            "confidence_percent": 0.0,
            "source_labels": [],
            "windows_supported": 0,
            "evidence_windows": [],
        }

    signal_values: list[float] = []
    source_labels: list[str] = []
    evidence_windows: list[dict[str, Any]] = []
    for label, invert in signals:
        values = [_probability(window, label) for window in windows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        source_labels.append(label)
        value = sum(values) / len(values)
        signal_values.append(1.0 - value if invert else value)

    for window in windows:
        signal_details = []
        for label, invert in signals:
            probability = _probability(window, label)
            if probability is None:
                continue
            signal_details.append(
                {
                    "label": label,
                    "probability": round(probability, 4),
                    "contribution": round(1.0 - probability if invert else probability, 4),
                    "inverted": invert,
                }
            )
        if signal_details:
            evidence_windows.append(
                {
                    "window_id": window.get("window_id"),
                    "start_seconds": window.get("start_seconds"),
                    "end_seconds": window.get("end_seconds"),
                    "signals": signal_details,
                }
            )

    if not signal_values:
        return 0, {
            "evidence_status": "missing_model_signal",
            "confidence_percent": 0.0,
            "source_labels": [label for label, _ in signals],
            "windows_supported": 0,
            "evidence_windows": evidence_windows,
        }

    aggregate = sum(signal_values) / len(signal_values)
    score = _score_from_probability(aggregate)
    return score, {
        "evidence_status": "supported",
        "confidence_percent": round(aggregate * 100.0, 2),
        "source_labels": source_labels,
        "windows_supported": len(windows),
        "evidence_windows": evidence_windows,
    }


def build_demuca_from_predictions(
    *,
    session_id: str,
    model_predictions: dict[str, Any],
) -> dict[str, Any]:
    """Return all DEMUCA domains from the model's observable event windows."""
    windows = model_predictions.get("windows")
    if not isinstance(windows, list):
        windows = []

    result: dict[str, Any] = {
        "sesion_id": session_id,
        "escala": "DEMUCA",
        "version": "mvp-projection-1.1",
        "score_scale": DEMUCA_SCALE,
        "status": "provisional" if windows else "unavailable",
        "score_source": "multimodal_observable_event_projection",
        "model_type": model_predictions.get("model_type"),
        "model_status": model_predictions.get("status", "unavailable"),
        "model_path": model_predictions.get("model_path"),
        "window_count": len(windows),
        "review_required": True,
        "ground_truth": False,
        "training_authorized": False,
        "calibration_status": "not_calibrated",
        "notice": (
            "Puntuación DEMUCA provisional derivada de eventos observables. "
            "No sustituye la evaluación del musicoterapeuta ni representa una "
            "probabilidad clínica calibrada."
        ),
        "evidence": {},
    }

    domains: dict[str, dict[str, int]] = {}
    for key in SIGNAL_MAP:
        domain, item = key.split(".", 1)
        score, evidence = _project_item(windows, SIGNAL_MAP[key])
        domains.setdefault(domain, {})[item] = score
        result["evidence"][key] = {"score": score, **evidence}

    result.update(domains)
    supported = sum(
        1 for item in result["evidence"].values()
        if item["evidence_status"] == "supported"
    )
    result["coverage_percent"] = round(
        (supported / len(result["evidence"]) * 100.0) if result["evidence"] else 0.0,
        2,
    )
    return result
