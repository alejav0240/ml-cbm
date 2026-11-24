"""Create a therapist-editable DEMUCA draft without inventing clinical scores."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEMUCA_DRAFT = {
    "comportamientos_restrictivos": (
        "estereotipias", "agresividad", "desinteres", "pasividad",
        "aislamiento", "resistencia", "berrinche_crisis",
    ),
    "exploracion_vocal": (
        "vocalizaciones", "balbuceos", "silabas_canonicas",
        "imitacion_de_canciones", "creacion_vocal", "comunicacion_verbal",
    ),
    "interaccion_social_cognitiva": (
        "contacto_visual", "interaccion_con_otros_objetos",
        "interaccion_con_instrumentos_musicales",
        "interaccion_con_el_educador_musicoterapeuta",
        "interaccion_con_sus_padres", "interaccion_con_sus_pares",
        "atencion", "imitacion",
    ),
    "movimiento_corporal": (
        "andar", "correr", "parar", "gesticular", "danzar",
        "movimiento_en_su_lugar", "saltar",
    ),
    "exploracion_sonora": (
        "sonido_silencio", "timbre", "planos_de_altura",
        "movimiento_sonoro", "contrastes_de_intensidad",
    ),
    "exploracion_ritmica": (
        "repeticion_de_ideas_ritmicas_melodicas", "sensacion_de_conclusion",
        "pulso_interno", "regulacion_temporal", "ritmo_real", "apoyo",
        "contrastes_de_tempo",
    ),
}


def build_demuca_draft(session_id: str) -> dict[str, Any]:
    if not session_id or not isinstance(session_id, str):
        raise ValueError("invalid_session_id")
    return {
        "sesion_id": session_id,
        **{
            domain: {item: 0 for item in items}
            for domain, items in DEMUCA_DRAFT.items()
        },
    }


def write_demuca_draft(session_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    metadata_path = session_dir / "metadata" / "session.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    session_id = metadata.get("session_id") or session_dir.name
    output_path = session_dir / "processed" / "features" / "demuca_draft.json"
    if output_path.exists() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    draft = build_demuca_draft(session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return draft
