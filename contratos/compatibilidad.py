"""Inventario explícito de artefactos nuevos y legacy."""

from __future__ import annotations

RUTA_ACUSTICA_NUEVA = "processed/features/acoustic_features.json"
RUTAS_LEGACY = (
    "processed/features/audio_features.json",
    "processed/features/visual_motion.json",
)


def es_artefacto_legacy(relative_path: str) -> bool:
    """Indica si una ruta pertenece a la salida histórica preservada."""
    return relative_path in RUTAS_LEGACY
