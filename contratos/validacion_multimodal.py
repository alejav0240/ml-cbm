"""Fachada de validación para conservar el punto de entrada histórico."""

from __future__ import annotations

from pathlib import Path

from validate_multimodal_artifacts import validate_multimodal_artifacts


def validar_sesion_multimodal(session_dir: Path) -> list[str]:
    """Valida una sesión usando el contrato público vigente."""
    return validate_multimodal_artifacts(session_dir)
