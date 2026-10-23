"""Adaptador de herramientas externas de medios."""

from __future__ import annotations

import subprocess


class AdaptadorFFmpeg:
    """Ejecuta FFmpeg/FFprobe sin filtrar detalles de subprocess al dominio."""

    def ejecutar(self, comando: list[str]) -> str:
        resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
        return resultado.stdout.strip()
