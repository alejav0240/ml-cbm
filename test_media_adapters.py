import unittest
from unittest.mock import patch

from infraestructura.medios import AdaptadorFFmpeg


class AdaptadorFFmpegTests(unittest.TestCase):
    def test_devuelve_salida_estandar_del_comando(self) -> None:
        with patch("infraestructura.medios.subprocess.run") as ejecutar:
            ejecutar.return_value.stdout = "  12.5\n"
            self.assertEqual(AdaptadorFFmpeg().ejecutar(["ffprobe", "archivo.mp4"]), "12.5")
            ejecutar.assert_called_once_with(
                ["ffprobe", "archivo.mp4"], check=True, capture_output=True, text=True
            )
