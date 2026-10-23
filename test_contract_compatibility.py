import unittest

from contratos.compatibilidad import RUTA_ACUSTICA_NUEVA, RUTAS_LEGACY, es_artefacto_legacy


class CompatibilidadDeArtefactosTests(unittest.TestCase):
    def test_no_mezcla_ruta_nueva_con_legacy(self) -> None:
        self.assertNotIn(RUTA_ACUSTICA_NUEVA, RUTAS_LEGACY)
        self.assertTrue(es_artefacto_legacy("processed/features/audio_features.json"))
        self.assertTrue(es_artefacto_legacy("processed/features/visual_motion.json"))
        self.assertFalse(es_artefacto_legacy(RUTA_ACUSTICA_NUEVA))


if __name__ == "__main__":
    unittest.main()
