"""
=============================================================
MOTOR LSC - NUCLEO DE RECONOCIMIENTO LIGERO Y MODULAR
Lengua de Señas Colombiana (LSC)
=============================================================
Arquitectura basada en:
- Extracción de landmarks estabilizados con OneEuroFilter 3D.
- Preprocesamiento óptico anti-ruido con CLAHE en luminancia.
- Segmentación en cuadrantes espaciales (Signing Space).
- Base de datos vectorial con búsqueda por similitud (<1ms en CPU).
- Ventana de consenso temporal e histéresis anti-jitter.
- Ensamblado de frases offline (LSC -> Español).
- Síntesis de voz local asíncrona (pyttsx3).
"""

from .cuadrantes import CuadranteEspacial, clasificar_cuadrante
from .extractor import ExtractorLandmarks, OneEuroFilter3D, preprocesar_imagen_anti_ruido
from .base_vectores import BaseVectoresLSC
from .ensamblador_frases import EnsambladorFrases, VentanaConsenso
from .tts_local import MotorVozLocal

__all__ = [
    "CuadranteEspacial",
    "clasificar_cuadrante",
    "ExtractorLandmarks",
    "OneEuroFilter3D",
    "preprocesar_imagen_anti_ruido",
    "BaseVectoresLSC",
    "EnsambladorFrases",
    "VentanaConsenso",
    "MotorVozLocal",
]

