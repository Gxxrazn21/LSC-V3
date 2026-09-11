"""
=============================================================
CONFIGURACION CENTRAL DEL PROYECTO LSC v3.0
Sistema de Reconocimiento de Lengua de Señas Colombiana
=============================================================
Arquitectura basada en:
- Vectores de Landmarks 3D (105 dimensiones) con OneEuroFilter.
- Preprocesamiento de cámara mejorado v3.0 (CLAHE + AWB + Sharpening).
- MediaPipe Holistic para detección unificada manos + cuerpo.
- Base de datos vectorial con búsqueda por similitud coseno (<1ms en CPU).
- Cuadrantes Espaciales (Signing Space) + Ventana de Consenso Temporal.
- Ensamblador de Frases Offline (LSC → Español).
- Síntesis de voz local asíncrona (pyttsx3).
"""

import os

# ─────────────────────────────────────────────
# RUTAS DEL PROYECTO + CARGA DE .env
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    except Exception as e:
        print(f"  [config] Aviso: no se pudo leer .env ({e})")


_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _env_path(var: str, default: str) -> str:
    value = os.environ.get(var, default)
    if not os.path.isabs(value):
        value = os.path.join(PROJECT_ROOT, value)
    return value


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ.get(var, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ.get(var, str(default)))
    except (TypeError, ValueError):
        return default


# Rutas de almacenamiento
DATASETS_DIR = _env_path("LSC_DATASETS_DIR", "datasets")
MODELOS_DIR = _env_path("LSC_MODELOS_DIR", "modelos_guardados")
RESULTADOS_DIR = _env_path("LSC_RESULTADOS_DIR", "resultados")

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Base de datos vectorial (modelo completo)
BASE_VECTORES_PATH = os.path.join(MODELOS_DIR, "base_senas_lsc.npz")
CLASES_LSC_PATH = os.path.join(MODELOS_DIR, "clases_lsc.json")

# Base de datos vectorial (modelo 5 palabras de validación)
BASE_5PALABRAS_PATH = os.path.join(MODELOS_DIR, "base_5palabras.npz")
CLASES_5P_PATH = os.path.join(MODELOS_DIR, "clases_5palabras.json")

# Datos capturados por el usuario
DIR_CAPTURADO = os.path.join(DATASETS_DIR, "capturado")

# Datasets de referencia (opcionales)
LSC54_JSON_PATH = os.path.join(DATASETS_DIR, "LSC54", "sample.json")
LSC70_PATH = os.path.join(DATASETS_DIR, "LSC70")
LSC50_PATH = os.path.join(DATASETS_DIR, "LSC50")

# Las 5 palabras de validación
SENAS_5_PALABRAS = ["HOLA", "GRACIAS", "BIEN", "CASA", "AGUA"]

# ─────────────────────────────────────────────
# HIPERPARAMETROS DE RECONOCIMIENTO Y CAMARA
# ─────────────────────────────────────────────

CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)
UMBRAL_SIMILITUD = _env_float("UMBRAL_SIMILITUD", 0.72)
FRAMES_ESTABLES = _env_int("FRAMES_ESTABLES", 6)
SEGUNDOS_SILENCIO = _env_float("SEGUNDOS_SILENCIO", 1.2)
BONO_CUADRANTE = _env_float("BONO_CUADRANTE", 0.10)

# Mejora de cámara v3.0 (pipeline preprocesamiento de imagen)
CAMARA_CLAHE = True           # Realce de contraste adaptativo (CLAHE)
CAMARA_DENOISE = True         # Bilateral filter anti-ruido
CAMARA_SHARPENING = True      # Kernel sharpening para bordes de dedos
CAMARA_AWB = True             # Auto White Balance (Gray World)
CAMARA_FPS_TARGET = 30        # FPS objetivo de captura
CAMARA_RESOLUCION = (1280, 720)  # Resolucón preferida

# ─────────────────────────────────────────────
# SINTESIS DE VOZ LOCAL (TTS)
# ─────────────────────────────────────────────

TTS_RATE = _env_int("TTS_RATE", 160)
TTS_COOLDOWN_SEG = _env_float("TTS_COOLDOWN_SEG", 1.5)

# ─────────────────────────────────────────────
# SERVIDOR WEB Y WEBSOCKETS
# ─────────────────────────────────────────────

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = _env_int("SERVER_PORT", 8000)


def resumen():
    """Imprime un resumen de la configuración activa."""
    print("=" * 60)
    print("  CONFIGURACION LSC (SISTEMA BASADO EN VECTORES Y CUADRANTES)")
    print("=" * 60)
    print(f"  Directorio raíz:         {PROJECT_ROOT}")
    print(f"  Base Vectorial:          {BASE_VECTORES_PATH}")
    print(f"  Catálogo Clases:         {CLASES_LSC_PATH}")
    print(f"  Umbral Similitud Mín:    {UMBRAL_SIMILITUD * 100:.0f}%")
    print(f"  Frames Estabilidad:      {FRAMES_ESTABLES}")
    print(f"  Silencio para Oración:   {SEGUNDOS_SILENCIO}s")
    print(f"  Cámara Índice:           {CAMERA_INDEX}")
    print(f"  Servidor Web/WS:         http://{SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)


if __name__ == "__main__":
    resumen()
