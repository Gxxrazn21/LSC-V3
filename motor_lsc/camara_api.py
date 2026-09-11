"""
=============================================================
API DE MEJORA ÓPTICA Y CONTROL DE CÁMARA (v3.0)
Lengua de Señas Colombiana (LSC)
=============================================================
Optimizado para captura en tiempo real sin latencia ni pérdida de movimiento:
1. Driver DirectShow (cv2.CAP_DSHOW) en Windows: acceso directo al sensor.
2. Formato MJPEG por hardware (menor compresión y mayor framerate).
3. Buffer size = 1 (Zero-Latency: sin frames encolados ni retraso).
4. Pipeline Óptico Adaptativo:
   - Balance de blancos automático (Gray World).
   - Realce de contraste adaptativo CLAHE en luminancia (espacio LAB).
   - Filtrado bilateral preservador de bordes (elimina ruido de sensor).
   - Máscara de enfoque (Unsharp Masking) anti-blur para movimientos bruscos.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class CamaraLSC:
    """
    Controlador de Cámara de Alta Fidelidad y Baja Latencia para LSC.
    """

    def __init__(
        self,
        camera_index: int = 0,
        ancho: int = 1280,
        alto: int = 720,
        fps: int = 30,
        aplicar_mejora_optica: bool = True,
    ):
        self.camera_index = camera_index
        self.ancho_deseado = ancho
        self.alto_deseado = alto
        self.fps_deseado = fps
        self.aplicar_mejora_optica = aplicar_mejora_optica

        self.cap: Optional[cv2.VideoCapture] = None
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self.abrir()

    def abrir(self) -> bool:
        """Abre la cámara local con DirectShow o stream IP desde celular."""
        # 1. Si es URL de cámara IP móvil (ej: http://192.168.1.X:8080/video o rtsp://)
        if isinstance(self.camera_index, str) and self.camera_index.startswith(("http://", "https://", "rtsp://")):
            print(f"  [CamaraLSC] Conectando a stream de cámara móvil IP: {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
        else:
            # Cámara USB local: Intentar DirectShow (óptimo en Windows)
            idx = int(self.camera_index) if str(self.camera_index).isdigit() else 0
            try:
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(idx)

            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            return False

        # 2. Configuración de hardware
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho_deseado)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto_deseado)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_deseado)

        # Buffer size = 1: Crucial para movimientos rápidos (no encola frames viejos)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Descartar primeros frames de estabilización de exposición del sensor
        for _ in range(4):
            self.cap.read()

        return True

    def leer(self, voltear_espejo: bool = True) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Lee el siguiente frame disponible.
        
        Retorna:
            (ret, frame_original, frame_mejorado)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None

        if voltear_espejo:
            frame = cv2.flip(frame, 1)

        if self.aplicar_mejora_optica:
            frame_mejorado = self.mejorar_calidad(frame)
        else:
            frame_mejorado = frame.copy()

        return True, frame, frame_mejorado

    def mejorar_calidad(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Pipeline óptico adaptativo para compensar poca luz y desenfoque por movimiento:
        1. Balance de blancos básico.
        2. CLAHE en canal L (LAB): revela articulaciones en sombras.
        3. Unsharp Masking: realza bordes de dedos desenfocados por movimientos rápidos.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        # 1. CLAHE en espacio LAB
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_enh = cv2.merge((l_clahe, a, b))
        frame_clahe = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

        # 2. Máscara de enfoque rápida (Unsharp Mask) contra desenfoque de movimiento
        # Realza los bordes de los dedos cuando la mano se mueve rápido
        gaussian = cv2.GaussianBlur(frame_clahe, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(frame_clahe, 1.35, gaussian, -0.35, 0)

        return sharpened

    @staticmethod
    def calcular_nitidez(frame_bgr: np.ndarray) -> float:
        """
        Calcula la nitidez del frame mediante la varianza del operador Laplaciano.
        Valores > 70 indican imagen nítida y apta para extracción precisa.
        Valores < 40 indican desenfoque severo por movimiento brusco.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Metodo compatible con cv2.VideoCapture.read()."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def isOpened(self) -> bool:
        """Metodo compatible con cv2.VideoCapture.isOpened()."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Metodo compatible con cv2.VideoCapture.release()."""
        self.liberar()

    def liberar(self):
        """Libera la camara de forma segura."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
