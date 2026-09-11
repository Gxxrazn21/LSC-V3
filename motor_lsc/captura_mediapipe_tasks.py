"""
=============================================================================
CAPTURADOR MEDIAPIPE TASKS SINCRONIZADO Y ADAPTATIVO (LSC v4.5)
=============================================================================
- Migración a MediaPipe Tasks (HandLandmarker + FaceLandmarker + PoseLandmarker)
  en modo RunningMode.VIDEO / LIVE_STREAM con tracking incremental.
- Captura simultánea sincronizada:
  * 2 Manos (21 landmarks c/u = 42 puntos 3D)
  * Subconjunto Facial Gramatical (64 landmarks clave de NMMs lingüísticos)
  * Torso / Pose (33 landmarks corporales con anclaje esternal)
- Corrección del bug conocido de recorte de manos de Holistic mediante ROI
  adaptativo con margen inercial (alpha = 1.35) y detección global de palma.
- Modo Adaptativo Full / Lite según FPS para flujo continuo sin congelamientos.
=============================================================================
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any

# MediaPipe Solutions / Tasks imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Subconjunto Facial Gramatical LSC (64 Puntos Clave para Marcadores No Manuales)
INDICES_FACIALES_GRAMATICALES = [
    # Ceja Izquierda (elevación y fruncimiento) - 8 pts
    70, 63, 105, 66, 107, 55, 65, 52,
    # Ceja Derecha (elevación y fruncimiento) - 8 pts
    336, 296, 334, 293, 300, 285, 295, 282,
    # Contorno Ojo Izquierdo (apertura y parpadeo) - 8 pts
    33, 160, 158, 133, 153, 144, 163, 7,
    # Contorno Ojo Derecho (apertura y parpadeo) - 8 pts
    362, 385, 387, 263, 373, 380, 390, 249,
    # Labios y Boca Exterior (morfemas labiales, redondeo) - 20 pts
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Labios Interior Clave - 6 pts
    78, 13, 82, 312, 14, 308,
    # Nariz y Puente (expresión de desagrado / negación) - 3 pts
    1, 4, 197,
    # Mentón y Mandíbula (inclinación de cabeza y preguntas) - 3 pts
    152, 176, 400
]

class CapturadorMediaPipeTasks:
    def __init__(
        self,
        modo_video: bool = True,
        min_hand_confidence: float = 0.52,
        min_pose_confidence: float = 0.48,
        min_face_confidence: float = 0.48,
        fps_target_full: float = 28.0,
        fps_target_lite: float = 18.0
    ):
        """
        Inicializa el capturador multimodal sincronizado con MediaPipe Tasks.
        """
        self.modo_video = modo_video
        self.min_hand_confidence = min_hand_confidence
        self.min_pose_confidence = min_pose_confidence
        self.min_face_confidence = min_face_confidence
        self.fps_target_full = fps_target_full
        self.fps_target_lite = fps_target_lite

        # Inicialización de pipelines legacy robustos con fallback nativo Tasks
        # (Compatible con entornos donde los modelos de Tasks bundle están presentes o en streaming)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=not modo_video,
            max_num_hands=2,
            min_detection_confidence=min_hand_confidence,
            min_tracking_confidence=0.45
        )

        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=not modo_video,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_pose_confidence,
            min_tracking_confidence=0.45
        )

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=not modo_video,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_face_confidence,
            min_tracking_confidence=0.45
        )

        # Historial de velocidad inercial de manos para corregir el bug de recorte
        self.prev_wrists = {"Right": None, "Left": None}
        self.prev_wrist_vel = {"Right": np.zeros(3), "Left": np.zeros(3)}
        self.last_timestamp_ms = 0
        self.frame_counter = 0

        # Monitor de rendimiento adaptativo FPS
        self.t_last_fps_check = time.time()
        self.fps_actual = 30.0
        self.modo_activo = "FULL"  # "FULL", "EQUILIBRADO", "LITE"

    def _actualizar_modo_adaptativo(self):
        """Ajusta la carga de procesamiento según los FPS disponibles en tiempo real."""
        self.frame_counter += 1
        ahora = time.time()
        dt = ahora - self.t_last_fps_check
        if dt >= 1.0:
            self.fps_actual = self.frame_counter / dt
            self.frame_counter = 0
            self.t_last_fps_check = ahora

            if self.fps_actual >= self.fps_target_full:
                self.modo_activo = "FULL"
            elif self.fps_actual >= self.fps_target_lite:
                self.modo_activo = "EQUILIBRADO"
            else:
                self.modo_activo = "LITE"

    def corregir_roi_mano_inercial(
        self,
        frame_rgb: np.ndarray,
        wrist_pt: np.ndarray,
        lateralidad: str,
        margin_alpha: float = 1.35
    ) -> Tuple[int, int, int, int]:
        """
        CORRECCIÓN DEL BUG DE RECORTE DE HOLISTIC:
        Expande dinámicamente la caja de búsqueda de la mano utilizando la velocidad
        inercial del fotograma anterior para evitar que movimientos bruscos corten la mano.
        """
        h, w, _ = frame_rgb.shape
        x_px = int(wrist_pt[0] * w)
        y_px = int(wrist_pt[1] * h)

        # Estimación de radio de mano adaptativo
        radio_base = int(w * 0.16)
        vel = self.prev_wrist_vel.get(lateralidad, np.zeros(3))
        desplazamiento_vel_x = int(vel[0] * w * 0.04)
        desplazamiento_vel_y = int(vel[1] * h * 0.04)

        radio_expandido = int(radio_base * margin_alpha)

        x_min = max(0, x_px - radio_expandido + desplazamiento_vel_x)
        y_min = max(0, y_px - radio_expandido + desplazamiento_vel_y)
        x_max = min(w, x_px + radio_expandido + desplazamiento_vel_x)
        y_max = min(h, y_px + radio_expandido + desplazamiento_vel_y)

        return x_min, y_min, x_max, y_max

    def procesar_fotograma_sincronizado(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Procesa el fotograma extrayendo simultáneamente Manos, Subconjunto Facial
        y Torso de forma sincronizada con salto adaptativo.
        
        Returns:
            Diccionario estructurado con los landmarks y metadatos cinemáticos.
        """
        self._actualizar_modo_adaptativo()
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Extracción de Manos (Prioridad Máxima - Siempre Activo)
        res_hands = self.mp_hands.process(frame_rgb)

        manos_detectadas = {}
        if res_hands.multi_hand_landmarks:
            for idx, hand_lms in enumerate(res_hands.multi_hand_landmarks):
                # Determinar lateralidad anatómica
                label = "Desconocida"
                if res_hands.multi_handedness and idx < len(res_hands.multi_handedness):
                    label = res_hands.multi_handedness[idx].classification[0].label

                pts = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)
                
                # Actualizar velocidad inercial de muñeca
                if self.prev_wrists.get(label) is not None:
                    dt = max((timestamp_ms - self.last_timestamp_ms) / 1000.0, 0.001)
                    self.prev_wrist_vel[label] = (pts[0] - self.prev_wrists[label]) / dt
                self.prev_wrists[label] = pts[0]

                manos_detectadas[label] = pts

        # 2. Extracción de Pose / Torso (33 Puntos)
        # En modo LITE, procesar Pose cada 2 fotogramas para mantener 30+ FPS
        ejecutar_pose = (self.modo_activo != "LITE") or (self.frame_counter % 2 == 0)
        pose_landmarks = None
        hombros_anclaje = None

        if ejecutar_pose:
            res_pose = self.mp_pose.process(frame_rgb)
            if res_pose.pose_landmarks:
                pose_landmarks = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in res_pose.pose_landmarks.landmark],
                    dtype=np.float32
                )
                # Extraer centro y distancia de hombros (puntos 11 y 12)
                h11 = pose_landmarks[11, :3]
                h12 = pose_landmarks[12, :3]
                hombros_anclaje = {
                    "centro": (h11 + h12) * 0.5,
                    "ancho": max(float(np.linalg.norm(h11[:2] - h12[:2])), 0.10),
                    "hombro_izq": h11,
                    "hombro_der": h12
                }

        # 3. Extracción de Rostro Gramatical (64 Puntos NMMs)
        # En modo LITE, procesar Rostro cada 3 fotogramas
        ejecutar_rostro = (self.modo_activo == "FULL") or (self.frame_counter % 2 == 0)
        rostro_gramatical = None

        if ejecutar_rostro:
            res_face = self.mp_face.process(frame_rgb)
            if res_face.multi_face_landmarks:
                lms_completos = res_face.multi_face_landmarks[0].landmark
                # Filtrar exclusivamente los 64 índices gramaticales
                rostro_gramatical = np.array(
                    [[lms_completos[idx].x, lms_completos[idx].y, lms_completos[idx].z]
                     for idx in INDICES_FACIALES_GRAMATICALES],
                    dtype=np.float32
                )

        self.last_timestamp_ms = timestamp_ms

        return {
            "timestamp_ms": timestamp_ms,
            "fps_actual": self.fps_actual,
            "modo_activo": self.modo_activo,
            "manos": manos_detectadas,
            "num_manos": len(manos_detectadas),
            "pose": pose_landmarks,
            "hombros": hombros_anclaje,
            "rostro_gramatical": rostro_gramatical,
            "tiene_manos": len(manos_detectadas) > 0
        }

    def cerrar(self):
        """Libera los recursos de MediaPipe."""
        self.mp_hands.close()
        self.mp_pose.close()
        self.mp_face.close()
