"""
=============================================================
EXTRACTOR ARTICULATORIO Y CANONICO DE LANDMARKS PARA LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Extrae un descriptor invariante de 101 dimensiones:
1. Coordenadas 3D Canónicas Ortonormales (63 dims):
   - Origen en muñeca (0).
   - Eje Y a lo largo de muñeca -> nudillo medio (9).
   - Eje Z perpendicular al plano de la palma (5 x 17).
   - Eje X ortogonal.
   - 100% Invariante a escala, traslación y rotación 3D de muñeca.
2. Estados continuos de extensión de los 5 dedos (5 dims: Pulgar, Índice, Medio, Anular, Meñique).
3. Coseno de ángulos de flexión articular en MCP, PIP, DIP (15 dims).
4. Distancias interdigitales entre puntas de los dedos (10 dims).
5. Distancias de puntas al centro de la palma (5 dims).
6. Vector normal de orientación de la palma (3 dims).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class ExtractorLandmarks:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.55,
        umbral_velocidad_estabilidad: float = 0.040,
    ):
        """
        Inicializa MediaPipe Hands y MediaPipe Pose con configuración de alta precisión.
        """
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.umbral_velocidad_estabilidad = umbral_velocidad_estabilidad
        self.prev_coords_manos: Dict[str, np.ndarray] = {}

    def procesar_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Procesa una imagen BGR y extrae el descriptor articular de 101 dimensiones.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alto, ancho, _ = frame_bgr.shape

        # Inferencia MediaPipe
        res_hands = self.hands.process(frame_rgb)
        res_pose = self.pose.process(frame_rgb)

        pose_anchors = {}
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            pose_anchors["nariz"] = (lm[0].x, lm[0].y, lm[0].z)
            pose_anchors["hombro_izq"] = (lm[11].x, lm[11].y, lm[11].z)
            pose_anchors["hombro_der"] = (lm[12].x, lm[12].y, lm[12].z)

        manos_info = []
        if res_hands.multi_hand_landmarks:
            for idx, hand_lms in enumerate(res_hands.multi_hand_landmarks):
                label = "Right"
                if res_hands.multi_handedness and len(res_hands.multi_handedness) > idx:
                    label = res_hands.multi_handedness[idx].classification[0].label

                coords = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(hand_lms.landmark):
                    coords[i] = [lm.x, lm.y, lm.z]

                # Métricas cinéticas
                velocidad_desplazamiento = 0.0
                es_estable = True
                if label in self.prev_coords_manos:
                    delta = np.linalg.norm(coords - self.prev_coords_manos[label], axis=1)
                    velocidad_desplazamiento = float(np.mean(delta))
                    es_estable = (velocidad_desplazamiento <= self.umbral_velocidad_estabilidad)

                self.prev_coords_manos[label] = coords.copy()

                # Extracción del vector articular canónico
                vector_articular, ext_estados = self.extraer_descriptor_articular(coords)
                muneca = tuple(coords[0])

                manos_info.append({
                    "tipo": label,
                    "landmarks_raw": coords,
                    "vector_normalizado": vector_articular,
                    "finger_extensions": ext_estados,
                    "muneca": muneca,
                    "velocidad": velocidad_desplazamiento,
                    "es_estable": es_estable,
                })
        else:
            self.prev_coords_manos.clear()

        return {
            "hay_manos": len(manos_info) > 0,
            "manos": manos_info,
            "pose_anchors": pose_anchors,
            "alto_frame": alto,
            "ancho_frame": ancho,
        }

    @classmethod
    def extraer_descriptor_articular(cls, coords: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Calcula el descriptor canónico 3D ortonormal y articular de 101 dimensiones.
        
        coords: shape (21, 3)
        Retorna:
            (vector_101d, [ext_pulgar, ext_indice, ext_medio, ext_anular, ext_menique])
        """
        p0 = coords[0]     # Muñeca
        p5 = coords[5]     # Nudillo Índice
        p9 = coords[9]     # Nudillo Medio
        p17 = coords[17]   # Nudillo Meñique

        # 1. Base Ortonormal Canónica de la Palma
        escala_palma = np.linalg.norm(p9 - p0)
        if escala_palma < 1e-4:
            escala_palma = 1.0

        uy = (p9 - p0) / escala_palma
        v_idx = p5 - p0
        v_pnk = p17 - p0
        uz = np.cross(v_idx, v_pnk)
        uz_norm = np.linalg.norm(uz)
        uz = uz / uz_norm if uz_norm > 1e-4 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ux = np.cross(uy, uz)
        ux_norm = np.linalg.norm(ux)
        ux = ux / ux_norm if ux_norm > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Proyección de los 21 puntos al sistema canónico (63 dims)
        rel = coords - p0
        p_canon = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            p_canon[i, 0] = np.dot(rel[i], ux) / escala_palma
            p_canon[i, 1] = np.dot(rel[i], uy) / escala_palma
            p_canon[i, 2] = np.dot(rel[i], uz) / escala_palma

        # 2. Estados de Extensión de los 5 Dedos (5 dims)
        ext_dedos = []
        # Pulgar (Punta 4 vs Nudillo 2 relativo a muñeca 0)
        d_p4_p2 = np.linalg.norm(coords[4] - coords[2])
        d_p2_p0 = np.linalg.norm(coords[2] - p0)
        ext_pulgar = float(np.clip((d_p4_p2 / (d_p2_p0 + 1e-4) - 0.45) / 0.40, 0.0, 1.0))
        ext_dedos.append(ext_pulgar)

        # 4 Dedos (Índice 8/6, Medio 12/10, Anular 16/14, Meñique 20/18)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            d_tip_p0 = np.linalg.norm(coords[tip] - p0)
            d_pip_p0 = np.linalg.norm(coords[pip] - p0)
            # Rango continuo [0.0 = totalmente cerrado/puño, 1.0 = totalmente extendido]
            ratio = d_tip_p0 / (d_pip_p0 + 1e-4)
            ext_val = float(np.clip((ratio - 0.95) / 0.25, 0.0, 1.0))
            ext_dedos.append(ext_val)

        # 3. Coseno de Ángulos Articulares (15 dims)
        articulaciones = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),      # Pulgar
            (0, 5, 6), (5, 6, 7), (6, 7, 8),      # Índice
            (0, 9, 10), (9, 10, 11), (10, 11, 12),# Medio
            (0, 13, 14), (13, 14, 15), (14, 15, 16),# Anular
            (0, 17, 18), (17, 18, 19), (18, 19, 20),# Meñique
        ]
        angulos = []
        for a, b, c in articulaciones:
            va = coords[a] - coords[b]
            vb = coords[c] - coords[b]
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a > 1e-4 and norm_b > 1e-4:
                cos_ang = float(np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0))
            else:
                cos_ang = 1.0
            angulos.append(cos_ang)

        # 4. Distancias Interdigitales entre Puntas (10 dims)
        tips = [4, 8, 12, 16, 20]
        distancias_tips = []
        for i in range(5):
            for j in range(i + 1, 5):
                d_ij = float(np.linalg.norm(p_canon[tips[i]] - p_canon[tips[j]]))
                distancias_tips.append(d_ij)

        # 5. Distancia de Puntas al Centro de la Palma (5 dims)
        centro_palma = 0.5 * (p_canon[0] + p_canon[9])
        dist_centro = [float(np.linalg.norm(p_canon[t] - centro_palma)) for t in tips]

        # 6. Vector Normal de la Palma (3 dims)
        palm_norm = [float(uz[0]), float(uz[1]), float(uz[2])]

        # Concatenar vector final (63 + 5 + 15 + 10 + 5 + 3 = 101 dims)
        descriptor_total = np.concatenate([
            p_canon.flatten(),
            np.array(ext_dedos, dtype=np.float32) * 2.0,      # Ponderación reforzada
            np.array(angulos, dtype=np.float32),
            np.array(distancias_tips, dtype=np.float32) * 1.5,
            np.array(dist_centro, dtype=np.float32),
            np.array(palm_norm, dtype=np.float32) * 0.8,
        ]).astype(np.float32)

        return descriptor_total, ext_dedos

    # Mantener compatibilidad con llamadas directas
    @classmethod
    def normalizar_mano(cls, coords: np.ndarray) -> np.ndarray:
        vec, _ = cls.extraer_descriptor_articular(coords)
        return vec

    def dibujar_overlays(
        self,
        frame_bgr: np.ndarray,
        resultado_procesamiento: Dict[str, Any],
        cuadrante_str: str = "",
        color_cuadrante: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Dibuja los esqueletos y puntos de control en el frame para visualización.
        """
        out = frame_bgr.copy()
        alto, ancho, _ = out.shape

        for mano in resultado_procesamiento.get("manos", []):
            coords = mano["landmarks_raw"]
            es_estable = mano.get("es_estable", True)
            color_puntos = (0, 255, 128) if es_estable else (0, 180, 255)

            # Dibujar puntos articulares
            for i in range(21):
                px = int(coords[i, 0] * ancho)
                py = int(coords[i, 1] * alto)
                cv2.circle(out, (px, py), 4, color_puntos, -1)

            # Resaltar puntas de dedos con anillo de estado
            tips = [4, 8, 12, 16, 20]
            exts = mano.get("finger_extensions", [0]*5)
            for idx_t, t in enumerate(tips):
                tx = int(coords[t, 0] * ancho)
                ty = int(coords[t, 1] * alto)
                color_tip = (0, 255, 0) if exts[idx_t] > 0.6 else (0, 0, 255)
                cv2.circle(out, (tx, ty), 6, color_tip, 2)

            # Ancla de muñeca
            wx = int(mano["muneca"][0] * ancho)
            wy = int(mano["muneca"][1] * alto)
            cv2.circle(out, (wx, wy), 9, color_cuadrante, 2)

        return out

    def liberar(self):
        if hasattr(self, "hands"):
            self.hands.close()
        if hasattr(self, "pose"):
            self.pose.close()
