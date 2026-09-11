"""
=============================================================
EXTRACTOR ARTICULATORIO v3.0 — LSC (Lengua de Señas Colombiana)
=============================================================
Pipeline cinemático con mejora de cámara integrada:
1. Preprocesamiento Óptico Mejorado (v3.0):
   - Bilateral Filter: preserva bordes de dedos, suprime ruido de sensor.
   - CLAHE adaptativo en canal L (espacio LAB): realza contraste en iluminación baja.
   - Sharpening kernel 3×3: acentúa articulaciones y contornos digitales.
   - Auto White Balance (Gray World Assumption): normaliza dominante de color.
2. Estabilización temporal OneEuroFilter 3D (elimina jitter de cámara).
3. Coordenadas 3D canónicas ortonormales en la base de la palma (63 dims).
4. Estados continuos de extensión digital (5 dims).
5. Cosenos de ángulos articulares MCP/PIP/DIP (15 dims).
6. Distancias interdigitales entre puntas (10 dims).
7. Distancias de puntas al centro de la palma (5 dims).
8. Métricas de contacto y proximidad del pulgar (4 dims).
9. Vector normal de orientación espacial de la palma (3 dims).
Descriptor total: 105 dimensiones perfectamente balanceadas.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────
# 1. FILTRO TEMPORAL ADAPTATIVO ONE-EURO (ZERO-LAG / ANTI-JITTER)
# ─────────────────────────────────────────────────────────────

class OneEuroFilter3D:
    """
    Filtro paso-bajo adaptativo de primer orden para señales 3D (landmarks).
    - A bajas velocidades (mano quieta): filtra agresivamente para eliminar temblores/ruido.
    - A altas velocidades (movimiento rápido): aumenta la frecuencia de corte para eliminar retraso (lag=0).
    """
    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.03,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: np.ndarray) -> np.ndarray:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def filter(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = time.time()

        if self.t_prev is None or self.x_prev is None:
            self.x_prev = np.copy(x)
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return np.copy(x)

        t_e = max(t - self.t_prev, 1e-4)

        # 1. Estimar velocidad (derivada filtrada)
        a_d = self._smoothing_factor(t_e, np.full_like(x, self.d_cutoff))
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # 2. Frecuencia de corte adaptativa basada en velocidad instantánea
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        
        # 3. Filtrado de la señal
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = np.copy(x_hat)
        self.dx_prev = np.copy(dx_hat)
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO OPTICO MEJORADO v3.0 (MEJORA DE CÁMARA)
# ─────────────────────────────────────────────────────────────

# Kernel de sharpening adaptativo para realce de articulaciones digitales
_KERNEL_SHARPEN = np.array([
    [ 0, -0.5,  0],
    [-0.5,  3, -0.5],
    [ 0, -0.5,  0],
], dtype=np.float32)


def auto_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Corrección de balance de blancos por Gray World Assumption.
    Normaliza la dominante de color para estabilizar la detección de piel
    en diferentes temperaturas de luz (LED fría, fluorescente, luz solar).
    """
    result = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    result[:, :, 2] = result[:, :, 2] - (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def preprocesar_imagen_anti_ruido(
    frame_bgr: np.ndarray,
    aplicar_clahe: bool = True,
    suavizar_ruido: bool = True,
    aplicar_sharpening: bool = True,
    aplicar_awb: bool = True,
) -> np.ndarray:
    """
    Pipeline de mejora de cámara v3.0 para captura de señas LSC.
    Optimiza la imagen para condiciones de iluminación variable y cámaras
    de baja calidad (webcams integradas, cámaras de portátil).

    Pipeline:
      1. Auto White Balance  — estabiliza dominante de color
      2. Bilateral Filter    — preserva bordes, suprime ruido de sensor
      3. CLAHE en LAB        — realza contraste en iluminación tenue
      4. Sharpening 3×3      — acentúa contornos de dedos para MediaPipe
    """
    out = frame_bgr

    # Paso 1: Auto White Balance (Gray World)
    if aplicar_awb:
        out = auto_white_balance(out)

    # Paso 2: Bilateral Filter — suprime ruido preservando bordes de articulaciones
    if suavizar_ruido:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=45, sigmaSpace=45)

    # Paso 3: CLAHE adaptativo en luminancia LAB
    if aplicar_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Paso 4: Sharpening adaptativo para realzar contornos de dedos
    if aplicar_sharpening:
        out = cv2.filter2D(out, -1, _KERNEL_SHARPEN)
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


# ─────────────────────────────────────────────────────────────
# 3. EXTRACTOR ARTICULATORIO DE ALTA PRECISION (MEDIAPIPE HOLISTIC)
# ─────────────────────────────────────────────────────────────

class ExtractorLandmarks:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.55,
        umbral_velocidad_estabilidad: float = 0.035,
        usar_filtro_temporal: bool = True,
        usar_denoise_imagen: bool = True,
    ):
        """
        Inicializa MediaPipe Holistic para detección unificada de manos + cuerpo.
        Holistic realiza una sola inferencia que comparte contexto entre pose, manos y cara,
        lo que produce landmarks de mano más estables y consistentes que Hands aislado.
        """
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Holistic: inferencia unificada pose + manos con contexto cruzado
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Fallback: MediaPipe Hands para detección cuando Holistic no detecta manos
        self.hands_fallback = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.umbral_velocidad_estabilidad = umbral_velocidad_estabilidad
        self.usar_filtro_temporal = usar_filtro_temporal
        self.usar_denoise_imagen = usar_denoise_imagen

        # Filtros OneEuro dedicados por mano y pose
        self.filtros_manos: Dict[str, OneEuroFilter3D] = {
            "Right": OneEuroFilter3D(min_cutoff=0.9, beta=0.04),
            "Left": OneEuroFilter3D(min_cutoff=0.9, beta=0.04),
        }
        self.filtro_pose = OneEuroFilter3D(min_cutoff=0.6, beta=0.02)
        
        self.prev_coords_manos: Dict[str, np.ndarray] = {}
        self.conteo_frames_perdidos: Dict[str, int] = {"Right": 0, "Left": 0}
        self.pose_wrist_prev: Dict[str, np.ndarray] = {}

    def _extraer_mano_holistic(
        self, hand_lms, label: str, alto: int, ancho: int, timestamp: float, pose_anchors: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Extrae datos de una mano detectada por Holistic o Hands fallback."""
        if hand_lms is None:
            return None

        coords_raw = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_lms.landmark):
            coords_raw[i] = [lm.x, lm.y, lm.z]

        # Aplicar filtro temporal OneEuro
        if self.usar_filtro_temporal:
            if label not in self.filtros_manos:
                self.filtros_manos[label] = OneEuroFilter3D(min_cutoff=0.9, beta=0.04)
            coords = self.filtros_manos[label].filter(coords_raw, timestamp)
        else:
            coords = coords_raw

        # Métricas cinéticas
        velocidad_desplazamiento = 0.0
        es_estable = True
        if label in self.prev_coords_manos:
            delta = np.linalg.norm(coords - self.prev_coords_manos[label], axis=1)
            velocidad_desplazamiento = float(np.mean(delta))
            es_estable = (velocidad_desplazamiento <= self.umbral_velocidad_estabilidad)

        self.prev_coords_manos[label] = coords.copy()

        # Extracción del vector articular canónico (105D) y multimodal con hombros (109D)
        vector_105d, ext_estados = self.extraer_descriptor_articular(coords)
        vector_109d, spatial_dict = self.extraer_descriptor_multimodal(coords, pose_anchors)
        muneca = tuple(coords[0])

        return {
            "tipo": label,
            "landmarks_raw": coords,
            "vector_normalizado": vector_109d,
            "vector_105d": vector_105d,
            "vector_109d": vector_109d,
            "spatial_coords": spatial_dict,
            "finger_extensions": ext_estados,
            "muneca": muneca,
            "velocidad": velocidad_desplazamiento,
            "es_estable": es_estable,
        }

    def procesar_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Procesa una imagen BGR con MediaPipe Holistic (inferencia unificada).
        Fallback a MediaPipe Hands si Holistic no detecta manos.
        """
        if timestamp is None:
            timestamp = time.time()

        alto, ancho, _ = frame_bgr.shape

        # 1. Preprocesamiento óptico anti-ruido (Bilateral + CLAHE)
        if self.usar_denoise_imagen:
            frame_limpio = preprocesar_imagen_anti_ruido(frame_bgr, aplicar_clahe=True)
        else:
            frame_limpio = frame_bgr

        frame_rgb = cv2.cvtColor(frame_limpio, cv2.COLOR_BGR2RGB)

        # 2. Inferencia principal: MediaPipe Holistic (pose + manos en un solo pase)
        res_holistic = self.holistic.process(frame_rgb)

        # 3. Filtrado de puntos Pose clave
        pose_anchors = {}
        if res_holistic.pose_landmarks:
            lm = res_holistic.pose_landmarks.landmark
            raw_pose_mat = np.array([
                [lm[0].x, lm[0].y, lm[0].z],    # Nariz
                [lm[11].x, lm[11].y, lm[11].z],  # Hombro Izq
                [lm[12].x, lm[12].y, lm[12].z],  # Hombro Der
            ], dtype=np.float32)

            if self.usar_filtro_temporal:
                pose_mat = self.filtro_pose.filter(raw_pose_mat, timestamp)
            else:
                pose_mat = raw_pose_mat

            pose_anchors["nariz"] = tuple(pose_mat[0])
            pose_anchors["hombro_izq"] = tuple(pose_mat[1])
            pose_anchors["hombro_der"] = tuple(pose_mat[2])
        else:
            self.filtro_pose.reset()

        # 4. Procesamiento de manos desde Holistic
        manos_info = []
        manos_detectadas = set()

        # Holistic devuelve right_hand_landmarks y left_hand_landmarks
        # Nota: en Holistic, "Right" y "Left" están desde la perspectiva del usuario (espejado)
        for label, hand_lms in [("Right", res_holistic.right_hand_landmarks),
                                 ("Left", res_holistic.left_hand_landmarks)]:
            info = self._extraer_mano_holistic(hand_lms, label, alto, ancho, timestamp, pose_anchors)
            if info is not None:
                manos_info.append(info)
                manos_detectadas.add(label)

        # 5. Fallback a MediaPipe Hands si Holistic no detectó manos
        if not manos_info:
            res_hands = self.hands_fallback.process(frame_rgb)
            if res_hands.multi_hand_landmarks:
                for idx, hand_lms in enumerate(res_hands.multi_hand_landmarks):
                    label = "Right"
                    if res_hands.multi_handedness and len(res_hands.multi_handedness) > idx:
                        label = res_hands.multi_handedness[idx].classification[0].label
                    info = self._extraer_mano_holistic(hand_lms, label, alto, ancho, timestamp, pose_anchors)
                    if info is not None:
                        manos_info.append(info)
                        manos_detectadas.add(label)

        # 6. Recuperación Cinemática por Pose en Movimientos Bruscos
        # Si la mano se movió muy rápido y los detectores globales fallaron:
        if not manos_info and res_holistic.pose_landmarks:
            plm = res_holistic.pose_landmarks.landmark
            for label, w_idx in [("Right", 16), ("Left", 15)]:
                w_lm = plm[w_idx]
                vis = getattr(w_lm, "visibility", 1.0)
                if vis > 0.35:
                    w_px = int(w_lm.x * ancho)
                    w_py = int(w_lm.y * alto)

                    # Intentar crop centrado en la muñeca (alta densidad)
                    tam_crop = int(min(alto, ancho) * 0.40)
                    x1 = max(0, w_px - tam_crop // 2)
                    y1 = max(0, w_py - tam_crop // 2)
                    x2 = min(ancho, x1 + tam_crop)
                    y2 = min(alto, y1 + tam_crop)

                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.shape[0] > 40 and crop.shape[1] > 40:
                        res_crop = self.hands_fallback.process(crop)
                        if res_crop.multi_hand_landmarks:
                            hand_crop = res_crop.multi_hand_landmarks[0]
                            coords_full = np.zeros((21, 3), dtype=np.float32)
                            crop_h, crop_w = crop.shape[:2]
                            for idx_pt, pt in enumerate(hand_crop.landmark):
                                coords_full[idx_pt] = [
                                    (x1 + pt.x * crop_w) / ancho,
                                    (y1 + pt.y * crop_h) / alto,
                                    pt.z,
                                ]
                            vec_105d, ext_est = self.extraer_descriptor_articular(coords_full)
                            vec_109d, spatial_dict = self.extraer_descriptor_multimodal(coords_full, pose_anchors)
                            manos_info.append({
                                "tipo": label,
                                "landmarks_raw": coords_full,
                                "vector_normalizado": vec_109d,
                                "vector_105d": vec_105d,
                                "vector_109d": vec_109d,
                                "spatial_coords": spatial_dict,
                                "finger_extensions": ext_est,
                                "muneca": tuple(coords_full[0]),
                                "velocidad": 0.05,
                                "es_estable": False,  # Flag de movimiento rápido activo
                                "recuperado_inercial": True,
                            })
                            manos_detectadas.add(label)

        # 7. Gestión de Histéresis Anti-Pérdida (No resetear filtros bruscamente)
        for lbl in list(self.filtros_manos.keys()):
            if lbl not in manos_detectadas:
                self.conteo_frames_perdidos[lbl] = self.conteo_frames_perdidos.get(lbl, 0) + 1
                if self.conteo_frames_perdidos[lbl] >= 6:
                    self.filtros_manos[lbl].reset()
                    if lbl in self.prev_coords_manos:
                        del self.prev_coords_manos[lbl]
            else:
                self.conteo_frames_perdidos[lbl] = 0

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
        Calcula el descriptor canónico 3D ortonormal y articular de 105 dimensiones.
        
        coords: shape (21, 3)
        Retorna:
            (vector_105d, [ext_pulgar, ext_indice, ext_medio, ext_anular, ext_menique])
        """
        p0 = coords[0]     # Muñeca
        p5 = coords[5]     # Nudillo Índice
        p9 = coords[9]     # Nudillo Medio
        p17 = coords[17]   # Nudillo Meñique

        # 1. Base Ortonormal Canónica de la Palma
        escala_palma = float(np.linalg.norm(p9 - p0))
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

        # 2. Estados Cinemáticos de Extensión de los 5 Dedos (5 dims)
        # Alcance 3D directo vs longitud de segmentos anatómicos (100% Invariante 3D)
        tips = [4, 8, 12, 16, 20]
        ext_dedos = []

        # Pulgar (Punta 4, IP 3, MCP 2, CMC 1)
        len_thumb_bones = float(np.linalg.norm(coords[4] - coords[3]) + np.linalg.norm(coords[3] - coords[2]) + 1e-5)
        d_thumb_mcp = float(np.linalg.norm(coords[4] - coords[2]))
        ratio_thumb = d_thumb_mcp / len_thumb_bones
        d_4_17 = float(np.linalg.norm(coords[4] - coords[17]))
        d_5_17 = float(np.linalg.norm(coords[5] - coords[17]) + 1e-5)
        ratio_apertura = d_4_17 / d_5_17
        ext_pulgar = float(np.clip((ratio_thumb - 0.60) / 0.35 * 0.5 + (ratio_apertura - 0.70) / 0.40 * 0.5, 0.0, 1.0))
        ext_dedos.append(ext_pulgar)

        # 4 Dedos (Índice 8, Medio 12, Anular 16, Meñique 20)
        dedos_indices = [(8, 7, 6, 5), (12, 11, 10, 9), (16, 15, 14, 13), (20, 19, 18, 17)]
        for tip, dip, pip, mcp in dedos_indices:
            len_bones = float(
                np.linalg.norm(coords[tip] - coords[dip]) + 
                np.linalg.norm(coords[dip] - coords[pip]) + 
                np.linalg.norm(coords[pip] - coords[mcp]) + 1e-5
            )
            d_tip_mcp = float(np.linalg.norm(coords[tip] - coords[mcp]))
            ratio_recto = d_tip_mcp / len_bones
            ext_val = float(np.clip((ratio_recto - 0.42) / 0.38, 0.0, 1.0))
            ext_dedos.append(ext_val)

        # 3. Coseno de Ángulos Articulares MCP, PIP, DIP (15 dims)
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
        distancias_tips = []
        for i in range(5):
            for j in range(i + 1, 5):
                d_ij = float(np.linalg.norm(p_canon[tips[i]] - p_canon[tips[j]]))
                distancias_tips.append(d_ij)

        # 5. Distancia de Puntas al Centro de la Palma (5 dims)
        centro_palma = 0.5 * (p_canon[0] + p_canon[9])
        dist_centro = [float(np.linalg.norm(p_canon[t] - centro_palma)) for t in tips]

        # 6. Métricas de Contacto y Proximidad del Pulgar (4 dims)
        # Desambigua configuraciones de puño cerrado (A, S, T, M, N, E) y anillas (F, 9, O)
        d_p4_p8 = float(np.linalg.norm(p_canon[4] - p_canon[8]))   # Pulgar a Índice
        d_p4_p6 = float(np.linalg.norm(p_canon[4] - p_canon[6]))   # Pulgar a PIP Índice (T / S)
        d_p4_p10 = float(np.linalg.norm(p_canon[4] - p_canon[10])) # Pulgar a PIP Medio (N)
        d_p4_p14 = float(np.linalg.norm(p_canon[4] - p_canon[14])) # Pulgar a PIP Anular (M)
        contacto_pulgar = [d_p4_p8, d_p4_p6, d_p4_p10, d_p4_p14]

        # 7. Vector Normal de Orientación de la Palma (3 dims)
        palm_norm = [float(uz[0]), float(uz[1]), float(uz[2])]

        # Concatenar vector final perfectamente balanceado (63 + 5 + 15 + 10 + 5 + 4 + 3 = 105 dims)
        descriptor_total = np.concatenate([
            p_canon.flatten() * 0.55,                          # Coordenadas canónicas escaladas
            np.array(ext_dedos, dtype=np.float32) * 3.2,       # Extensión de dedos reforzada
            np.array(angulos, dtype=np.float32) * 1.2,         # Ángulos articulares
            np.array(distancias_tips, dtype=np.float32) * 2.0, # Separación entre puntas
            np.array(dist_centro, dtype=np.float32) * 1.2,     # Centro palma
            np.array(contacto_pulgar, dtype=np.float32) * 1.8, # Matriz de contacto pulgar
            np.array(palm_norm, dtype=np.float32) * 1.4,       # Normal de orientación
        ]).astype(np.float32)

        return descriptor_total, ext_dedos

    @classmethod
    def extraer_descriptor_multimodal(
        cls,
        coords: np.ndarray,
        pose_anchors: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calcula el descriptor multimodal de 109 dimensiones:
        - 105 dims: cinemática articular canónica de la mano (invariante a escala y rotación).
        - 4 dims: anclaje espacial corporal relativo a hombros de Pose (invariante a censura facial).
        """
        vec_105d, _ = cls.extraer_descriptor_articular(coords)
        p0 = coords[0]  # Muñeca [x, y, z]

        if pose_anchors and "hombro_izq" in pose_anchors and "hombro_der" in pose_anchors:
            h_izq = np.array(pose_anchors["hombro_izq"], dtype=np.float32)
            h_der = np.array(pose_anchors["hombro_der"], dtype=np.float32)
            c_h = 0.5 * (h_izq + h_der)
            w_h = max(float(np.linalg.norm(h_izq[:2] - h_der[:2])), 0.10)

            dx = float((p0[0] - c_h[0]) / w_h)
            dy = float((p0[1] - c_h[1]) / w_h)
            dz = float((p0[2] - c_h[2]) / w_h)
        else:
            # Fallback centrado si hombros no están en cuadro (plano cerrado)
            dx = float((p0[0] - 0.5) * 2.0)
            dy = float((p0[1] - 0.5) * 2.0)
            dz = float(p0[2])

        dist_cuerpo = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        # Ponderación calibrada para que las 4 coordenadas espaciales separen de forma inequívoca señas homónimas
        cuerpo_coords = np.array([dx, dy, dz, dist_cuerpo], dtype=np.float32) * 2.5
        vec_109d = np.concatenate([vec_105d, cuerpo_coords]).astype(np.float32)

        info_zonas = cls.extraer_zonas_corporales_pose(coords, pose_anchors)

        spatial_dict = {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dist": dist_cuerpo,
            "zona": info_zonas["zona"] if info_zonas else "PECHO_TORSO",
            "info_zonas": info_zonas
        }
        return vec_109d, spatial_dict

    @classmethod
    def extraer_zonas_corporales_pose(
        cls, coords: np.ndarray, pose_anchors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calcula el descriptor de ubicación fonológico (TAB) relativo al cuerpo
        a partir de landmarks de Pose y centroide de la mano.
        """
        p0 = coords[0]
        centroide_mano = np.mean(coords, axis=0)
        p_mano = 0.4 * p0 + 0.6 * centroide_mano

        if pose_anchors and "hombro_izq" in pose_anchors and "hombro_der" in pose_anchors:
            h_izq = np.array(pose_anchors["hombro_izq"], dtype=np.float32)
            h_der = np.array(pose_anchors["hombro_der"], dtype=np.float32)
            c_h = 0.5 * (h_izq + h_der)
            w_h = max(float(np.linalg.norm(h_izq[:2] - h_der[:2])), 0.10)
            c_nariz = np.array(pose_anchors["nariz"], dtype=np.float32) if "nariz" in pose_anchors else None
        else:
            c_h = np.array([0.50, 0.68, 0.0], dtype=np.float32)
            w_h = 0.38
            c_nariz = None

        c_cabeza = c_nariz if c_nariz is not None else np.array([c_h[0], c_h[1] - 0.70 * w_h, c_h[2]], dtype=np.float32)
        c_cuello = np.array([c_h[0], c_h[1] - 0.15 * w_h, c_h[2]], dtype=np.float32)
        c_pecho = np.array([c_h[0], c_h[1] + 0.35 * w_h, c_h[2]], dtype=np.float32)
        c_abdomen = np.array([c_h[0], c_h[1] + 0.85 * w_h, c_h[2]], dtype=np.float32)

        dx = float((p0[0] - c_h[0]) / w_h)
        dy = float((p0[1] - c_h[1]) / w_h)
        dz = float((p0[2] - c_h[2]) / w_h)

        if dy < -0.22:
            zona = "CABEZA_ROSTRO"
        elif -0.22 <= dy <= 0.12:
            zona = "CUELLO_GARGANTA" if abs(dx) <= 0.65 else "CABEZA_ROSTRO"
        elif 0.12 < dy <= 0.70:
            zona = "PECHO_TORSO" if abs(dx) <= 0.55 else "ESPACIO_CENTRAL"
        elif 0.70 < dy <= 1.45:
            zona = "ABDOMEN_CADERA"
        else:
            zona = "LATERAL_BAJO"

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "zona": zona,
            "dist_cabeza": float(np.linalg.norm(p_mano - c_cabeza) / w_h),
            "dist_cuello": float(np.linalg.norm(p_mano - c_cuello) / w_h),
            "dist_pecho": float(np.linalg.norm(p_mano - c_pecho) / w_h),
            "dist_abdomen": float(np.linalg.norm(p_mano - c_abdomen) / w_h),
        }

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
        Dibuja los esqueletos estabilizados y puntos de control en el frame.
        """
        out = frame_bgr.copy()
        alto, ancho, _ = out.shape

        for mano in resultado_procesamiento.get("manos", []):
            coords = mano["landmarks_raw"]
            es_estable = mano.get("es_estable", True)
            color_puntos = (0, 255, 128) if es_estable else (0, 180, 255)

            # Conexiones de los dedos
            conexiones = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
                (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
                (0, 9), (9, 10), (10, 11), (11, 12),   # Medio
                (0, 13), (13, 14), (14, 15), (15, 16), # Anular
                (0, 17), (17, 18), (18, 19), (19, 20), # Meñique
                (5, 9), (9, 13), (13, 17),             # Palma
            ]

            for a, b in conexiones:
                p_a = (int(coords[a, 0] * ancho), int(coords[a, 1] * alto))
                p_b = (int(coords[b, 0] * ancho), int(coords[b, 1] * alto))
                cv2.line(out, p_a, p_b, (40, 45, 55), 2, cv2.LINE_AA)

            # Dibujar puntos articulares
            for i in range(21):
                px = int(coords[i, 0] * ancho)
                py = int(coords[i, 1] * alto)
                cv2.circle(out, (px, py), 4, color_puntos, -1, cv2.LINE_AA)

            # Resaltar puntas de dedos con anillo de estado
            tips = [4, 8, 12, 16, 20]
            exts = mano.get("finger_extensions", [0]*5)
            for idx_t, t in enumerate(tips):
                tx = int(coords[t, 0] * ancho)
                ty = int(coords[t, 1] * alto)
                color_tip = (0, 255, 0) if exts[idx_t] > 0.55 else (0, 0, 255)
                cv2.circle(out, (tx, ty), 6, color_tip, 2, cv2.LINE_AA)

            # Ancla de muñeca
            wx = int(mano["muneca"][0] * ancho)
            wy = int(mano["muneca"][1] * alto)
            cv2.circle(out, (wx, wy), 9, color_cuadrante, 2, cv2.LINE_AA)

        return out

    def liberar(self):
        if hasattr(self, "holistic"):
            self.holistic.close()
        if hasattr(self, "hands_fallback"):
            self.hands_fallback.close()

