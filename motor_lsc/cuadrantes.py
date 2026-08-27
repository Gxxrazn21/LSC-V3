"""
=============================================================
SEGMENTACION ESPACIAL POR CUADRANTES Y SUB-ZONAS (SIGNING SPACE)
Lengua de Señas Colombiana (LSC)
=============================================================
En LSC, el punto de articulación (ubicación espacial de las manos
respecto a la cabeza, cuello, hombros y torso) es fundamental
para desambiguar señas que comparten configuraciones manuales idénticas.

Zonas Anatómicas:
1. CABEZA_ROSTRO: Frente, sienes, mejillas, orejas (ej: HOLA, SABER, PENSAR).
2. CUELLO_GARGANTA: Mandíbula, garganta, cuello (ej: LICOR, AGUA, SED).
3. PECHO_TORSO: Esternón, clavículas, pecho central (ej: YO, NOMBRE, GUSTAR, SENTIR).
4. ESPACIO_CENTRAL: Frente al torso centrado a dos manos (ej: APOYAR, AYUDAR, BAÑO, BIENVENIDO).
5. ESPACIO_LATERAL: Al frente lateral a media altura (ej: Alfabeto A-Z, Números 1-10).
6. LATERAL_BAJO: Costados o posición baja de reposo.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Any
import numpy as np


class CuadranteEspacial(str, Enum):
    CABEZA_ROSTRO = "CABEZA_ROSTRO"
    CUELLO_GARGANTA = "CUELLO_GARGANTA"
    PECHO_TORSO = "PECHO_TORSO"
    ESPACIO_CENTRAL = "ESPACIO_CENTRAL"
    ESPACIO_LATERAL = "ESPACIO_LATERAL"
    LATERAL_BAJO = "LATERAL_BAJO"
    DESCONOCIDO = "DESCONOCIDO"


# Colores BGR para visualización en OpenCV
COLORES_CUADRANTE = {
    CuadranteEspacial.CABEZA_ROSTRO: (0, 165, 255),       # Naranja brillante
    CuadranteEspacial.CUELLO_GARGANTA: (0, 220, 255),     # Amarillo ámbar
    CuadranteEspacial.PECHO_TORSO: (255, 105, 180),       # Rosa/Púrpura
    CuadranteEspacial.ESPACIO_CENTRAL: (50, 205, 50),     # Verde esmeralda
    CuadranteEspacial.ESPACIO_LATERAL: (255, 191, 0),     # Azul cielo
    CuadranteEspacial.LATERAL_BAJO: (150, 150, 150),      # Gris neutro
    CuadranteEspacial.DESCONOCIDO: (80, 80, 80),
}


def clasificar_cuadrante(
    muneca_coords: Tuple[float, float, float],
    pose_landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None,
    alto_frame: int = 480,
    ancho_frame: int = 640,
) -> Tuple[CuadranteEspacial, Dict[str, float]]:
    """
    Determina a qué zona anatómica del espacio de señación pertenece la mano.
    
    Args:
        muneca_coords: (x, y, z) de la muñeca en coordenadas normalizadas [0.0, 1.0].
        pose_landmarks: Diccionario con 'nariz', 'hombro_izq', 'hombro_der'.
        alto_frame: Alto en píxeles.
        ancho_frame: Ancho en píxeles.
        
    Returns:
        (CuadranteEspacial, metricas_relativas)
    """
    wx, wy, wz = muneca_coords

    # 1. Modo Anclaje Corporal Multimodal (con Pose)
    if pose_landmarks and "hombro_izq" in pose_landmarks and "hombro_der" in pose_landmarks:
        h_izq = pose_landmarks["hombro_izq"]
        h_der = pose_landmarks["hombro_der"]
        
        cx_hombros = (h_izq[0] + h_der[0]) / 2.0
        cy_hombros = (h_izq[1] + h_der[1]) / 2.0
        ancho_hombros = max(abs(h_izq[0] - h_der[0]), 0.08)

        cy_nariz = pose_landmarks["nariz"][1] if "nariz" in pose_landmarks else (cy_hombros - ancho_hombros * 0.7)
        dy_nariz = wy - cy_nariz
        dy_hombros = wy - cy_hombros
        dx_centro = wx - cx_hombros

        metricas = {
            "dx_centro": float(dx_centro),
            "dy_hombros": float(dy_hombros),
            "ancho_hombros": float(ancho_hombros),
        }

        # A. Cabeza y Rostro (arriba del mentón / a nivel de la nariz y ojos)
        if wy <= (cy_hombros - 0.25 * ancho_hombros):
            return CuadranteEspacial.CABEZA_ROSTRO, metricas

        # B. Cuello y Garganta (entre mentón y línea de clavículas)
        if (cy_hombros - 0.25 * ancho_hombros) < wy <= (cy_hombros + 0.08 * ancho_hombros):
            if abs(dx_centro) <= ancho_hombros * 0.55:
                return CuadranteEspacial.CUELLO_GARGANTA, metricas
            else:
                return CuadranteEspacial.CABEZA_ROSTRO, metricas

        # C. Pecho y Esternón (a nivel del torso central)
        if (cy_hombros + 0.08 * ancho_hombros) < wy <= (cy_hombros + 0.95 * ancho_hombros):
            if abs(dx_centro) <= ancho_hombros * 0.45:
                return CuadranteEspacial.PECHO_TORSO, metricas
            elif abs(dx_centro) <= ancho_hombros * 1.0:
                return CuadranteEspacial.ESPACIO_CENTRAL, metricas
            else:
                return CuadranteEspacial.ESPACIO_LATERAL, metricas

        # D. Espacio Neutro Frente al Cuerpo (Media Altura)
        if wy <= (cy_hombros + 1.4 * ancho_hombros):
            if abs(dx_centro) <= ancho_hombros * 0.65:
                return CuadranteEspacial.ESPACIO_CENTRAL, metricas
            else:
                return CuadranteEspacial.ESPACIO_LATERAL, metricas

        # E. Reposo / Lateral Bajo
        return CuadranteEspacial.LATERAL_BAJO, metricas

    # 2. Modo Fallback (Sin Pose - Basado en Cuadrícula Relativa Normalizada)
    metricas = {"wx": wx, "wy": wy, "wz": wz}

    if wy < 0.32:
        return CuadranteEspacial.CABEZA_ROSTRO, metricas
    elif 0.32 <= wy < 0.45:
        if 0.35 <= wx <= 0.65:
            return CuadranteEspacial.CUELLO_GARGANTA, metricas
        else:
            return CuadranteEspacial.CABEZA_ROSTRO, metricas
    elif 0.45 <= wy < 0.68:
        if 0.38 <= wx <= 0.62:
            return CuadranteEspacial.PECHO_TORSO, metricas
        elif 0.22 <= wx <= 0.78:
            return CuadranteEspacial.ESPACIO_CENTRAL, metricas
        else:
            return CuadranteEspacial.ESPACIO_LATERAL, metricas
    else:
        return CuadranteEspacial.LATERAL_BAJO, metricas
