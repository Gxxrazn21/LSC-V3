"""
=============================================================
EXPORTADOR DE POSES 3D DESDE BASE VECTORIAL LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Decodifica los vectores articulares entrenados (105 dimensiones)
de vuelta a poses 3D de 21 landmarks para renderizado en avatar.

Pipeline:
1. Carga la base de vectores NPZ con 4,842+ muestras.
2. Agrupa por clase/seña y calcula centroides representativos.
3. Decodifica cada centroide a 21 landmarks 3D + extensión de dedos + normal.
4. Genera keyframes JSON para animación en Three.js.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from .base_vectores import (
    BaseVectoresLSC,
    DIMS_COORDS,
    DIMS_EXT_DEDOS,
    DIMS_ANGULOS,
    DIMS_INTER_DIGITS,
    DIMS_DIST_CENTRO,
    DIMS_CONTACTO_PULGAR,
    DIMS_NORMAL_PALMA,
)
from .catalogo_senas import DICCIONARIO_EDUCATIVO_LSC


# ─── Pesos de ponderación del extractor (inversos para decodificación) ───
PESO_COORDS = 0.55
PESO_EXT_DEDOS = 3.2
PESO_ANGULOS = 1.2
PESO_INTER_DIGITS = 2.0
PESO_DIST_CENTRO = 1.2
PESO_CONTACTO_PULGAR = 1.8
PESO_NORMAL_PALMA = 1.4

# ─── Topología de MediaPipe Hand (21 landmarks) ───
# Índices de las falanges por dedo para reconstrucción articular
TOPOLOGIA_DEDOS = {
    "pulgar":  [0, 1, 2, 3, 4],
    "indice":  [0, 5, 6, 7, 8],
    "medio":   [0, 9, 10, 11, 12],
    "anular":  [0, 13, 14, 15, 16],
    "menique": [0, 17, 18, 19, 20],
}

# Nombres de los 21 landmarks de MediaPipe
NOMBRES_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Duraciones base por tipo de seña (ms)
DURACION_LETRA = 800       # Letras del alfabeto (rápidas)
DURACION_NUMERO = 900      # Números
DURACION_PALABRA = 1400    # Palabras/gestos normales
DURACION_DINAMICA = 1800   # Señas con movimiento corporal

# Pausa entre señas (ms)
PAUSA_ENTRE_SENAS = 400


class ExportadorPoses3D:
    """
    Decodifica vectores articulares de la base entrenada a poses 3D
    renderizables por un avatar Three.js.
    """

    def __init__(self, base_vectores: BaseVectoresLSC):
        self.base = base_vectores
        self._centroides: Dict[str, np.ndarray] = {}
        self._poses_cache: Dict[str, Dict[str, Any]] = {}
        self._compilar_centroides()

    def _compilar_centroides(self):
        """Calcula el centroide representativo por cada clase/seña."""
        if self.base.vectores is None or len(self.base.vectores) == 0:
            return

        # Agrupar índices por etiqueta
        por_clase: Dict[str, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.base.etiquetas):
            por_clase[lbl].append(idx)

        for lbl, indices in por_clase.items():
            vecs = self.base.vectores[indices]
            centroide = np.mean(vecs, axis=0)
            self._centroides[lbl] = centroide

        print(f"  [Exportador3D] Centroides calculados: {len(self._centroides)} señas.")

    def decodificar_vector_a_pose(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Decodifica un vector articular de 105 dims a una pose 3D legible.

        Returns:
            Dict con landmarks_3d (21×3), dedos_extension (5), normal_palma (3),
            angulos_flexion (15), distancias_inter (10), contacto_pulgar (4).
        """
        v = vector.flatten().astype(np.float64)
        D = len(v)

        # 1. Coordenadas 3D canónicas (21 puntos × 3) — quitar peso
        coords_raw = v[DIMS_COORDS] / PESO_COORDS if D >= 63 else np.zeros(63)
        landmarks_3d = coords_raw.reshape(21, 3).tolist()

        # 2. Extensión de dedos [0.0-1.0] — quitar peso
        ext_dedos = (v[DIMS_EXT_DEDOS] / PESO_EXT_DEDOS).clip(0, 1).tolist() if D >= 68 else [0.5] * 5

        # 3. Ángulos articulares (15 cosenos de flexión MCP/PIP/DIP)
        angulos = (v[DIMS_ANGULOS] / PESO_ANGULOS).tolist() if D >= 83 else [0.0] * 15

        # 4. Distancias interdigitales
        dist_inter = (v[DIMS_INTER_DIGITS] / PESO_INTER_DIGITS).tolist() if D >= 93 else [0.0] * 10

        # 5. Distancias al centro de palma
        dist_centro = (v[DIMS_DIST_CENTRO] / PESO_DIST_CENTRO).tolist() if D >= 98 else [0.0] * 5

        # 6. Contacto del pulgar
        contacto = (v[DIMS_CONTACTO_PULGAR] / PESO_CONTACTO_PULGAR).tolist() if D >= 102 else [0.0] * 4

        # 7. Normal de la palma (orientación)
        normal = (v[DIMS_NORMAL_PALMA] / PESO_NORMAL_PALMA).tolist() if D >= 105 else [0.0, 0.0, 1.0]

        # Normalizar la normal
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-6:
            normal = (np.array(normal) / norm_mag).tolist()

        return {
            "landmarks_3d": landmarks_3d,
            "dedos_extension": [round(e, 3) for e in ext_dedos],
            "angulos_flexion": [round(a, 3) for a in angulos],
            "distancias_inter": [round(d, 4) for d in dist_inter],
            "distancias_centro": [round(d, 4) for d in dist_centro],
            "contacto_pulgar": [round(c, 4) for c in contacto],
            "normal_palma": [round(n, 4) for n in normal],
        }

    def obtener_pose_sena(self, glosa: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la pose 3D decodificada para una seña específica.

        Args:
            glosa: Identificador de la seña (ej: "HOLA", "A", "5").

        Returns:
            Dict con datos de pose 3D o None si la seña no existe.
        """
        glosa_upper = glosa.strip().upper()

        # Cache
        if glosa_upper in self._poses_cache:
            return self._poses_cache[glosa_upper]

        if glosa_upper not in self._centroides:
            return None

        centroide = self._centroides[glosa_upper]
        pose = self.decodificar_vector_a_pose(centroide)

        # Enriquecer con metadatos del catálogo educativo
        info_catalogo = DICCIONARIO_EDUCATIVO_LSC.get(glosa_upper, {})
        pose["glosa"] = glosa_upper
        pose["nombre"] = info_catalogo.get("nombre", glosa_upper)
        pose["cuadrante"] = info_catalogo.get("cuadrante", "ESPACIO_LATERAL")
        pose["descripcion"] = info_catalogo.get("descripcion", f"Seña para {glosa_upper}")
        pose["consejo"] = info_catalogo.get("consejo", "")
        pose["dedos_esperados"] = info_catalogo.get("dedos", pose["dedos_extension"])
        pose["duracion_ms"] = self._calcular_duracion(glosa_upper, info_catalogo)

        self._poses_cache[glosa_upper] = pose
        return pose

    def _calcular_duracion(self, glosa: str, info: Dict) -> int:
        """Calcula la duración de animación según el tipo de seña."""
        categoria = info.get("categoria", "")

        if len(glosa) == 1 and glosa.isalpha():
            return DURACION_LETRA
        elif glosa.isdigit() or glosa in ["MIL", "MILLON"]:
            return DURACION_NUMERO
        elif categoria in ["Saludos", "Emergencia", "Solidaridad"]:
            return DURACION_DINAMICA
        else:
            return DURACION_PALABRA

    def generar_secuencia_animacion(
        self,
        glosas: List[str],
        pausa_entre_ms: int = PAUSA_ENTRE_SENAS,
    ) -> Dict[str, Any]:
        """
        Genera la secuencia completa de keyframes para animar el avatar.

        Args:
            glosas: Lista de glosas LSC en orden.
            pausa_entre_ms: Milisegundos de pausa entre señas.

        Returns:
            Dict con la secuencia de poses y metadatos de animación.
        """
        secuencia = []
        tiempo_acumulado = 0

        for glosa in glosas:
            pose = self.obtener_pose_sena(glosa)
            if pose is None:
                # Si no hay datos 3D, generar pose neutra con metadatos básicos
                pose = self._generar_pose_neutra(glosa)

            frame = {
                **pose,
                "tiempo_inicio_ms": tiempo_acumulado,
                "pausa_despues_ms": pausa_entre_ms,
            }
            secuencia.append(frame)
            tiempo_acumulado += pose["duracion_ms"] + pausa_entre_ms

        return {
            "total_senas": len(secuencia),
            "duracion_total_ms": tiempo_acumulado,
            "pausa_entre_ms": pausa_entre_ms,
            "secuencia": secuencia,
        }

    def _generar_pose_neutra(self, glosa: str) -> Dict[str, Any]:
        """Genera una pose neutra/reposo para señas sin datos articulares."""
        # Posición de mano relajada en reposo
        landmarks_reposo = self._construir_mano_reposo()

        return {
            "glosa": glosa,
            "nombre": glosa,
            "cuadrante": "ESPACIO_LATERAL",
            "descripcion": f"Seña para {glosa} (datos articulares no disponibles)",
            "consejo": "",
            "landmarks_3d": landmarks_reposo,
            "dedos_extension": [0.3, 0.3, 0.3, 0.3, 0.3],
            "dedos_esperados": [0, 0, 0, 0, 0],
            "angulos_flexion": [0.0] * 15,
            "distancias_inter": [0.0] * 10,
            "distancias_centro": [0.0] * 5,
            "contacto_pulgar": [0.0] * 4,
            "normal_palma": [0.0, 0.0, 1.0],
            "duracion_ms": DURACION_PALABRA,
        }

    @staticmethod
    def _construir_mano_reposo() -> List[List[float]]:
        """Construye una mano en posición de reposo anatómico."""
        # Posiciones aproximadas de una mano relajada (21 landmarks)
        return [
            [0.0, 0.0, 0.0],       # 0: WRIST
            [-0.04, 0.02, 0.01],    # 1: THUMB_CMC
            [-0.07, 0.04, 0.02],    # 2: THUMB_MCP
            [-0.09, 0.06, 0.02],    # 3: THUMB_IP
            [-0.10, 0.08, 0.02],    # 4: THUMB_TIP
            [-0.02, 0.08, 0.0],     # 5: INDEX_MCP
            [-0.02, 0.12, -0.01],   # 6: INDEX_PIP
            [-0.02, 0.14, -0.02],   # 7: INDEX_DIP
            [-0.02, 0.16, -0.02],   # 8: INDEX_TIP
            [0.0, 0.08, 0.0],       # 9: MIDDLE_MCP
            [0.0, 0.13, -0.01],     # 10: MIDDLE_PIP
            [0.0, 0.15, -0.02],     # 11: MIDDLE_DIP
            [0.0, 0.17, -0.02],     # 12: MIDDLE_TIP
            [0.02, 0.08, 0.0],      # 13: RING_MCP
            [0.02, 0.12, -0.01],    # 14: RING_PIP
            [0.02, 0.14, -0.02],    # 15: RING_DIP
            [0.02, 0.16, -0.02],    # 16: RING_TIP
            [0.04, 0.07, 0.01],     # 17: PINKY_MCP
            [0.04, 0.10, -0.01],    # 18: PINKY_PIP
            [0.04, 0.12, -0.02],    # 19: PINKY_DIP
            [0.04, 0.13, -0.02],    # 20: PINKY_TIP
        ]

    def obtener_senas_disponibles(self) -> List[str]:
        """Retorna la lista de señas con datos 3D disponibles."""
        return sorted(list(self._centroides.keys()))

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retorna estadísticas del exportador."""
        return {
            "total_senas_con_3d": len(self._centroides),
            "total_vectores_base": self.base.total_senas if self.base else 0,
            "senas_disponibles": self.obtener_senas_disponibles(),
            "cache_size": len(self._poses_cache),
        }
