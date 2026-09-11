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
import json
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


# Señas bimanuales identificadas en el catálogo de LSC
SENAS_BIMANUALES = {
    "APOYAR", "AYUDAR", "BIENVENIDO", "NOCHES", "BAÑO", "FAMILIA", "TRABAJAR",
    "AMIGO", "COLEGIO", "CASA", "GRACIAS", "PORFAVOR", "FIESTA", "REUNION",
    "IGUAL", "DIFERENTE", "JUNTOS", "COMPARTIR", "COMUNIDAD", "ENTENDER"
}


class ExportadorPoses3D:
    """
    Decodifica vectores articulares de la base entrenada a poses 3D
    renderizables por un avatar Three.js con soporte bimanual, torso y cabeza.
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
        """
        v = vector.flatten().astype(np.float64)
        D = len(v)

        coords_raw = v[DIMS_COORDS] / PESO_COORDS if D >= 63 else np.zeros(63)
        landmarks_3d = coords_raw.reshape(21, 3).tolist()

        ext_dedos = (v[DIMS_EXT_DEDOS] / PESO_EXT_DEDOS).clip(0, 1).tolist() if D >= 68 else [0.5] * 5
        angulos = (v[DIMS_ANGULOS] / PESO_ANGULOS).tolist() if D >= 83 else [0.0] * 15
        dist_inter = (v[DIMS_INTER_DIGITS] / PESO_INTER_DIGITS).tolist() if D >= 93 else [0.0] * 10
        dist_centro = (v[DIMS_DIST_CENTRO] / PESO_DIST_CENTRO).tolist() if D >= 98 else [0.0] * 5
        contacto = (v[DIMS_CONTACTO_PULGAR] / PESO_CONTACTO_PULGAR).tolist() if D >= 102 else [0.0] * 4

        normal = (v[DIMS_NORMAL_PALMA] / PESO_NORMAL_PALMA).tolist() if D >= 105 else [0.0, 0.0, 1.0]
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

    def _generar_dinamica_corporal(self, glosa: str, categoria: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """
        Genera rotación de torso, cabeza y expresión facial basada en la lingüística LSC.
        """
        if categoria in ["Saludos"]:
            torso = {"pitch": 0.04, "yaw": 0.0, "roll": 0.01}
            cabeza = {"pitch": 0.06, "yaw": -0.02, "tilt": 0.03}
            expresion = {"cejas": 0.35, "boca": 0.4, "tipo": "sonrisa_cordial"}
        elif categoria in ["Emergencia", "Identidad"] or glosa in ["AYUDAR", "BAÑO"]:
            torso = {"pitch": 0.08, "yaw": 0.0, "roll": 0.0}
            cabeza = {"pitch": -0.04, "yaw": 0.0, "tilt": 0.08}
            expresion = {"cejas": 0.65, "boca": 0.5, "tipo": "pregunta_atencion"}
        elif categoria in ["Solidaridad", "Emociones"] or glosa in ["APOYAR", "GUSTAR", "BIEN"]:
            torso = {"pitch": 0.05, "yaw": 0.0, "roll": 0.0}
            cabeza = {"pitch": 0.08, "yaw": 0.0, "tilt": 0.0}
            expresion = {"cejas": 0.2, "boca": 0.35, "tipo": "afirmativo"}
        elif glosa in ["YO"]:
            torso = {"pitch": 0.02, "yaw": 0.05, "roll": 0.0}
            cabeza = {"pitch": 0.05, "yaw": 0.02, "tilt": 0.0}
            expresion = {"cejas": 0.1, "boca": 0.15, "tipo": "auto_referencia"}
        else:
            torso = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
            cabeza = {"pitch": 0.0, "yaw": 0.0, "tilt": 0.0}
            expresion = {"cejas": 0.0, "boca": 0.0, "tipo": "neutro"}

        return torso, cabeza, expresion

    def obtener_pose_sena(self, glosa: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la pose 3D decodificada completa (ambas manos, torso, cabeza).
        """
        glosa_upper = glosa.strip().upper()

        if glosa_upper in self._poses_cache:
            return self._poses_cache[glosa_upper]

        if glosa_upper not in self._centroides:
            return None

        centroide = self._centroides[glosa_upper]
        pose = self.decodificar_vector_a_pose(centroide)

        info_catalogo = DICCIONARIO_EDUCATIVO_LSC.get(glosa_upper, {})
        categoria = info_catalogo.get("categoria", "")
        cuadrante_der = info_catalogo.get("cuadrante", "ESPACIO_LATERAL")
        es_bimanual = glosa_upper in SENAS_BIMANUALES

        # Generar mano izquierda (espejada si es bimanual, o reposo si es monomanual)
        landmarks_der = pose["landmarks_3d"]
        if es_bimanual:
            landmarks_izq = [[-p[0], p[1], p[2]] for p in landmarks_der]
            cuadrante_izq = cuadrante_der
            dedos_izq = pose["dedos_extension"]
        else:
            landmarks_izq = self._construir_mano_reposo(es_izquierda=True)
            cuadrante_izq = "REPOSO"
            dedos_izq = [0.2, 0.2, 0.2, 0.2, 0.2]

        torso, cabeza, expresion = self._generar_dinamica_corporal(glosa_upper, categoria)

        pose.update({
            "glosa": glosa_upper,
            "nombre": info_catalogo.get("nombre", glosa_upper),
            "cuadrante": cuadrante_der,
            "cuadrante_der": cuadrante_der,
            "cuadrante_izq": cuadrante_izq,
            "es_bimanual": es_bimanual,
            "landmarks_3d_derecha": landmarks_der,
            "landmarks_3d_izquierda": landmarks_izq,
            "dedos_extension_izq": dedos_izq,
            "torso": torso,
            "cabeza": cabeza,
            "expresion": expresion,
            "descripcion": info_catalogo.get("descripcion", f"Seña para {glosa_upper}"),
            "consejo": info_catalogo.get("consejo", ""),
            "dedos_esperados": info_catalogo.get("dedos", pose["dedos_extension"]),
            "duracion_ms": self._calcular_duracion(glosa_upper, info_catalogo),
        })

        self._poses_cache[glosa_upper] = pose
        return pose

    def _calcular_duracion(self, glosa: str, info: Dict) -> int:
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
        secuencia = []
        tiempo_acumulado = 0

        for glosa in glosas:
            pose = self.obtener_pose_sena(glosa)
            if pose is None:
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
        landmarks_reposo_der = self._construir_mano_reposo(es_izquierda=False)
        landmarks_reposo_izq = self._construir_mano_reposo(es_izquierda=True)
        es_bimanual = glosa.upper() in SENAS_BIMANUALES

        return {
            "glosa": glosa,
            "nombre": glosa,
            "cuadrante": "ESPACIO_LATERAL",
            "cuadrante_der": "ESPACIO_LATERAL",
            "cuadrante_izq": "ESPACIO_CENTRAL" if es_bimanual else "REPOSO",
            "es_bimanual": es_bimanual,
            "descripcion": f"Seña para {glosa} (datos articulares no disponibles)",
            "consejo": "",
            "landmarks_3d": landmarks_reposo_der,
            "landmarks_3d_derecha": landmarks_reposo_der,
            "landmarks_3d_izquierda": landmarks_reposo_izq,
            "dedos_extension": [0.3, 0.3, 0.3, 0.3, 0.3],
            "dedos_extension_izq": [0.3, 0.3, 0.3, 0.3, 0.3],
            "dedos_esperados": [0, 0, 0, 0, 0],
            "angulos_flexion": [0.0] * 15,
            "distancias_inter": [0.0] * 10,
            "distancias_centro": [0.0] * 5,
            "contacto_pulgar": [0.0] * 4,
            "normal_palma": [0.0, 0.0, 1.0],
            "torso": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "cabeza": {"pitch": 0.0, "yaw": 0.0, "tilt": 0.0},
            "expresion": {"cejas": 0.0, "boca": 0.0, "tipo": "neutro"},
            "duracion_ms": DURACION_PALABRA,
        }

    @staticmethod
    def _construir_mano_reposo(es_izquierda: bool = False) -> List[List[float]]:
        signo_x = -1.0 if es_izquierda else 1.0
        return [
            [0.0, 0.0, 0.0],
            [signo_x * -0.04, 0.02, 0.01],
            [signo_x * -0.07, 0.04, 0.02],
            [signo_x * -0.09, 0.06, 0.02],
            [signo_x * -0.10, 0.08, 0.02],
            [signo_x * -0.02, 0.08, 0.0],
            [signo_x * -0.02, 0.12, -0.01],
            [signo_x * -0.02, 0.14, -0.02],
            [signo_x * -0.02, 0.16, -0.02],
            [0.0, 0.08, 0.0],
            [0.0, 0.13, -0.01],
            [0.0, 0.15, -0.02],
            [0.0, 0.17, -0.02],
            [signo_x * 0.02, 0.08, 0.0],
            [signo_x * 0.02, 0.12, -0.01],
            [signo_x * 0.02, 0.14, -0.02],
            [signo_x * 0.02, 0.16, -0.02],
            [signo_x * 0.04, 0.07, 0.01],
            [signo_x * 0.04, 0.10, -0.01],
            [signo_x * 0.04, 0.12, -0.02],
            [signo_x * 0.04, 0.13, -0.02],
        ]

    def exportar_a_blender_json(self, glosa: str, ruta_archivo: Optional[str] = None) -> Dict[str, Any]:
        """
        Exporta una seña al formato JSON consumible por apply_lsc_keyframes_blender.py
        para animar rigs de Rigify directamente en Blender.
        """
        pose = self.obtener_pose_sena(glosa)
        if pose is None:
            raise ValueError(f"Seña '{glosa}' no encontrada en el catálogo 3D.")

        ext_dedos = pose.get("dedos_extension", [0.5] * 5)
        curls = {
            "thumb": round(float(np.clip(1.0 - ext_dedos[0], 0.0, 1.0)), 3),
            "index": round(float(np.clip(1.0 - ext_dedos[1], 0.0, 1.0)), 3),
            "middle": round(float(np.clip(1.0 - ext_dedos[2], 0.0, 1.0)), 3),
            "ring": round(float(np.clip(1.0 - ext_dedos[3], 0.0, 1.0)), 3),
            "pinky": round(float(np.clip(1.0 - ext_dedos[4], 0.0, 1.0)), 3),
        }

        cuadrante = pose.get("cuadrante", "ESPACIO_CENTRAL")
        if cuadrante == "CABEZA_ROSTRO":
            shoulder_raise = 1.65
            elbow_curl = 0.85
        elif cuadrante == "PECHO_TORSO":
            shoulder_raise = 1.15
            elbow_curl = 0.65
        else:
            shoulder_raise = 0.85
            elbow_curl = 0.45

        fps = 30.0
        duracion_frames = int(round((pose.get("duracion_ms", 1200) / 1000.0) * fps))

        frames = []
        for i in range(duracion_frames):
            t = i / fps
            p = 0.5 - 0.5 * float(np.cos(np.pi * (i / max(1, duracion_frames - 1))))
            frames.append({
                "t": round(t, 4),
                "elbow_curl": round(elbow_curl * p, 3),
                "shoulder_raise_rad": round(shoulder_raise * p, 3),
                "curls": {k: round(v * p, 3) for k, v in curls.items()},
            })

        data = {
            "glosa": glosa.upper(),
            "fps": fps,
            "side": "right",
            "frames": frames,
        }

        if ruta_archivo:
            with open(ruta_archivo, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

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
