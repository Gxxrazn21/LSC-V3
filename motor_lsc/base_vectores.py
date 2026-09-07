"""
=============================================================
BASE DE DATOS VECTORIAL Y MOTOR DE BUSQUEDA ARTICULAR LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Búsqueda ultrarrápida (<1ms en CPU) basada en:
1. Similitud Coseno de vectores canónicos 3D balanceados (105 dimensiones).
2. Penalización ARTICULAR CUÁDRUPLE:
   a) Extensión de dedos (55%) — separa posturas abiertas vs cerradas.
   b) Ángulos articulares MCP/PIP/DIP (40%) — diferencia curvatura fina.
   c) Distancias interdigitales (30%) — configuraciones de separación/contacto (U vs V).
   d) Contacto de pulgar (35%) — desambigua señas de puño cerrado (A, S, T, M, N, E).
3. Ponderación por Cuadrantes Anatómicos (Signing Space).
4. Búsqueda estricta con margen mínimo obligatorio (anti-adivinanza).
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ─── Mapa de dimensiones del descriptor de 105 dims ───
# [0:63]    Coordenadas canónicas 3D (21 puntos × 3) × 0.55
# [63:68]   Extensión de dedos (5 dims) × 3.2
# [68:83]   Coseno de ángulos articulares (15 dims) × 1.2
# [83:93]   Distancias interdigitales (10 dims) × 2.0
# [93:98]   Distancias al centro de palma (5 dims) × 1.2
# [98:102]  Métricas de contacto del pulgar (4 dims) × 1.8
# [102:105] Normal de la palma (3 dims) × 1.4

DIMS_COORDS = slice(0, 63)
DIMS_EXT_DEDOS = slice(63, 68)
DIMS_ANGULOS = slice(68, 83)
DIMS_INTER_DIGITS = slice(83, 93)
DIMS_DIST_CENTRO = slice(93, 98)
DIMS_CONTACTO_PULGAR = slice(98, 102)
DIMS_NORMAL_PALMA = slice(102, 105)


class BaseVectoresLSC:
    def __init__(self, umbral_min_similitud: float = 0.80):
        self.umbral_min_similitud = umbral_min_similitud
        
        # Vectores estáticos
        self.vectores: Optional[np.ndarray] = None  # Shape (N, D)
        self.vectores_norm: Optional[np.ndarray] = None  # Shape (N, D) unitarios
        self.etiquetas: List[str] = []
        self.cuadrantes: List[str] = []
        self.categorias: List[str] = []
        
        # Plantillas dinámicas (secuencias temporales)
        self.plantillas_dinamicas: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def total_senas(self) -> int:
        return len(self.etiquetas)

    @property
    def clases_unicas(self) -> List[str]:
        return sorted(list(set(self.etiquetas + list(self.plantillas_dinamicas.keys()))))

    def agregar_referencia_estatica(
        self,
        vector: np.ndarray,
        etiqueta: str,
        cuadrante: str = "ESPACIO_LATERAL",
        categoria: str = "General",
    ):
        """Agrega un vector de referencia estático al catálogo."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if self.vectores is None:
            self.vectores = np.array([vec], dtype=np.float32)
        else:
            self.vectores = np.vstack([self.vectores, vec])

        self.etiquetas.append(etiqueta.strip().upper())
        self.cuadrantes.append(cuadrante)
        self.categorias.append(categoria)
        self._actualizar_normas()

    def agregar_referencia_dinamica(
        self,
        secuencia: np.ndarray,
        etiqueta: str,
        cuadrante: str = "ESPACIO_CENTRAL",
        categoria: str = "General",
    ):
        """Agrega una secuencia temporal (T, D) de referencia para una seña dinámica."""
        sec = np.asarray(secuencia, dtype=np.float32)
        et_clean = etiqueta.strip().upper()
        if et_clean not in self.plantillas_dinamicas:
            self.plantillas_dinamicas[et_clean] = []
            
        self.plantillas_dinamicas[et_clean].append({
            "cuadrante": cuadrante,
            "categoria": categoria,
            "secuencia": sec,
        })

    def _actualizar_normas(self):
        """Precalcula los vectores unitarios para acelerar producto punto masivo."""
        if self.vectores is not None and len(self.vectores) > 0:
            normas = np.linalg.norm(self.vectores, axis=1, keepdims=True)
            normas[normas < 1e-7] = 1.0
            self.vectores_norm = self.vectores / normas

    def _calcular_penalizaciones_articulares(
        self,
        q: np.ndarray,
        similitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Aplica penalizaciones articulares multi-capa al vector de similitudes.
        """
        D = self.vectores.shape[1]
        
        # Capa 1: Extensión de dedos (dims 63-68, ponderadas ×3.2 en extractor)
        # Penalización suave: cada dedo discrepante resta ~0.12 al score
        if D >= 68 and len(q) >= 68:
            ext_q = q[DIMS_EXT_DEDOS] / 3.2
            ext_refs = self.vectores[:, DIMS_EXT_DEDOS] / 3.2
            diff_ext = np.abs(ext_refs - ext_q)
            pen_ext = np.sum(diff_ext, axis=1) * 0.12
            similitudes -= pen_ext

        # Capa 2: Ángulos articulares (dims 68-83, 15 cosenos de flexión × 1.2)
        if D >= 83 and len(q) >= 83:
            ang_q = q[DIMS_ANGULOS] / 1.2
            ang_refs = self.vectores[:, DIMS_ANGULOS] / 1.2
            diff_ang = np.abs(ang_refs - ang_q)
            similitudes -= np.mean(diff_ang, axis=1) * 0.18

        # Capa 3: Distancias interdigitales (dims 83-93, ×2.0 en extractor)
        if D >= 93 and len(q) >= 93:
            inter_q = q[DIMS_INTER_DIGITS] / 2.0
            inter_refs = self.vectores[:, DIMS_INTER_DIGITS] / 2.0
            diff_inter = np.abs(inter_refs - inter_q)
            similitudes -= np.mean(diff_inter, axis=1) * 0.15

        # Capa 4: Contacto de pulgar (dims 98-102, ×1.8 en extractor)
        if D >= 102 and len(q) >= 102:
            cont_q = q[DIMS_CONTACTO_PULGAR] / 1.8
            cont_refs = self.vectores[:, DIMS_CONTACTO_PULGAR] / 1.8
            diff_cont = np.abs(cont_refs - cont_q)
            similitudes -= np.mean(diff_cont, axis=1) * 0.20

        return similitudes

    def buscar_similar(
        self,
        vector_query: np.ndarray,
        cuadrante_query: Optional[str] = None,
        top_k: int = 3,
        bono_cuadrante: float = 0.05,
    ) -> List[Tuple[str, float, str, str]]:
        """
        Busca las señas más similares con penalización articular cuádruple.
        """
        if self.vectores_norm is None or len(self.vectores_norm) == 0:
            return []

        q = np.asarray(vector_query, dtype=np.float32).flatten()
        norm_q = np.linalg.norm(q)
        if norm_q < 1e-7:
            return []
        q_unit = q / norm_q

        # 1. Similitud Coseno vectorizada masiva (<0.1 ms)
        similitudes = np.dot(self.vectores_norm, q_unit).copy()

        # 2. Penalización articular CUÁDRUPLE
        similitudes = self._calcular_penalizaciones_articulares(q, similitudes)

        # 3. Ponderación por Cuadrante Anatómico (Signing Space)
        if cuadrante_query:
            for i, cuad in enumerate(self.cuadrantes):
                et = self.etiquetas[i]
                es_dactilologia = (len(et) == 1 or et.isdigit() or et in ["MIL", "MILLON", "NN", "Ñ"])
                
                if es_dactilologia:
                    if cuadrante_query in ["ESPACIO_LATERAL", "ESPACIO_CENTRAL", "PECHO_TORSO"]:
                        similitudes[i] += 0.02
                else:
                    if cuad == cuadrante_query:
                        similitudes[i] += bono_cuadrante
                    else:
                        similitudes[i] -= (bono_cuadrante * 0.5)

        # 4. Top-K ordenados
        indices_ordenados = np.argsort(-similitudes)[:top_k]
        resultados = []
        for idx in indices_ordenados:
            score = float(similitudes[idx])
            resultados.append((
                self.etiquetas[idx],
                score,
                self.cuadrantes[idx],
                self.categorias[idx],
            ))

        return resultados

    def buscar_estricto(
        self,
        vector_query: np.ndarray,
        cuadrante_query: Optional[str] = None,
        umbral_minimo: float = 0.75,
        margen_minimo: float = 0.03,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Búsqueda ESTRICTA anti-adivinanza.
        Solo devuelve un resultado si:
          1. El score del top-1 supera umbral_minimo.
          2. El margen entre top-1 y top-2 supera margen_minimo.
        """
        candidatos = self.buscar_similar(
            vector_query=vector_query,
            cuadrante_query=cuadrante_query,
            top_k=5,
        )
        
        if not candidatos:
            return None
        
        score_1 = candidatos[0][1]
        if score_1 < umbral_minimo:
            return None
        
        # Calcular margen con el segundo candidato
        score_2 = candidatos[1][1] if len(candidatos) > 1 else 0.0
        margen = score_1 - score_2
        
        if margen < margen_minimo:
            return None
        
        return (candidatos[0][0], score_1, margen)

    def podar_outliers_por_centroide(self, umbral_similitud_centroide: float = 0.80):
        """
        Elimina muestras de referencia atípicas o ruidosas cuya similitud con el
        centroide de su respectiva clase sea inferior al umbral.
        """
        if self.vectores is None or len(self.vectores) == 0:
            return

        from collections import defaultdict
        por_clase = defaultdict(list)
        for idx, (vec, lbl) in enumerate(zip(self.vectores_norm, self.etiquetas)):
            por_clase[lbl].append(idx)

        indices_validos = []
        for lbl, indices in por_clase.items():
            if len(indices) <= 2:
                indices_validos.extend(indices)
                continue

            sub_vecs = self.vectores_norm[indices]
            centroide = np.mean(sub_vecs, axis=0)
            norm_c = np.linalg.norm(centroide)
            if norm_c > 1e-6:
                centroide = centroide / norm_c

            sims = np.dot(sub_vecs, centroide)
            for idx_local, s in enumerate(sims):
                if s >= umbral_similitud_centroide:
                    indices_validos.append(indices[idx_local])

        indices_validos = sorted(indices_validos)
        total_antes = len(self.etiquetas)
        total_despues = len(indices_validos)

        self.vectores = self.vectores[indices_validos]
        self.etiquetas = [self.etiquetas[i] for i in indices_validos]
        self.cuadrantes = [self.cuadrantes[i] for i in indices_validos]
        self.categorias = [self.categorias[i] for i in indices_validos]
        self._actualizar_normas()

        print(f"  [Poda de Outliers] Vectores depurados: {total_antes} -> {total_despues} (eliminadas {total_antes - total_despues} muestras atípicas)")

    def guardar(self, ruta_npz: str, ruta_json_clases: Optional[str] = None):
        """Guarda la base de vectores en disco."""
        os.makedirs(os.path.dirname(ruta_npz) or ".", exist_ok=True)
        np.savez_compressed(
            ruta_npz,
            vectores=self.vectores if self.vectores is not None else np.array([]),
            etiquetas=np.array(self.etiquetas, dtype=object),
            cuadrantes=np.array(self.cuadrantes, dtype=object),
            categorias=np.array(self.categorias, dtype=object),
        )

        if ruta_json_clases:
            with open(ruta_json_clases, "w", encoding="utf-8") as f:
                json.dump(self.clases_unicas, f, ensure_ascii=False, indent=2)

    def cargar(self, ruta_npz: str):
        """Carga la base de vectores desde disco."""
        if not os.path.exists(ruta_npz):
            raise FileNotFoundError(f"No se encontró el archivo de base de datos: {ruta_npz}")

        data = np.load(ruta_npz, allow_pickle=True)
        self.vectores = data["vectores"] if len(data["vectores"]) > 0 else None
        self.etiquetas = list(data["etiquetas"])
        self.cuadrantes = list(data["cuadrantes"])
        self.categorias = list(data["categorias"])
        self._actualizar_normas()

