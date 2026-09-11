"""
=============================================================
CLASIFICADOR DE INTELIGENCIA ARTIFICIAL PARA LSC (v3.0)
Lengua de Señas Colombiana (LSC)
=============================================================
Arquitectura de Ensamble Híbrido:
1. Red Neuronal Profunda (MLP): mapeo continuo no lineal en 105 dimensiones.
2. Bosques Aleatorios Extra-Trees: fronteras de decisión ortogonales sobre estados articulares.
3. Fusión Probabilística Calibrada (Soft Voting).
4. Motor Anti-Adivinanza (Anti-Guessing Engine):
   - Clase negativa de control: REPOSO_TRANSICION.
   - Umbral de confianza estricto (P_max >= 0.72).
   - Margen mínimo de ambigüedad (Top1 - Top2 >= 0.15).
   - Supresión cinemática por velocidad (OneEuroFilter).
   - Gating espacial anatómico (Signing Space).
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


CLASE_REPOSO = "REPOSO_TRANSICION"


class ClasificadorIALSC:
    """
    Clasificador de Inteligencia Artificial con Ensamble Híbrido y
    Filtros Anti-Adivinanza para Lengua de Señas Colombiana.
    """

    def __init__(
        self,
        modelo_path: Optional[str] = None,
        umbral_confianza: float = 0.72,
        margen_minimo: float = 0.15,
        peso_mlp: float = 0.50,
        peso_et: float = 0.50,
    ):
        self.umbral_confianza = umbral_confianza
        self.margen_minimo = margen_minimo
        self.peso_mlp = peso_mlp
        self.peso_et = peso_et

        self.mlp_model = None
        self.et_model = None
        self.scaler = None
        self.clases: List[str] = []
        self.cuadrantes_map: Dict[str, str] = {}
        self.esta_cargado = False
        self.modo_numpy = False

        # Pesos para modo puro NumPy
        self.mlp_w0 = None
        self.mlp_b0 = None
        self.mlp_w1 = None
        self.mlp_b1 = None
        self.mlp_w2 = None
        self.mlp_b2 = None
        self.scaler_mean = None
        self.scaler_scale = None

        if modelo_path and os.path.exists(modelo_path):
            self.cargar(modelo_path)

    def cargar(self, ruta_archivo: str):
        """Carga el paquete de modelos (.npz universal o .joblib)."""
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontro el modelo: {ruta_archivo}")

        ruta_npz = ruta_archivo.replace(".joblib", ".npz")
        
        # Preferir .npz si la ruta ya es .npz o si existe como alternativa universal
        if ruta_archivo.endswith(".npz"):
            self._cargar_npz(ruta_archivo)
            return

        # Intentar cargar .joblib
        try:
            paquete = joblib.load(ruta_archivo)
            self.mlp_model = paquete.get("mlp_model")
            self.et_model = paquete.get("et_model")
            self.scaler = paquete.get("scaler")
            self.clases = list(paquete["clases"])
            self.cuadrantes_map = paquete.get("cuadrantes_map", {})
            self.peso_mlp = paquete.get("peso_mlp", 0.50)
            self.peso_et = paquete.get("peso_et", 0.50)
            self.umbral_confianza = paquete.get("umbral_confianza", self.umbral_confianza)
            self.margen_minimo = paquete.get("margen_minimo", self.margen_minimo)
            self.modo_numpy = False
            self.esta_cargado = True
        except Exception as e:
            # Fallback a .npz si joblib falla por version mismatch de numpy/sklearn
            if os.path.exists(ruta_npz):
                print(f"  [AVISO] joblib no compatible con este Python. Cargando version universal NPZ: {ruta_npz}")
                self._cargar_npz(ruta_npz)
            else:
                raise e

    def _cargar_npz(self, ruta_npz: str):
        """Carga pesos puros en NumPy para inferencia sin dependencias de version."""
        data = np.load(ruta_npz, allow_pickle=True)
        self.mlp_w0 = data["mlp_w0"]
        self.mlp_b0 = data["mlp_b0"]
        self.mlp_w1 = data["mlp_w1"]
        self.mlp_b1 = data["mlp_b1"]
        self.mlp_w2 = data["mlp_w2"]
        self.mlp_b2 = data["mlp_b2"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]
        self.clases = list(data["clases"])
        self.cuadrantes_map = {}
        if "cuadrantes_keys" in data and "cuadrantes_vals" in data:
            keys = list(data["cuadrantes_keys"])
            vals = list(data["cuadrantes_vals"])
            self.cuadrantes_map = dict(zip(keys, vals))
        self.modo_numpy = True
        self.esta_cargado = True

    def guardar(self, ruta_archivo: str, metricas: Optional[Dict[str, Any]] = None):
        """Guarda tanto el paquete .joblib como el .npz universal."""
        os.makedirs(os.path.dirname(os.path.abspath(ruta_archivo)), exist_ok=True)
        
        # 1. Guardar Joblib
        paquete = {
            "mlp_model": self.mlp_model,
            "et_model": self.et_model,
            "scaler": self.scaler,
            "clases": self.clases,
            "cuadrantes_map": self.cuadrantes_map,
            "peso_mlp": self.peso_mlp,
            "peso_et": self.peso_et,
            "umbral_confianza": self.umbral_confianza,
            "margen_minimo": self.margen_minimo,
            "metricas": metricas or {},
        }
        try:
            joblib.dump(paquete, ruta_archivo, compress=3)
        except Exception:
            pass

        # 2. Guardar NPZ universal (compatible con cualquier Python/NumPy)
        ruta_npz = ruta_archivo.replace(".joblib", ".npz")
        if self.mlp_model is not None and self.scaler is not None:
            cuad_keys = np.array(list(self.cuadrantes_map.keys()), dtype=str)
            cuad_vals = np.array(list(self.cuadrantes_map.values()), dtype=str)
            np.savez_compressed(
                ruta_npz,
                mlp_w0=self.mlp_model.coefs_[0],
                mlp_b0=self.mlp_model.intercepts_[0],
                mlp_w1=self.mlp_model.coefs_[1],
                mlp_b1=self.mlp_model.intercepts_[1],
                mlp_w2=self.mlp_model.coefs_[2],
                mlp_b2=self.mlp_model.intercepts_[2],
                scaler_mean=self.scaler.mean_,
                scaler_scale=self.scaler.scale_,
                clases=np.array(self.clases, dtype=str),
                cuadrantes_keys=cuad_keys,
                cuadrantes_vals=cuad_vals,
            )

    def predecir(
        self,
        vector_105d: np.ndarray,
        cuadrante: Optional[str] = None,
        mano_estable: bool = True,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Ejecuta la inferencia del ensamble y aplica los filtros anti-adivinanza.
        """
        if not self.esta_cargado:
            return {
                "etiqueta": "---",
                "confianza": 0.0,
                "margen": 0.0,
                "es_valida": False,
                "estado": "NO_MODELO",
                "candidatos": [],
            }

        # 1. Filtro Cinemático Anti-Jitter
        if not mano_estable:
            return {
                "etiqueta": "---",
                "confianza": 0.0,
                "margen": 0.0,
                "es_valida": False,
                "estado": "TRANSICION",
                "candidatos": [],
            }

        # Adaptación dimensional dinámica (105D / 109D)
        raw_x = np.asarray(vector_105d, dtype=np.float32).flatten()
        n_esperado = self.scaler_mean.shape[0] if self.modo_numpy and self.scaler_mean is not None else (
            self.scaler.n_features_in_ if self.scaler is not None and hasattr(self.scaler, "n_features_in_") else len(raw_x)
        )
        if len(raw_x) < n_esperado:
            pad = np.zeros(n_esperado - len(raw_x), dtype=np.float32)
            x = np.concatenate([raw_x, pad]).reshape(1, -1)
        elif len(raw_x) > n_esperado:
            x = raw_x[:n_esperado].reshape(1, -1)
        else:
            x = raw_x.reshape(1, -1)

        # 2. Inferencia de Probabilidades
        if self.modo_numpy:
            # Forward pass en NumPy puro (inmune a versiones de sklearn)
            x_norm = (x - self.scaler_mean) / np.where(self.scaler_scale > 1e-6, self.scaler_scale, 1.0)
            z1 = np.maximum(0, x_norm @ self.mlp_w0 + self.mlp_b0)
            z2 = np.maximum(0, z1 @ self.mlp_w1 + self.mlp_b1)
            logits = z2 @ self.mlp_w2 + self.mlp_b2
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            p_ens = (exp_l / np.sum(exp_l, axis=-1, keepdims=True))[0]
        else:
            if self.scaler is not None:
                x_norm = self.scaler.transform(x)
            else:
                x_norm = x
            p_mlp = self.mlp_model.predict_proba(x_norm)[0] if self.mlp_model else np.zeros(len(self.clases))
            p_et = self.et_model.predict_proba(x_norm)[0] if self.et_model else np.zeros(len(self.clases))
            p_ens = (self.peso_mlp * p_mlp) + (self.peso_et * p_et)

        # 3. Gating Espacial Anatómico (Signing Space)
        if cuadrante and self.cuadrantes_map:
            for idx, c in enumerate(self.clases):
                if c == CLASE_REPOSO:
                    continue
                cuad_esperado = self.cuadrantes_map.get(c)
                if cuad_esperado:
                    if cuad_esperado == cuadrante:
                        p_ens[idx] *= 1.08  # Bonificación por coincidencia de zona
                    else:
                        p_ens[idx] *= 0.70  # Penalización por zona anatómica incorrecta

            # Renormalizar probabilidades
            suma = np.sum(p_ens)
            if suma > 1e-6:
                p_ens = p_ens / suma

        # 4. Ordenamiento Top-K
        indices_ordenados = np.argsort(-p_ens)
        top1_idx = int(indices_ordenados[0])
        top1_label = self.clases[top1_idx]
        top1_prob = float(p_ens[top1_idx])

        top2_prob = float(p_ens[indices_ordenados[1]]) if len(indices_ordenados) > 1 else 0.0
        margen = top1_prob - top2_prob

        candidatos = [
            (self.clases[idx], float(p_ens[idx]))
            for idx in indices_ordenados[:top_k]
            if self.clases[idx] != CLASE_REPOSO
        ]

        # 5. Reglas del Motor Anti-Adivinanza (Anti-Guessing)
        if top1_label == CLASE_REPOSO:
            return {
                "etiqueta": "---",
                "confianza": top1_prob,
                "margen": margen,
                "es_valida": False,
                "estado": "REPOSO",
                "candidatos": candidatos,
            }

        if top1_prob < self.umbral_confianza:
            return {
                "etiqueta": "---",
                "confianza": top1_prob,
                "margen": margen,
                "es_valida": False,
                "estado": "BAJA_CONFIANZA",
                "candidatos": candidatos,
            }

        if margen < self.margen_minimo:
            return {
                "etiqueta": "---",
                "confianza": top1_prob,
                "margen": margen,
                "es_valida": False,
                "estado": "DUDOSO",
                "candidatos": candidatos,
            }

        # ¡Seña confirmada con alta certeza y sin ambigüedad!
        return {
            "etiqueta": top1_label,
            "confianza": top1_prob,
            "margen": margen,
            "es_valida": True,
            "estado": "SEÑA_DETECTADA",
            "candidatos": candidatos,
        }
