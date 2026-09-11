"""
=============================================================================
DECODIFICADOR CONTINUO CTC Y VENTANA DESLIZANTE PARA LSC
=============================================================================
Pipeline de decodificación en tiempo real para traducción continua (no solo
señas aisladas) de oraciones en Lengua de Señas Colombiana:
1. Buffer de ventana deslizante temporal (W = 40 frames, stride = 8).
2. Decodificador CTC (Greedy + Prefix Collapse de tokens [BLANK] y duplicados).
3. Ensamblador de oraciones con debounce temporal e integración directa a TTS.
=============================================================================
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

class DecodificadorContinuoCTC:
    def __init__(
        self,
        clases: List[str],
        tamano_ventana: int = 40,
        stride_fotogramas: int = 8,
        umbral_confianza_min: float = 0.82,
        tiempo_cooldown_misma_sena: float = 1.35
    ):
        """
        Inicializa el decodificador continuo para oraciones LSC.
        
        Args:
            clases: Lista de nombres de señas (índice 0 reservado para [BLANK]).
            tamano_ventana: Número de fotogramas en la ventana deslizante.
            stride_fotogramas: Desplazamiento de fotogramas entre inferencias continuas.
            umbral_confianza_min: Probabilidad mínima para emitir una seña confirmada.
            tiempo_cooldown_misma_sena: Tiempo mínimo en segundos para volver a repetir
                                       la misma seña si no ha habido reposo.
        """
        # El índice 0 siempre es el token BLANK de CTC
        self.clases = ["_BLANK_"] + [c for c in clases if c != "_BLANK_"]
        self.idx_to_class = {i: c for i, c in enumerate(self.clases)}
        self.tamano_ventana = tamano_ventana
        self.stride_fotogramas = stride_fotogramas
        self.umbral_confianza_min = umbral_confianza_min
        self.tiempo_cooldown_misma_sena = tiempo_cooldown_misma_sena

        # Buffers temporales
        self.buffer_manos: List[np.ndarray] = []
        self.buffer_rostro: List[np.ndarray] = []
        self.buffer_pose: List[np.ndarray] = []
        self.contador_fotogramas = 0

        # Historial de señas emitidas
        self.ultima_sena_emitida = ""
        self.timestamp_ultima_emision = 0.0
        self.oracion_actual: List[str] = []
        self.callback_emision: Optional[Callable[[str, Dict], None]] = None

    def registrar_callback_emision(self, callback: Callable[[str, Dict], None]):
        """Registra la función oyente cuando se confirma una palabra en la oración."""
        self.callback_emision = callback

    def ingresar_fotograma(
        self,
        vec_manos: np.ndarray,
        vec_rostro: np.ndarray,
        vec_pose: np.ndarray,
        modelo_inferencia_fn: Optional[Callable] = None
    ) -> Optional[Dict]:
        """
        Ingresa un fotograma al buffer de ventana deslizante y, si se cumple el stride,
        ejecuta la inferencia continua con decodificación CTC.
        """
        self.buffer_manos.append(vec_manos)
        self.buffer_rostro.append(vec_rostro)
        self.buffer_pose.append(vec_pose)
        self.contador_fotogramas += 1

        # Mantener tamaño fijo de ventana
        if len(self.buffer_manos) > self.tamano_ventana:
            self.buffer_manos.pop(0)
            self.buffer_rostro.pop(0)
            self.buffer_pose.pop(0)

        # Evaluar únicamente cuando la ventana está llena y en múltiplos del stride
        if len(self.buffer_manos) >= self.tamano_ventana and (self.contador_fotogramas % self.stride_fotogramas == 0):
            if modelo_inferencia_fn is not None:
                x_h = np.array(self.buffer_manos, dtype=np.float32)[np.newaxis, ...]
                x_f = np.array(self.buffer_rostro, dtype=np.float32)[np.newaxis, ...]
                x_p = np.array(self.buffer_pose, dtype=np.float32)[np.newaxis, ...]

                # Inferencia con Conformer
                salida_modelo = modelo_inferencia_fn(x_h, x_f, x_p)
                ctc_logits = salida_modelo.get("ctc_logits")

                if ctc_logits is not None:
                    return self.decodificar_ctc_greedy(ctc_logits[0])

        return None

    def decodificar_ctc_greedy(self, logits_secuencia: np.ndarray) -> Dict:
        """
        Decodifica la secuencia de logits con el algoritmo CTC Greedy:
        1. Argmax en cada fotograma.
        2. Colapso de repeticiones consecutivas (ej: A A A B B -> A B).
        3. Eliminación de tokens [BLANK], REPOSO y TRANSICION.
        4. Verificación de confianza y emisión con debounce.
        """
        # Softmax numéricamente estable
        exp_logits = np.exp(logits_secuencia - np.max(logits_secuencia, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        best_indices = np.argmax(probs, axis=-1)
        best_probs = np.max(probs, axis=-1)

        # 1. Colapso CTC
        colapsado_idx = []
        colapsado_probs = []
        prev_idx = -1

        for idx, p in zip(best_indices, best_probs):
            if idx != prev_idx:
                if idx != 0:  # 0 es BLANK
                    colapsado_idx.append(idx)
                    colapsado_probs.append(p)
                prev_idx = idx

        # 2. Filtrado de palabras
        senas_detectadas = []
        ahora = time.time()

        for idx, conf in zip(colapsado_idx, colapsado_probs):
            nombre_sena = self.idx_to_class.get(idx, "")
            
            # Descartar estados no comunicativos
            if nombre_sena in ["_BLANK_", "REPOSO", "TRANSICION", "TRANSICIÓN", ""]:
                continue

            # Umbral de confianza estricto
            if conf >= self.umbral_confianza_min:
                es_misma = (nombre_sena == self.ultima_sena_emitida)
                tiempo_desde = ahora - self.timestamp_ultima_emision

                # Emitir si es seña diferente o si ha pasado el cooldown
                if not es_misma or tiempo_desde > self.tiempo_cooldown_misma_sena:
                    self.ultima_sena_emitida = nombre_sena
                    self.timestamp_ultima_emision = ahora
                    self.oracion_actual.append(nombre_sena)
                    senas_detectadas.append({
                        "sena": nombre_sena,
                        "confianza": float(conf),
                        "timestamp": ahora
                    })

                    if self.callback_emision:
                        self.callback_emision(nombre_sena, {
                            "confianza": float(conf),
                            "oracion_completa": " ".join(self.oracion_actual)
                        })

        return {
            "senas_nuevas": senas_detectadas,
            "oracion_actual": list(self.oracion_actual),
            "texto_oracion": " ".join(self.oracion_actual).replace("_", " ")
        }

    def reiniciar_oracion(self):
        """Limpia el acumulador de la oración actual."""
        self.oracion_actual.clear()
        self.ultima_sena_emitida = ""

    def reiniciar_buffers(self):
        """Limpia todos los buffers temporales (ej. cuando no hay manos)."""
        self.buffer_manos.clear()
        self.buffer_rostro.clear()
        self.buffer_pose.clear()
        self.contador_fotogramas = 0
        self.ultima_sena_emitida = ""
