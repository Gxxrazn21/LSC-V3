"""
=============================================================
ENSAMBLADOR AVANZADO DE FRASES LSC -> ESPANOL (100% OFFLINE)
Lengua de Señas Colombiana (LSC)
=============================================================
Motor lingüístico estructurado que:
1. Filtra ruido de transición (anti-jitter) y espera estabilidad real.
2. Agrupa deletreo (secuencias de letras A-Z -> palabra).
3. Interpreta relaciones sintácticas: Sujeto + Verbo + Objeto + Modificadores.
4. Aplica reglas gramaticales y plantillas idiomáticas de LSC a Español natural.
5. Cierra y emite la oración por reposo/pausa, enviándola al TTS local.
"""

import time
import re
from typing import List, Optional, Tuple, Callable, Dict, Any


# ─────────────────────────────────────────────────────────────
# 1. CATEGORIZACION LEXICA DE SEÑAS LSC
# ─────────────────────────────────────────────────────────────

SUJETOS_PRONOMBRES = {
    "YO": {"pronombre": "yo", "posesivo": "mi", "objeto": "me", "persona": 1},
    "TU": {"pronombre": "tú", "posesivo": "tu", "objeto": "te", "persona": 2},
    "EL": {"pronombre": "él", "posesivo": "su", "objeto": "le", "persona": 3},
    "ELLA": {"pronombre": "ella", "posesivo": "su", "objeto": "la", "persona": 3},
    "NOSOTROS": {"pronombre": "nosotros", "posesivo": "nuestro", "objeto": "nos", "persona": 4},
    "AMIGO": {"sustantivo": "amigo", "articulo": "un", "persona": 3},
}

VERBOS = {
    "AYUDAR": {1: "te ayudo", 2: "me ayudas", 3: "ayuda", 4: "ayudamos", "inf": "ayudar"},
    "APOYAR": {1: "te apoyo", 2: "me apoyas", 3: "apoya", 4: "apoyamos", "inf": "apoyar"},
    "GUSTAR": {1: "me gusta", 2: "te gusta", 3: "le gusta", 4: "nos gusta", "inf": "gustar"},
    "QUERER": {1: "quiero", 2: "quieres", 3: "quiere", 4: "queremos", "inf": "querer"},
    "NECESITAR": {1: "necesito", 2: "necesitas", 3: "necesita", 4: "necesitamos", "inf": "necesitar"},
    "ESTAR": {1: "estoy", 2: "estás", 3: "está", 4: "estamos", "inf": "estar"},
    "TENER": {1: "tengo", 2: "tienes", 3: "tiene", 4: "tenemos", "inf": "tener"},
}

SUSTANTIVOS_OBJETOS = {
    "BAÑO": {"texto": "el baño", "prep": "al"},
    "CASA": {"texto": "la casa", "prep": "a la"},
    "LICOR": {"texto": "el licor", "prep": "con"},
    "NOMBRE": {"texto": "nombre"},
    "ANNOS": {"texto": "años"},
}

INTERROGATIVOS = {
    "DONDE": "¿Dónde",
    "COMO": "¿Cómo",
    "QUE": "¿Qué",
    "CUANDO": "¿Cuándo",
    "CUANTO": "¿Cuánto",
}

# ─────────────────────────────────────────────────────────────
# 2. PLANTILLAS IDIOMATICAS EXACTAS (PRIORIDAD ALTA)
# ─────────────────────────────────────────────────────────────

PLANTILLAS_EXACTAS: Dict[Tuple[str, ...], str] = {
    # Saludos y Bienvenidas
    ("HOLA",): "¡Hola!",
    ("BUENAS", "DIAS"): "¡Buenos días!",
    ("BUENAS", "TARDES"): "¡Buenas tardes!",
    ("BUENAS", "NOCHES"): "¡Buenas noches!",
    ("HOLA", "BUENAS", "DIAS"): "¡Hola, muy buenos días!",
    ("HOLA", "BUENAS", "TARDES"): "¡Hola, muy buenas tardes!",
    ("HOLA", "BUENAS", "NOCHES"): "¡Hola, muy buenas noches!",
    ("HOLA", "BIENVENIDO"): "¡Hola, bienvenido!",
    ("BIENVENIDO",): "¡Bienvenido!",
    ("GRACIAS",): "Muchas gracias.",
    ("BIEN", "GRACIAS"): "Muy bien, muchas gracias.",
    ("BIEN",): "Todo está bien.",

    # Necesidades y preguntas
    ("BAÑO",): "¿Dónde queda el baño?",
    ("DONDE", "BAÑO"): "¿Dónde queda el baño?",
    ("DONDE", "ESTAR", "BAÑO"): "¿Dónde está el baño?",
    ("NECESITAR", "BAÑO"): "Necesito ir al baño.",
    ("COMO", "ESTAR"): "¿Cómo estás?",
    ("COMO", "ESTAR", "TU"): "¿Cómo estás tú?",

    # Apoyo y Ayuda
    ("AYUDAR",): "Te ayudo.",
    ("YO", "AYUDAR"): "Yo te ayudo.",
    ("YO", "AYUDAR", "TU"): "Yo te puedo ayudar.",
    ("TU", "AYUDAR", "YO"): "¿Me puedes ayudar por favor?",
    ("TU", "AYUDAR"): "¿Me puedes ayudar?",
    ("APOYAR",): "Cuentas con mi apoyo.",
    ("YO", "APOYAR"): "Yo te apoyo.",
    ("YO", "APOYAR", "AMIGO"): "Yo apoyo a mi amigo.",
    ("APOYAR", "AMIGO"): "Apoyo a un amigo.",

    # Gustos y Preferencias
    ("YO", "GUSTAR"): "A mí me gusta.",
    ("GUSTAR", "LICOR"): "Me gusta el licor.",
    ("YO", "GUSTAR", "LICOR"): "A mí me gusta el licor.",
}


# ─────────────────────────────────────────────────────────────
# 3. VENTANA DE CONSENSO TEMPORAL (ANTI-JITTER / ANTI-ADIVINANZA)
# ─────────────────────────────────────────────────────────────

from collections import deque, Counter

class VentanaConsenso:
    """
    Suavizado temporal por mayoría de votos en ventana deslizante.
    
    Solo acepta una seña cuando:
      1. Al menos `min_consenso_pct`% de los últimos `tamano_ventana` frames
         coinciden en la MISMA seña.
      2. El score promedio de esa seña en la ventana supera `umbral_score_promedio`.
      3. Si ya hay una seña aceptada, se necesitan `frames_histeresis` frames 
         consecutivos con otra seña diferente para cambiar (anti-jitter).
    """
    def __init__(
        self,
        tamano_ventana: int = 10,
        min_consenso_pct: float = 0.70,
        umbral_score_promedio: float = 0.78,
        frames_histeresis: int = 3,
    ):
        self.tamano_ventana = tamano_ventana
        self.min_consenso_pct = min_consenso_pct
        self.umbral_score_promedio = umbral_score_promedio
        self.frames_histeresis = frames_histeresis
        
        self.buffer: deque = deque(maxlen=tamano_ventana)
        self.sena_aceptada_actual: Optional[str] = None
        self.contador_otra_sena: int = 0
        self.ultima_otra_sena: Optional[str] = None

    def alimentar(
        self,
        sena: Optional[str],
        score: float,
    ) -> Tuple[Optional[str], float, float]:
        """
        Registra una predicción de frame y devuelve la seña consensuada.
        
        Returns:
            (seña_consensuada, score_promedio, porcentaje_consenso)
        """
        self.buffer.append((sena, score))
        votos = [(s, sc) for s, sc in self.buffer if s is not None]
        
        if not votos:
            self.sena_aceptada_actual = None
            self.contador_otra_sena = 0
            return (None, 0.0, 0.0)

        conteo = Counter(s for s, _ in votos)
        sena_top, n_votos = conteo.most_common(1)[0]
        pct_consenso = n_votos / self.tamano_ventana

        scores_top = [sc for s, sc in votos if s == sena_top]
        score_promedio = sum(scores_top) / len(scores_top)

        if pct_consenso < self.min_consenso_pct or score_promedio < self.umbral_score_promedio:
            return (None, score_promedio, pct_consenso)

        # Histéresis
        if self.sena_aceptada_actual is not None and sena_top != self.sena_aceptada_actual:
            if sena_top == self.ultima_otra_sena:
                self.contador_otra_sena += 1
            else:
                self.ultima_otra_sena = sena_top
                self.contador_otra_sena = 1

            if self.contador_otra_sena < self.frames_histeresis:
                return (self.sena_aceptada_actual, score_promedio, pct_consenso)

        self.sena_aceptada_actual = sena_top
        self.contador_otra_sena = 0
        self.ultima_otra_sena = None

        return (sena_top, score_promedio, pct_consenso)

    def limpiar(self):
        self.buffer.clear()
        self.sena_aceptada_actual = None
        self.contador_otra_sena = 0
        self.ultima_otra_sena = None


class EnsambladorFrases:
    def __init__(
        self,
        frames_para_aceptar: int = 7,
        segundos_silencio_cierre: float = 1.2,
        callback_frase_lista: Optional[Callable[[str], None]] = None,
        callback_glosa_confirmada: Optional[Callable[[str], None]] = None,
    ):
        self.frames_para_aceptar = frames_para_aceptar
        self.segundos_silencio_cierre = segundos_silencio_cierre
        self.callback_frase_lista = callback_frase_lista
        self.callback_glosa_confirmada = callback_glosa_confirmada

        # Buffer y estado
        self.glosa_candidata: Optional[str] = None
        self.conteo_frames: int = 0
        self.glosas_acumuladas: List[str] = []
        self.ultima_glosa_confirmada: Optional[str] = None
        
        self.tiempo_ultimo_gesto: float = time.time()
        self.frase_en_construccion: bool = False

    def registrar_prediccion(
        self,
        sena_detectada: Optional[str],
        score: float,
        mano_estable: bool = True,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Alimenta el motor con la seña detectada en el frame actual.
        
        Args:
            sena_detectada: Nombre de la seña o None.
            score: Nivel de confianza [0.0, 1.0].
            mano_estable: True si la mano no está en movimiento brusco de transición.
            
        Returns:
            (glosas_actuales, frase_completa_si_terminó)
        """
        ahora = time.time()
        frase_emitida = None

        if sena_detectada and mano_estable and score >= 0.70:
            self.tiempo_ultimo_gesto = ahora
            sena_clean = sena_detectada.strip().upper()

            if sena_clean == self.glosa_candidata:
                self.conteo_frames += 1
            else:
                self.glosa_candidata = sena_clean
                self.conteo_frames = 1

            # Verificar si se cumple el tiempo mínimo de permanencia
            if (
                self.conteo_frames >= self.frames_para_aceptar
                and sena_clean != self.ultima_glosa_confirmada
            ):
                self.glosas_acumuladas.append(sena_clean)
                self.ultima_glosa_confirmada = sena_clean
                self.frase_en_construccion = True
                self.conteo_frames = 0

                if self.callback_glosa_confirmada:
                    self.callback_glosa_confirmada(sena_clean)
        else:
            # Si no hay mano o está en movimiento de transición
            self.glosa_candidata = None
            self.conteo_frames = 0

        # Evaluar cierre por pausa/silencio
        if self.frase_en_construccion:
            inactividad = ahora - self.tiempo_ultimo_gesto
            if inactividad >= self.segundos_silencio_cierre:
                frase_emitida = self.construir_oracion_inteligente(self.glosas_acumuladas)
                if self.callback_frase_lista and frase_emitida:
                    self.callback_frase_lista(frase_emitida)

                # Resetear buffer para la próxima oración
                self.glosas_acumuladas = []
                self.ultima_glosa_confirmada = None
                self.frase_en_construccion = False

        return self.glosas_acumuladas, frase_emitida

    def forzar_cierre_frase(self) -> Optional[str]:
        """Fuerza la construcción y pronunciación inmediata de la frase acumulada."""
        if not self.glosas_acumuladas:
            return None
        frase = self.construir_oracion_inteligente(self.glosas_acumuladas)
        self.glosas_acumuladas = []
        self.ultima_glosa_confirmada = None
        self.frase_en_construccion = False
        if self.callback_frase_lista and frase:
            self.callback_frase_lista(frase)
        return frase

    @classmethod
    def construir_oracion_inteligente(cls, glosas: List[str]) -> str:
        """
        Construye una oración fluida en español a partir de una secuencia de glosas LSC.
        """
        if not glosas:
            return ""

        # 1. Preprocesamiento: Agrupar secuencias de deletreo (letras A-Z)
        glosas_procesadas = cls._agrupar_deletreo(glosas)

        # 2. Coincidencia exacta con plantilla idiomática
        tupla_g = tuple(glosas_procesadas)
        if tupla_g in PLANTILLAS_EXACTAS:
            return PLANTILLAS_EXACTAS[tupla_g]

        # 3. Reglas sintácticas avanzadas (Sujeto + Nombre / Edad / Verbo)
        oracion_sintactica = cls._aplicar_reglas_sintacticas(glosas_procesadas)
        if oracion_sintactica:
            return oracion_sintactica

        # 4. Greedy Sub-template matching
        resultado_tokens = []
        i = 0
        n = len(glosas_procesadas)
        while i < n:
            emparejado = False
            for tam in [4, 3, 2]:
                if i + tam <= n:
                    sub_t = tuple(glosas_procesadas[i : i + tam])
                    if sub_t in PLANTILLAS_EXACTAS:
                        resultado_tokens.append(PLANTILLAS_EXACTAS[sub_t])
                        i += tam
                        emparejado = True
                        break
            if not emparejado:
                palabra = glosas_procesadas[i].capitalize()
                resultado_tokens.append(palabra)
                i += 1

        texto = " ".join(resultado_tokens).strip()
        if texto:
            texto = texto[0].upper() + texto[1:]
            if not texto.endswith((".", "!", "?")):
                texto += "."
        return texto

    @staticmethod
    def _agrupar_deletreo(glosas: List[str]) -> List[str]:
        """Agrupa letras consecutivas (deletreo dactilológico) en una sola palabra."""
        salida = []
        buffer_letras = []

        for g in glosas:
            # Si es una letra individual (longitud 1 y alfabética)
            if len(g) == 1 and g.isalpha():
                buffer_letras.append(g)
            else:
                if buffer_letras:
                    palabra_deletreada = "".join(buffer_letras).capitalize()
                    salida.append(palabra_deletreada)
                    buffer_letras = []
                salida.append(g)

        if buffer_letras:
            salida.append("".join(buffer_letras).capitalize())

        return salida

    @classmethod
    def _aplicar_reglas_sintacticas(cls, glosas: List[str]) -> Optional[str]:
        """Aplica reglas de estructura gramatical LSC -> Español."""
        n = len(glosas)
        if n == 0:
            return None

        # Regla A: Identificación de Nombre (ej: [YO, NOMBRE, JHON] o [NOMBRE, JHON])
        if "NOMBRE" in glosas:
            idx = glosas.index("NOMBRE")
            resto = [g for i, g in enumerate(glosas) if i != idx and g != "YO"]
            nombre_str = " ".join(resto) if resto else "alguien"
            if "YO" in glosas or idx == 0:
                return f"Mi nombre es {nombre_str.capitalize()}."
            elif "TU" in glosas:
                return "¿Cuál es tu nombre?"
            return f"Nombre: {nombre_str.capitalize()}."

        # Regla B: Edad / Años (ej: [YO, ANNOS, 10] o [YO, 10, ANNOS] o [5, ANNOS])
        if "ANNOS" in glosas:
            numeros = [g for g in glosas if g.isdigit() or g in ["MIL", "MILLON"]]
            num_str = numeros[0] if numeros else "algunos"
            if "YO" in glosas:
                return f"Yo tengo {num_str} años."
            elif "TU" in glosas:
                return f"¿Tienes {num_str} años?"
            return f"{num_str} años."

        # Regla C: Preguntas con Interrogativos (DONDE, COMO, etc.)
        primera = glosas[0]
        if primera in INTERROGATIVOS:
            interrogativo = INTERROGATIVOS[primera]
            resto_palabras = glosas[1:]
            if resto_palabras:
                sustantivo = resto_palabras[0]
                if sustantivo == "BAÑO":
                    return f"{interrogativo} queda el baño?"
                elif sustantivo == "CASA":
                    return f"{interrogativo} está la casa?"
                else:
                    return f"{interrogativo} está {sustantivo.lower()}?"
            return f"{interrogativo}?"

        # Regla D: Sujeto + Verbo + Objeto
        sujeto_info = None
        verbo_info = None
        objeto_str = ""

        for g in glosas:
            if g in SUJETOS_PRONOMBRES and sujeto_info is None:
                sujeto_info = SUJETOS_PRONOMBRES[g]
            elif g in VERBOS and verbo_info is None:
                verbo_info = VERBOS[g]
            elif g in SUSTANTIVOS_OBJETOS:
                objeto_str = SUSTANTIVOS_OBJETOS[g]["texto"]

        if sujeto_info and verbo_info:
            persona = sujeto_info.get("persona", 1)
            verbo_conjugado = verbo_info.get(persona, verbo_info.get("inf", ""))
            
            if objeto_str:
                return f"{sujeto_info['pronombre'].capitalize()} {verbo_conjugado} {objeto_str}."
            else:
                return f"{sujeto_info['pronombre'].capitalize()} {verbo_conjugado}."

        return None
