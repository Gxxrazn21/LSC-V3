"""
=============================================================
MOTOR DE SINTESIS DE VOZ LOCAL (TTS OFFLINE)
Lengua de Señas Colombiana (LSC)
=============================================================
Pronunciación asíncrona de frases en hilo daemon (no bloqueante),
100% offline utilizando pyttsx3.
"""

import queue
import threading
import time
from typing import Optional


class MotorVozLocal:
    def __init__(self, rate: int = 160, cooldown_segundos: float = 1.5):
        self.rate = rate
        self.cooldown_segundos = cooldown_segundos
        
        self.cola_voz: queue.Queue = queue.Queue()
        self.ultima_frase_hablada: Optional[str] = None
        self.tiempo_ultimo_hablado: float = 0.0
        
        # Iniciar hilo daemon de pronunciación
        self.hilo = threading.Thread(target=self._bucle_trabajador, daemon=True)
        self.hilo.start()

    def _bucle_trabajador(self):
        """Worker en segundo plano que inicializa pyttsx3 y consume la cola."""
        try:
            import pyttsx3
            motor = pyttsx3.init()
            motor.setProperty("rate", self.rate)

            # Intentar seleccionar una voz en español si está disponible
            voces = motor.getProperty("voices")
            for voz in voces:
                if "spanish" in voz.name.lower() or "es" in voz.id.lower() or "helena" in voz.name.lower() or "sabina" in voz.name.lower():
                    motor.setProperty("voice", voz.id)
                    break
        except Exception as e:
            print(f"[TTS] Advertencia al inicializar pyttsx3: {e}")
            motor = None

        while True:
            try:
                texto = self.cola_voz.get()
                if texto is None:
                    break

                if motor:
                    motor.say(texto)
                    motor.runAndWait()
                else:
                    print(f"[TTS Audio Simulado]: {texto}")

                self.cola_voz.task_done()
            except Exception as err:
                print(f"[TTS] Error al pronunciar: {err}")
                time.sleep(0.1)

    def hablar(self, texto: str, ignorar_cooldown: bool = False):
        """
        Encola un texto para pronunciación sin congelar el hilo principal.
        """
        if not texto or not texto.strip():
            return

        texto_clean = texto.strip()
        ahora = time.time()

        if (
            not ignorar_cooldown
            and texto_clean == self.ultima_frase_hablada
            and (ahora - self.tiempo_ultimo_hablado) < self.cooldown_segundos
        ):
            return

        self.ultima_frase_hablada = texto_clean
        self.tiempo_ultimo_hablado = ahora
        self.cola_voz.put(texto_clean)

    def detener(self):
        """Detiene el hilo de voz."""
        self.cola_voz.put(None)
