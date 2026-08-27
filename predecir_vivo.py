"""
=============================================================
RECONOCIMIENTO DE LSC EN VIVO POR CAMARA (CON HUD Y VOZ)
Lengua de Señas Colombiana (LSC)
=============================================================
Pipeline en tiempo real:
1. Captura de cámara (OpenCV).
2. Detección de manos y pose con telemetría articular (MediaPipe).
3. Clasificación de cuadrante espacial (Signing Space).
4. Búsqueda vectorial instantánea en 4,800+ vectores (<1ms).
5. Ensamblado lingüístico de frases LSC a Español natural.
6. Síntesis de voz local no bloqueante (pyttsx3).

Uso:
  python predecir_vivo.py
  python predecir_vivo.py --camera 0 --umbral 0.72
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

from motor_lsc import (
    ExtractorLandmarks,
    CuadranteEspacial,
    clasificar_cuadrante,
    BaseVectoresLSC,
    EnsambladorFrases,
    MotorVozLocal,
)
from motor_lsc.cuadrantes import COLORES_CUADRANTE


def dibujar_hud_profesional(
    frame: np.ndarray,
    fps: float,
    cuadrante_str: str,
    color_cuadrante: tuple,
    estado_mano_str: str,
    color_estado_mano: tuple,
    dedos_str: str,
    top_candidatos: list,
    glosas_acumuladas: list,
    frase_actual: str,
    umbral: float,
) -> np.ndarray:
    """
    Dibuja una interfaz visual moderna sobre el frame de video.
    """
    alto, ancho, _ = frame.shape
    hud = frame.copy()

    # 1. Barra Superior (Encabezado)
    overlay_top = hud.copy()
    cv2.rectangle(overlay_top, (0, 0), (ancho, 54), (18, 22, 30), -1)
    cv2.addWeighted(overlay_top, 0.88, hud, 0.12, 0, hud)

    # Título y FPS
    cv2.putText(hud, "SEÑA LSC - IA EN VIVO", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"FPS: {fps:4.1f}", (ancho - 110, 26), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)

    # Telemetría de Dedos
    cv2.putText(hud, f"DEDOS: {dedos_str}", (16, 46), cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 210, 255), 1, cv2.LINE_AA)

    # Badge de Cuadrante Activo
    cv2.rectangle(hud, (230, 8), (430, 36), color_cuadrante, -1)
    cv2.putText(hud, f"ZONA: {cuadrante_str}", (236, 26), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # Badge de Estado Cinético (Estable vs Transición)
    cv2.rectangle(hud, (440, 8), (620, 36), color_estado_mano, -1)
    cv2.putText(hud, estado_mano_str, (448, 26), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Panel Lateral Derecho: Top Candidatos
    if top_candidatos:
        px = ancho - 230
        py = 64
        overlay_side = hud.copy()
        cv2.rectangle(overlay_side, (px - 10, py - 5), (ancho - 10, py + 120), (18, 22, 30), -1)
        cv2.addWeighted(overlay_side, 0.85, hud, 0.15, 0, hud)
        cv2.putText(hud, "CANDIDATOS DETECTADOS:", (px, py + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (180, 190, 200), 1, cv2.LINE_AA)

        for idx, (sena, score, cuad, cat) in enumerate(top_candidatos[:3]):
            item_y = py + 44 + (idx * 26)
            color_score = (0, 255, 128) if score >= umbral else (100, 180, 255)
            cv2.putText(hud, f"{idx+1}. {sena[:10]}", (px, item_y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"{int(score*100)}%", (ancho - 55, item_y), cv2.FONT_HERSHEY_DUPLEX, 0.45, color_score, 1, cv2.LINE_AA)

    # 3. Barra Inferior: Glosas Acumuladas y Oración Generada
    overlay_bot = hud.copy()
    cv2.rectangle(overlay_bot, (0, alto - 90), (ancho, alto), (15, 18, 26), -1)
    cv2.addWeighted(overlay_bot, 0.92, hud, 0.08, 0, hud)

    # Glosas en progreso
    glosas_str = " ".join([f"[{g}]" for g in glosas_acumuladas]) if glosas_acumuladas else "[Esperando seña firme...]"
    cv2.putText(hud, f"Glosas LSC: {glosas_str}", (16, alto - 58), cv2.FONT_HERSHEY_DUPLEX, 0.50, (140, 200, 255), 1, cv2.LINE_AA)

    # Frase traducida y hablada
    frase_mostrar = frase_actual if frase_actual else "Realiza las señas y baja la mano para pronunciar la frase..."
    cv2.putText(hud, f"Español: {frase_mostrar}", (16, alto - 20), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

    return hud


def ejecutar_reconocimiento_en_vivo(
    camera_index: int = 0,
    base_npz_path: str = "modelos_guardados/base_senas_lsc.npz",
    umbral: float = 0.72,
):
    print("=" * 60)
    print("  INICIANDO RECONOCIMIENTO LSC EN VIVO (MOTOR ARTICULAR)")
    print("=" * 60)

    # 1. Cargar Base de Vectores
    base_vectores = BaseVectoresLSC(umbral_min_similitud=umbral)
    if os.path.exists(base_npz_path):
        base_vectores.cargar(base_npz_path)
        print(f"  Base vectorial cargada: {base_vectores.total_senas} vectores de referencia.")
        print(f"  Clases disponibles: {base_vectores.clases_unicas}")
    else:
        print(f"  [AVISO] No se encontró {base_npz_path}. Ejecutando compilación inicial...")
        from motor_lsc.generador_referencias import generar_base_completa
        base_vectores = generar_base_completa(salida_npz=base_npz_path)

    # 2. Inicializar Módulos
    extractor = ExtractorLandmarks()
    motor_voz = MotorVozLocal(rate=160, cooldown_segundos=1.5)

    ultima_frase_emitida = ""

    def al_confirmar_glosa(g: str):
        print(f"  [Seña confirmada]: +[{g}]")

    def al_finalizar_frase(frase: str):
        nonlocal ultima_frase_emitida
        ultima_frase_emitida = frase
        print(f"\n📢 [VOZ TTS]: \"{frase}\"\n")
        motor_voz.hablar(frase)

    ensamblador = EnsambladorFrases(
        frames_para_aceptar=6,          # 6 frames de estabilidad sostenida
        segundos_silencio_cierre=1.2,   # Cierra la frase tras 1.2s de descanso
        callback_frase_lista=al_finalizar_frase,
        callback_glosa_confirmada=al_confirmar_glosa,
    )

    # 3. Inicializar Cámara
    print(f"  Abriendo cámara #{camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir la cámara {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n  [CONTROLES]")
    print("    [Q]     - Salir")
    print("    [C]     - Limpiar buffer de glosas")
    print("    [ESPACIO]- Forzar pronunciación de frase actual")
    print("=" * 60 + "\n")

    t_previo = time.time()
    top_candidatos_actuales = []
    cuadrante_nombre = "DESCONOCIDO"
    color_cuad = (100, 100, 100)

    estado_mano_str = "ESPERANDO MANO"
    color_estado_mano = (60, 60, 60)
    dedos_str = "[P:- I:- M:- A:- M:-]"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            t_ahora = time.time()
            fps = 1.0 / max(t_ahora - t_previo, 1e-4)
            t_previo = t_ahora

            # Extracción de características
            res = extractor.procesar_frame(frame)

            sena_detectada = None
            score_max = 0.0
            mano_estable = False

            if res["hay_manos"]:
                mano = res["manos"][0]
                mano_estable = mano.get("es_estable", True)

                # Formatear telemetría de dedos
                exts = mano.get("finger_extensions", [0]*5)
                nombres_dedos = ["P", "I", "M", "A", "m"]
                dedos_str = " ".join([f"{nombres_dedos[i]}:{'1' if exts[i] > 0.55 else '0'}" for i in range(5)])

                # Clasificación de cuadrante
                cuad, _ = clasificar_cuadrante(
                    mano["muneca"],
                    res.get("pose_anchors"),
                    res["alto_frame"],
                    res["ancho_frame"],
                )
                cuadrante_nombre = cuad.value
                color_cuad = COLORES_CUADRANTE.get(cuad, (0, 255, 0))

                if mano_estable:
                    estado_mano_str = "SEÑA ESTABLE"
                    color_estado_mano = (0, 140, 60)

                    # Búsqueda articular instantánea (<1ms)
                    candidatos = base_vectores.buscar_similar(
                        vector_query=mano["vector_normalizado"],
                        cuadrante_query=cuadrante_nombre,
                        top_k=3,
                    )
                    top_candidatos_actuales = candidatos

                    # Filtro de no-adivinanza
                    if candidatos and candidatos[0][1] >= umbral:
                        margen_valido = True
                        if len(candidatos) > 1:
                            margen_valido = (candidatos[0][1] - candidatos[1][1]) >= 0.02
                        
                        if margen_valido:
                            sena_detectada = candidatos[0][0]
                            score_max = candidatos[0][1]
                else:
                    estado_mano_str = "MOVIENDO..."
                    color_estado_mano = (0, 100, 200)
                    top_candidatos_actuales = []
                    sena_detectada = None

                # Dibujar esqueleto de la mano
                frame = extractor.dibujar_overlays(
                    frame, res, cuadrante_nombre, color_cuad
                )
            else:
                cuadrante_nombre = "ESPERANDO"
                color_cuad = (80, 80, 80)
                estado_mano_str = "SIN MANO"
                color_estado_mano = (60, 60, 60)
                dedos_str = "[P:- I:- M:- A:- M:-]"
                top_candidatos_actuales = []

            # Actualizar ensamblador de frases
            glosas, _ = ensamblador.registrar_prediccion(
                sena_detectada, score_max, mano_estable=mano_estable
            )

            # Dibujar HUD profesional
            hud_frame = dibujar_hud_profesional(
                frame=frame,
                fps=fps,
                cuadrante_str=cuadrante_nombre,
                color_cuadrante=color_cuad,
                estado_mano_str=estado_mano_str,
                color_estado_mano=color_estado_mano,
                dedos_str=dedos_str,
                top_candidatos=top_candidatos_actuales,
                glosas_acumuladas=glosas,
                frase_actual=ultima_frase_emitida,
                umbral=umbral,
            )

            cv2.imshow("Seña LSC - Reconocimiento Inteligente", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("c") or key == ord("C"):
                ensamblador.glosas_acumuladas = []
                ensamblador.ultima_glosa_confirmada = None
                ensamblador.frase_en_construccion = False
                ultima_frase_emitida = ""
                print("  [Buffer de glosas limpiado]")
            elif key == ord(" "):
                ensamblador.forzar_cierre_frase()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.liberar()
        motor_voz.detener()
        print("\n[Cerrado] Reconocimiento finalizado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconocimiento LSC en vivo")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0)")
    parser.add_argument("--umbral", type=float, default=0.72, help="Umbral de similitud (default: 0.72)")
    args = parser.parse_args()

    ejecutar_reconocimiento_en_vivo(camera_index=args.camera, umbral=args.umbral)
