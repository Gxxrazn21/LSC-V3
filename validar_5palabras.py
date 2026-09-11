"""
=============================================================
VALIDACION EN VIVO - RECONOCIMIENTO LSC CON IA (v3.0)
Lengua de Señas Colombiana (LSC)
=============================================================
Soporta:
1. Modelo de IA (Ensamble MLP + Extra-Trees entrenado con LSC70).
2. Motor Anti-Adivinanza (Anti-Guessing Engine):
   - Muestra explícitamente cuando la mano está en REPOSO o TRANSICION.
   - Rechaza predicciones ambiguas (margen insuficiente entre candidatos).
   - Inmune a falsos positivos en mano relajada.
3. Modo Vectorial tradicional (fallback .npz).

Uso:
  python validar_5palabras.py
  python validar_5palabras.py --camera 0 --umbral 0.72 --margen 0.15
  python validar_5palabras.py --vectorial
"""

import os
import sys
import time
import argparse
import collections
import numpy as np
import cv2

from motor_lsc import ExtractorLandmarks, clasificar_cuadrante, BaseVectoresLSC, ClasificadorIALSC, CamaraLSC
from motor_lsc.ensamblador_frases import VentanaConsenso

# ─────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────

MODELO_IA_PATH = os.path.join("modelos_guardados", "modelo_ia_lsc70.joblib")
MODELO_5P_PATH = os.path.join("modelos_guardados", "base_5palabras.npz")

# Colores (BGR)
COLOR_VERDE    = (0, 220, 120)
COLOR_AZUL     = (255, 160, 50)
COLOR_NARANJA  = (0, 140, 255)
COLOR_ROJO     = (60, 60, 230)
COLOR_BLANCO   = (240, 240, 240)
COLOR_GRIS     = (120, 120, 130)
COLOR_FONDO    = (20, 20, 28)
COLOR_AMARILLO = (0, 220, 220)


# ─────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────

def _texto(frame, texto, x, y, color, escala=0.6, grosor=1):
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0,0,0), grosor+2, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def _barra(frame, x, y, ancho, alto, valor, color_barra, max_val=1.0, color_bg=(40,40,50)):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color_bg, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (70, 70, 80), 1)
    fill = int(ancho * min(valor / max_val, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + alto), color_barra, -1)


def _panel(frame, x1, y1, x2, y2, alpha=0.72):
    roi = frame[y1:y2, x1:x2]
    bg = np.full_like(roi, COLOR_FONDO)
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, bg, alpha, 0)


def dibujar_hud_validacion(
    frame, fps, latencia_ms, candidatos, consenso_sena,
    consenso_pct, umbral, historial, total_preds,
    correctas, sesion_accuracy, estado_ia="ESPERANDO_MANO", margen_ia=0.0, modo_ia=True
):
    alto, ancho = frame.shape[:2]
    
    # ── Panel superior ──
    _panel(frame, 0, 0, ancho, 72)
    titulo = "LSC v3.0 - IA Anti-Adivinanza (LSC70 MLP+ExtraTrees)" if modo_ia else "LSC v3.0 - Validacion Vectorial"
    _texto(frame, titulo, 14, 26, COLOR_AZUL, 0.68, 2)
    _texto(frame, f"FPS: {fps:.1f}  |  Latencia: {latencia_ms:.1f}ms  |"
                  f"  Umbral: {umbral:.0%}  |  Margen: {margen_ia:.0%}", 14, 56, COLOR_GRIS, 0.48, 1)

    # ── Panel derecho: candidatos ──
    px = ancho - 310
    _panel(frame, px - 10, 80, ancho - 10, 330)
    _texto(frame, "TOP CANDIDATOS:", px, 102, COLOR_NARANJA, 0.55, 1)

    for i, cand in enumerate(candidatos[:3]):
        etq, sim = cand[0], cand[1]
        y_c = 130 + i * 60
        color_c = COLOR_VERDE if i == 0 and sim >= umbral else \
                  COLOR_AMARILLO if i == 0 else COLOR_GRIS
        _texto(frame, f"#{i+1}  {etq[:12]}", px, y_c, color_c, 0.62, 1 if i > 0 else 2)
        _texto(frame, f"{sim:.1%}", px + 200, y_c, color_c, 0.62, 1)
        _barra(frame, px, y_c + 8, 270, 10, sim, color_c)

    # ── Prediccion principal (Centro) ──
    txt_sub = ""
    if consenso_sena and consenso_pct >= umbral:
        txt_pred = consenso_sena
        color_pred = COLOR_VERDE
        txt_sub = f"CONSENSO CONFIRMADO ({consenso_pct:.0%})"
    elif estado_ia == "REPOSO":
        txt_pred = "[REPOSO]"
        color_pred = COLOR_GRIS
        txt_sub = "Mano neutra / sin seña (Anti-adivinanza activo)"
    elif estado_ia == "TRANSICION":
        txt_pred = "[TRANSICION]"
        color_pred = COLOR_NARANJA
        txt_sub = "Mano en movimiento rapido"
    elif estado_ia == "DUDOSO":
        txt_pred = "[INCIERTO]"
        color_pred = COLOR_AMARILLO
        txt_sub = f"Margen ambiguo ({margen_ia:.1%}) - No adivina"
    elif candidatos and candidatos[0][1] >= umbral:
        txt_pred = candidatos[0][0]
        color_pred = COLOR_VERDE
        txt_sub = f"Confianza: {candidatos[0][1]:.1%}"
    else:
        txt_pred = "---"
        color_pred = COLOR_GRIS
        txt_sub = "Esperando seña firme..."

    _panel(frame, 0, alto // 2 - 65, px - 20, alto // 2 + 65)
    (w_t, h_t), _ = cv2.getTextSize(txt_pred, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)
    cx = max(14, (px - 20 - w_t) // 2)
    _texto(frame, txt_pred, cx, alto // 2 + h_t // 2 - 8, color_pred, 2.2, 4)

    if txt_sub:
        (w_s, _), _ = cv2.getTextSize(txt_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cx_s = max(14, (px - 20 - w_s) // 2)
        _texto(frame, txt_sub, cx_s, alto // 2 + 48, COLOR_BLANCO, 0.50, 1)

    # ── Panel inferior: historial ──
    _panel(frame, 0, alto - 130, ancho, alto)
    _texto(frame, "HISTORIAL CONFIRMADO:", 14, alto - 108, COLOR_NARANJA, 0.5, 1)
    hist_str = "  ->  ".join(list(historial)[-8:]) if historial else "(sin predicciones aun)"
    _texto(frame, hist_str, 14, alto - 82, COLOR_BLANCO, 0.55, 1)

    # Metricas de sesion
    acc_str = f"Sesion: {total_preds} detecciones firmes | Confianza prom: {sesion_accuracy:.1%}"
    _texto(frame, acc_str, 14, alto - 50, COLOR_GRIS, 0.5, 1)
    _texto(frame, "[Q/ESC] Salir | [R] Resetear stats | [C] Captura test", 14, alto - 22,
           COLOR_GRIS, 0.45, 1)

    return frame


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validacion en vivo LSC con IA y Anti-Adivinanza")
    parser.add_argument("--camera", default="0", help="Indice de camara USB (ej: 0) o URL de camara IP del celular (ej: http://192.168.1.X:8080/video)")
    parser.add_argument("--umbral", type=float, default=0.70, help="Umbral minimo de confianza (default: 0.70)")
    parser.add_argument("--margen", type=float, default=0.12, help="Margen minimo entre Top1 y Top2 (default: 0.12)")
    parser.add_argument("--modelo", default=None, help="Ruta al modelo (.joblib para IA o .npz para vectorial)")
    parser.add_argument("--vectorial", action="store_true", help="Forzar uso del modelo vectorial .npz")
    args = parser.parse_args()

    # Determinar modo (IA vs Vectorial)
    usar_ia = False
    ruta_modelo = args.modelo

    if not args.vectorial:
        if ruta_modelo and ruta_modelo.endswith(".joblib") and os.path.exists(ruta_modelo):
            usar_ia = True
        elif (ruta_modelo is None or ruta_modelo == MODELO_IA_PATH) and os.path.exists(MODELO_IA_PATH):
            usar_ia = True
            ruta_modelo = MODELO_IA_PATH

    if not usar_ia:
        if ruta_modelo is None:
            ruta_modelo = MODELO_5P_PATH

    print("=" * 65)
    print("  LSC v3.0 - SISTEMA DE VALIDACION EN VIVO")
    print("=" * 65)

    clasificador_ia = None
    base_vectorial = None

    if usar_ia:
        print(f"  [MODO IA ACTIVO] Cargando ensamble híbrido (MLP + ExtraTrees)...")
        if not os.path.exists(ruta_modelo):
            print(f"  ERROR: No se encontró el modelo de IA en {ruta_modelo}")
            print("  Ejecuta primero: python entrenar_ia_lsc70.py")
            sys.exit(1)
        clasificador_ia = ClasificadorIALSC(
            modelo_path=ruta_modelo,
            umbral_confianza=args.umbral,
            margen_minimo=args.margen,
        )
        print(f"  Modelo IA cargado : {ruta_modelo}")
        print(f"  Clases IA ({len(clasificador_ia.clases)}): {clasificador_ia.clases}")
        print(f"  Umbral confianza  : {clasificador_ia.umbral_confianza:.0%}")
        print(f"  Margen minimo     : {clasificador_ia.margen_minimo:.0%}")
    else:
        print(f"  [MODO VECTORIAL ACTIVO] Cargando base de vectores...")
        if not os.path.exists(ruta_modelo):
            print(f"  ERROR: Modelo no encontrado: {ruta_modelo}")
            sys.exit(1)
        base_vectorial = BaseVectoresLSC(umbral_min_similitud=args.umbral)
        base_vectorial.cargar(ruta_modelo)
        print(f"  Modelo cargado    : {ruta_modelo}")
        print(f"  Vectores totales  : {base_vectorial.total_senas}")
        print(f"  Clases            : {base_vectorial.clases_unicas}")

    print("=" * 65)

    # Inicializar Camara con API DirectShow y realce optico adaptativo
    cam = CamaraLSC(
        camera_index=args.camera,
        ancho=1280,
        alto=720,
        fps=30,
        aplicar_mejora_optica=True,
    )
    if not cam.cap or not cam.cap.isOpened():
        print(f"ERROR: No se puede abrir camara #{args.camera}")
        sys.exit(1)

    # Extractor con estabilización
    extractor = ExtractorLandmarks(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
        usar_filtro_temporal=True,
        usar_denoise_imagen=True,
    )

    # Estado de sesión y consenso
    ventana_consenso = VentanaConsenso(
        tamano_ventana=10,
        min_consenso_pct=0.65,
        umbral_score_promedio=args.umbral,
        frames_histeresis=3,
    )
    historial = collections.deque(maxlen=20)
    total_preds = 0
    suma_confianzas = 0.0
    ultima_prediccion = ""
    sesion_accuracy = 0.0

    t_fps = time.time()
    fps = 0.0
    frame_count = 0

    print("\n  Iniciando validacion en vivo... Presiona [Q] o [ESC] para salir.\n")

    while True:
        ret, frame_raw, frame = cam.leer(voltear_espejo=True)
        if not ret or frame is None:
            break

        t0 = time.time()
        resultado = extractor.procesar_frame(frame)
        latencia_ms = (time.time() - t0) * 1000

        frame_viz = extractor.dibujar_overlays(frame, resultado)

        candidatos = []
        consenso_sena = ""
        consenso_pct = 0.0
        estado_ia = "ESPERANDO_MANO"
        margen_ia = 0.0

        if resultado["hay_manos"]:
            mano = resultado["manos"][0]
            vector = mano["vector_normalizado"]
            mano_estable = mano.get("es_estable", True)

            cuad_obj, _ = clasificar_cuadrante(
                mano.get("muneca", (0.5, 0.5, 0)),
                resultado.get("pose_anchors", {}),
            )
            cuadrante_str = cuad_obj.value if hasattr(cuad_obj, "value") else str(cuad_obj)

            if usar_ia and clasificador_ia:
                res_ia = clasificador_ia.predecir(
                    vector_105d=vector,
                    cuadrante=cuadrante_str,
                    mano_estable=mano_estable,
                    top_k=3,
                )
                candidatos = res_ia["candidatos"]
                estado_ia = res_ia["estado"]
                margen_ia = res_ia["margen"]
                es_valida = res_ia["es_valida"]

                pred_label = res_ia["etiqueta"] if es_valida else None
                score_pred = res_ia["confianza"] if es_valida else 0.0

                consenso_sena, score_prom, consenso_pct = ventana_consenso.alimentar(
                    pred_label, score_pred
                )

                if consenso_sena and consenso_sena != ultima_prediccion:
                    ultima_prediccion = consenso_sena
                    historial.append(consenso_sena)
                    total_preds += 1
                    suma_confianzas += score_pred
                    print(f"  -> [IA] {consenso_sena:15s} ({score_pred:.1%}) | Margen: {margen_ia:.1%}")
            else:
                # Búsqueda vectorial tradicional
                candidatos = base_vectorial.buscar_similar(vector, cuadrante_query=cuadrante_str, top_k=3)
                estado_ia = "SEÑA_DETECTADA" if (candidatos and candidatos[0][1] >= args.umbral) else "BAJA_CONFIANZA"
                if candidatos and len(candidatos) > 1:
                    margen_ia = candidatos[0][1] - candidatos[1][1]

                if candidatos and candidatos[0][1] >= args.umbral:
                    pred_label = candidatos[0][0]
                    consenso_sena, score_prom, consenso_pct = ventana_consenso.alimentar(
                        pred_label, candidatos[0][1]
                    )
                    if consenso_sena and consenso_sena != ultima_prediccion:
                        ultima_prediccion = consenso_sena
                        historial.append(consenso_sena)
                        total_preds += 1
                        suma_confianzas += candidatos[0][1]
                        print(f"  -> [VECT] {consenso_sena:15s} ({candidatos[0][1]:.1%})")
                else:
                    consenso_sena, _, consenso_pct = ventana_consenso.alimentar(None, 0.0)

        # FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - t_fps)
            t_fps = time.time()

        sesion_accuracy = (suma_confianzas / total_preds) if total_preds > 0 else 0.0

        frame_viz = dibujar_hud_validacion(
            frame_viz, fps, latencia_ms, candidatos, consenso_sena,
            consenso_pct, args.umbral, historial, total_preds,
            0, sesion_accuracy, estado_ia=estado_ia, margen_ia=margen_ia, modo_ia=usar_ia
        )

        cv2.imshow("LSC v3.0 - Validacion", frame_viz)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            historial.clear()
            total_preds = 0
            suma_confianzas = 0.0
            ultima_prediccion = ""
            ventana_consenso.reset() if hasattr(ventana_consenso, "reset") else None
            print("  Estadisticas de sesion reseteadas.")
        elif key in (ord("c"), ord("C")):
            ts = int(time.time())
            os.makedirs("resultados", exist_ok=True)
            cv2.imwrite(f"resultados/validacion_test_{ts}.png", frame_viz)
            print(f"  Captura guardada: resultados/validacion_test_{ts}.png")

    # Resumen final
    print("\n" + "=" * 65)
    print("  RESUMEN DE SESION DE VALIDACION")
    print("=" * 65)
    print(f"  Modo utilizado     : {'IA (Ensamble LSC70)' if usar_ia else 'Vectorial'}")
    print(f"  Total predicciones : {total_preds}")
    print(f"  Confianza promedio : {sesion_accuracy:.1%}")
    print(f"  Historial completo : {list(historial)}")
    print("=" * 65)

    extractor.liberar()
    cam.liberar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
