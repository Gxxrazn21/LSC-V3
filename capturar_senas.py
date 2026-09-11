"""
=============================================================
CAPTURA INTERACTIVA DE SEÑAS LSC v3.1 — 5 PALABRAS
Lengua de Señas Colombiana (LSC)
=============================================================
Mejoras v3.1:
  - Auto-captura: cuando la mano está quieta X segundos, captura sola.
  - FPS estable: preprocesamiento ligero en modo captura.
  - Barra de progreso visual de estabilidad (countdown animado).
  - Pantalla de transición corregida.
  - Feedback claro en consola y en pantalla.

Las 5 señas objetivo:
  HOLA    - Mano abierta, saludo lateral
  GRACIAS - Yemas de dedos al mentón -> adelante
  BIEN    - Pulgar arriba (puño cerrado)
  CASA    - Triángulo con ambas manos (techo)
  AGUA    - Forma W (índice+medio+anular) en labios

Uso:
  python capturar_senas.py
  python capturar_senas.py --camera 1 --muestras 8 --auto
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

from motor_lsc import ExtractorLandmarks, clasificar_cuadrante, CamaraLSC
from motor_lsc.extractor import preprocesar_imagen_anti_ruido

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

SENAS_OBJETIVO = ["HOLA", "GRACIAS", "BIEN", "CASA", "AGUA"]

# Instrucciones extendidas — se puede pasar cualquier seña
INSTRUCCIONES_SENAS = {
    "HOLA": [
        "Mano derecha abierta, 5 dedos extendidos.",
        "Palma hacia el frente.",
        "Para la foto: mantén la mano quieta.",
    ],
    "GRACIAS": [
        "Junta los dedos de la mano derecha.",
        "Toca el mentón con las yemas.",
        "Luego extiende hacia adelante.",
    ],
    "BIEN": [
        "Extiende el pulgar derecho hacia arriba",
        "(thumbs up / like). Los otros dedos",
        "cerrados en puño. Palma hacia ti.",
    ],
    "CASA": [
        "Junta los índices y pulgares de ambas manos",
        "formando un triángulo (techo de casa).",
        "Coloca las manos frente a ti.",
    ],
    "AGUA": [
        "Forma W: extiende índice, medio y anular",
        "de la mano dominante. Toca los labios",
        "con el dedo medio brevemente.",
    ],
    "BUENOS_DIAS": [
        "Abre la mano dominante en forma de saludo.",
        "Mueve la mano desde la frente hacia afuera",
        "(como un saludo militar suave).",
    ],
    "BUENAS_TARDES": [
        "Mano dominante abierta, dedos juntos.",
        "Mueve la mano desde el pecho hacia adelante",
        "con un ligero arco descendente.",
    ],
    "BUENAS_NOCHES": [
        "Junta los dedos de ambas manos frente",
        "al pecho y llévalas hacia abajo",
        "(gesto de 'cerrar el día').",
    ],
    "NOMBRE": [
        "Forma una N con los dedos dominantes",
        "y muévela ligeramente hacia adelante.",
        "Dos dedos (índice+medio) sobre el pulgar.",
    ],
    "POR_FAVOR": [
        "Mano abierta en el pecho, palma adentro.",
        "Realiza un movimiento circular",
        "en sentido horario sobre el pecho.",
    ],
}

DIR_SALIDA = os.path.join("datasets", "capturado")

# ─────────────────────────────────────────────────────────────
# COLORES (BGR)
# ─────────────────────────────────────────────────────────────
C_VERDE   = (0, 210, 110)
C_AZUL    = (220, 140, 40)
C_NARANJA = (0, 140, 255)
C_ROJO    = (50, 50, 210)
C_BLANCO  = (240, 240, 240)
C_FONDO   = (18, 18, 26)
C_GRIS    = (110, 110, 120)
C_AMARILLO= (0, 215, 215)


# ─────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────

def _txt(frame, texto, x, y, color, escala=0.58, grosor=1):
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


def _barra(frame, x, y, w, h, pct, color, bg=(38, 38, 50)):
    pct = max(0.0, min(1.0, pct))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 80), 1)
    if pct > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * pct), y + h), color, -1)


def _panel(frame, x1, y1, x2, y2, alpha=0.72):
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if y2 > y1 and x2 > x1:
        roi = frame[y1:y2, x1:x2]
        bg = np.full_like(roi, C_FONDO)
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, bg, alpha, 0)


# ─────────────────────────────────────────────────────────────
# HUD PRINCIPAL
# ─────────────────────────────────────────────────────────────

def dibujar_hud(frame, sena, idx_sena, total_senas, n_capturadas, n_total,
                hay_mano, estable, hold_pct, fase, modo_auto):
    alto, ancho = frame.shape[:2]

    # ── Panel superior ──────────────────────────────────────
    _panel(frame, 0, 0, ancho, 80)
    _txt(frame, "LSC v3.0  Captura de Senas", 14, 24, C_AZUL, 0.65, 2)

    progreso_txt = f"Sena {idx_sena+1}/{total_senas}:  {sena}"
    _txt(frame, progreso_txt, 14, 54, C_BLANCO, 0.9, 2)

    prog_pct = idx_sena / total_senas
    _barra(frame, 14, 65, ancho - 28, 10, prog_pct, C_AZUL)

    # ── Panel instrucciones (derecha) ───────────────────────
    pw = 310
    px = ancho - pw - 8
    _panel(frame, px, 88, ancho - 8, 88 + 145)
    _txt(frame, "INSTRUCCIONES:", px + 8, 108, C_NARANJA, 0.52, 1)
    for i, linea in enumerate(INSTRUCCIONES_SENAS.get(sena, [])):
        _txt(frame, linea, px + 8, 132 + i * 26, C_BLANCO, 0.45, 1)

    # ── Panel inferior ──────────────────────────────────────
    _panel(frame, 0, alto - 120, ancho, alto)

    # Estado mano
    if hay_mano and estable:
        estado = "Mano ESTABLE - listo para capturar"
        ce = C_VERDE
    elif hay_mano:
        estado = "Mano detectada - espera que se estabilice..."
        ce = C_AMARILLO
    else:
        estado = "No se detecta mano - coloca la mano frente a la camara"
        ce = C_ROJO

    _txt(frame, estado, 14, alto - 98, ce, 0.56, 1)

    # Barra de estabilidad / countdown
    _txt(frame, "Estabilidad:", 14, alto - 72, C_GRIS, 0.48, 1)
    color_barra = C_VERDE if hold_pct >= 1.0 else (C_NARANJA if hold_pct > 0.3 else C_GRIS)
    _barra(frame, 14, alto - 58, 340, 14, hold_pct, color_barra)

    # Conteo muestras
    muestras_txt = f"Muestras: {n_capturadas} / {n_total}"
    _txt(frame, muestras_txt, 14, alto - 35, C_BLANCO, 0.65, 2)
    _barra(frame, 14, alto - 20, 200, 10,
           n_capturadas / n_total if n_total else 0, C_VERDE)

    # Modo y controles
    modo_txt = "[AUTO]" if modo_auto else "[MANUAL]"
    color_modo = C_VERDE if modo_auto else C_AZUL
    _txt(frame, modo_txt, ancho - 100, alto - 35, color_modo, 0.55, 1)
    ctrl = "[ESPACIO] Capturar  [A] Auto  [Q] Saltar  [ESC] Salir"
    _txt(frame, ctrl, 14, alto - 8, C_GRIS, 0.42, 1)

    # ── Flash al capturar ───────────────────────────────────
    if fase == "capturado":
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (ancho, alto), (0, 180, 90), -1)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
        (wt, ht), _ = cv2.getTextSize("CAPTURADO!", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cx = (ancho - wt) // 2
        cy = alto // 2
        _txt(frame, "CAPTURADO!", cx, cy, C_VERDE, 2.0, 4)

    # ── Número grande de muestra ────────────────────────────
    if fase == "esperar" and n_capturadas > 0:
        num = str(n_capturadas)
        (wt, ht), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
        # esquina inferior-izquierda en transparente
        cx = ancho // 2 - wt // 2
        cy = alto // 2 + ht // 2
        cv2.putText(frame, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    4.0, (0, 0, 0, 80), 10, cv2.LINE_AA)
        cv2.putText(frame, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    4.0, (40, 40, 50), 6, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────
# PANTALLA DE TRANSICIÓN
# ─────────────────────────────────────────────────────────────

def pantalla_transicion(cap, sena, idx_sena, duracion=2.5):
    """Muestra pantalla de transición entre señas."""
    t0 = time.time()
    while time.time() - t0 < duracion:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
        else:
            alto, ancho = 480, 640
            frame = np.zeros((alto, ancho, 3), dtype=np.uint8)

        # Oscurecer el fondo
        ov = np.full_like(frame, C_FONDO)
        frame = cv2.addWeighted(frame, 0.2, ov, 0.8, 0)

        restante = duracion - (time.time() - t0)
        cx = ancho // 2

        _txt(frame, "Siguiente sena:", cx - 130, alto // 2 - 70, C_GRIS, 0.75, 1)
        _txt(frame, f"#{idx_sena+1}   {sena}", cx - 130, alto // 2 - 20, C_BLANCO, 1.4, 3)

        for i, linea in enumerate(INSTRUCCIONES_SENAS.get(sena, [])):
            _txt(frame, linea, cx - 130, alto // 2 + 40 + i * 28, C_AZUL, 0.5, 1)

        _txt(frame, f"Comenzando en {restante:.1f}s...", cx - 100, alto - 50, C_NARANJA, 0.65, 1)
        _txt(frame, "[ESPACIO] Saltar espera", cx - 80, alto - 20, C_GRIS, 0.48, 1)

        cv2.imshow("LSC v3.0 — Captura", frame)
        if cv2.waitKey(1) & 0xFF in (ord(" "), 27):
            break


# ─────────────────────────────────────────────────────────────
# LOOP DE CAPTURA POR SEÑA
# ─────────────────────────────────────────────────────────────

def capturar_sena(extractor, cap, sena, idx_sena, n_total, dir_sena, modo_auto):
    """
    Loop de captura para una seña. Retorna True si completó (o se saltó),
    False si el usuario quiere salir completamente.
    """
    os.makedirs(dir_sena, exist_ok=True)

    # Contar muestras ya guardadas
    existentes = sorted([f for f in os.listdir(dir_sena) if f.endswith(".npy")])
    n_guardadas = len(existentes)

    if n_guardadas >= n_total:
        print(f"  [{sena}] Ya tiene {n_guardadas}/{n_total} muestras. Saltando.")
        return True

    print(f"\n[{idx_sena+1}/{len(SENAS_OBJETIVO)}] Capturando: {sena}")
    print(f"  Existentes: {n_guardadas} / {n_total}  ->  {dir_sena}")

    # Estado del loop
    DURACION_HOLD = 1.5       # segundos de quietud requeridos (auto)
    DURACION_HOLD_MANUAL = 0.8 # segundos en modo manual (más rápido)
    DELAY_POST_CAPTURA = 1.0  # segundos de pausa después de capturar

    fase = "esperar"
    hold_inicio = None
    t_capturado = 0.0
    hold_pct = 0.0

    while n_guardadas < n_total:
        ret, frame = cap.read()
        if not ret:
            print("  ERROR: No se puede leer la cámara.")
            return False

        frame = cv2.flip(frame, 1)

        # Procesamiento con MediaPipe (CLAHE activado para mejor detección)
        resultado = extractor.procesar_frame(frame)
        hay_mano = resultado["hay_manos"]
        mano = resultado["manos"][0] if hay_mano else None
        nitidez = CamaraLSC.calcular_nitidez(frame)
        estable = (mano["es_estable"] and nitidez >= 45.0) if mano else False

        # Dibujar esqueleto
        frame_viz = extractor.dibujar_overlays(frame, resultado)

        # ── Lógica de fases ─────────────────────────────────
        duracion_hold = DURACION_HOLD if modo_auto else DURACION_HOLD_MANUAL

        if fase == "esperar":
            hold_pct = 0.0
            if hay_mano and estable and (modo_auto or hold_inicio is not None):
                # En modo auto: empieza el hold automáticamente cuando hay mano estable
                if modo_auto and hold_inicio is None:
                    hold_inicio = time.time()
                if hold_inicio:
                    elapsed = time.time() - hold_inicio
                    hold_pct = min(elapsed / duracion_hold, 1.0)
                    if elapsed >= duracion_hold:
                        fase = "capturando"
            elif not (hay_mano and estable):
                hold_inicio = None

        elif fase == "capturando":
            if mano is not None:
                vector = mano["vector_normalizado"]
                n_guardadas += 1
                ruta = os.path.join(dir_sena, f"muestra_{n_guardadas:03d}.npy")
                np.save(ruta, vector)
                print(f"  OK  Muestra {n_guardadas}/{n_total} -> {os.path.basename(ruta)}")
                fase = "capturado"
                t_capturado = time.time()
                hold_inicio = None
                hold_pct = 0.0
            else:
                # La mano desapareció justo antes de capturar
                fase = "esperar"
                hold_inicio = None

        elif fase == "capturado":
            hold_pct = 0.0
            if time.time() - t_capturado >= DELAY_POST_CAPTURA:
                fase = "esperar"
                if n_guardadas >= n_total:
                    break  # terminamos esta seña

        # ── Dibujo del HUD ──────────────────────────────────
        frame_viz = dibujar_hud(
            frame_viz, sena, idx_sena, len(SENAS_OBJETIVO),
            n_guardadas, n_total, hay_mano, estable,
            hold_pct, fase, modo_auto,
        )

        cv2.imshow("LSC v3.0 — Captura", frame_viz)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC -> salir completamente
            return False
        elif key in (ord("q"), ord("Q")):  # Q -> saltar esta seña
            print(f"  Sena {sena} saltada.")
            return True
        elif key in (ord("a"), ord("A")):  # A -> alternar modo auto
            modo_auto = not modo_auto
            hold_inicio = None
            print(f"  Modo: {'AUTO' if modo_auto else 'MANUAL'}")
        elif key == ord(" "):  # ESPACIO -> capturar en modo manual
            if not modo_auto:
                if hay_mano and estable and fase == "esperar":
                    if hold_inicio is None:
                        hold_inicio = time.time()
                        print("  Manteniendo... (suelta ESPACIO para capturar)")
                elif hay_mano and not estable:
                    print("  Mano en movimiento, espera a que se estabilice.")
                elif not hay_mano:
                    print("  No se detecta mano.")
            else:
                # En modo auto, ESPACIO también dispara captura inmediata si hay mano
                if hay_mano and mano and fase == "esperar":
                    fase = "capturando"

    print(f"  OK  Sena '{sena}' completa: {n_guardadas} muestras.")
    time.sleep(0.5)
    return True


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Captura interactiva de senas LSC — 5 palabras"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Indice de camara (default: 0)")
    parser.add_argument("--muestras", type=int, default=5,
                        help="Muestras por sena (default: 5)")
    parser.add_argument("--senas", nargs="+", default=None,
                        help="Senas especificas a capturar")
    parser.add_argument("--auto", action="store_true", default=False,
                        help="Iniciar en modo auto-captura")
    parser.add_argument("--forzar", action="store_true", default=False,
                        help="Sobreescribir muestras existentes")
    args = parser.parse_args()

    senas = [s.upper() for s in args.senas] if args.senas else SENAS_OBJETIVO
    # Sin validación restrictiva — acepta cualquier nombre de seña

    print("=" * 60)
    print("  LSC v3.0 — CAPTURA DE SENAS (5 PALABRAS)")
    print("=" * 60)
    print(f"  Camara       : #{args.camera}")
    print(f"  Muestras/sena: {args.muestras}")
    print(f"  Modo inicial : {'AUTO' if args.auto else 'MANUAL'}")
    print(f"  Senas        : {senas}")
    print("=" * 60)
    print()
    print("  CONTROLES:")
    print("   [ESPACIO] : Capturar (manual) / Captura inmediata (auto)")
    print("   [A]       : Alternar modo auto / manual")
    print("   [Q]       : Saltar sena actual")
    print("   [ESC]     : Salir completamente")
    print()

    # Limpiar muestras si --forzar
    if args.forzar:
        import shutil
        for sena in senas:
            d = os.path.join(DIR_SALIDA, sena)
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"  Borradas muestras de {sena}")

    # Inicializar camara con API DirectShow y buffer cero
    print("  Abriendo camara con DirectShow (baja latencia)...")
    cap = CamaraLSC(
        camera_index=args.camera,
        ancho=1280,
        alto=720,
        fps=30,
        aplicar_mejora_optica=False,  # En captura se procesa despues para maximo FPS
    )
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir camara #{args.camera}")
        sys.exit(1)
    print("  Camara OK")

    # Inicializar extractor
    # NOTA: usar_denoise_imagen=False en captura para mayor FPS
    # La mejora de imagen se aplica solo si la detección falla
    print("  Inicializando MediaPipe Holistic...")
    extractor = ExtractorLandmarks(
        min_detection_confidence=0.50,
        min_tracking_confidence=0.45,
        usar_filtro_temporal=True,
        usar_denoise_imagen=False,   # desactivado para mayor FPS en captura
    )
    print("  MediaPipe OK\n")

    modo_auto = args.auto

    # Loop por señas
    for idx, sena in enumerate(senas):
        dir_sena = os.path.join(DIR_SALIDA, sena)

        # Transición si no es la primera
        if idx > 0:
            pantalla_transicion(cap, sena, idx)

        continuar = capturar_sena(
            extractor, cap, sena, idx, args.muestras, dir_sena, modo_auto
        )
        if not continuar:
            print("\n  Captura interrumpida por el usuario.")
            break

    # ── Resumen ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN DE CAPTURA")
    print("=" * 60)
    total = 0
    for sena in senas:
        d = os.path.join(DIR_SALIDA, sena)
        n = len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0
        total += n
        ok = "OK" if n >= args.muestras else f"!! ({n}/{args.muestras})"
        print(f"  {ok:14s}  {sena}: {n} muestras")

    print(f"\n  Total vectores: {total}")
    print(f"  Directorio   : {os.path.abspath(DIR_SALIDA)}")
    print()
    print("  Siguiente paso:")
    print("    python entrenar_5palabras.py")
    print("=" * 60)

    extractor.liberar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
