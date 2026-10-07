"""Captura de señas propias con la webcam (para ampliar el vocabulario).

Guarda landmarks crudos (manos + hombros) en datasets/capturas_propias.jsonl,
en el mismo formato que scripts/extraer_landmarks_lsc70.py, así que el
vectorizador y el entrenamiento los incorporan sin pasos extra.

Para que una seña nueva generalice, captúrala con VARIAS personas distintas
(--persona) — el entrenamiento valida por persona y una seña con una sola
persona no puede evaluarse.

Uso:
    python capturar_senas.py --persona P01 --senas GRACIAS BIEN --muestras 30
Controles: ESPACIO inicia/pausa la grabación de la seña actual, N pasa a la
siguiente, Q sale.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp

SALIDA = Path(__file__).resolve().parent / "datasets" / "capturas_propias.jsonl"


def _lista(lms) -> list:
    return [[round(p.x, 6), round(p.y, 6), round(p.z, 6)] for p in lms.landmark]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", required=True, help="Seudónimo de quien firma, p. ej. P01")
    parser.add_argument("--senas", nargs="+", required=True, help="Etiquetas a capturar, p. ej. GRACIAS BIEN")
    parser.add_argument("--muestras", type=int, default=30, help="Frames a guardar por seña")
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--intervalo", type=float, default=0.12, help="Segundos entre frames guardados")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")
    holistic = mp.solutions.holistic.Holistic(model_complexity=1)
    dibujo = mp.solutions.drawing_utils

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    i_sena, guardadas, grabando, ultimo = 0, 0, False, 0.0
    with SALIDA.open("a", encoding="utf-8") as out:
        while i_sena < len(args.senas):
            ok, frame = cap.read()
            if not ok:
                break
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            manos = [
                {"label": etq, "coords": _lista(lms)}
                for etq, lms in (("Right", res.right_hand_landmarks), ("Left", res.left_hand_landmarks))
                if lms is not None
            ]
            pose = None
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                pose = {k: [lm[i].x, lm[i].y, lm[i].z] for k, i in (("nariz", 0), ("hombro_izq", 11), ("hombro_der", 12))}

            ahora = time.time()
            if grabando and manos and pose and ahora - ultimo >= args.intervalo:
                out.write(json.dumps({
                    "subconjunto": "PROPIAS", "persona": args.persona, "clase": args.senas[i_sena],
                    "archivo": f"{args.persona}_{args.senas[i_sena]}_{int(ahora * 1000)}", "pose": pose, "manos": manos,
                }) + "\n")
                guardadas += 1
                ultimo = ahora
                if guardadas >= args.muestras:
                    grabando, guardadas = False, 0
                    i_sena += 1
                    continue

            for lms in (res.right_hand_landmarks, res.left_hand_landmarks):
                if lms is not None:
                    dibujo.draw_landmarks(frame, lms, mp.solutions.holistic.HAND_CONNECTIONS)
            vista = cv2.flip(frame, 1)
            estado = "GRABANDO" if grabando else "ESPACIO para grabar"
            aviso = "" if (manos and pose) else "  (sin manos u hombros visibles)"
            cv2.putText(vista, f"{args.senas[i_sena]}  {guardadas}/{args.muestras}  {estado}{aviso}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if grabando else (40, 200, 40), 2)
            cv2.imshow("Captura LSC", vista)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                grabando = not grabando
            if tecla == ord("n"):
                grabando, guardadas = False, 0
                i_sena += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Capturas guardadas en {SALIDA}. Siguiente paso: node scripts/vectorizar_landmarks.js")


if __name__ == "__main__":
    main()
