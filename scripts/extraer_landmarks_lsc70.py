"""Extrae landmarks crudos (manos + hombros) de las imágenes LSC70.

Guarda una línea JSON por imagen con las coordenadas normalizadas que entrega
MediaPipe, sin calcular descriptores. El vector 109D se calcula después con la
misma función JS que ejecuta la app (scripts/vectorizar_landmarks.js), lo que
garantiza paridad exacta entre entrenamiento e inferencia.

Uso:
    python scripts/extraer_landmarks_lsc70.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LSC70 = ROOT / "datasets" / "LSC70"
SUBCONJUNTOS = ("LSC70W", "LSC70AN")
SALIDA = ROOT / "datasets" / "landmarks_lsc70.jsonl"

_holistic = None
_hands = None


def _init_worker() -> None:
    global _holistic, _hands
    import mediapipe as mp

    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.5
    )
    _hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )


def _lista(lms) -> list:
    return [[round(p.x, 6), round(p.y, 6), round(p.z, 6)] for p in lms.landmark]


def _procesar(tarea: tuple) -> dict | None:
    import cv2

    subconjunto, persona, clase, ruta = tarea
    img = cv2.imread(ruta)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _holistic.process(rgb)

    pose = None
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        pose = {
            "nariz": [lm[0].x, lm[0].y, lm[0].z],
            "hombro_izq": [lm[11].x, lm[11].y, lm[11].z],
            "hombro_der": [lm[12].x, lm[12].y, lm[12].z],
        }

    manos = []
    # Holistic etiqueta desde la perspectiva de la persona
    for etiqueta, lms in (("Right", res.right_hand_landmarks), ("Left", res.left_hand_landmarks)):
        if lms is not None:
            manos.append({"label": etiqueta, "coords": _lista(lms)})
    if not manos:
        rh = _hands.process(rgb)
        if rh.multi_hand_landmarks:
            for i, lms in enumerate(rh.multi_hand_landmarks):
                etiqueta = rh.multi_handedness[i].classification[0].label
                manos.append({"label": etiqueta, "coords": _lista(lms), "fallback": True})

    return {
        "subconjunto": subconjunto,
        "persona": persona,
        "clase": clase,
        "archivo": os.path.basename(ruta),
        "pose": pose,
        "manos": manos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    tareas = []
    for sub in SUBCONJUNTOS:
        for p_dir in sorted((LSC70 / sub).iterdir()):
            if not p_dir.is_dir() or not p_dir.name.startswith("Per"):
                continue
            for c_dir in sorted(p_dir.iterdir()):
                if not c_dir.is_dir():
                    continue
                for img in sorted(c_dir.iterdir()):
                    if img.suffix.lower() in (".jpg", ".png"):
                        tareas.append((sub, p_dir.name, c_dir.name, str(img)))
    print(f"Imágenes a procesar: {len(tareas)}", flush=True)

    hechas = sin_mano = 0
    with Pool(args.workers, initializer=_init_worker) as pool, SALIDA.open("w", encoding="utf-8") as out:
        for reg in pool.imap_unordered(_procesar, tareas, chunksize=16):
            hechas += 1
            if reg is None:
                continue
            if not reg["manos"]:
                sin_mano += 1
            out.write(json.dumps(reg) + "\n")
            if hechas % 1000 == 0:
                print(f"  {hechas}/{len(tareas)} (sin mano: {sin_mano})", flush=True)
    print(f"Listo: {hechas} imágenes, {sin_mano} sin mano detectada -> {SALIDA}")


if __name__ == "__main__":
    main()
