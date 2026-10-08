"""Genera web/senas_avatar.json: cómo mover el avatar para cada seña del modelo.

Para cada seña elige, entre las 70 personas de LSC70, la secuencia más
representativa (la más cercana al promedio de su clase, señada con la mano
derecha) y vuelve a pasar sus cuadros por MediaPipe guardando datos 3D:
  - Pose world landmarks: hombro, codo y muñeca de cada brazo (metros).
  - Hands world landmarks: las 21 articulaciones de cada mano (metros).
De ahí salen, por cuadro, las direcciones de brazo, antebrazo, palma y cada
falange en los ejes del avatar (x a la derecha del espectador, y arriba,
z hacia la cámara). web/avatar_lsc.js las convierte en rotaciones de huesos.

Requiere datasets/landmarks_lsc70.jsonl y datasets/vectores_lsc70_109d.json.
Uso: python scripts/generar_senas_avatar.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = ROOT / "datasets" / "landmarks_lsc70.jsonl"
VECTORES = ROOT / "datasets" / "vectores_lsc70_109d.json"
SALIDA = ROOT / "web" / "senas_avatar.json"
RENOMBRAR = {"ANNOS": "AÑOS"}
# Señas con movimiento propio duran más; las posturas estáticas se sostienen
DINAMICAS = {"J", "Z", "NN", "MILLON", "MIL", "10"}
CADENAS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def a_avatar(v):
    """MediaPipe (x der. imagen, y abajo, z lejos) -> avatar (x der., y arriba, z hacia cámara)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[1], -v[2]])


def unit(v):
    n = np.linalg.norm(v)
    return (v / n).round(4).tolist() if n > 1e-6 else None


def secuencias_candidatas():
    """(sub, persona, clase) -> archivos ordenados, solo si la mano activa es la derecha en todos."""
    seqs = defaultdict(list)
    for linea in LANDMARKS.open(encoding="utf-8"):
        r = json.loads(linea)
        if not r["pose"] or not r["manos"]:
            continue
        m = re.search(r"_(\d+)_?\.\w+$", r["archivo"])
        if not m:
            continue
        activa = min(r["manos"], key=lambda h: h["coords"][0][1])
        seqs[(r["subconjunto"], r["persona"], r["clase"])].append(
            (int(m.group(1)), r["archivo"], activa["label"] == "Right"))
    out = {}
    for k, cuadros in seqs.items():
        cuadros.sort()
        if len(cuadros) >= 5 and all(c[2] for c in cuadros):
            out[k] = [c[1] for c in cuadros]
    return out


def elegir_representativas(candidatas):
    """La secuencia cuyo vector medio está más cerca del promedio de la clase."""
    datos = json.loads(VECTORES.read_text(encoding="utf-8"))
    X = np.asarray(datos["X"], dtype=np.float32)
    medias = {}
    for s in datos["secuencias"]:
        medias[(s["persona"], s["clase"])] = X[s["filas"]].mean(0)
    por_clase = defaultdict(list)
    for (sub, persona, clase), archivos in candidatas.items():
        if (persona, clase) in medias:
            por_clase[clase].append((sub, persona, archivos, medias[(persona, clase)]))
    elegidas = {}
    for clase, lista in por_clase.items():
        centro = np.median(np.stack([v for *_, v in lista]), axis=0)
        lista.sort(key=lambda t: np.linalg.norm(t[3] - centro))
        elegidas[clase] = [(sub, persona, archivos) for sub, persona, archivos, _ in lista[:6]]
    return elegidas


def brazo(pw, hombro, codo, muneca):
    return unit(a_avatar(pw[codo]) - a_avatar(pw[hombro])), unit(a_avatar(pw[muneca]) - a_avatar(pw[codo]))


# MediaPipe subestima la flexión cuando los dedos se tapan (puños, garras):
# se amplifica el ángulo de cada articulación ya doblada más de UMBRAL_FLEXION.
AMPLIFICAR_FLEXION = 1.35
UMBRAL_FLEXION = np.radians(20)


def doblar_mas(prev, seg):
    """Gira `seg` alejándolo de `prev` en su mismo plano, multiplicando el ángulo."""
    a, b = prev / np.linalg.norm(prev), seg / np.linalg.norm(seg)
    ang = np.arccos(np.clip(a @ b, -1, 1))
    if ang < UMBRAL_FLEXION:
        return seg
    eje = np.cross(a, b)
    if np.linalg.norm(eje) < 1e-6:
        return seg
    eje /= np.linalg.norm(eje)
    nuevo = min(ang * AMPLIFICAR_FLEXION, np.radians(115))
    # Rodrigues: rotar `a` alrededor de `eje` el ángulo nuevo
    return a * np.cos(nuevo) + np.cross(eje, a) * np.sin(nuevo) + eje * (eje @ a) * (1 - np.cos(nuevo))


# Correcciones a mano: señas cuyo puño MediaPipe no resuelve (dedos tapados).
# Se cierran índice..meñique hacia la palma, en la dirección de curvatura detectada.
PUNO_CERRADO = {"A"}
ANGULOS_PUNO = np.radians([85, 185, 245])  # acumulados: MCP, PIP, DIP


def rotar(v, eje, ang):
    return v * np.cos(ang) + np.cross(eje, v) * np.sin(ang) + eje * (eje @ v) * (1 - np.cos(ang))


def cerrar_puno(info):
    """Reescribe las falanges de índice a meñique como un puño cerrado."""
    palma = np.array(info["dir"])
    for i in range(1, 5):
        seg1 = np.array(info["dedos"][i][0])
        eje = np.cross(palma, seg1)
        if np.linalg.norm(eje) < 0.05:  # dedo recto: usar el eje lateral de la mano
            eje = np.array(info["lateral"])
        eje /= np.linalg.norm(eje)
        info["dedos"][i] = [unit(rotar(palma, eje, a)) for a in ANGULOS_PUNO]
    return info


def mano(hw):
    p = [a_avatar(q) for q in hw]
    palma = p[9] - p[0]
    dedos = []
    for i, c in enumerate(CADENAS):
        segs = [p[c[k + 1]] - p[c[k]] for k in range(3)]
        if i > 0:  # el pulgar se deja tal cual
            prev = palma
            for k in range(3):
                segs[k] = doblar_mas(prev, segs[k])
                prev = segs[k]
        dedos.append([unit(s) for s in segs])
    return {"dir": unit(palma), "lateral": unit(p[17] - p[9]), "dedos": dedos}


def cuadros_de_secuencia(holistic, hands, sub, persona, clase, archivos):
    """Pasa una secuencia por MediaPipe 3D y devuelve sus cuadros útiles para el avatar."""
    cuadros = []
    for archivo in archivos:
        img = cv2.imread(str(ROOT / "datasets" / "LSC70" / sub / persona / clase / archivo))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rh = holistic.process(rgb)
        if not rh.pose_world_landmarks or not rh.pose_landmarks:
            continue
        pw = [[l.x, l.y, l.z] for l in rh.pose_world_landmarks.landmark]
        pimg = rh.pose_landmarks.landmark
        rhands = hands.process(rgb)
        # Emparejar cada mano con la muñeca de la pose (las etiquetas de Hands asumen imagen espejada)
        manos_mundo = {}
        if rhands.multi_hand_landmarks:
            for lms, wl in zip(rhands.multi_hand_landmarks, rhands.multi_hand_world_landmarks):
                w = lms.landmark[0]
                d_der = (w.x - pimg[16].x) ** 2 + (w.y - pimg[16].y) ** 2
                d_izq = (w.x - pimg[15].x) ** 2 + (w.y - pimg[15].y) ** 2
                manos_mundo["Right" if d_der < d_izq else "Left"] = [[l.x, l.y, l.z] for l in wl.landmark]
        cuadro = {}
        # Pose: 12/14/16 = hombro/codo/muñeca derechos de la persona; 11/13/15 izquierdos
        for lado, (h, c, m) in (("Right", (12, 14, 16)), ("Left", (11, 13, 15))):
            levantada = pw[m][1] < pw[23 if lado == "Left" else 24][1] - 0.12  # muñeca sobre la cadera
            if lado == "Left" and not levantada:
                continue  # la mano pasiva descansa
            b, a = brazo(pw, h, c, m)
            cuadro[lado] = {"brazo": b, "antebrazo": a}
            if lado in manos_mundo:
                info = mano(manos_mundo[lado])
                if lado == "Right" and clase in PUNO_CERRADO:
                    info = cerrar_puno(info)
                cuadro[lado]["mano"] = {"dir": info["dir"], "lateral": info["lateral"]}
                cuadro[lado]["dedos"] = info["dedos"]
        if "Right" in cuadro and "mano" in cuadro["Right"]:
            cuadros.append(cuadro)
    return cuadros


def main():
    elegidas = elegir_representativas(secuencias_candidatas())
    holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                     min_detection_confidence=0.4)
    senas = {}
    for clase, opciones in sorted(elegidas.items()):
        # Probar las secuencias más representativas hasta encontrar una nítida (>= 5 cuadros útiles)
        mejor = None
        for sub, persona, archivos in opciones:
            cuadros = cuadros_de_secuencia(holistic, hands, sub, persona, clase, archivos)
            if mejor is None or len(cuadros) > len(mejor[2]):
                mejor = (sub, persona, cuadros)
            if len(cuadros) >= 5:
                break
        sub, persona, cuadros = mejor
        if len(cuadros) < 2:
            print(f"  {clase:7} sin cuadros suficientes ({len(cuadros)})")
            continue
        nombre = RENOMBRAR.get(clase, clase)
        dinamica = sub == "LSC70W" or clase in DINAMICAS
        senas[nombre] = {"origen": f"{sub}/{persona}", "duracion": 1.3 if dinamica else 0.9,
                         "cuadros": cuadros}
        print(f"  {nombre:7} {persona} ({len(cuadros)} cuadros)")
    SALIDA.write_text(json.dumps({"version": 1, "senas": senas}, ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
    print(f"{len(senas)} señas -> {SALIDA} ({SALIDA.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
