"""Genera web/senas_avatar.json: cómo mover el avatar para cada seña del modelo.

Fuente: el dataset LSC70 (70 personas). Para cada seña se eligen las grabaciones
más representativas (las más cercanas al promedio de su clase, señadas con la
mano derecha) y se pasan de nuevo por MediaPipe en 3D (pose + manos, en metros).

Todo se expresa en el marco del TORSO de la persona (no de la cámara), así una
persona algo girada no tuerce al avatar:
  x = hacia el hombro izquierdo de la persona (derecha del espectador)
  y = de la cadera hacia los hombros
  z = hacia adelante (hacia la cámara)

Por cuadro y por brazo se guarda:
  muneca  posición de la muñeca respecto al hombro, dividida por el largo del brazo
  codo    posición del codo respecto al hombro, dividida por el largo del brazo
  mano    dirección de la palma (muñeca -> nudillo medio) y normal de la palma
  dedos   ángulos en grados por dedo (índice..meñique): [base, apertura, medio, punta]
  pulgar  dirección de sus 3 falanges en el marco de la mano (D, N, L)

Robustez:
  - Letras y números estáticos: la forma de la mano es la MEDIANA de varias
    personas y de todos sus cuadros (mano firme, sin temblor).
  - Señas con movimiento: trayectoria de una sola persona, suavizada en el tiempo.
  - Límites anatómicos en cada articulación; la punta va acoplada a la media.

Uso: python scripts/generar_senas_avatar.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = ROOT / "datasets" / "landmarks_lsc70.jsonl"
VECTORES = ROOT / "datasets" / "vectores_lsc70_109d.json"
SALIDA = ROOT / "web" / "senas_avatar.json"
RENOMBRAR = {"ANNOS": "AÑOS"}

# Letras/números con movimiento: forma de mano fija, se mueve el brazo
DINAMICAS_FORMA_FIJA = {"J", "Z", "NN", "MILLON", "MIL", "10"}
PERSONAS_FORMA = 12        # personas para la mediana de la forma de la mano
PERSONAS_TRAYECTORIA = 6   # candidatas para elegir la trayectoria de una seña con movimiento

DEDOS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
LIMITES = [(-15, 95), (-20, 20), (0, 110), (0, 85)]   # base, apertura, medio, punta
# Corrección a mano: puño que MediaPipe no resuelve (dedos tapados entre sí)
PUNO = [85.0, 0.0, 100.0, 65.0]
PUNO_CERRADO = {"A"}

_holistic = None
_hands = None


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def a_avatar(v):
    """MediaPipe (x der. imagen, y abajo, z lejos) -> (x der. espectador, y arriba, z hacia cámara)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[1], -v[2]])


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def marco_torso(pw):
    """Matriz (filas x, y, z) del torso a partir de pose world en ejes del avatar."""
    hi, hd = pw[11], pw[12]          # hombro izquierdo / derecho de la persona
    ci, cd = pw[23], pw[24]
    x = norm(hi - hd)
    y = (hi + hd) / 2 - (ci + cd) / 2
    y = norm(y - x * (y @ x))
    return np.stack([x, y, np.cross(x, y)])


def marco_mano(h, signo):
    """Base (D, N, L): D muñeca -> nudillo medio, N normal hacia la palma, L = N x D."""
    d = norm(h[9] - h[0])
    n = norm(np.cross(h[5] - h[0], h[17] - h[0])) * signo
    n = norm(n - d * (n @ d))
    return d, n, np.cross(n, d)


def angulo(a, b):
    return float(np.degrees(np.arccos(np.clip(norm(a) @ norm(b), -1, 1))))


def angulos_dedos(h, d, n, l):
    out = []
    for mcp, pip, dip, tip in DEDOS:
        meta = h[mcp] - h[0]
        meta = norm(meta - n * (meta @ n))
        s1, s2, s3 = h[pip] - h[mcp], h[dip] - h[pip], h[tip] - h[dip]
        base = np.degrees(np.arctan2(s1 @ n, s1 @ meta))
        lat = norm(np.cross(n, meta))
        apertura = np.degrees(np.arctan2(s1 @ lat, s1 @ meta))
        out.append([base, apertura, angulo(s1, s2), angulo(s2, s3)])
    return np.array(out)


def limpiar_dedos(a):
    """Dedos firmes: zona muerta (MediaPipe nunca da 0° en un dedo estirado),
    apertura amplificada, límites anatómicos y punta acoplada a la articulación media."""
    a = np.array(a, dtype=float)
    base, apertura, medio, punta = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    base = np.where(base > 10, (base - 10) * 95 / 85, base * 0.3)
    medio = np.maximum(0, medio - 18) * 110 / 92
    punta = np.maximum(0, punta - 15) * 85 / 70
    apertura = apertura * 1.5
    a = np.stack([base, apertura, medio, punta], axis=1)
    a[:, 3] = 0.5 * a[:, 3] + 0.5 * (0.67 * a[:, 2])
    for j, (lo, hi) in enumerate(LIMITES):
        a[:, j] = np.clip(a[:, j], lo, hi)
    return np.round(a, 1)


def pulgar(h, d, n, l):
    base = np.stack([d, n, l])
    return np.array([norm(base @ (h[b] - h[a])) for a, b in ((1, 2), (2, 3), (3, 4))])


def suavizar(arr, pesos=(0.25, 0.5, 0.25)):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        return arr
    pad = np.concatenate([arr[:1], arr, arr[-1:]])
    return pesos[0] * pad[:-2] + pesos[1] * pad[1:-1] + pesos[2] * pad[2:]


def mediana3(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        return arr
    pad = np.concatenate([arr[:1], arr, arr[-1:]])
    return np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)


def r4(v):
    return [round(float(x), 4) for x in v]


# ---------------------------------------------------------------------------
# MediaPipe en paralelo
# ---------------------------------------------------------------------------
def _init():
    global _holistic, _hands
    import mediapipe as mp
    _holistic = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2)
    _hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                      min_detection_confidence=0.4)


def _procesar(ruta):
    """Landmarks 3D de una imagen: pose world y manos world emparejadas con la pose."""
    import cv2
    img = cv2.imdecode(np.fromfile(ruta, np.uint8), 1)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rh = _holistic.process(rgb)
    if not rh.pose_world_landmarks or not rh.pose_landmarks:
        return None
    pimg = rh.pose_landmarks.landmark
    manos = {}
    r = _hands.process(rgb)
    if r.multi_hand_landmarks:
        for lms, wl in zip(r.multi_hand_landmarks, r.multi_hand_world_landmarks):
            w = lms.landmark[0]
            d_der = (w.x - pimg[16].x) ** 2 + (w.y - pimg[16].y) ** 2
            d_izq = (w.x - pimg[15].x) ** 2 + (w.y - pimg[15].y) ** 2
            manos["Right" if d_der < d_izq else "Left"] = [[l.x, l.y, l.z] for l in wl.landmark]
    alto, ancho = img.shape[:2]
    return {"pw": [[l.x, l.y, l.z] for l in rh.pose_world_landmarks.landmark],
            "pimg": [[l.x * ancho, l.y * alto] for l in pimg], "manos": manos}


def hibrido(crudo):
    """Pose con x, y de la IMAGEN (fiables) escaladas a metros y z del modelo 3D.

    El modelo 3D de MediaPipe baja la muñeca de forma sistemática (0.15-0.2 anchos
    de hombro frente a la imagen); la imagen da bien la altura y el lado.
    """
    pw, pi = np.asarray(crudo["pw"], float), np.asarray(crudo["pimg"], float)
    escala = np.linalg.norm(pw[11] - pw[12]) / max(np.linalg.norm(pi[11] - pi[12]), 1e-6)
    cadera = (pi[23] + pi[24]) / 2
    out = pw.copy()
    out[:, :2] = (pi - cadera) * escala
    # La profundidad del modelo 3D exagera cuánto se adelanta la mano. Con la posición
    # 2D y el largo de cada hueso la profundidad queda determinada; del 3D solo se usa
    # el signo (hacia adelante o hacia atrás).
    for hombro, codo, muneca in ((11, 13, 15), (12, 14, 16)):
        for a, b in ((hombro, codo), (codo, muneca)):
            largo = np.linalg.norm(pw[b] - pw[a])
            plano = np.linalg.norm(out[b, :2] - out[a, :2])
            prof = np.sqrt(max(largo * largo - plano * plano, 0.0))
            signo = 1.0 if pw[b, 2] >= pw[a, 2] else -1.0
            out[b, 2] = out[a, 2] + signo * prof
    return out


def rasgos(crudo, signo_palma):
    """Rasgos de un cuadro en el marco del torso, por lado."""
    pw = np.array([a_avatar(p) for p in hibrido(crudo)])
    M = marco_torso(pw)
    out = {}
    for lado, (ih, ic, im, icad) in (("Right", (12, 14, 16, 24)), ("Left", (11, 13, 15, 23))):
        hombro, codo, muneca = pw[ih], pw[ic], pw[im]
        largo = np.linalg.norm(codo - hombro) + np.linalg.norm(muneca - codo)
        info = {"muneca": M @ (muneca - hombro) / largo,
                "codo": M @ (codo - hombro) / largo,
                "levantada": bool((M @ (muneca - pw[icad]))[1] > 0.12)}
        if lado in crudo["manos"]:
            h = np.array([M @ a_avatar(p) for p in crudo["manos"][lado]])
            d, n, l = marco_mano(h, signo_palma * (1 if lado == "Right" else -1))
            info.update(D=d, N=n, dedos=angulos_dedos(h, d, n, l), pulgar=pulgar(h, d, n, l))
        out[lado] = info
    return out


# ---------------------------------------------------------------------------
# Selección de grabaciones
# ---------------------------------------------------------------------------
def secuencias_candidatas():
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
    return {k: [c[1] for c in sorted(v)] for k, v in seqs.items()
            if len(v) >= 5 and all(c[2] for c in v)}


def ranking_por_clase(candidatas):
    datos = json.loads(VECTORES.read_text(encoding="utf-8"))
    X = np.asarray(datos["X"], dtype=np.float32)
    medias = {(s["persona"], s["clase"]): X[s["filas"]].mean(0) for s in datos["secuencias"]}
    por_clase = defaultdict(list)
    for (sub, persona, clase), archivos in candidatas.items():
        if (persona, clase) in medias:
            por_clase[clase].append((sub, persona, archivos, medias[(persona, clase)]))
    out = {}
    for clase, lista in por_clase.items():
        centro = np.median(np.stack([v for *_, v in lista]), axis=0)
        lista.sort(key=lambda t: np.linalg.norm(t[3] - centro))
        out[clase] = [(sub, persona, archivos) for sub, persona, archivos, _ in lista]
    return out


# ---------------------------------------------------------------------------
# Construcción de cada seña
# ---------------------------------------------------------------------------
def forma_mediana(cuadros_lado):
    dedos = np.median(np.stack([c["dedos"] for c in cuadros_lado]), axis=0)
    pul = np.median(np.stack([c["pulgar"] for c in cuadros_lado]), axis=0)
    return limpiar_dedos(dedos), np.array([norm(v) for v in pul])


def lado_json(c, dedos, pul):
    d = norm(c["D"])
    n = norm(c["N"] - d * (c["N"] @ d))
    return {"muneca": r4(c["muneca"]), "codo": r4(c["codo"]), "mano": {"dir": r4(d), "normal": r4(n)},
            "dedos": np.asarray(dedos).tolist(), "pulgar": [r4(v) for v in pul]}


def construir_estatica(por_persona):
    """Una sola pose firme: medianas sobre todas las personas y cuadros."""
    der = [f["Right"] for _, _, s in por_persona for f in s if "D" in f["Right"]]
    if len(der) < 3:
        return None
    dedos, pul = forma_mediana(der)
    pose = {k: np.median(np.stack([c[k] for c in der]), axis=0) for k in ("muneca", "codo")}
    pose.update({k: np.mean(np.stack([c[k] for c in der]), axis=0) for k in ("D", "N")})
    return [{"Right": lado_json(pose, dedos, pul)}]


def construir_dinamica(cuadros, forma_fija=None):
    """Trayectoria de una persona, suavizada; con forma_fija la mano no cambia de forma."""
    cuadros = [f for f in cuadros if "D" in f["Right"]]
    if len(cuadros) < 3:
        return None
    salida = [dict() for _ in cuadros]
    usar_izq = sum(f["Left"]["levantada"] and "D" in f["Left"] for f in cuadros) >= len(cuadros) / 2
    for lado in (("Right", "Left") if usar_izq else ("Right",)):
        serie = [dict(f[lado]) for f in cuadros]
        validos = [i for i, c in enumerate(serie) if "D" in c]
        for i, c in enumerate(serie):
            if "D" not in c:
                j = min(validos, key=lambda k: abs(k - i))
                c.update({k: serie[j][k] for k in ("D", "N", "dedos", "pulgar")})
        mun = suavizar(mediana3([c["muneca"] for c in serie]))
        cod = suavizar(mediana3([c["codo"] for c in serie]))
        D = suavizar([c["D"] for c in serie])
        N = suavizar([c["N"] for c in serie])
        if forma_fija is not None and lado == "Right":
            dedos_t, pul_t = [forma_fija[0]] * len(serie), [forma_fija[1]] * len(serie)
        else:
            dedos_t = [limpiar_dedos(x) for x in suavizar(mediana3([c["dedos"] for c in serie]))]
            pul_t = [np.array([norm(v) for v in x]) for x in suavizar([c["pulgar"] for c in serie])]
        for i in range(len(serie)):
            salida[i][lado] = lado_json({"muneca": mun[i], "codo": cod[i], "D": D[i], "N": N[i]},
                                        dedos_t[i], pul_t[i])
    return salida


def main():
    ranking = ranking_por_clase(secuencias_candidatas())
    trabajos = {}
    for clase, opciones in ranking.items():
        estatica = opciones[0][0] == "LSC70AN" and clase not in DINAMICAS_FORMA_FIJA
        n = PERSONAS_FORMA if (estatica or clase in DINAMICAS_FORMA_FIJA) else PERSONAS_TRAYECTORIA
        for sub, persona, archivos in opciones[:n]:
            for a in archivos:
                trabajos[str(ROOT / "datasets" / "LSC70" / sub / persona / clase / a)] = clase
    print(f"Imágenes a procesar en 3D: {len(trabajos)}", flush=True)
    rutas = list(trabajos)
    with Pool(10, initializer=_init) as pool:
        crudos = dict(zip(rutas, pool.map(_procesar, rutas, chunksize=8)))

    # Signo de la normal de la palma: en un puño las yemas van hacia la palma
    muestras = []
    for ruta, cr in crudos.items():
        if cr and trabajos[ruta] in ("A", "S", "E", "T", "M", "N") and "Right" in cr["manos"]:
            h = np.array([a_avatar(p) for p in cr["manos"]["Right"]])
            nn = norm(np.cross(h[5] - h[0], h[17] - h[0]))
            muestras += [(h[t] - h[m]) @ nn for m, _, _, t in DEDOS]
    signo_palma = 1.0 if np.median(muestras) > 0 else -1.0
    print(f"Signo de la palma: {signo_palma:+.0f} ({len(muestras)} muestras)", flush=True)

    senas = {}
    for clase, opciones in sorted(ranking.items()):
        por_persona = []
        for sub, persona, archivos in opciones:
            rutas_p = [str(ROOT / "datasets" / "LSC70" / sub / persona / clase / a) for a in archivos]
            if rutas_p[0] not in crudos:
                continue
            rs = [crudos[r] for r in rutas_p if crudos.get(r) is not None]
            if rs:
                por_persona.append((sub, persona, [rasgos(r, signo_palma) for r in rs]))
        if not por_persona:
            print(f"  {clase:7} sin datos")
            continue
        estatica = opciones[0][0] == "LSC70AN" and clase not in DINAMICAS_FORMA_FIJA
        if estatica:
            cuadros = construir_estatica(por_persona)
            origen = f"mediana de {len(por_persona)} personas"
        else:
            forma = None
            if clase in DINAMICAS_FORMA_FIJA:
                forma = forma_mediana([f["Right"] for _, _, s in por_persona for f in s if "D" in f["Right"]])
            mejor = next((p for p in por_persona if sum("D" in f["Right"] for f in p[2]) >= 5), por_persona[0])
            cuadros = construir_dinamica(mejor[2], forma)
            origen = f"{mejor[0]}/{mejor[1]}" + (" + forma mediana" if forma else "")
        if not cuadros:
            print(f"  {clase:7} sin datos suficientes")
            continue
        if clase in PUNO_CERRADO:
            for c in cuadros:
                c["Right"]["dedos"] = [PUNO[:] for _ in range(4)]
        nombre = RENOMBRAR.get(clase, clase)
        senas[nombre] = {"origen": origen, "estatica": estatica,
                         "duracion": 0.9 if estatica else 1.4, "cuadros": cuadros}
        print(f"  {nombre:7} {origen} ({len(cuadros)} cuadros)", flush=True)

    SALIDA.write_text(json.dumps({"version": 2, "signo_palma": signo_palma, "senas": senas},
                                 ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(senas)} señas -> {SALIDA} ({SALIDA.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
