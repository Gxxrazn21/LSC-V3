"""
Entrenamiento del modelo LSC (palabras + abecedario + números + REPOSO).

Pipeline reproducible:
  1. python scripts/extraer_landmarks_lsc70.py   -> datasets/landmarks_lsc70.jsonl
  2. node scripts/vectorizar_landmarks.js         -> datasets/vectores_lsc70_109d.json
  3. python entrenar_modelo_lsc.py                -> web/modelo_ia_cliente.js + métricas
  4. python scripts/publish_web_release.py --version X.Y.Z

Validación honesta: GroupKFold por persona. Ninguna persona del conjunto de
prueba aparece en entrenamiento, así que la métrica refleja el desempeño con
usuarios nuevos. No hay aumentación sintética ni sobremuestreo.
"""

import json
import os
import sys
import time
from collections import Counter

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

VERSION = "8.1.0"
VECTORES = os.path.join("datasets", "vectores_lsc70_109d.json")
ARQ = (256, 128)
MAX_REPOSO = 2500  # REPOSO es ~1/3 de los cuadros: se acota para no inflar métricas ni sesgar la red

RENOMBRAR = {"ANNOS": "AÑOS"}
PALABRAS = ["AÑOS", "BUENAS", "DIAS", "GUSTAR", "HOLA", "LICOR", "NOCHES", "NOMBRE", "TARDES", "YO"]
ABECEDARIO = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "NN",
              "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
NUMEROS = ["1", "4", "5", "6", "7", "8", "9", "10", "MIL", "MILLON"]


def nuevo_mlp(seed=42):
    return MLPClassifier(
        hidden_layer_sizes=ARQ,
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=seed,
    )


def grupos_confundibles(cm, clases, y, umbral=0.04, max_tam=4):
    """Une en grupos las señas de un mismo modo que se confunden entre sí más que `umbral`."""
    n = np.array([(y == c).sum() for c in clases])
    modo = {c: m for m, lista in (("palabras", PALABRAS), ("abecedario", ABECEDARIO), ("numeros", NUMEROS))
            for c in lista}
    pares = []
    for i in range(len(clases)):
        for j in range(i + 1, len(clases)):
            a, b = clases[i], clases[j]
            if a not in modo or modo.get(a) != modo.get(b):
                continue
            tasa = (cm[i, j] + cm[j, i]) / max(n[i] + n[j], 1)
            if tasa > umbral:
                pares.append((tasa, a, b))
    grupos = []
    for _, a, b in sorted(pares, reverse=True):
        ga = next((g for g in grupos if a in g), None)
        gb = next((g for g in grupos if b in g), None)
        if ga is None and gb is None:
            grupos.append({a, b})
        elif ga is not None and gb is None and len(ga) < max_tam:
            ga.add(b)
        elif gb is not None and ga is None and len(gb) < max_tam:
            gb.add(a)
    return [sorted(g) for g in grupos]


def entrenar_validadores(secuencias, fila_nueva, prob_cv, clases, cm, y):
    """Segunda etapa para señas parecidas: regresión logística sobre
    [probabilidades medias del grupo durante la seña, rasgos de movimiento].
    Se entrena con probabilidades fuera de pliegue (stacking honesto) y solo se
    activa si mejora la precisión por secuencia con personas no vistas."""
    idx = {c: i for i, c in enumerate(clases)}
    validadores, resumen = [], []
    for grupo in grupos_confundibles(cm, clases, y):
        cols = [idx[c] for c in grupo]
        F, t, g, base = [], [], [], []
        for s in secuencias:
            clase = RENOMBRAR.get(s["clase"], s["clase"])
            filas = [fila_nueva[f] for f in s["filas"] if f in fila_nueva]
            if clase not in grupo or not filas:
                continue
            medias = prob_cv[filas][:, cols].mean(0)
            F.append(np.concatenate([medias, s["rasgos"]]))
            t.append(grupo.index(clase))
            g.append(s["persona"])
            base.append(int(np.argmax(medias)))
        if len(set(t)) < 2:
            continue
        F, t, g, base = np.array(F), np.array(t), np.array(g), np.array(base)
        pred = np.zeros_like(t)
        for tr, te in GroupKFold(5).split(F, t, g):
            sc = StandardScaler().fit(F[tr])
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(F[tr]), t[tr])
            pred[te] = lr.predict(sc.transform(F[te]))
        acc_base, acc_val = float((base == t).mean()), float((pred == t).mean())
        activo = acc_val >= acc_base + 0.01
        resumen.append({"grupo": grupo, "secuencias": int(len(t)), "sin_validador": acc_base,
                        "con_validador": acc_val, "activo": bool(activo)})
        print(f"  Validador {'/'.join(grupo):24} {acc_base * 100:5.1f}% -> {acc_val * 100:5.1f}%"
              f" {'(activo)' if activo else '(descartado)'}")
        if not activo:
            continue
        sc = StandardScaler().fit(F)
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(F), t)
        if len(grupo) == 2:  # binaria -> dos logits equivalentes (z0 = 0)
            w = [[0.0] * F.shape[1], lr.coef_[0].tolist()]
            b = [0.0, float(lr.intercept_[0])]
        else:
            w, b = lr.coef_.tolist(), lr.intercept_.tolist()
        validadores.append({"clases": grupo, "media": sc.mean_.tolist(), "escala": sc.scale_.tolist(),
                            "w": w, "b": b})
    return validadores, resumen


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(VECTORES, encoding="utf-8") as f:
        datos = json.load(f)
    X = np.asarray(datos["X"], dtype=np.float32)
    y = np.array([RENOMBRAR.get(c, c) for c in datos["y"]])
    grupos = np.asarray(datos["personas"])
    idx_rep = np.where(y == "REPOSO")[0]
    keep = np.arange(len(y))
    if len(idx_rep) > MAX_REPOSO:
        rng = np.random.default_rng(42)
        keep = np.sort(np.concatenate([np.where(y != "REPOSO")[0], rng.choice(idx_rep, MAX_REPOSO, replace=False)]))
        X, y, grupos = X[keep], y[keep], grupos[keep]
    fila_nueva = {int(orig): i for i, orig in enumerate(keep)}
    print(f"Muestras: {len(X)} | personas: {len(set(grupos))} | clases: {len(set(y))}")
    for c, n in sorted(Counter(y).items()):
        print(f"  {c:8} {n}")

    clases = sorted(set(y))
    idx = {c: i for i, c in enumerate(clases)}
    y_enc = np.array([idx[c] for c in y])

    # ---- Validación cruzada por persona ----
    gkf = GroupKFold(n_splits=5)
    pred_cv = np.zeros_like(y_enc)
    prob_cv = np.zeros((len(X), len(clases)))
    scores = []
    for k, (tr, te) in enumerate(gkf.split(X, y_enc, grupos), 1):
        sc = StandardScaler().fit(X[tr])
        m = nuevo_mlp(42 + k).fit(sc.transform(X[tr]), y_enc[tr])
        prob_cv[te] = m.predict_proba(sc.transform(X[te]))
        pred_cv[te] = prob_cv[te].argmax(1)
        scores.append(accuracy_score(y_enc[te], pred_cv[te]))
        print(f"  Fold {k}: {scores[-1] * 100:.2f}% ({len(set(grupos[te]))} personas no vistas)")
    acc_cv, std_cv = float(np.mean(scores)), float(np.std(scores))
    print(f"Precisión con personas no vistas (GroupKFold 5, incluye REPOSO): {acc_cv * 100:.2f}% ± {std_cv * 100:.2f}%")

    # Métricas solo sobre señas (sin REPOSO): modo Todo y cada modo de la app,
    # enmascarando las clases fuera del modo igual que predecirRedNeuronal
    es_sena = y != "REPOSO"
    acc_senas = float((pred_cv[es_sena] == y_enc[es_sena]).mean())
    acc_modos = {}
    for modo, lista in {"palabras": PALABRAS, "abecedario": ABECEDARIO, "numeros": NUMEROS}.items():
        permitidas = np.array([c in lista or c == "REPOSO" for c in clases])
        sel = np.isin(y, lista)
        acc_modos[modo] = float(((prob_cv[sel] * permitidas).argmax(1) == y_enc[sel]).mean())
    resumen_modos = ", ".join(f"{k} {v * 100:.1f}%" for k, v in acc_modos.items())
    print(f"Precisión SOLO SEÑAS (modo Todo): {acc_senas * 100:.2f}% | por modo: {resumen_modos}")

    rep = classification_report(y_enc, pred_cv, target_names=clases, output_dict=True, zero_division=0)
    rep_txt = classification_report(y_enc, pred_cv, target_names=clases, digits=3, zero_division=0)
    print(rep_txt)

    cm = confusion_matrix(y_enc, pred_cv)
    pares = []
    for i in range(len(clases)):
        for j in range(len(clases)):
            if i != j and cm[i, j] > 0:
                pares.append((int(cm[i, j]), clases[i], clases[j]))
    pares.sort(reverse=True)
    print("Pares más confundidos (real -> predicho):")
    for n, a, b in pares[:10]:
        print(f"  {a} -> {b}: {n}")

    validadores, resumen_validadores = entrenar_validadores(
        datos.get("secuencias", []), fila_nueva, prob_cv, clases, cm, y)

    # ---- Modelo final con todos los datos ----
    t0 = time.time()
    scaler = StandardScaler().fit(X)
    mlp = nuevo_mlp(42).fit(scaler.transform(X), y_enc)
    print(f"Modelo final entrenado en {time.time() - t0:.1f}s")

    os.makedirs("modelos_guardados", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    joblib.dump({"mlp": mlp, "scaler": scaler, "clases": clases},
                os.path.join("modelos_guardados", "modelo_lsc.joblib"))

    plt.figure(figsize=(16, 14))
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(clases)), clases, rotation=90, fontsize=8)
    plt.yticks(range(len(clases)), clases, fontsize=8)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión con personas no vistas — solo señas {acc_senas * 100:.1f}%")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join("resultados", "matriz_confusion.png"), dpi=150)
    plt.close()

    categorias = {
        "palabras": [c for c in PALABRAS if c in idx] + ["REPOSO"],
        "abecedario": [c for c in ABECEDARIO if c in idx] + ["REPOSO"],
        "numeros": [c for c in NUMEROS if c in idx] + ["REPOSO"],
        "todo": clases,
    }
    metricas = {
        "version": VERSION,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validacion": "GroupKFold(5) por persona, sin aumentación",
        "muestras": int(len(X)),
        "personas": int(len(set(grupos))),
        "clases": clases,
        "categorias": categorias,
        "arquitectura": [109, *ARQ, len(clases)],
        "precision_solo_senas": acc_senas,
        "precision_por_modo": acc_modos,
        "precision_incluyendo_reposo": acc_cv,
        "desviacion_folds": std_cv,
        "reposo_acotado_a": MAX_REPOSO,
        "validadores_senas_parecidas": resumen_validadores,
        "folds": [float(s) for s in scores],
        "por_clase": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "soporte": int(rep[c]["support"])} for c in clases},
        "pares_confundidos": [{"real": a, "predicho": b, "n": n} for n, a, b in pares[:15]],
    }
    with open(os.path.join("resultados", "metricas_actuales.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    with open(os.path.join("resultados", "reporte_clasificacion.txt"), "w", encoding="utf-8") as f:
        f.write(f"Precisión solo señas, personas no vistas: {acc_senas * 100:.2f}%\n"
                f"Por modo: {resumen_modos}\n"
                f"Incluyendo REPOSO: {acc_cv * 100:.2f}% ± {std_cv * 100:.2f}%\n\n{rep_txt}")

    modelo = {
        "clases": clases,
        "categorias": categorias,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": [np.round(w, 6).tolist() for w in mlp.coefs_],
        "biases": [np.round(b, 6).tolist() for b in mlp.intercepts_],
        "layers": [109, *ARQ, len(clases)],
        "version": VERSION,
        "precision_cv": acc_senas,
        "precision_modos": acc_modos,
        "validadores": validadores,
    }
    js = f"""/**
 * Modelo LSC v{VERSION} (on-device). Generado por entrenar_modelo_lsc.py — no editar a mano.
 * Precisión solo señas con personas no vistas (GroupKFold por persona): {acc_senas * 100:.2f}%
 * Por modo: {resumen_modos}
 * MLP 109D -> {' -> '.join(map(str, ARQ))} -> {len(clases)} clases
 */
const VERSION_MODELO_LSC = "{VERSION}";
const METADATOS_MODELO_LSC = {{
  version: "{VERSION}",
  precision_cv: "{acc_senas * 100:.1f}%",
  clases: {len(clases)},
  fecha: "{time.strftime('%Y-%m-%d')}"
}};
const _MODELO_LSC_DATA = {json.dumps(modelo, ensure_ascii=False, separators=(',', ':'))};
if (typeof window !== 'undefined') {{
  window.MODELO_LSC = _MODELO_LSC_DATA;
  window.VERSION_MODELO_LSC = VERSION_MODELO_LSC;
  window.METADATOS_MODELO_LSC = METADATOS_MODELO_LSC;
}}
if (typeof module !== 'undefined' && module.exports) {{
  module.exports = {{ ..._MODELO_LSC_DATA, VERSION_MODELO_LSC, METADATOS_MODELO_LSC }};
}}
"""
    with open(os.path.join("web", "modelo_ia_cliente.js"), "w", encoding="utf-8") as f:
        f.write(js)
    print("Modelo exportado a web/modelo_ia_cliente.js (publicar con scripts/publish_web_release.py)")


if __name__ == "__main__":
    main()
