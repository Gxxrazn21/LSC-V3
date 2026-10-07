"""Evalúa si clasificadores expertos por grupo de señas parecidas reducen las confusiones.

Para cada grupo (p. ej. 1/6) compara, con GroupKFold por persona y sobre las
muestras cuyo top-1 del modelo principal cae dentro del grupo:
  - acierto del modelo principal
  - acierto del modelo principal + experto del grupo (cascada, como en la app)

Uso: python scripts/analizar_confusiones.py
"""

import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import entrenar_modelo_lsc as E  # noqa: E402

GRUPOS = [["1", "6"], ["4", "9"], ["W", "8"], ["V", "7"], ["N", "NN"], ["I", "J"],
          ["DIAS", "NOCHES", "TARDES"], ["F", "L"], ["R", "H", "U"], ["1", "S"], ["MIL", "MILLON"]]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with open(E.VECTORES, encoding="utf-8") as f:
        d = json.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.array([E.RENOMBRAR.get(c, c) for c in d["y"]])
    g = np.asarray(d["personas"])
    rep = np.where(y == "REPOSO")[0]
    keep = np.sort(np.concatenate([np.where(y != "REPOSO")[0],
                                   np.random.default_rng(42).choice(rep, E.MAX_REPOSO, replace=False)]))
    X, y, g = X[keep], y[keep], g[keep]
    clases = sorted(set(y))
    idx = {c: i for i, c in enumerate(clases)}
    ye = np.array([idx[c] for c in y])

    prob = np.zeros((len(X), len(clases)))
    pred_exp = {tuple(gr): np.full(len(X), -1) for gr in GRUPOS}
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, ye, g), 1):
        sc = StandardScaler().fit(X[tr])
        prob[te] = E.nuevo_mlp(42 + k).fit(sc.transform(X[tr]), ye[tr]).predict_proba(sc.transform(X[te]))
        for gr in GRUPOS:
            m_tr = tr[np.isin(y[tr], gr)]
            sce = StandardScaler().fit(X[m_tr])
            exp = MLPClassifier((128, 64), alpha=1e-3, max_iter=400, early_stopping=True,
                                n_iter_no_change=20, random_state=k).fit(sce.transform(X[m_tr]), ye[m_tr])
            pred_exp[tuple(gr)][te] = exp.predict(sce.transform(X[te]))
        print(f"fold {k} listo", flush=True)

    top1 = prob.argmax(1)
    print(f"\n{'grupo':22} {'n':>6} {'principal':>10} {'+experto':>10}")
    resultados = {}
    for gr in GRUPOS:
        ids = [idx[c] for c in gr]
        sel = np.isin(ye, ids) & np.isin(top1, ids)
        base = float((top1[sel] == ye[sel]).mean())
        expt = float((pred_exp[tuple(gr)][sel] == ye[sel]).mean())
        resultados["/".join(gr)] = {"n": int(sel.sum()), "principal": base, "experto": expt}
        print(f"{'/'.join(gr):22} {sel.sum():6d} {base * 100:9.1f}% {expt * 100:9.1f}%")
    with open(os.path.join("resultados", "analisis_confusiones.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
