"""
=============================================================================
ENTRENAMIENTO ULTRA-PRECISO DEL MODELO DE IA LSC70 v6.4.0 UNIFICADO
Lengua de Señas Colombiana — Red Neuronal Profunda Multimodal 109D (49 Clases)
=============================================================================
Integra:
  - 11 Palabras de uso frecuente + REPOSO (12 clases)
  - 27 Letras del Abecedario LSC (A-Z, NN/Ñ)
  - 10 Números y Cantidades LSC (1, 4, 5, 6, 7, 8, 9, 10, MIL, MILLON)
Total: 49 Clases Puras con balanceo cinemático 3D y Validación Cruzada.
Soporta segmentación lógica por modos (Palabras, Abecedario, Números, Todo).
=============================================================================
"""

import os
import sys
import json
import time
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import joblib

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 75)
    print("  ENTRENAMIENTO UNIFICADO LSC70 v6.4.0 (49 CLASES: PALABRAS + ALFABETO + NÚMEROS)")
    print("=" * 75)

    # Preferir caché consolidado si existe, sino caché base
    cache_path = os.path.join("datasets", "cache_lsc70_109d_completo.npz")
    if not os.path.exists(cache_path):
        cache_path = os.path.join("datasets", "cache_lsc70_109d.npz")
    
    if not os.path.exists(cache_path):
        print(f"Error: No existe el archivo de caché {cache_path}")
        return

    print(f"  Cargando dataset desde: {cache_path}...")
    with open(cache_path, 'rb') as f:
        buf = io.BytesIO(f.read())
    cache = np.load(buf, allow_pickle=True)
    X_raw, y_raw = cache["X"], cache["y"]
    print(f"  Datos crudos cargados: {len(X_raw)} muestras, {X_raw.shape[1]} dimensiones")

    # Mapeo de ANNOS a AÑOS en etiquetas crudas
    y_raw = np.array(['AÑOS' if s == 'ANNOS' else s for s in y_raw])

    # Definición de Categorías Lingüísticas
    valid_words = ['AÑOS', 'BUENAS', 'DIAS', 'GRACIAS', 'GUSTAR', 'HOLA', 'LICOR', 'NOCHES', 'NOMBRE', 'TARDES', 'YO']
    valid_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'NN', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    valid_numbers = ['1', '4', '5', '6', '7', '8', '9', '10', 'MIL', 'MILLON']
    
    all_target_signs = valid_words + valid_alphabet + valid_numbers
    
    # 1. Depuración Anatómica y Filtrado
    clean_samples = {}
    reposo_samples = []

    for c in all_target_signs:
        idx = np.where(y_raw == c)[0]
        if len(idx) == 0:
            continue
        Xc = X_raw[idx]
        dys = Xc[:, 106]
        fingers = Xc[:, 63:68] / 3.2
        
        # Filtros específicos para palabras dinámicas
        if c == 'AÑOS':
            is_clean = (fingers[:, 1] <= 0.65) & (fingers[:, 2] <= 0.65)
        elif c == 'BUENAS':
            is_clean = (fingers[:, 1] >= 0.60) & (fingers[:, 2] >= 0.60) & (dys <= 3.8)
        elif c == 'DIAS':
            is_clean = (fingers[:, 1] >= 0.50) & (dys <= 3.2)
        elif c == 'HOLA':
            is_clean = (fingers[:, 1] >= 0.55) & (fingers[:, 2] >= 0.55) & (dys <= 1.2)
        elif c == 'GRACIAS':
            is_clean = (fingers[:, 1] >= 0.70) & (fingers[:, 2] >= 0.70)
        elif c == 'GUSTAR':
            is_clean = (dys >= -0.50) & (dys <= 4.0) & (fingers[:, 1] >= 0.40)
        elif c == 'LICOR':
            is_clean = (fingers[:, 0] >= 0.45) & (dys <= 2.2)
        elif c == 'NOCHES':
            is_clean = (dys >= -1.80) & (dys <= 3.0)
        elif c == 'NOMBRE':
            is_clean = (fingers[:, 1] >= 0.55) & (fingers[:, 2] >= 0.55)
        elif c == 'TARDES':
            is_clean = (fingers[:, 1] >= 0.55) & (dys >= 0.20)
        elif c == 'YO':
            is_clean = (fingers[:, 1] >= 0.25) & (dys <= 5.5)
        else:
            # Letras y números (posturas estáticas controladas en LSC70AN)
            is_clean = np.ones(len(Xc), dtype=bool)
        
        clean_samples[c] = Xc[is_clean]

        # Enriquecimiento cinemático fonológico para GRACIAS
        if c == 'GRACIAS' and len(clean_samples['GRACIAS']) > 0:
            base_gracias = clean_samples['GRACIAS']
            n_enrich = 300
            idx_g = np.random.choice(len(base_gracias), n_enrich, replace=True)
            X_gracias_dyn = base_gracias[idx_g].copy()
            dys_traj = np.random.uniform(-0.35, 0.15, n_enrich)
            dzs_traj = np.random.uniform(-0.25, 0.20, n_enrich)
            dxs_traj = np.random.uniform(0.05, 0.22, n_enrich)
            X_gracias_dyn[:, 105] = dxs_traj * 2.5
            X_gracias_dyn[:, 106] = dys_traj * 2.5
            X_gracias_dyn[:, 107] = dzs_traj * 2.5
            X_gracias_dyn[:, 90] = np.random.uniform(0.04, 0.15, n_enrich)
            X_gracias_dyn[:, :105] += np.random.normal(0, 0.005, (n_enrich, 105))
            clean_samples['GRACIAS'] = np.vstack([clean_samples['GRACIAS'], X_gracias_dyn])

    # 2. Muestras de REPOSO
    if 'REPOSO_TRANSICION' in y_raw:
        rt = X_raw[y_raw == 'REPOSO_TRANSICION']
        reposo_samples.append(rt[rt[:, 106] > 3.0])

    for c in ['HOLA', 'BUENAS', 'TARDES', 'GUSTAR']:
        idx_src = np.where(y_raw == c)[0]
        if len(idx_src) > 0:
            Xsrc = X_raw[idx_src]
            n_s = 150
            idx_s = np.random.choice(len(Xsrc), n_s, replace=True)
            Xs = Xsrc[idx_s].copy()
            Xs[:, 106] = np.random.uniform(2.2, 4.5, n_s) * 2.5
            Xs[:, 105] = np.random.uniform(-0.6, 0.6, n_s) * 2.5
            factor_relajacion = np.random.uniform(0.40, 1.0, (n_s, 5))
            Xs[:, 63:68] = np.clip(Xs[:, 63:68] * factor_relajacion, 0.0, 3.2)
            Xs[:, :105] += np.random.normal(0, 0.005, (n_s, 105))
            reposo_samples.append(Xs)

    clean_samples['REPOSO'] = np.vstack(reposo_samples)

    print(f"\n  Filtrado y Limpieza Completados: {len(clean_samples)} clases activas:")
    for c, arr in sorted(clean_samples.items()):
        print(f"    - {c:10}: {len(arr):3} muestras base")

    # 3. Balanceo y Aumento Fino 3D Canónico
    np.random.seed(42)
    X_final_list = []
    y_final_list = []
    target_por_clase = 300  # 300 muestras por clase balanceadas (~14,700 total)

    def aplicar_rotacion_3d(X_in, max_grados=12.0):
        X_rot = X_in.copy()
        n_samples = len(X_in)
        angulos_rad = np.radians(np.random.uniform(-max_grados, max_grados, n_samples))
        cos_a = np.cos(angulos_rad)
        sin_a = np.sin(angulos_rad)

        for i in range(21):
            px = X_rot[:, i * 3]
            pz = X_rot[:, i * 3 + 2]
            X_rot[:, i * 3]     = px * cos_a + pz * sin_a
            X_rot[:, i * 3 + 2] = -px * sin_a + pz * cos_a

        nx = X_rot[:, 102]
        nz = X_rot[:, 104]
        X_rot[:, 102] = nx * cos_a + nz * sin_a
        X_rot[:, 104] = -nx * sin_a + nz * cos_a
        return X_rot

    for c, Xc in clean_samples.items():
        reps = int(np.ceil(target_por_clase / len(Xc)))
        for r in range(reps):
            if r == 0:
                X_rep = Xc.copy()
            else:
                X_rep = aplicar_rotacion_3d(Xc, max_grados=10.0)
                noise = np.zeros_like(Xc)
                noise[:, :105] = np.random.normal(0, 0.003, (len(Xc), 105))
                noise[:, 105] = np.random.normal(0, 0.015, len(Xc)) * 2.5
                noise[:, 106] = np.random.normal(0, 0.020, len(Xc)) * 2.5
                noise[:, 107] = np.random.normal(0, 0.015, len(Xc)) * 2.5
                noise[:, 108] = np.random.normal(0, 0.015, len(Xc)) * 2.5
                X_rep += noise
                
            X_final_list.append(X_rep)
            y_final_list.append(np.array([c] * len(Xc)))

    X_all_raw = np.vstack(X_final_list)
    y_all_raw = np.concatenate(y_final_list)

    # Balanceo exacto
    X_exact = []
    y_exact = []
    for c in np.unique(y_all_raw):
        idx_c = np.where(y_all_raw == c)[0]
        sub = np.random.choice(idx_c, target_por_clase, replace=False)
        X_exact.append(X_all_raw[sub])
        y_exact.append(y_all_raw[sub])

    X = np.vstack(X_exact)
    y = np.concatenate(y_exact)
    print(f"\n  Total de muestras balanceadas: {len(X)} en {len(np.unique(y))} clases:")
    clases_ordenadas = sorted(list(np.unique(y)))
    print(f"  Clases ({len(clases_ordenadas)}): {clases_ordenadas}")

    # 4. Normalización
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # 5. Validación Cruzada Estratificada (5-Fold CV)
    print("\n  Ejecutando Validación Cruzada Estratificada (5-Fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    arch = (640, 384, 192)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_norm, y_enc), 1):
        mlp_cv = MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            alpha=0.00012,
            learning_rate_init=0.0008,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.10,
            random_state=42 + fold
        )
        mlp_cv.fit(X_norm[train_idx], y_enc[train_idx])
        acc_fold = accuracy_score(y_enc[test_idx], mlp_cv.predict(X_norm[test_idx]))
        cv_scores.append(acc_fold)
        print(f"    Fold {fold}: {acc_fold * 100:.2f}% de precisión")

    acc_media = np.mean(cv_scores)
    std_media = np.std(cv_scores)
    print(f"  [EXITO] Precisión Media 5-Fold: {acc_media * 100:.2f}% (+/- {std_media * 100:.2f}%)")

    # 6. Modelo Final de Producción
    print(f"\n  Entrenando Modelo Final de Producción ({len(clases_ordenadas)} clases, Arquitectura: {arch})...")
    t0 = time.time()
    mlp_final = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        alpha=0.00012,
        learning_rate_init=0.0008,
        max_iter=800,
        random_state=42
    )
    mlp_final.fit(X_norm, y_enc)
    t_train = time.time() - t0
    print(f"  Entrenamiento completado en {t_train:.1f}s")

    preds_final = mlp_final.predict(X_norm)
    acc_final = accuracy_score(y_enc, preds_final)
    f1_final = f1_score(y_enc, preds_final, average='weighted')
    report = classification_report(y_enc, preds_final, target_names=clases_ordenadas, output_dict=True)

    print(f"\n" + "=" * 75)
    print(f"  RESULTADO FINAL: Precisión Global = {acc_final * 100:.2f}% | F1-Score = {f1_final * 100:.2f}%")
    print("=" * 75)

    # 7. Guardar Matriz de Confusión
    os.makedirs("modelos_guardados", exist_ok=True)
    cm = confusion_matrix(y_enc, preds_final)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=clases_ordenadas, yticklabels=clases_ordenadas)
    plt.title(f"Matriz de Confusión LSC70 v6.4 (49 Clases) — Precisión: {acc_final*100:.1f}%", fontsize=14, fontweight='bold')
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Etiqueta Real", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join("modelos_guardados", "matriz_confusion.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"  [OK] Matriz de confusión guardada: {cm_path}")

    # 8. Exportación a NPZ y Joblib
    npz_path = os.path.join("modelos_guardados", "modelo_ia_lsc70.npz")
    joblib_path = os.path.join("modelos_guardados", "modelo_ia_lsc70.joblib")

    joblib.dump({
        "mlp": mlp_final,
        "scaler": scaler,
        "clases": clases_ordenadas,
        "label_encoder": le
    }, joblib_path)

    coefs = mlp_final.coefs_
    intercepts = mlp_final.intercepts_
    
    save_dict = {
        'clases': np.array(clases_ordenadas),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'num_layers': np.array([len(coefs)])
    }
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        save_dict[f'w{i}'] = w
        save_dict[f'b{i}'] = b
    
    np.savez_compressed(npz_path, **save_dict)
    print(f"  [OK] Modelo guardado en {npz_path} y {joblib_path}")

    # 9. Definición de Categorías para el Selector de Modos de la App
    categorias_dict = {
        "palabras": [c for c in valid_words if c in clases_ordenadas] + ["REPOSO"],
        "abecedario": [c for c in valid_alphabet if c in clases_ordenadas] + ["REPOSO"],
        "numeros": [c for c in valid_numbers if c in clases_ordenadas] + ["REPOSO"],
        "todo": clases_ordenadas
    }

    # 10. Exportación Directa a JSON y JavaScript On-Device
    weights_export = [w.tolist() for w in coefs]
    biases_export = [b.tolist() for b in intercepts]
    layers_list = [109] + list(arch) + [len(clases_ordenadas)]

    modelo_json = {
        "clases": clases_ordenadas,
        "categorias": categorias_dict,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": weights_export,
        "biases": biases_export,
        "layers": layers_list,
        "version": "6.4.0",
        "precision_cv": float(acc_media),
        "precision_global": float(acc_final)
    }

    json_path = os.path.join("modelos_guardados", "modelo_lsc_movil.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(modelo_json, f)
    print(f"  [OK] Modelo JSON exportado a: {json_path}")

    js_code = f"""/**
 * MODELO DE INTELIGENCIA ARTIFICIAL LSC v6.4.0 UNIFICADO (ON-DEVICE / ZERO SERVER)
 * Precisión Validación Cruzada: {acc_media*100:.2f}% | Precisión Global: {acc_final*100:.2f}%
 * Arquitectura: MLP 109D -> {' -> '.join(str(x) for x in arch)} -> {len(clases_ordenadas)} Clases
 * Clases: {json.dumps(clases_ordenadas)}
 * Soporte Multi-Modo: Palabras ({len(categorias_dict['palabras'])}), Abecedario ({len(categorias_dict['abecedario'])}), Números ({len(categorias_dict['numeros'])})
 */
const VERSION_MODELO_LSC = "6.4.0";
const BUILD_FECHA_LSC = "{time.strftime('%Y-%m-%d')}";
const METADATOS_MODELO_LSC = {{
  version: "6.4.0",
  subversion: "Unificado-49Clases-Multimodo",
  precision: "{acc_final*100:.2f}%",
  precision_cv: "{acc_media*100:.2f}%",
  clases: {len(clases_ordenadas)},
  fecha: "{time.strftime('%Y-%m-%d')}"
}};
const _MODELO_LSC_DATA = {json.dumps(modelo_json)};
if (typeof window !== 'undefined') {{
  window.MODELO_LSC = _MODELO_LSC_DATA;
  window.VERSION_MODELO_LSC = VERSION_MODELO_LSC;
  window.METADATOS_MODELO_LSC = METADATOS_MODELO_LSC;
}}
if (typeof module !== 'undefined' && module.exports) {{
  module.exports = {{
    ..._MODELO_LSC_DATA,
    VERSION_MODELO_LSC,
    METADATOS_MODELO_LSC
  }};
}}
"""
    for dest in ["estilo/modelo_ia_cliente.js", "app_lsc/assets/web/modelo_ia_cliente.js", "docs/modelo_ia_cliente.js"]:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(js_code)
        print(f"  [OK] Modelo cliente JS actualizado en: {dest}")

    # 11. Guardar Métricas
    metricas = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "6.4.0",
        "total_muestras": len(X),
        "total_clases": len(clases_ordenadas),
        "precision_global": float(acc_final),
        "precision_cv_media": float(acc_media),
        "f1_score": float(f1_final),
        "clases": clases_ordenadas,
        "categorias": categorias_dict,
        "arquitectura": layers_list,
        "cv_scores": [float(s) for s in cv_scores],
        "metricas_por_clase": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"])
            }
            for c in clases_ordenadas
        }
    }
    with open(os.path.join("modelos_guardados", "metricas_ia_lsc70.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    print("  [OK] Métricas guardadas en modelos_guardados/metricas_ia_lsc70.json")

    # 12. Actualizar Carpeta resultados/
    os.makedirs("resultados", exist_ok=True)
    report_str = classification_report(y_enc, preds_final, target_names=clases_ordenadas, digits=4)
    with open(os.path.join("resultados", "reporte_clasificacion.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)
    
    with open(os.path.join("resultados", "metricas_actuales.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    
    # Copiar matriz de confusión a resultados/
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=clases_ordenadas, yticklabels=clases_ordenadas)
    plt.title(f"Matriz de Confusión LSC70 v6.4 (49 Clases)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("resultados", "matriz_confusion.png"), dpi=180)
    plt.close()

    print(f"\n" + "=" * 75)
    print(f"  ENTRENAMIENTO Y DESPLIEGUE MULTIMODO COMPLETADO CON ÉXITO")
    print(f"=" * 75)

if __name__ == '__main__':
    main()
