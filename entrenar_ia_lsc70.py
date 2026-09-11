"""
=============================================================
ENTRENAMIENTO DE INTELIGENCIA ARTIFICIAL CON LSC70 (v3.0)
Lengua de Señas Colombiana (LSC) - Descriptor Multimodal 109D
=============================================================
Pipeline:
1. Ingesta de LSC70W (70 signantes reales) con anclaje en hombros Pose:
   - Invariante a la censura / pixelado del rostro en LSC70.
   - Fotogramas 1 a 5: ejecución y ápice del gesto (HOLA, GUSTAR, etc.).
   - Fotograma 0: inicio desde reposo -> alimentado como clase REPOSO_TRANSICION.
2. Ingesta de señas capturadas locales (ej: GRACIAS).
3. Control negativo balanceado (REPOSO_TRANSICION) para eliminar adivinanza.
4. Descriptores multimodales de 109 dimensiones:
   - 105D cinemática canónica articular (rotación/escala invariante).
   - 4D anclaje espacial corporal relativo a hombros (altura y lateralidad).
5. Caché en datasets/cache_lsc70_109d.npz para aceleración inmediata.
6. Ensamble Híbrido Calibrado:
   - Red Neuronal Profunda MLP (256, 128) con regularización L2.
   - Extra-Trees Classifier (150 estimadores balanceados).
7. Validación cruzada estratificada (5-Fold Stratified CV).
8. Exportación dual: .joblib y .npz universal (pure-NumPy sin version lock).
"""

import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import cv2
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from motor_lsc import ExtractorLandmarks, clasificar_cuadrante, CuadranteEspacial, ClasificadorIALSC
from motor_lsc.clasificador_ia import CLASE_REPOSO


# ─────────────────────────────────────────────────────────────
# CONFIGURACION Y RUTAS
# ─────────────────────────────────────────────────────────────

DIR_LSC70W = os.path.join("datasets", "LSC70", "LSC70W")
DIR_CAPTURADO = os.path.join("datasets", "capturado")
CACHE_FEATURES_PATH = os.path.join("datasets", "cache_lsc70_109d.npz")
MODELO_SALIDA_PATH = os.path.join("modelos_guardados", "modelo_ia_lsc70.joblib")
METRICAS_SALIDA_PATH = os.path.join("modelos_guardados", "metricas_ia_lsc70.json")

# Mapeo anatómico verificado para LSC
MAPA_CUADRANTES = {
    "HOLA": CuadranteEspacial.CABEZA_ROSTRO.value,       # Sien / oreja / lateral cabeza
    "BUENAS": CuadranteEspacial.PECHO_TORSO.value,       # Pecho / esternón hacia afuera
    "DIAS": CuadranteEspacial.CABEZA_ROSTRO.value,       # Cabeza / barbilla
    "TARDES": CuadranteEspacial.PECHO_TORSO.value,       # Pecho / torso descendente
    "NOCHES": CuadranteEspacial.ESPACIO_CENTRAL.value,   # Espacio central frente al cuerpo
    "GRACIAS": CuadranteEspacial.CABEZA_ROSTRO.value,    # Barbilla / boca hacia adelante
    "BUENOS_DIAS": CuadranteEspacial.CABEZA_ROSTRO.value,
    "BUENAS_TARDES": CuadranteEspacial.PECHO_TORSO.value,
    "BUENAS_NOCHES": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "GUSTAR": CuadranteEspacial.PECHO_TORSO.value,       # Centro del pecho / corazón
    "NOMBRE": CuadranteEspacial.ESPACIO_LATERAL.value,   # Espacio neutro lateral
    "YO": CuadranteEspacial.PECHO_TORSO.value,           # Apuntando al pecho
    "LICOR": CuadranteEspacial.CUELLO_GARGANTA.value,    # Pulgar al cuello / garganta
    "ANNOS": CuadranteEspacial.CABEZA_ROSTRO.value,      # Mentón / mejilla
    CLASE_REPOSO: CuadranteEspacial.LATERAL_BAJO.value,
}


# ─────────────────────────────────────────────────────────────
# EXTRACCION Y CACHE DE CARACTERISTICAS (109 DIMENSIONES)
# ─────────────────────────────────────────────────────────────

def extraer_caracteristicas_lsc70(
    directorio_lsc70w: str,
    directorio_capturado: str,
    max_personas: int = 70,
    muestras_por_persona: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae descriptores multimodales de 109 dimensiones de LSC70W y capturas locales.
    Separa ápices (fotogramas 1-5) para gestos, y fotogramas 0 para reposo/transición.
    """
    print("\n" + "=" * 65)
    print("  EXTRACCION MULTIMODAL 109D (LSC70 + ANCLAJE CORPORAL)")
    print("=" * 65)

    extractor = ExtractorLandmarks(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
        usar_filtro_temporal=False,  # Procesamiento de imágenes estáticas
        usar_denoise_imagen=True,
    )

    X_list = []
    y_list = []
    personas_list = []
    cuadrantes_list = []

    # 1. Ingesta de LSC70W
    if os.path.exists(directorio_lsc70w):
        personas = sorted([
            d for d in os.listdir(directorio_lsc70w)
            if os.path.isdir(os.path.join(directorio_lsc70w, d)) and d.startswith("Per")
        ])[:max_personas]

        print(f"  Procesando {len(personas)} personas de LSC70W...")
        t_inicio = time.time()
        total_imgs = 0
        total_ok = 0
        total_reposo_real = 0

        for p_idx, p_name in enumerate(personas, 1):
            p_dir = os.path.join(directorio_lsc70w, p_name)
            for clase_dir in os.listdir(p_dir):
                ruta_clase = os.path.join(p_dir, clase_dir)
                if not os.path.isdir(ruta_clase):
                    continue

                etiqueta = clase_dir.upper().strip()
                imgs = sorted(glob.glob(os.path.join(ruta_clase, "*.jpg")))[:muestras_por_persona]

                for img_path in imgs:
                    total_imgs += 1
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # Determinar si es fotograma de reposo/inicio (frame 0) o ápice (frames 1-5)
                    nombre_archivo = os.path.basename(img_path)
                    es_frame_reposo = nombre_archivo.endswith(("_0.jpg", "_0.png"))

                    res = extractor.procesar_frame(img)
                    if res["hay_manos"]:
                        mano = res["manos"][0]
                        vec_109d = mano["vector_normalizado"]  # Vector 109D multimodal

                        # Determinar cuadrante
                        cuad_obj, _ = clasificar_cuadrante(
                            mano.get("muneca", (0.5, 0.5, 0)),
                            res.get("pose_anchors", {}),
                        )
                        cuad_str = cuad_obj.value if hasattr(cuad_obj, "value") else str(cuad_obj)

                        if es_frame_reposo:
                            # Frame 0 es posición de inicio/reposo: usar como control negativo real
                            X_list.append(vec_109d)
                            y_list.append(CLASE_REPOSO)
                            personas_list.append(p_name)
                            cuadrantes_list.append(CuadranteEspacial.LATERAL_BAJO.value)
                            total_reposo_real += 1
                        else:
                            # Frames 1-5 son la ejecución activa del gesto
                            X_list.append(vec_109d)
                            y_list.append(etiqueta)
                            personas_list.append(p_name)
                            cuadrantes_list.append(cuad_str)
                            total_ok += 1

            if p_idx % 10 == 0 or p_idx == len(personas):
                print(f"    [{p_idx}/{len(personas)}] Gestos: {total_ok} | Reposos reales: {total_reposo_real} | Evaluadas: {total_imgs}")

        t_fin = time.time()
        print(f"  [LSC70W] Finalizado en {t_fin - t_inicio:.1f}s: {total_ok} muestras de señas + {total_reposo_real} reposos reales.")

    # 2. Ingesta de Datasets Capturados Locales (ej: GRACIAS)
    if os.path.exists(directorio_capturado):
        print(f"\n  Procesando capturas locales en {directorio_capturado}...")
        muestras_locales = 0
        for sena_dir in os.listdir(directorio_capturado):
            ruta_sena = os.path.join(directorio_capturado, sena_dir)
            if not os.path.isdir(ruta_sena):
                continue
            etiqueta = sena_dir.upper().strip()
            archivos_npy = sorted(glob.glob(os.path.join(ruta_sena, "*.npy")))

            # Posición corporal representativa por defecto si viene de captura 105D antigua
            if etiqueta == "GRACIAS":
                coords_def = np.array([0.15, -0.25, -0.20, 0.35], dtype=np.float32) * 2.5
            elif "HOLA" in etiqueta:
                coords_def = np.array([0.70, -0.75, -0.15, 1.05], dtype=np.float32) * 2.5
            elif "NOCHES" in etiqueta:
                coords_def = np.array([0.10, 0.10, -0.25, 0.30], dtype=np.float32) * 2.5
            else:
                coords_def = np.array([0.10, 0.35, -0.20, 0.45], dtype=np.float32) * 2.5

            for f_npy in archivos_npy:
                try:
                    datos = np.load(f_npy, allow_pickle=True)
                    if datos.dtype == object and hasattr(datos, "item"):
                        d = datos.item()
                        vec = d.get("vector_normalizado", d.get("vector", None))
                    elif isinstance(datos, np.ndarray):
                        vec = datos.flatten()
                    else:
                        continue

                    if vec is not None:
                        if len(vec) == 105:
                            vec_109 = np.concatenate([vec, coords_def]).astype(np.float32)
                        elif len(vec) == 109:
                            vec_109 = vec
                        else:
                            continue

                        cuad = MAPA_CUADRANTES.get(etiqueta, CuadranteEspacial.CABEZA_ROSTRO.value)
                        factor_rep = 5 if etiqueta == "GRACIAS" else 2

                        for _ in range(factor_rep):
                            ruido = np.random.normal(0, 0.012, 109).astype(np.float32)
                            X_list.append(vec_109 + ruido)
                            y_list.append(etiqueta)
                            personas_list.append("LocalUser")
                            cuadrantes_list.append(cuad)
                            muestras_locales += 1
                except Exception:
                    pass

        print(f"  [CAPTURADO] Se integraron {muestras_locales} muestras locales adaptadas a 109D.")

    # 3. Muestras Sintéticas Adicionales de Control: REPOSO_TRANSICION
    print(f"\n  Generando control negativo sintético [{CLASE_REPOSO}]...")
    n_reposo_sintetico = 120
    for _ in range(n_reposo_sintetico):
        vec_rep = np.zeros(109, dtype=np.float32)
        # 63 dims canónicas relajadas
        vec_rep[:63] = np.random.normal(0.0, 0.04, 63)
        # 5 extensiones bajas (dedos caídos / relajados)
        vec_rep[63:68] = np.random.uniform(0.05, 0.28, 5)
        # Ángulos y distancias neutras
        vec_rep[68:105] = np.random.uniform(0.15, 0.50, 105 - 68)
        # Coordenadas corporales de reposo (manos abajo, dy > 1.2)
        dx_rep = np.random.uniform(-0.3, 0.3)
        dy_rep = np.random.uniform(1.1, 1.8)
        dz_rep = np.random.uniform(0.0, 0.4)
        dist_rep = np.sqrt(dx_rep**2 + dy_rep**2 + dz_rep**2)
        vec_rep[105:109] = np.array([dx_rep, dy_rep, dz_rep, dist_rep], dtype=np.float32) * 2.5

        X_list.append(vec_rep)
        y_list.append(CLASE_REPOSO)
        personas_list.append("ControlSintetico")
        cuadrantes_list.append(CuadranteEspacial.LATERAL_BAJO.value)

    extractor.liberar()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=str)
    personas = np.array(personas_list, dtype=str)
    cuadrantes = np.array(cuadrantes_list, dtype=str)

    # Guardar en caché
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_FEATURES_PATH)), exist_ok=True)
    np.savez_compressed(
        CACHE_FEATURES_PATH,
        X=X,
        y=y,
        personas=personas,
        cuadrantes=cuadrantes,
    )
    print(f"\n  [CACHE] Guardado exitosamente en: {CACHE_FEATURES_PATH}")
    print(f"  Dimensiones del dataset: X={X.shape} | Clases ({len(np.unique(y))}): {sorted(np.unique(y))}")
    return X, y, personas, cuadrantes


# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO Y EVALUACION DEL ENSAMBLE HIBRIDO
# ─────────────────────────────────────────────────────────────

def entrenar_ensamble_ia(
    X: np.ndarray,
    y: np.ndarray,
    cuadrantes: np.ndarray,
    n_estimators: int = 150,
) -> Tuple[ClasificadorIALSC, Dict]:
    """
    Entrena el ensamble híbrido (MLP + ExtraTrees) y evalúa mediante 5-Fold Stratified CV.
    """
    print("\n" + "=" * 65)
    print("  ENTRENAMIENTO DEL ENSAMBLE HIBRIDO (MLP + EXTRA-TREES)")
    print("=" * 65)

    clases_unicas = sorted(np.unique(y))
    print(f"  Distribución de clases ({len(clases_unicas)} clases):")
    for c in clases_unicas:
        n_c = np.sum(y == c)
        print(f"    - {c:20s}: {n_c:4d} muestras")

    # 1. Validación Cruzada Estratificada (5 Folds)
    print("\n  Ejecutando 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs_mlp = []
    accs_et = []
    accs_ensamble = []
    f1s_ensamble = []

    y_va_all = []
    y_pred_all = []

    # Métricas de confusión específicas para HOLA
    hola_total = 0
    hola_correctas = 0
    hola_confundidas_gustar = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_va_norm = scaler.transform(X_va)

        # Modelo A: Red Neuronal MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate="adaptive",
            max_iter=350,
            early_stopping=True,
            n_iter_no_change=14,
            random_state=42 + fold,
        )
        mlp.fit(X_tr_norm, y_tr)

        # Modelo B: Extra-Trees Classifier
        et = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=24,
            min_samples_split=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42 + fold,
        )
        et.fit(X_tr_norm, y_tr)

        # Fusión Probabilística Calibrada
        p_mlp = mlp.predict_proba(X_va_norm)
        p_et = et.predict_proba(X_va_norm)
        p_ens = 0.50 * p_mlp + 0.50 * p_et

        y_pred_mlp = mlp.classes_[np.argmax(p_mlp, axis=1)]
        y_pred_et = et.classes_[np.argmax(p_et, axis=1)]
        y_pred_ens = mlp.classes_[np.argmax(p_ens, axis=1)]

        y_va_all.extend(y_va)
        y_pred_all.extend(y_pred_ens)

        acc_mlp = accuracy_score(y_va, y_pred_mlp)
        acc_et = accuracy_score(y_va, y_pred_et)
        acc_ens = accuracy_score(y_va, y_pred_ens)
        f1_ens = f1_score(y_va, y_pred_ens, average="weighted")

        accs_mlp.append(acc_mlp)
        accs_et.append(acc_et)
        accs_ensamble.append(acc_ens)
        f1s_ensamble.append(f1_ens)

        # Conteo específico para HOLA
        h_mask = (y_va == "HOLA")
        if np.any(h_mask):
            hola_total += int(np.sum(h_mask))
            hola_correctas += int(np.sum(y_pred_ens[h_mask] == "HOLA"))
            hola_confundidas_gustar += int(np.sum(y_pred_ens[h_mask] == "GUSTAR"))

        print(f"    Fold {fold}: MLP={acc_mlp:.1%} | ExtraTrees={acc_et:.1%} | Ensamble={acc_ens:.1%} (F1={f1_ens:.3f})")

    prom_mlp = float(np.mean(accs_mlp))
    prom_et = float(np.mean(accs_et))
    prom_ens = float(np.mean(accs_ensamble))
    prom_f1 = float(np.mean(f1s_ensamble))

    print("\n" + "-" * 65)
    print(f"  ACCURACY PROMEDIO (5-Fold CV):")
    print(f"    - Red Neuronal MLP : {prom_mlp:.2%}")
    print(f"    - Extra-Trees      : {prom_et:.2%}")
    print(f"    - ENSAMBLE HIBRIDO : {prom_ens:.2%}  (F1-Weighted: {prom_f1:.4f})")
    if hola_total > 0:
        prec_hola = hola_correctas / hola_total
        print(f"    - Precision 'HOLA' : {prec_hola:.1%} ({hola_correctas}/{hola_total})")
        print(f"    - Confusión con 'GUSTAR': {hola_confundidas_gustar} veces")
    print("-" * 65)

    # 2. Entrenamiento Final sobre Todo el Dataset
    print("\n  Entrenando modelo final sobre el 100% de los datos...")
    scaler_final = StandardScaler()
    X_full_norm = scaler_final.fit_transform(X)

    mlp_final = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=16,
        random_state=42,
    )
    mlp_final.fit(X_full_norm, y)

    et_final = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=26,
        min_samples_split=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    et_final.fit(X_full_norm, y)

    # Mapeo de cuadrantes por clase
    cuadrantes_map = {}
    for c in clases_unicas:
        if c in MAPA_CUADRANTES:
            cuadrantes_map[c] = MAPA_CUADRANTES[c]
        else:
            idxs = np.where(y == c)[0]
            if len(idxs) > 0:
                vals, counts = np.unique(cuadrantes[idxs], return_counts=True)
                cuadrantes_map[c] = str(vals[np.argmax(counts)])

    # Empaquetar en el ClasificadorIALSC
    clasificador = ClasificadorIALSC(
        umbral_confianza=0.70,
        margen_minimo=0.12,
        peso_mlp=0.50,
        peso_et=0.50,
    )
    clasificador.mlp_model = mlp_final
    clasificador.et_model = et_final
    clasificador.scaler = scaler_final
    clasificador.clases = list(mlp_final.classes_)
    clasificador.cuadrantes_map = cuadrantes_map
    clasificador.esta_cargado = True

    # Generar todos los artefactos visuales y estadísticos en 'resultados/'
    from motor_lsc.generador_metricas import generar_reporte_completo_entrenamiento

    reporte_artefactos = generar_reporte_completo_entrenamiento(
        y_true=np.array(y_va_all),
        y_pred=np.array(y_pred_all),
        clases=clases_unicas,
        mlp_model=mlp_final,
        acc_ensamble=prom_ens,
        f1_ensamble=prom_f1,
        acc_mlp=prom_mlp,
        acc_et=prom_et,
        cv_accs=accs_ensamble,
        cv_f1s=f1s_ensamble,
        total_muestras=len(X),
        dimensiones_vector=int(X.shape[1]),
        carpeta_salida="resultados",
    )

    metricas = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_muestras": int(len(X)),
        "dimensiones_vector": int(X.shape[1]),
        "clases": list(clases_unicas),
        "cv_folds": 5,
        "accuracy_mlp": prom_mlp,
        "accuracy_extra_trees": prom_et,
        "accuracy_ensamble": prom_ens,
        "f1_score_ensamble": prom_f1,
        "hola_evaluadas": hola_total,
        "hola_aciertos": hola_correctas,
        "hola_confusion_gustar": hola_confundidas_gustar,
        "n_estimators": n_estimators,
        **reporte_artefactos,
    }

    return clasificador, metricas


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenador de IA para LSC con LSC70 (109D)")
    parser.add_argument("--max-personas", type=int, default=70, help="Personas de LSC70W (default: 70)")
    parser.add_argument("--muestras-por-persona", type=int, default=6, help="Muestras por persona (default: 6)")
    parser.add_argument("--forzar-extraccion", action="store_true", help="Re-extraer caracteristicas ignorando cache")
    parser.add_argument("--n-estimators", type=int, default=150, help="Estimadores de ExtraTrees (default: 150)")
    parser.add_argument("--salida-modelo", default=MODELO_SALIDA_PATH, help="Ruta de guardado del modelo .joblib")
    parser.add_argument("--salida-metricas", default=METRICAS_SALIDA_PATH, help="Ruta de metricas .json")
    args = parser.parse_args()

    # 1. Cargar o Extraer Características (109D)
    if os.path.exists(CACHE_FEATURES_PATH) and not args.forzar_extraccion:
        print(f"\n  [CACHE] Cargando descriptores 109D pre-extraidos desde: {CACHE_FEATURES_PATH}")
        data = np.load(CACHE_FEATURES_PATH)
        X = data["X"]
        y = data["y"]
        personas = data["personas"]
        cuadrantes = data["cuadrantes"]
        print(f"  Cargadas {len(X)} muestras (dims: {X.shape[1]}) de {len(np.unique(y))} clases.")
    else:
        X, y, personas, cuadrantes = extraer_caracteristicas_lsc70(
            directorio_lsc70w=DIR_LSC70W,
            directorio_capturado=DIR_CAPTURADO,
            max_personas=args.max_personas,
            muestras_por_persona=args.muestras_por_persona,
        )

    # 2. Entrenar y Evaluar Ensamble
    clasificador, metricas = entrenar_ensamble_ia(
        X, y, cuadrantes, n_estimators=args.n_estimators
    )

    # 3. Guardar Paquete del Modelo (.joblib y .npz universal) y Métricas
    clasificador.guardar(args.salida_modelo, metricas=metricas)
    print(f"\n  [OK] Modelo exportado (.joblib y .npz) en: {args.salida_modelo}")

    with open(args.salida_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metricas exportadas en: {args.salida_metricas}")

    print("\n" + "=" * 65)
    print("  ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"  Dimensiones del vector      : {metricas['dimensiones_vector']}D")
    print(f"  Precision Global Ensamble   : {metricas['accuracy_ensamble']:.2%}")
    if metricas.get("hola_evaluadas", 0) > 0:
        prec_h = metricas["hola_aciertos"] / metricas["hola_evaluadas"]
        print(f"  Precision 'HOLA'            : {prec_h:.1%}")
        print(f"  Colisiones HOLA con GUSTAR  : {metricas['hola_confusion_gustar']}")
    print("  Siguiente paso:")
    print("    python validar_5palabras.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
