"""
=============================================================
ENTRENAMIENTO DINAMICO — REGISTRO EN BASE VECTORIAL LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Lee TODOS los vectores capturados por capturar_senas.py
(de datasets/capturado/<SENA>/) y los registra en la
BaseVectoresLSC, generando el modelo listo para uso en vivo.

Detección automática: cualquier carpeta con archivos .npy
en datasets/capturado/ es tratada como una seña a registrar.

Uso:
  python entrenar_5palabras.py
  python entrenar_5palabras.py --senas HOLA BUENOS_DIAS
  python entrenar_5palabras.py --ampliar
  python entrenar_5palabras.py --umbral 0.72
"""

import os
import sys
import json
import argparse
import numpy as np

from motor_lsc.base_vectores import BaseVectoresLSC

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

# Señas por defecto (si no se especifica nada)
SENAS_OBJETIVO = ["HOLA", "GRACIAS", "BIEN", "CASA", "AGUA"]

DIR_CAPTURADO = os.path.join("datasets", "capturado")
MODELO_5P_PATH = os.path.join("modelos_guardados", "base_5palabras.npz")
CLASES_5P_PATH = os.path.join("modelos_guardados", "clases_5palabras.json")

# Cuadrante espacial estimado por seña
# (se puede ampliar para cualquier seña nueva)
CUADRANTES_SENAS = {
    # Señas originales
    "HOLA":           "ESPACIO_LATERAL",
    "GRACIAS":        "CABEZA_ROSTRO",
    "BIEN":           "ESPACIO_LATERAL",
    "CASA":           "ESPACIO_CENTRAL",
    "AGUA":           "CABEZA_ROSTRO",
    # Saludos temporales (frases compuestas)
    "BUENOS_DIAS":    "CABEZA_ROSTRO",
    "BUENAS_TARDES":  "PECHO_TORSO",
    "BUENAS_NOCHES":  "ESPACIO_CENTRAL",
    # Otros comunes
    "POR_FAVOR":      "PECHO_TORSO",
    "NOMBRE":         "ESPACIO_LATERAL",
    "AYUDA":          "ESPACIO_CENTRAL",
    "SI":             "ESPACIO_LATERAL",
    "NO":             "ESPACIO_LATERAL",
}


# ─────────────────────────────────────────────────────────────
# CARGA DE MUESTRAS
# ─────────────────────────────────────────────────────────────

def cargar_muestras(sena: str, dir_base: str = DIR_CAPTURADO):
    """Carga todos los vectores .npy de una seña."""
    dir_sena = os.path.join(dir_base, sena)
    if not os.path.isdir(dir_sena):
        return []

    archivos = sorted([f for f in os.listdir(dir_sena) if f.endswith(".npy")])
    vectores = []
    for archivo in archivos:
        ruta = os.path.join(dir_sena, archivo)
        try:
            v = np.load(ruta)
            if v.ndim == 1 and len(v) == 105:
                vectores.append(v)
            else:
                print(f"  !  {archivo}: forma inesperada {v.shape}, omitido.")
        except Exception as e:
            print(f"  !  {archivo}: error al cargar ({e}), omitido.")

    return vectores


# ─────────────────────────────────────────────────────────────
# ANÁLISIS DE CALIDAD
# ─────────────────────────────────────────────────────────────

def analizar_calidad(vectores: list, sena: str):
    """Calcula métricas de calidad del conjunto de muestras."""
    if len(vectores) < 2:
        return {"varianza": 0.0, "similitud_intra": 0.0, "n": len(vectores)}

    mat = np.array(vectores, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / np.where(norms > 1e-6, norms, 1.0)

    # Similitud coseno intra-clase (promedio entre todos los pares)
    sim_matrix = mat_norm @ mat_norm.T
    n = len(vectores)
    # Excluir diagonal
    mask = ~np.eye(n, dtype=bool)
    similitud_intra = float(sim_matrix[mask].mean())

    varianza = float(np.mean(np.var(mat, axis=0)))

    return {
        "n": n,
        "similitud_intra": similitud_intra,
        "varianza": varianza,
    }


def similitud_inter_clase(base: BaseVectoresLSC, sena_ref: str, vectores_ref: list) -> float:
    """Calcula similitud máxima entre esta seña y las demás registradas."""
    if base.vectores is None or not vectores_ref:
        return 0.0

    # Promedio de vectores de esta seña
    vec_prom = np.mean(vectores_ref, axis=0).astype(np.float32)
    vec_prom_norm = vec_prom / (np.linalg.norm(vec_prom) + 1e-6)

    # Comparar contra todos los vectores de OTRAS clases
    otras_idx = [i for i, e in enumerate(base.etiquetas) if e != sena_ref.upper()]
    if not otras_idx or base.vectores_norm is None:
        return 0.0

    otras_vecs = base.vectores_norm[otras_idx]
    sims = otras_vecs @ vec_prom_norm
    return float(np.max(sims))


# ─────────────────────────────────────────────────────────────
# REGISTRO EN BASE VECTORIAL
# ─────────────────────────────────────────────────────────────

def registrar_sena(base: BaseVectoresLSC, sena: str, vectores: list, estrategia: str = "todos"):
    """
    Registra las muestras de una seña en la base vectorial.
    
    Estrategias:
      'todos'    : Registra cada muestra como referencia separada (más robusto).
      'promedio' : Registra solo el vector promedio (más compacto).
      'mixto'    : Registra promedio + muestras de los extremos del espacio.
    """
    cuadrante = CUADRANTES_SENAS.get(sena, "ESPACIO_LATERAL")
    n_reg = 0

    if estrategia == "promedio":
        vec_prom = np.mean(vectores, axis=0).astype(np.float32)
        base.agregar_referencia_estatica(vec_prom, sena, cuadrante, "5Palabras")
        n_reg = 1

    elif estrategia == "todos":
        for vec in vectores:
            base.agregar_referencia_estatica(vec, sena, cuadrante, "5Palabras")
        n_reg = len(vectores)

    elif estrategia == "mixto":
        # Promedio + los 2 más extremos (máxima cobertura del espacio)
        vec_prom = np.mean(vectores, axis=0).astype(np.float32)
        base.agregar_referencia_estatica(vec_prom, sena, cuadrante, "5Palabras")
        n_reg = 1
        if len(vectores) >= 3:
            mat = np.array(vectores, dtype=np.float32)
            prom_norm = vec_prom / (np.linalg.norm(vec_prom) + 1e-6)
            mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)
            sims = mat_norm @ prom_norm
            idx_min1 = int(np.argmin(sims))
            sims[idx_min1] = 1.0
            idx_min2 = int(np.argmin(sims))
            for idx in [idx_min1, idx_min2]:
                base.agregar_referencia_estatica(vectores[idx], sena, cuadrante, "5Palabras")
                n_reg += 1

    return n_reg


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento dinamico LSC")
    parser.add_argument("--senas", nargs="+", default=None,
                        help="Senas a registrar. Por defecto: auto-detecta todo en datasets/capturado/")
    parser.add_argument("--ampliar", action="store_true",
                        help="Ampliar modelo existente en lugar de crear uno nuevo")
    parser.add_argument("--estrategia", choices=["todos", "promedio", "mixto"],
                        default="todos", help="Estrategia de registro (default: todos)")
    parser.add_argument("--umbral", type=float, default=0.72,
                        help="Umbral minimo de similitud (default: 0.72)")
    args = parser.parse_args()

    # Auto-detectar señas si no se especifican
    if args.senas:
        senas = [s.upper() for s in args.senas]
    elif os.path.isdir(DIR_CAPTURADO):
        # Detectar todas las carpetas con .npy dentro
        senas = sorted([
            d for d in os.listdir(DIR_CAPTURADO)
            if os.path.isdir(os.path.join(DIR_CAPTURADO, d))
            and any(f.endswith(".npy")
                    for f in os.listdir(os.path.join(DIR_CAPTURADO, d)))
        ])
        if not senas:
            print("ERROR: No hay senas capturadas en datasets/capturado/")
            print("Ejecuta primero: python capturar_senas.py --auto")
            sys.exit(1)
    else:
        print("ERROR: No existe datasets/capturado/")
        print("Ejecuta primero: python capturar_senas.py --auto")
        sys.exit(1)

    print("=" * 60)
    print("  LSC v3.0 — ENTRENAMIENTO DINAMICO")
    print("=" * 60)
    print(f"  Senas detectadas  : {senas}")
    print(f"  Estrategia        : {args.estrategia}")
    print(f"  Umbral similitud  : {args.umbral}")
    print(f"  Modo              : {'ampliar' if args.ampliar else 'nuevo'}")
    print(f"  Fuente datos      : {os.path.abspath(DIR_CAPTURADO)}")
    print(f"  Modelo salida     : {os.path.abspath(MODELO_5P_PATH)}")
    print("=" * 60)

    # Inicializar base
    base = BaseVectoresLSC(umbral_min_similitud=args.umbral)

    # Cargar modelo existente si se amplía
    if args.ampliar and os.path.exists(MODELO_5P_PATH):
        base.cargar(MODELO_5P_PATH)
        print(f"\n  Cargado modelo existente: {base.total_senas} vectores, {len(base.clases_unicas)} clases")

    # Procesar cada seña
    print("\n  Procesando muestras:")
    resultados = {}
    errores = []

    for sena in senas:
        vectores = cargar_muestras(sena)

        if not vectores:
            print(f"  X  {sena}: sin muestras en {os.path.join(DIR_CAPTURADO, sena)}/")
            print(f"       -> Ejecuta: python capturar_senas.py --senas {sena}")
            errores.append(sena)
            continue

        calidad = analizar_calidad(vectores, sena)
        n_reg = registrar_sena(base, sena, vectores, args.estrategia)

        # Calidad inter-clase
        sim_inter = similitud_inter_clase(base, sena, vectores)

        resultados[sena] = {
            "muestras": len(vectores),
            "registrados": n_reg,
            "similitud_intra": calidad["similitud_intra"],
            "similitud_inter_max": sim_inter,
            "varianza": calidad["varianza"],
        }

        # Indicador de calidad
        if calidad["similitud_intra"] >= 0.90:
            q = "[V] Excelente"
        elif calidad["similitud_intra"] >= 0.75:
            q = "[A] Buena"
        else:
            q = "[X] Baja (considera recapturar)"

        print(f"\n  OK  {sena}:")
        print(f"       Muestras: {len(vectores)} | Registradas: {n_reg}")
        print(f"       Similitud intra-clase: {calidad['similitud_intra']:.3f} — {q}")
        if sim_inter > 0:
            conf_sep = "[V]" if sim_inter < 0.85 else "[X] (riesgo confusión!)"
            print(f"       Similitud inter-clase máx: {sim_inter:.3f} {conf_sep}")

    if errores:
        print(f"\n  ! Señas sin datos: {errores}")
        if len(errores) == len(senas):
            print("  No hay datos para ninguna seña. Abortar.")
            sys.exit(1)

    if base.total_senas == 0:
        print("\n  ERROR: Base vectorial vacía. Verifica que capturar_senas.py haya corrido.")
        sys.exit(1)

    # Guardar modelo
    os.makedirs("modelos_guardados", exist_ok=True)
    base.guardar(MODELO_5P_PATH)
    print(f"\n  OK Modelo guardado: {MODELO_5P_PATH}")
    print(f"    Vectores totales: {base.total_senas}")
    print(f"    Clases únicas   : {base.clases_unicas}")

    # Guardar metadatos de clases
    meta = {
        "version": "3.0",
        "clases": base.clases_unicas,
        "total_vectores": base.total_senas,
        "estrategia": args.estrategia,
        "umbral": args.umbral,
        "resultados": resultados,
    }
    with open(CLASES_5P_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"    Metadatos       : {CLASES_5P_PATH}")

    # Reporte final
    print("\n" + "=" * 60)
    print("  REPORTE DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"  {'SEÑA':<12} {'MUESTRAS':>8} {'REG':>5} {'INTRA':>8} {'INTER':>8}")
    print("  " + "-" * 48)
    for sena, r in resultados.items():
        print(f"  {sena:<12} {r['muestras']:>8} {r['registrados']:>5} "
              f"{r['similitud_intra']:>8.3f} {r['similitud_inter_max']:>8.3f}")

    print()
    print("  Siguiente paso:")
    print("    python validar_5palabras.py        (validar en vivo)")
    print("    python predecir_vivo.py            (modo producción)")
    print("=" * 60)


if __name__ == "__main__":
    main()
