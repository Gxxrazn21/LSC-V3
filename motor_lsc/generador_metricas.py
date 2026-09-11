"""
=============================================================
GENERADOR DE MÉTRICAS, CURVAS Y REPORTES VISUALES (v3.0)
Lengua de Señas Colombiana (LSC)
=============================================================
Genera y almacena en la carpeta 'resultados/':
1. Matriz de confusión dual (conteos absolutos y % de recall normalizado).
2. Curvas de aprendizaje y pérdida (Loss Curve de la Red Neuronal MLP).
3. Comparativa de Precision, Recall y F1-Score por seña individual.
4. Reporte textual detallado (reporte_clasificacion.txt).
5. Historial cronológico acumulativo (historial_entrenamientos.json).
6. Gráfico de evolución histórica (comparativa_historica.png) para medir
   si cada nuevo entrenamiento mejora o desmejora respecto al anterior.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos headless y servidores
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


def generar_reporte_completo_entrenamiento(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases: List[str],
    mlp_model: Any,
    acc_ensamble: float,
    f1_ensamble: float,
    acc_mlp: float,
    acc_et: float,
    cv_accs: List[float],
    cv_f1s: List[float],
    total_muestras: int,
    dimensiones_vector: int = 109,
    carpeta_salida: str = "resultados",
) -> Dict[str, Any]:
    """
    Genera todos los artefactos visuales y estadísticos en la carpeta de resultados.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_id = time.strftime("%Y%m%d_%H%M%S")

    # 1. Matriz de Confusión Dual
    ruta_matriz = os.path.join(carpeta_salida, "matriz_confusion.png")
    _generar_grafico_matriz_confusion(y_true, y_pred, clases, ruta_matriz)

    # 2. Curvas de Aprendizaje y Pérdida
    ruta_curvas = os.path.join(carpeta_salida, "curvas_aprendizaje.png")
    _generar_grafico_curvas_aprendizaje(mlp_model, cv_accs, cv_f1s, ruta_curvas)

    # 3. Métricas de Precision, Recall y F1 por Clase
    ruta_barras = os.path.join(carpeta_salida, "metricas_por_clase.png")
    dict_reporte = _generar_grafico_metricas_clase(y_true, y_pred, clases, ruta_barras)

    # 4. Reporte Textual en TXT
    ruta_txt = os.path.join(carpeta_salida, "reporte_clasificacion.txt")
    txt_reporte = classification_report(y_true, y_pred, labels=clases, zero_division=0)
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  REPORTE DE CLASIFICACION - LSC v3.0 (IA LSC70)\n")
        f.write(f"  Fecha: {timestamp_str} | Muestras: {total_muestras} | Dims: {dimensiones_vector}D\n")
        f.write(f"  Accuracy Global Ensamble: {acc_ensamble:.2%} | F1-Score: {f1_ensamble:.4f}\n")
        f.write("=" * 70 + "\n\n")
        f.write(txt_reporte)
        f.write("\n" + "=" * 70 + "\n")

    # 5. Historial Acumulativo de Entrenamientos
    ruta_historial = os.path.join(carpeta_salida, "historial_entrenamientos.json")
    registro_actual = {
        "id": timestamp_id,
        "fecha": timestamp_str,
        "total_muestras": int(total_muestras),
        "dimensiones": int(dimensiones_vector),
        "accuracy_ensamble": float(acc_ensamble),
        "f1_score_ensamble": float(f1_ensamble),
        "accuracy_mlp": float(acc_mlp),
        "accuracy_extra_trees": float(acc_et),
        "cv_folds_accuracy": [float(a) for a in cv_accs],
        "precision_hola": float(dict_reporte.get("HOLA", {}).get("precision", 0.0)),
        "recall_hola": float(dict_reporte.get("HOLA", {}).get("recall", 0.0)),
        "precision_gustar": float(dict_reporte.get("GUSTAR", {}).get("precision", 0.0)),
        "recall_gustar": float(dict_reporte.get("GUSTAR", {}).get("recall", 0.0)),
    }

    historial = []
    if os.path.exists(ruta_historial):
        try:
            with open(ruta_historial, "r", encoding="utf-8") as f:
                historial = json.load(f)
        except Exception:
            historial = []

    # Calcular deltas de mejora respecto al entrenamiento previo
    if historial:
        previo = historial[-1]
        delta_acc = acc_ensamble - previo["accuracy_ensamble"]
        delta_f1 = f1_ensamble - previo["f1_score_ensamble"]
        registro_actual["delta_accuracy"] = float(delta_acc)
        registro_actual["delta_f1"] = float(delta_f1)
        registro_actual["ha_mejorado"] = bool(delta_acc > 0 or delta_f1 > 0)
    else:
        registro_actual["delta_accuracy"] = 0.0
        registro_actual["delta_f1"] = 0.0
        registro_actual["ha_mejorado"] = True

    historial.append(registro_actual)
    with open(ruta_historial, "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)

    # 6. Gráfico de Comparativa Histórica de Entrenamientos
    ruta_comparativa = os.path.join(carpeta_salida, "comparativa_historica.png")
    _generar_grafico_comparativa_historica(historial, ruta_comparativa)

    # 7. Resumen de la sesión actual
    ruta_actual = os.path.join(carpeta_salida, "metricas_actuales.json")
    with open(ruta_actual, "w", encoding="utf-8") as f:
        json.dump(registro_actual, f, indent=2, ensure_ascii=False)

    print(f"\n  [RESULTADOS] Todos los artefactos generados exitosamente en '{carpeta_salida}/':")
    print(f"    - Matriz de confusión     : {ruta_matriz}")
    print(f"    - Curvas de aprendizaje   : {ruta_curvas}")
    print(f"    - Métricas por clase      : {ruta_barras}")
    print(f"    - Reporte detallado (.txt): {ruta_txt}")
    print(f"    - Historial acumulativo   : {ruta_historial} ({len(historial)} sesiones registradas)")
    print(f"    - Comparativa histórica   : {ruta_comparativa}")
    if "delta_accuracy" in registro_actual and len(historial) > 1:
        signo = "+" if registro_actual["delta_accuracy"] >= 0 else ""
        print(f"    -> Variación Accuracy vs sesión anterior: {signo}{registro_actual['delta_accuracy']:.2%}")

    return registro_actual


def _generar_grafico_matriz_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, clases: List[str], ruta_salida: str
):
    """Genera una imagen con dos matrices de confusión: conteo absoluto y recall normalizado."""
    cm_counts = confusion_matrix(y_true, y_pred, labels=clases)
    cm_norm = confusion_matrix(y_true, y_pred, labels=clases, normalize="true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), dpi=150)
    fig.patch.set_facecolor("#181920")

    # Matriz 1: Conteos Absolutos
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clases,
        yticklabels=clases,
        ax=ax1,
        cbar=True,
        linewidths=0.5,
        linecolor="#252630",
    )
    ax1.set_title("Matriz de Confusión - Conteos Absolutos", color="#E5E7EB", fontsize=14, pad=12, fontweight="bold")
    ax1.set_xlabel("Predicción del Modelo", color="#D1D5DB", fontsize=12)
    ax1.set_ylabel("Etiqueta Real", color="#D1D5DB", fontsize=12)
    ax1.tick_params(colors="#D1D5DB", labelsize=10, rotation=45)

    # Matriz 2: Recall Normalizado (%)
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=clases,
        yticklabels=clases,
        ax=ax2,
        cbar=True,
        linewidths=0.5,
        linecolor="#252630",
    )
    ax2.set_title("Matriz de Confusión - Recall Normalizado (%)", color="#E5E7EB", fontsize=14, pad=12, fontweight="bold")
    ax2.set_xlabel("Predicción del Modelo", color="#D1D5DB", fontsize=12)
    ax2.set_ylabel("Etiqueta Real", color="#D1D5DB", fontsize=12)
    ax2.tick_params(colors="#D1D5DB", labelsize=10, rotation=45)

    plt.tight_layout()
    plt.savefig(ruta_salida, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _generar_grafico_curvas_aprendizaje(
    mlp_model: Any, cv_accs: List[float], cv_f1s: List[float], ruta_salida: str
):
    """Genera gráfico con la curva de pérdida de MLP y el rendimiento por fold de CV."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor("#181920")

    # Subplot 1: Curva de Pérdida (Loss Curve)
    ax1.set_facecolor("#22232E")
    if hasattr(mlp_model, "loss_curve_") and mlp_model.loss_curve_:
        losses = mlp_model.loss_curve_
        ax1.plot(range(1, len(losses) + 1), losses, color="#38BDF8", linewidth=2.4, label="Pérdida Entrenamiento (L2 Loss)")
        ax1.scatter([len(losses)], [losses[-1]], color="#F43F5E", s=60, zorder=5, label=f"Pérdida Final: {losses[-1]:.4f}")
        ax1.set_title("Curva de Aprendizaje - Pérdida de la Red Neuronal (MLP)", color="#E5E7EB", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Iteraciones / Épocas", color="#D1D5DB", fontsize=11)
        ax1.set_ylabel("Función de Pérdida (Cross-Entropy)", color="#D1D5DB", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.3, color="#4B5563")
        ax1.legend(facecolor="#2A2B38", edgecolor="#374151", labelcolor="#E5E7EB")
        ax1.tick_params(colors="#D1D5DB")
    else:
        ax1.text(0.5, 0.5, "Curva de pérdida no disponible", color="#9CA3AF", ha="center", va="center")

    # Subplot 2: Rendimiento por Fold (Cross-Validation)
    ax2.set_facecolor("#22232E")
    folds = list(range(1, len(cv_accs) + 1))
    ax2.plot(folds, [a * 100 for a in cv_accs], marker="o", color="#34D399", linewidth=2.2, label="Accuracy (%)")
    ax2.plot(folds, [f * 100 for f in cv_f1s], marker="s", color="#FBBF24", linewidth=2.0, linestyle="--", label="F1-Score (%)")
    ax2.axhline(y=np.mean(cv_accs) * 100, color="#34D399", linestyle=":", alpha=0.7, label=f"Media Acc: {np.mean(cv_accs):.1%}")
    ax2.set_title("Consistencia en Validación Cruzada (5-Fold CV)", color="#E5E7EB", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Fold de Validación", color="#D1D5DB", fontsize=11)
    ax2.set_ylabel("Porcentaje (%)", color="#D1D5DB", fontsize=11)
    ax2.set_xticks(folds)
    ax2.set_ylim(min(cv_accs) * 100 - 6, max(cv_accs) * 100 + 6)
    ax2.grid(True, linestyle="--", alpha=0.3, color="#4B5563")
    ax2.legend(facecolor="#2A2B38", edgecolor="#374151", labelcolor="#E5E7EB")
    ax2.tick_params(colors="#D1D5DB")

    plt.tight_layout()
    plt.savefig(ruta_salida, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _generar_grafico_metricas_clase(
    y_true: np.ndarray, y_pred: np.ndarray, clases: List[str], ruta_salida: str
) -> Dict[str, Any]:
    """Genera gráfico de barras agrupadas comparando Precision, Recall y F1 por seña."""
    report_dict = classification_report(y_true, y_pred, labels=clases, output_dict=True, zero_division=0)

    precisions = [report_dict[c]["precision"] * 100 for c in clases]
    recalls = [report_dict[c]["recall"] * 100 for c in clases]
    f1s = [report_dict[c]["f1-score"] * 100 for c in clases]

    fig, ax = plt.subplots(figsize=(18, 7), dpi=150)
    fig.patch.set_facecolor("#181920")
    ax.set_facecolor("#22232E")

    x = np.arange(len(clases))
    width = 0.26

    rects1 = ax.bar(x - width, precisions, width, label="Precisión", color="#38BDF8")
    rects2 = ax.bar(x, recalls, width, label="Recall (Sensibilidad)", color="#34D399")
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#FBBF24")

    ax.set_title("Métricas de Reconocimiento por Seña (Precisión vs Recall vs F1)", color="#E5E7EB", fontsize=14, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=45, ha="right", color="#D1D5DB", fontsize=11)
    ax.set_ylabel("Porcentaje (%)", color="#D1D5DB", fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#4B5563")
    ax.legend(facecolor="#2A2B38", edgecolor="#374151", labelcolor="#E5E7EB", fontsize=11)
    ax.tick_params(colors="#D1D5DB")

    # Valores sobre barras destacadas (ej: HOLA y GUSTAR)
    for idx_c, c in enumerate(clases):
        if c in ("HOLA", "GUSTAR", "BUENAS", "REPOSO_TRANSICION"):
            ax.annotate(
                f"{recalls[idx_c]:.0f}%",
                (x[idx_c], recalls[idx_c] + 2),
                ha="center",
                va="bottom",
                color="#FFFFFF",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(ruta_salida, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return report_dict


def _generar_grafico_comparativa_historica(historial: List[Dict], ruta_salida: str):
    """Genera gráfico temporal comparativo de la evolución de entrenamientos."""
    if not historial:
        return

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#181920")
    ax.set_facecolor("#22232E")

    runs = list(range(1, len(historial) + 1))
    accs = [h["accuracy_ensamble"] * 100 for h in historial]
    f1s = [h["f1_score_ensamble"] * 100 for h in historial]
    hola_recalls = [h.get("recall_hola", 0.0) * 100 for h in historial]

    ax.plot(runs, accs, marker="o", color="#38BDF8", linewidth=2.5, markersize=8, label="Accuracy Ensamble (%)")
    ax.plot(runs, f1s, marker="s", color="#34D399", linewidth=2.2, linestyle="--", markersize=7, label="F1-Score (%)")
    ax.plot(runs, hola_recalls, marker="^", color="#F43F5E", linewidth=2.0, linestyle=":", markersize=7, label="Recall 'HOLA' (%)")

    # Anotar deltas
    for i in range(len(runs)):
        ax.annotate(f"{accs[i]:.1f}%", (runs[i], accs[i] + 1.2), ha="center", color="#E5E7EB", fontsize=10, fontweight="bold")

    ax.set_title("Evolución Histórica del Rendimiento del Modelo de IA", color="#E5E7EB", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Sesión de Entrenamiento (#)", color="#D1D5DB", fontsize=11)
    ax.set_ylabel("Rendimiento (%)", color="#D1D5DB", fontsize=11)
    ax.set_xticks(runs)
    ax.set_ylim(max(0, min(accs + f1s) - 10), 105)
    ax.grid(True, linestyle="--", alpha=0.3, color="#4B5563")
    ax.legend(facecolor="#2A2B38", edgecolor="#374151", labelcolor="#E5E7EB", fontsize=11)
    ax.tick_params(colors="#D1D5DB")

    plt.tight_layout()
    plt.savefig(ruta_salida, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
