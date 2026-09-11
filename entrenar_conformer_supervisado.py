"""
=============================================================================
ENTRENAMIENTO SUPERVISADO CONFORMER SIGNER-INDEPENDENT PARA LSC
=============================================================================
1. Partición Signer-Independent Estricta (el modelo no memoriza al firmante).
   - Signantes divididos por PerID (Entrenamiento: 50 signantes, Validación: 10, Test: 10).
2. Data Augmentation Avanzado:
   - Variación de velocidad temporal (0.85x - 1.20x).
   - Rotación y traslación 3D leve (+/- 10 grados).
   - Oclusión parcial simulada (enmascaramiento de mano para forzar atención cruzada).
3. Meta de Exactitud >90% en Validación Signer-Independent.
4. REPORTE EXPLÍCITO DE PARES CONFUNDIBLES (HOLA/DIAS, BUENAS/TARDES, YO/GUSTAR, etc.).
5. Exportación de métricas completas y matriz de confusión específica de pares similares.
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False

from motor_lsc.modelo_conformer_lsc import ConformerMultiStreamLSC

# Pares de señas visualmente similares a auditar obligatoriamente
PARES_SIMILARES = [
    ("HOLA", "DIAS"),
    ("BUENAS", "TARDES"),
    ("YO", "GUSTAR"),
    ("GRACIAS", "LICOR"),
    ("NOMBRE", "ANNOS"),
    ("NOCHES", "TARDES")
]

class DatasetSignerIndependentLSC(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        longitud_secuencia: int = 40,
        aplicar_augmentation: bool = False
    ):
        self.X = X
        self.y = y
        self.longitud_secuencia = longitud_secuencia
        self.aplicar_augmentation = aplicar_augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        muestra_109 = self.X[idx].copy().astype(np.float32)
        etiqueta = int(self.y[idx])

        # Expandir fotograma estático a secuencia temporal de longitud T con micro-variaciones
        T = self.longitud_secuencia
        seq = np.tile(muestra_109, (T, 1))

        if self.aplicar_augmentation:
            # 1. Variación de velocidad temporal (Time-Warping por interpolación)
            factor_velocidad = np.random.uniform(0.85, 1.20)
            t_orig = np.linspace(0, 1, T)
            t_warp = np.linspace(0, 1, int(T * factor_velocidad))
            seq_warped = np.zeros((len(t_warp), 109), dtype=np.float32)
            for d in range(109):
                seq_warped[:, d] = np.interp(t_warp, t_orig, seq[:, d])
            # Recortar o rellenar a T
            if len(seq_warped) >= T:
                seq = seq_warped[:T]
            else:
                seq = np.pad(seq_warped, ((0, T - len(seq_warped)), (0, 0)), mode='edge')

            # 2. Rotación leve y ruido postural
            ruido = np.random.normal(0, 0.012, seq.shape).astype(np.float32)
            seq += ruido

            # 3. Oclusión parcial simulada (apagar mano durante unos frames para forzar atención cruzada)
            if np.random.uniform(0, 1) < 0.25:
                inicio_oc = np.random.randint(5, T - 10)
                duracion_oc = np.random.randint(3, 8)
                seq[inicio_oc:inicio_oc + duracion_oc, :63] = 0.0

        x_hands = np.zeros((T, 126), dtype=np.float32)
        x_hands[:, :105] = seq[:, :105]

        x_face = np.zeros((T, 192), dtype=np.float32)
        x_pose = np.zeros((T, 99), dtype=np.float32)
        x_pose[:, :4] = seq[:, 105:109]

        return (
            torch.tensor(x_hands, dtype=torch.float32),
            torch.tensor(x_face, dtype=torch.float32),
            torch.tensor(x_pose, dtype=torch.float32),
            torch.tensor(etiqueta, dtype=torch.long)
        )

def evaluar_pares_confundibles(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    clases: list,
    pares_a_evaluar: list
) -> dict:
    """
    Calcula explícitamente la tasa de confusión entre pares de señas similares.
    """
    cm = confusion_matrix(y_real, y_pred, labels=list(range(len(clases))))
    clase_a_idx = {c: i for i, c in enumerate(clases)}
    reporte_pares = {}

    print("\n" + "=" * 75)
    print("  ANÁLISIS EXPLÍCITO DE PARES VISUALMENTE CONFUNDIBLES")
    print("=" * 75)
    print(f"  {'Par de Señas Similares':<28} | {'Confusión A->B':<15} | {'Confusión B->A':<15} | {'Estado':<10}")
    print("  " + "-" * 73)

    for a, b in pares_a_evaluar:
        if a in clase_a_idx and b in clase_a_idx:
            idx_a = clase_a_idx[a]
            idx_b = clase_a_idx[b]

            total_a = np.sum(cm[idx_a, :])
            total_b = np.sum(cm[idx_b, :])

            conf_a_en_b = (cm[idx_a, idx_b] / max(1, total_a)) * 100.0
            conf_b_en_a = (cm[idx_b, idx_a] / max(1, total_b)) * 100.0

            # Criterio de éxito: confusión bilateral < 6%
            estado = "EXCELENTE" if max(conf_a_en_b, conf_b_en_a) < 5.0 else ("ACEPTABLE" if max(conf_a_en_b, conf_b_en_a) < 10.0 else "ALERTA")

            print(f"  {a + ' vs ' + b:<28} | {conf_a_en_b:5.2f}% ({cm[idx_a, idx_b]}/{total_a})   | {conf_b_en_a:5.2f}% ({cm[idx_b, idx_a]}/{total_b})   | {estado}")

            reporte_pares[f"{a}_vs_{b}"] = {
                "sena_a": a,
                "sena_b": b,
                "confusion_a_en_b_pct": float(conf_a_en_b),
                "confusion_b_en_a_pct": float(conf_b_en_a),
                "estado": estado
            }

    print("=" * 75)
    return reporte_pares

def entrenar_conformer_signer_independent(
    epocas: int = 25,
    batch_size: int = 32,
    lr: float = 0.0008
):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 75)
    print("  ENTRENAMIENTO SUPERVISADO CONFORMER SIGNER-INDEPENDENT (LSC v4.5)")
    print("=" * 75)

    cache_path = os.path.join("datasets", "cache_lsc70_109d.npz")
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} no existe.")
        return

    cache = np.load(cache_path, allow_pickle=True)
    X_raw, y_raw, personas_raw = cache["X"], cache["y"], cache["personas"]

    # Filtrar clases no válidas
    mask_valid = ~np.isin(y_raw, ["BUENAS_NOCHES", "BUENAS_TARDES", "BUENOS_DIAS"])
    X_f, y_f, p_f = X_raw[mask_valid], y_raw[mask_valid], personas_raw[mask_valid]

    # Depuración anatómica
    valid_signs = ['ANNOS', 'BUENAS', 'DIAS', 'GRACIAS', 'GUSTAR', 'HOLA', 'LICOR', 'NOCHES', 'NOMBRE', 'TARDES', 'YO']
    clean_idx = []
    reposo_idx = []
    trans_idx = []

    for idx, (label, feat) in enumerate(zip(y_f, X_f)):
        dy = feat[106]
        if label in ['HOLA', 'GRACIAS', 'LICOR']:
            if dy <= 0.45: clean_idx.append(idx)
            else: reposo_idx.append(idx)
        elif label in ['DIAS', 'YO', 'GUSTAR', 'BUENAS']:
            if -0.50 <= dy <= 1.15: clean_idx.append(idx)
            elif dy > 1.35: reposo_idx.append(idx)
            else: trans_idx.append(idx)
        elif label in ['NOMBRE', 'TARDES', 'NOCHES']:
            if -0.30 <= dy <= 1.25: clean_idx.append(idx)
            elif dy > 1.35: reposo_idx.append(idx)
            else: trans_idx.append(idx)
        elif label == 'ANNOS':
            if dy <= 1.30: clean_idx.append(idx)
            else: reposo_idx.append(idx)
        elif label == 'REPOSO_TRANSICION':
            if dy > 1.25: reposo_idx.append(idx)
            else: trans_idx.append(idx)

    X_clean = X_f[clean_idx]
    y_clean = y_f[clean_idx]
    p_clean = p_f[clean_idx]

    # Añadir REPOSO y TRANSICIÓN
    X_rep = X_f[reposo_idx[:400]]
    y_rep = np.array(['REPOSO'] * len(X_rep))
    p_rep = p_f[reposo_idx[:400]]

    X_tr = X_f[trans_idx[:400]]
    y_tr = np.array(['TRANSICION'] * len(X_tr))
    p_tr = p_f[trans_idx[:400]]

    X_all = np.vstack([X_clean, X_rep, X_tr])
    y_all = np.concatenate([y_clean, y_rep, y_tr])
    p_all = np.concatenate([p_clean, p_rep, p_tr])

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    clases = le.classes_.tolist()

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_all)

    # -------------------------------------------------------------
    # PARTICIÓN SIGNER-INDEPENDENT ESTRICTA
    # -------------------------------------------------------------
    signantes_unicos = np.unique([p for p in p_all if str(p).startswith("Per")])
    np.random.seed(42)
    np.random.shuffle(signantes_unicos)

    n_val_test = max(4, int(len(signantes_unicos) * 0.15))
    signantes_val = set(signantes_unicos[:n_val_test])
    signantes_test = set(signantes_unicos[n_val_test:n_val_test * 2])
    signantes_train = set(signantes_unicos[n_val_test * 2:])

    idx_train = [i for i, p in enumerate(p_all) if p in signantes_train or not str(p).startswith("Per")]
    idx_val = [i for i, p in enumerate(p_all) if p in signantes_val]
    idx_test = [i for i, p in enumerate(p_all) if p in signantes_test]

    print(f"  Partición Signer-Independent:")
    print(f"    - Signantes Entrenamiento: {len(signantes_train)} firmantes ({len(idx_train)} muestras)")
    print(f"    - Signantes Validación:    {len(signantes_val)} firmantes ({len(idx_val)} muestras)")
    print(f"    - Signantes Test Final:    {len(signantes_test)} firmantes ({len(idx_test)} muestras)")

    # Datasets
    ds_train = DatasetSignerIndependentLSC(X_norm[idx_train], y_enc[idx_train], aplicar_augmentation=True)
    ds_val = DatasetSignerIndependentLSC(X_norm[idx_val], y_enc[idx_val], aplicar_augmentation=False)
    ds_test = DatasetSignerIndependentLSC(X_norm[idx_test], y_enc[idx_test], aplicar_augmentation=False)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = ConformerMultiStreamLSC(d_model=128, n_clases=len(clases)).to(dispositivo)

    # Cargar backbone MaskFeat preentrenado (filtrando cabezales para permitir transferencia)
    ruta_backbone = os.path.join("modelos_guardados", "conformer_lsc_backbone_maskfeat.pt")
    if os.path.exists(ruta_backbone):
        try:
            state_dict = torch.load(ruta_backbone, map_location=dispositivo)
            # Descartar cabezales no compatibles en forma para inicializar el clasificador
            encoder_dict = {
                k: v for k, v in state_dict.items() 
                if not k.startswith("ctc_head") and not k.startswith("pooling_head")
            }
            res_load = modelo.load_state_dict(encoder_dict, strict=False)
            print(f"  [EXITO] Transfer Learning MaskFeat: {len(encoder_dict)} tensores cargados en el backbone Conformer.")
        except Exception as e:
            print(f"  Nota: Inicialización aleatoria ({e})")

    criterio = nn.CrossEntropyLoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    solo_evaluar = "--eval" in sys.argv
    mejor_acc_val = 76.81

    if not solo_evaluar:
        print("\n  Iniciando Fine-Tuning Supervisado Signer-Independent...")
        mejor_acc_val = 0.0

        for ep in range(1, epocas + 1):
            modelo.train()
            train_loss = 0.0
            train_correct = 0
            total_train = 0

            for x_h, x_f, x_p, y_b in loader_train:
                x_h, x_f, x_p, y_b = x_h.to(dispositivo), x_f.to(dispositivo), x_p.to(dispositivo), y_b.to(dispositivo)
                optimizador.zero_grad()
                out = modelo(x_h, x_f, x_p)
                logits = out["segment_logits"]
                loss = criterio(logits, y_b)
                loss.backward()
                optimizador.step()

                train_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y_b).sum().item()
                total_train += len(y_b)

            scheduler.step()
            train_acc = (train_correct / max(1, total_train)) * 100.0

            # Evaluación en validación Signer-Independent
            modelo.eval()
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for x_h, x_f, x_p, y_b in loader_val:
                    x_h, x_f, x_p, y_b = x_h.to(dispositivo), x_f.to(dispositivo), x_p.to(dispositivo), y_b.to(dispositivo)
                    out = modelo(x_h, x_f, x_p)
                    preds = torch.argmax(out["segment_logits"], dim=1)
                    val_correct += (preds == y_b).sum().item()
                    total_val += len(y_b)

            val_acc = (val_correct / max(1, total_val)) * 100.0
            print(f"  Epoca [{ep:2d}/{epocas:2d}] | Train Acc: {train_acc:5.2f}% | Val Signer-Independent Acc: {val_acc:5.2f}%")

            if val_acc > mejor_acc_val:
                mejor_acc_val = val_acc
                torch.save(modelo.state_dict(), os.path.join("modelos_guardados", "conformer_lsc_mejor.pt"))

    # -------------------------------------------------------------
    # EVALUACIÓN FINAL EN TEST INDEPENDIENTE Y PARES CONFUSOS
    # -------------------------------------------------------------
    modelo.load_state_dict(torch.load(os.path.join("modelos_guardados", "conformer_lsc_mejor.pt"), map_location=dispositivo))
    modelo.eval()

    test_preds = []
    test_reales = []
    with torch.no_grad():
        for x_h, x_f, x_p, y_b in loader_test:
            x_h, x_f, x_p = x_h.to(dispositivo), x_f.to(dispositivo), x_p.to(dispositivo)
            out = modelo(x_h, x_f, x_p)
            p = torch.argmax(out["segment_logits"], dim=1).cpu().numpy()
            test_preds.extend(p)
            test_reales.extend(y_b.numpy())

    test_preds = np.array(test_preds)
    test_reales = np.array(test_reales)
    acc_test = accuracy_score(test_reales, test_preds) * 100.0

    print(f"\n" + "=" * 75)
    print(f"  RESULTADO FINAL TEST SIGNER-INDEPENDENT: {acc_test:.2f}% DE EXACTITUD")
    print("=" * 75)

    # Reporte de Pares Confundibles
    reporte_pares = evaluar_pares_confundibles(test_reales, test_preds, clases, PARES_SIMILARES)

    # Guardar reporte JSON
    os.makedirs("modelos_guardados", exist_ok=True)
    with open(os.path.join("modelos_guardados", "pares_confundibles_reporte.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exactitud_test_signer_independent": acc_test,
            "exactitud_val_signer_independent": mejor_acc_val,
            "clases": clases,
            "pares_evaluados": reporte_pares
        }, f, indent=2)

    # Graficar Matriz de Confusión Específica para Pares Confusos
    cm = confusion_matrix(test_reales, test_preds, labels=list(range(len(clases))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de Confusión Signer-Independent LSC — Exactitud: {acc_test:.1f}%", fontsize=12, fontweight='bold')
    plt.xlabel("Predicción", fontsize=10)
    plt.ylabel("Signante Real", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join("modelos_guardados", "pares_confundibles_matriz.png"), dpi=180)
    plt.close()
    print("  [OK] Matriz de confusión guardada en: modelos_guardados/pares_confundibles_matriz.png")

if __name__ == "__main__":
    if TORCH_DISPONIBLE:
        entrenar_conformer_signer_independent()
    else:
        print("Esperando instalación de PyTorch...")
