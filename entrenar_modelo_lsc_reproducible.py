"""Entrenamiento LSC reproducible con evaluación separada por firmante.

No sobrescribe los artefactos de producción v6.4. Los modelos v7 se publican
solo después de revisar sus métricas signer-independent y sus pruebas móviles.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEED = 20260922
ARCHITECTURE = (640, 384, 192)


def augment_train(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, target_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Balancea solo el conjunto de entrenamiento; nunca toca test/validación."""
    blocks, labels = [], []
    for label in np.unique(y):
        source = X[y == label]
        count = max(len(source), target_per_class)
        choices = rng.choice(len(source), size=count, replace=count > len(source))
        augmented = source[choices].copy()
        # El vector 109D se mantiene en el mismo contrato del extractor actual.
        augmented[:, :105] += rng.normal(0.0, 0.003, size=(count, 105))
        augmented[:, 105:109] += rng.normal(0.0, 0.015, size=(count, 4))
        blocks.append(augmented)
        labels.append(np.full(count, label))
    return np.vstack(blocks), np.concatenate(labels)


def export_model(model, scaler, classes: list[str], output_dir: Path, version: str, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_npz = output_dir / f"modelo_lsc_{version}.npz"
    model_joblib = output_dir / f"modelo_lsc_{version}.joblib"
    model_json = output_dir / f"modelo_lsc_{version}.json"
    weights = model.coefs_
    biases = model.intercepts_
    payload = {
        "schema_version": 1,
        "version": version,
        "input_dimensions": 109,
        "classes": classes,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": [weight.tolist() for weight in weights],
        "biases": [bias.tolist() for bias in biases],
        "layers": [109, *ARCHITECTURE, len(classes)],
        "evaluation": metrics,
    }
    np.savez_compressed(model_npz, classes=np.asarray(classes), scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, **{
        f"w{i}": weight for i, weight in enumerate(weights)
    }, **{f"b{i}": bias for i, bias in enumerate(biases)})
    joblib.dump({"mlp": model, "scaler": scaler, "classes": classes, "version": version}, model_joblib)
    model_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Artefactos v{version}: {model_npz.name}, {model_joblib.name}, {model_json.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena MLP LSC sin fuga de evaluación.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/curated/captures_109d_v1.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("modelos_guardados"))
    parser.add_argument("--version", default="7.0.0-candidate")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-per-class", type=int, default=300)
    parser.add_argument("--allow-legacy-sample-split", action="store_true", help="Solo diagnóstico: no afirma generalización por firmante.")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"No existe {args.dataset}. Primero ejecuta preparar_dataset_lsc.py.")
    data = np.load(args.dataset, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    if X.ndim != 2 or X.shape[1] != 109 or len(X) != len(y) or not np.isfinite(X).all():
        raise SystemExit("El dataset debe contener X finito de forma (n, 109) y y alineado.")

    has_groups = "signer_id" in data.files and len(set(data["signer_id"].astype(str))) >= args.folds
    if not has_groups and not args.allow_legacy_sample_split:
        raise SystemExit("Falta signer_id suficiente. No se publican métricas signer-independent sin manifiesto.")
    groups = data["signer_id"].astype(str) if has_groups else np.arange(len(y)).astype(str)
    protocol = "stratified_group_by_signer" if has_groups else "stratified_sample_legacy"
    splitter = (StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED) if has_groups
                else StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED))
    labels = LabelEncoder().fit_transform(y)
    classes = sorted(np.unique(y).tolist())
    fold_metrics, oof_truth, oof_pred = [], [], []

    splits = splitter.split(X, labels, groups) if has_groups else splitter.split(X, labels)
    for fold, (train_index, test_index) in enumerate(splits, 1):
        rng = np.random.default_rng(SEED + fold)
        X_train, y_train = augment_train(X[train_index], y[train_index], rng, args.target_per_class)
        encoder = LabelEncoder().fit(classes)
        y_train_encoded = encoder.transform(y_train)
        y_test_encoded = encoder.transform(y[test_index])
        scaler = StandardScaler().fit(X_train)
        model = MLPClassifier(hidden_layer_sizes=ARCHITECTURE, activation="relu", alpha=0.00012,
                              learning_rate_init=0.0008, max_iter=800, early_stopping=True,
                              validation_fraction=0.10, n_iter_no_change=30, random_state=SEED + fold)
        model.fit(scaler.transform(X_train), y_train_encoded)
        prediction = model.predict(scaler.transform(X[test_index]))
        oof_truth.extend(y_test_encoded.tolist())
        oof_pred.extend(prediction.tolist())
        fold_metrics.append({"fold": fold, "accuracy": accuracy_score(y_test_encoded, prediction),
                             "macro_f1": f1_score(y_test_encoded, prediction, average="macro"),
                             "test_signers": sorted(set(groups[test_index].tolist()))})

    report = classification_report(oof_truth, oof_pred, target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "version": args.version,
        "protocol": protocol, "dataset": str(args.dataset), "seed": SEED, "folds": fold_metrics,
        "oof_accuracy": accuracy_score(oof_truth, oof_pred), "oof_macro_f1": f1_score(oof_truth, oof_pred, average="macro"),
        "per_class": report, "class_count": len(classes), "sample_count": int(len(X)),
        "signer_count": int(len(set(groups))),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"metricas_lsc_{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / f"matriz_confusion_lsc_{args.version}.csv", confusion_matrix(oof_truth, oof_pred), delimiter=",", fmt="%d")

    # El modelo final se ajusta después de medir; la publicación OTA sigue siendo una decisión explícita.
    final_rng = np.random.default_rng(SEED)
    X_final, y_final = augment_train(X, y, final_rng, args.target_per_class)
    final_encoder = LabelEncoder().fit(classes)
    final_scaler = StandardScaler().fit(X_final)
    final_model = MLPClassifier(hidden_layer_sizes=ARCHITECTURE, activation="relu", alpha=0.00012,
                                learning_rate_init=0.0008, max_iter=800, early_stopping=True,
                                validation_fraction=0.10, n_iter_no_change=30, random_state=SEED)
    final_model.fit(final_scaler.transform(X_final), final_encoder.transform(y_final))
    export_model(final_model, final_scaler, classes, args.output_dir, args.version, metrics)
    print(f"OOF accuracy={metrics['oof_accuracy']:.4f} macro-F1={metrics['oof_macro_f1']:.4f} ({protocol})")


if __name__ == "__main__":
    main()
