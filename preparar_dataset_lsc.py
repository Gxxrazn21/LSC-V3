"""Construye un cache de entrenamiento auditable desde capturas con manifiesto."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida y consolida capturas LSC 109D.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/capturado/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/curated/captures_109d_v1.npz"))
    parser.add_argument("--min-sharpness", type=float, default=45.0)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"No existe el manifiesto: {args.manifest}")

    vectors, labels, signers, sessions, paths = [], [], [], [], []
    rejected = []
    for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            sample = Path(item["sample_path"])
            if not sample.exists():
                raise ValueError("archivo inexistente")
            if item.get("sha256") != file_hash(sample):
                raise ValueError("checksum no coincide")
            vector = np.load(sample, allow_pickle=False).astype(np.float32)
            if vector.shape != (109,) or not np.isfinite(vector).all():
                raise ValueError(f"vector inválido: {vector.shape}")
            if not item.get("consent"):
                raise ValueError("sin consentimiento")
            if float(item.get("sharpness", 0)) < args.min_sharpness:
                raise ValueError("nitidez por debajo del mínimo")
            if not item.get("signer_id") or not item.get("session_id"):
                raise ValueError("falta signer_id o session_id")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            rejected.append({"line": line_number, "reason": str(error)})
            continue
        vectors.append(vector)
        labels.append(str(item["label"]).upper())
        signers.append(str(item["signer_id"]))
        sessions.append(str(item["session_id"]))
        paths.append(sample.as_posix())

    if not vectors:
        raise SystemExit("No hay muestras válidas; revisa el manifiesto y la captura.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.vstack(vectors),
        y=np.asarray(labels),
        signer_id=np.asarray(signers),
        session_id=np.asarray(sessions),
        sample_path=np.asarray(paths),
        schema_version=np.asarray([1]),
    )
    report = args.output.with_suffix(".validation.json")
    report.write_text(json.dumps({
        "accepted": len(vectors), "rejected": rejected,
        "labels": sorted(set(labels)), "signers": sorted(set(signers)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset curado: {args.output} ({len(vectors)} muestras, {len(set(signers))} firmantes)")
    print(f"Reporte de validación: {report}")


if __name__ == "__main__":
    main()
