"""Trazabilidad de muestras de LSC.

Cada vector capturado debe poder responder: quién lo produjo (seudónimo), en
qué sesión y dispositivo, bajo qué condiciones y con qué versión de extractor.
El manifiesto JSONL es deliberadamente simple: se puede auditar, versionar y
convertir a Parquet cuando el volumen lo requiera.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_sample_record(
    manifest_path: Path,
    sample_path: Path,
    *,
    label: str,
    signer_id: str,
    session_id: str,
    device_id: str,
    lighting: str,
    camera_index: int,
    sharpness: float,
    extractor_version: str,
    consent: bool,
) -> Dict[str, Any]:
    """Añade un registro autocontenido y devuelve el registro creado."""
    if not consent:
        raise ValueError("No se puede registrar una muestra sin consentimiento.")
    if not signer_id or signer_id.lower() in {"anon", "unknown", "desconocido"}:
        raise ValueError("signer_id debe ser un seudónimo estable, no 'anon'.")
    if not session_id:
        raise ValueError("session_id es obligatorio para separar sesiones.")

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_path": sample_path.as_posix(),
        "sha256": sha256_file(sample_path),
        "label": label,
        "signer_id": signer_id,
        "session_id": session_id,
        "device_id": device_id or "unspecified",
        "lighting": lighting or "unspecified",
        "camera_index": camera_index,
        "sharpness": round(float(sharpness), 3),
        "extractor_version": extractor_version,
        "consent": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        target.flush()
    return record
