"""Publica una única fuente web en GitHub Pages y en los assets de Flutter.

Uso:
    python scripts/publish_web_release.py --version 7.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web"
TARGETS = (ROOT / "docs", ROOT / "app_lsc" / "assets" / "web", ROOT / "estilo")
EXCLUDED = {"release-manifest.json"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza el frontend y genera un manifiesto OTA verificable.")
    parser.add_argument("--version", required=True, help="Versión semántica del release, ej. 7.0.0")
    parser.add_argument("--min-shell-version", default="6.2.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SOURCE.is_dir():
        raise SystemExit(f"No existe la fuente canónica: {SOURCE}")

    files = {}
    for source_file in sorted(SOURCE.iterdir()):
        if not source_file.is_file() or source_file.name in EXCLUDED:
            continue
        data = source_file.read_bytes()
        files[source_file.name] = {"sha256": sha256(data), "bytes": len(data)}
    required = {"index.html", "motor_inferencia_local.js", "modelo_ia_cliente.js"}
    missing = required.difference(files)
    if missing:
        raise SystemExit(f"Faltan archivos requeridos en web/: {', '.join(sorted(missing))}")

    manifest = {
        "schema_version": 1,
        "version": args.version,
        "min_shell_version": args.min_shell_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    (SOURCE / "release-manifest.json").write_bytes(payload)
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for source_file in SOURCE.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, target / source_file.name)
    print(f"Release {args.version} publicado desde web/ en {len(TARGETS)} destinos ({len(files)} archivos verificados).")


if __name__ == "__main__":
    main()
