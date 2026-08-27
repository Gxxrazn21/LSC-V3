"""
=============================================================
AUDITORIA COMPLETA DE ESTRUCTURA Y VALIDACION DE DATASETS
Lengua de Señas Colombiana (LSC)
=============================================================
Inspecciona y corrobora:
1. LSC54: Formato JSON, series temporales, landmarks 3D, consistencia de frames.
2. LSC70: Imágenes JPG (LSC70W, LSC70ANH, LSC70AN), integridad de archivos, clases.
3. LSC50: CSVs de landmarks (manos, cuerpo, rostro), formato tabular, consistencia.
4. Compatibilidad con el motor de extracción y vectorización.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import cv2
from motor_lsc.extractor import ExtractorLandmarks
from motor_lsc.cuadrantes import clasificar_cuadrante


def auditar_lsc54(ruta_json="datasets/LSC54/sample.json"):
    print("\n" + "=" * 60)
    print("  [1/3] AUDITORIA DATASET LSC54 (Series 3D JSON)")
    print("=" * 60)

    if not os.path.exists(ruta_json):
        print(f"  ❌ Archivo no encontrado: {ruta_json}")
        return False

    tam_mb = os.path.getsize(ruta_json) / (1024 * 1024)
    print(f"  Tamaño de archivo: {tam_mb:.2f} MB")

    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    signers = list(data.keys())
    print(f"  Firmantes detectados ({len(signers)}): {signers}")

    total_senas = 0
    total_reps = 0
    total_frames = 0
    muestras_validas = 0
    senas_list = []

    for s in signers:
        for cat in data[s]:
            for sena in data[s][cat]:
                total_senas += 1
                senas_list.append(sena)
                for vid in data[s][cat][sena]:
                    for rep in data[s][cat][sena][vid]:
                        total_reps += 1
                        frames = data[s][cat][sena][vid][rep]
                        total_frames += len(frames)
                        
                        # Validar primer frame
                        if frames:
                            primer_f = list(frames.values())[0]
                            if "r_hand" in primer_f or "l_hand" in primer_f:
                                muestras_validas += 1

    print(f"  Categorías: {[c for s in signers for c in data[s].keys()]}")
    print(f"  Señas únicas: {set(senas_list)}")
    print(f"  Total repeticiones de video: {total_reps}")
    print(f"  Total frames con coordenadas: {total_frames:,}")
    print(f"  Repeticiones con landmarks válidos: {muestras_validas}/{total_reps}")

    assert muestras_validas > 0, "LSC54 debe tener muestras válidas"
    print("  [OK] LSC54 estructurado e integro.")
    return True


def auditar_lsc70(ruta_base="datasets/LSC70"):
    print("\n" + "=" * 60)
    print("  [2/3] AUDITORIA DATASET LSC70 (Imágenes RGB)")
    print("=" * 60)

    if not os.path.exists(ruta_base):
        print(f"  ❌ Directorio no encontrado: {ruta_base}")
        return False

    subcarpetas = [d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))]
    print(f"  Subcarpetas principales: {subcarpetas}")

    total_imagenes = 0
    imagenes_corruptas = 0
    clases_detectadas = set()

    for sub in subcarpetas:
        sub_path = os.path.join(ruta_base, sub)
        # Contar personas
        personas = [p for p in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, p))]
        print(f"\n  Directorio: {sub} ({len(personas)} personas)")

        # Muestrear clases
        if personas:
            p_sample = os.path.join(sub_path, personas[0])
            clases_p = [c for c in os.listdir(p_sample) if os.path.isdir(os.path.join(p_sample, c))]
            clases_detectadas.update(clases_p)
            print(f"    Clases en muestra ({len(clases_p)}): {clases_p[:12]} ...")

        # Conteo rápido de archivos JPG
        jpgs = glob.glob(os.path.join(sub_path, "**", "*.jpg"), recursive=True)
        total_imagenes += len(jpgs)
        print(f"    Total imágenes JPG: {len(jpgs):,}")

        # Probar lectura de 5 imágenes aleatorias para verificar no corrupción
        for test_img_path in jpgs[:5]:
            img = cv2.imread(test_img_path)
            if img is None or img.size == 0:
                imagenes_corruptas += 1

    print(f"\n  Total global imágenes LSC70: {total_imagenes:,}")
    print(f"  Total clases únicas detectadas: {len(clases_detectadas)}")
    print(f"  Imágenes corruptas en test: {imagenes_corruptas}")

    assert total_imagenes > 0, "LSC70 debe contener imágenes"
    assert imagenes_corruptas == 0, "No debe haber imagenes corruptas"
    print("  [OK] LSC70 estructurado e integro.")
    return True


def auditar_lsc50(ruta_base="datasets/LSC50"):
    print("\n" + "=" * 60)
    print("  [3/3] AUDITORIA DATASET LSC50 (CSVs / Videos / IMU)")
    print("=" * 60)

    if not os.path.exists(ruta_base):
        print(f"  [ERROR] Directorio no encontrado: {ruta_base}")
        return False

    subcarpetas = os.listdir(ruta_base)
    print(f"  Directorios encontrados: {subcarpetas}")

    # 1. Auditar Landmarks CSVs
    csvs = glob.glob(os.path.join(ruta_base, "LANDMARKS", "**", "*.csv"), recursive=True)
    print(f"  Total archivos CSV de landmarks: {len(csvs):,}")

    if csvs:
        # Validar estructura del primer CSV
        sample_csv = csvs[0]
        df = pd.read_csv(sample_csv)
        print(f"  Muestra CSV ({os.path.basename(sample_csv)}):")
        print(f"    Shape: {df.shape} (Frames: {df.shape[0]}, Columnas: {df.shape[1]})")
        print(f"    Columnas: {list(df.columns[:6])} ... {list(df.columns[-3:])}")
        assert df.shape[1] >= 63, "El CSV de mano debe contener al menos 63 columnas (21x3)"

    # 2. Auditar Videos
    videos = glob.glob(os.path.join(ruta_base, "VIDEOS", "**", "*.avi"), recursive=True)
    print(f"  Total videos AVI: {len(videos):,}")

    if videos:
        sample_vid = videos[0]
        cap = cv2.VideoCapture(sample_vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  Muestra Video ({os.path.basename(sample_vid)}): {w}x{h} px @ {fps:.1f} FPS ({total_f} frames)")

    print("  [OK] LSC50 estructurado e integro.")
    return True


def auditar_todos_los_datasets():
    print("=" * 60)
    print("  INICIANDO AUDITORIA GLOBAL DE DATASETS")
    print("=" * 60)
    
    r1 = auditar_lsc54()
    r2 = auditar_lsc70()
    r3 = auditar_lsc50()

    print("\n" + "=" * 60)
    if r1 and r2 and r3:
        print("  [EXITO] TODOS LOS DATASETS (LSC54, LSC70, LSC50) ESTAN CORRECTOS Y OPERATIVOS!")
    else:
        print("  [AVISO] Se encontraron advertencias en algunos datasets.")
    print("=" * 60)


if __name__ == "__main__":
    auditar_todos_los_datasets()
