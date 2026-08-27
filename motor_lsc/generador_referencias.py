"""
=============================================================
GENERADOR DE BASE DE DATOS VECTORIAL ARTICULAR LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Compila los datasets locales con descriptores articulares 3D de 101 dimensiones:
- LSC70ANH: Alfabeto dactilológico (A-Z) y Números (1-10, MIL, MILLON)
- LSC70W: Palabras del vocabulario LSC (HOLA, YO, NOMBRE, LICOR, GUSTAR, etc.)
- LSC70AN: Variaciones de alfabeto y números
- LSC54: Series temporales 3D (APOYAR, AYUDAR, BAÑO, BIEN, BIENVENIDO)

Genera 'modelos_guardados/base_senas_lsc.npz' y 'clases_lsc.json'.
"""

import os
import json
from typing import Optional, List, Dict
import numpy as np
import cv2

from .base_vectores import BaseVectoresLSC
from .extractor import ExtractorLandmarks
from .cuadrantes import clasificar_cuadrante, CuadranteEspacial


MAPA_CUADRANTES_SEÑAS: Dict[str, str] = {
    # Rostro / Cabeza
    "HOLA": CuadranteEspacial.CABEZA_ROSTRO.value,
    "PENSAR": CuadranteEspacial.CABEZA_ROSTRO.value,
    "SABER": CuadranteEspacial.CABEZA_ROSTRO.value,

    # Cuello / Garganta
    "LICOR": CuadranteEspacial.CUELLO_GARGANTA.value,
    "AGUA": CuadranteEspacial.CUELLO_GARGANTA.value,
    "SED": CuadranteEspacial.CUELLO_GARGANTA.value,

    # Pecho / Torso
    "YO": CuadranteEspacial.PECHO_TORSO.value,
    "NOMBRE": CuadranteEspacial.PECHO_TORSO.value,
    "GUSTAR": CuadranteEspacial.PECHO_TORSO.value,
    "SENTIR": CuadranteEspacial.PECHO_TORSO.value,
    "AMIGO": CuadranteEspacial.PECHO_TORSO.value,

    # Espacio Central Frente al Cuerpo (a dos manos o neutro)
    "APOYAR": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "AYUDAR": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "BAÑO": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "BIEN": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "BIENVENIDO": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "BUENAS": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "DIAS": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "TARDES": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "NOCHES": CuadranteEspacial.ESPACIO_CENTRAL.value,
    "CASA": CuadranteEspacial.ESPACIO_CENTRAL.value,
}


def compilar_desde_lsc54(
    json_path: str,
    base_vectores: BaseVectoresLSC,
    max_reps_por_sena: int = 8,
) -> int:
    """
    Extrae descriptores articulares 3D de LSC54 (sample.json).
    """
    if not os.path.exists(json_path):
        print(f"  [LSC54] Archivo no encontrado: {json_path}")
        return 0

    print(f"  [LSC54] Procesando: {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_agregadas = 0

    for signer, categorias in data.items():
        for cat_nombre, senas in categorias.items():
            for sena_nombre, vids in senas.items():
                sena_clean = sena_nombre.upper().strip()
                count_sena = 0
                cuad_sena = MAPA_CUADRANTES_SEÑAS.get(sena_clean, CuadranteEspacial.ESPACIO_CENTRAL.value)

                for vid_id, reps in vids.items():
                    for rep_id, frames in reps.items():
                        if count_sena >= max_reps_por_sena:
                            break

                        frame_keys = sorted(
                            list(frames.keys()),
                            key=lambda k: int(k.split("_")[-1]) if "_" in k else 0
                        )

                        secuencia_vectores = []

                        for fk in frame_keys:
                            f_data = frames[fk]
                            h_data = f_data.get("r_hand") or f_data.get("l_hand")
                            if not h_data or "x" not in h_data or len(h_data["x"]) < 21:
                                continue

                            coords = np.zeros((21, 3), dtype=np.float32)
                            for idx in range(21):
                                coords[idx, 0] = h_data["x"][idx]
                                coords[idx, 1] = h_data["y"][idx]
                                coords[idx, 2] = h_data["z"][idx] if "z" in h_data else 0.0

                            vec_articular, _ = ExtractorLandmarks.extraer_descriptor_articular(coords)
                            secuencia_vectores.append(vec_articular)

                        if len(secuencia_vectores) > 4:
                            idx_medio = len(secuencia_vectores) // 2
                            vec_representativo = secuencia_vectores[idx_medio]
                            
                            base_vectores.agregar_referencia_estatica(
                                vector=vec_representativo,
                                etiqueta=sena_clean,
                                cuadrante=cuad_sena,
                                categoria=f"LSC54_{cat_nombre}",
                            )

                            base_vectores.agregar_referencia_dinamica(
                                secuencia=np.array(secuencia_vectores),
                                etiqueta=sena_clean,
                                cuadrante=cuad_sena,
                                categoria=f"LSC54_{cat_nombre}",
                            )

                            count_sena += 1
                            total_agregadas += 1

    print(f"  [LSC54] Se agregaron {total_agregadas} muestras articulares de referencia.")
    return total_agregadas


def compilar_carpeta_lsc70(
    directorio_base: str,
    base_vectores: BaseVectoresLSC,
    extractor: ExtractorLandmarks,
    max_personas: int = 20,
    muestras_por_persona: int = 5,
    categoria_nombre: str = "LSC70",
) -> int:
    """
    Escanea carpetas de sujetos (Per01..Per20) y extrae descriptores articulares 3D.
    """
    if not os.path.exists(directorio_base):
        return 0

    print(f"  [LSC70] Escaneando {categoria_nombre} en: {directorio_base} ({max_personas} personas) ...")
    total_agregadas = 0

    personas = sorted([
        f.path for f in os.scandir(directorio_base) if f.is_dir() and f.name.startswith("Per")
    ])[:max_personas]

    for p_dir in personas:
        for c_entry in os.scandir(p_dir):
            if not c_entry.is_dir():
                continue
            nombre_clase = c_entry.name.upper().strip()
            cuad_clase = MAPA_CUADRANTES_SEÑAS.get(nombre_clase, CuadranteEspacial.ESPACIO_LATERAL.value)

            imgs = [f.path for f in os.scandir(c_entry.path) if f.name.lower().endswith((".jpg", ".png"))]
            imgs = imgs[:muestras_por_persona]

            for ruta_img in imgs:
                img = cv2.imread(ruta_img)
                if img is None:
                    continue

                res = extractor.procesar_frame(img)
                if res["hay_manos"]:
                    mano = res["manos"][0]
                    base_vectores.agregar_referencia_estatica(
                        vector=mano["vector_normalizado"],
                        etiqueta=nombre_clase,
                        cuadrante=cuad_clase,
                        categoria=categoria_nombre,
                    )
                    total_agregadas += 1

    print(f"  [{categoria_nombre}] Se agregaron {total_agregadas} vectores de referencia.")
    return total_agregadas


def generar_base_completa(
    dir_datasets: str = "datasets",
    salida_npz: str = "modelos_guardados/base_senas_lsc.npz",
    salida_json: str = "modelos_guardados/clases_lsc.json",
    max_personas: int = 20,
) -> BaseVectoresLSC:
    """
    Construye el catálogo completo compilando LSC54, LSC70W, LSC70ANH y LSC70AN.
    """
    print("=" * 60)
    print("  COMPILANDO BASE DE DATOS ARTICULAR LSC (101 DIMENSIONES)")
    print("=" * 60)

    base = BaseVectoresLSC()
    extractor = ExtractorLandmarks()

    # 1. LSC54 (Series 3D)
    ruta_lsc54 = os.path.join(dir_datasets, "LSC54", "sample.json")
    compilar_desde_lsc54(ruta_lsc54, base, max_reps_por_sena=10)

    # 2. LSC70W (Palabras: HOLA, YO, NOMBRE, LICOR, GUSTAR, etc.)
    ruta_lsc70w = os.path.join(dir_datasets, "LSC70", "LSC70W")
    compilar_carpeta_lsc70(ruta_lsc70w, base, extractor, max_personas=max_personas, muestras_por_persona=5, categoria_nombre="LSC70W")

    # 3. LSC70ANH (Alfabeto A-Z y Números 1-10, MIL, MILLON)
    ruta_lsc70anh = os.path.join(dir_datasets, "LSC70", "LSC70ANH")
    compilar_carpeta_lsc70(ruta_lsc70anh, base, extractor, max_personas=max_personas, muestras_por_persona=4, categoria_nombre="LSC70ANH")

    # 4. LSC70AN (Variaciones adicionales)
    ruta_lsc70an = os.path.join(dir_datasets, "LSC70", "LSC70AN")
    if os.path.exists(ruta_lsc70an):
        compilar_carpeta_lsc70(ruta_lsc70an, base, extractor, max_personas=10, muestras_por_persona=3, categoria_nombre="LSC70AN")

    # Guardar en disco
    base.guardar(salida_npz, salida_json)
    extractor.liberar()

    print("\n" + "=" * 60)
    print(f"  [EXITO] Base de datos guardada en: {salida_npz}")
    print(f"  Total vectores de referencia: {base.total_senas}")
    print(f"  Total clases únicas: {len(base.clases_unicas)}")
    print(f"  Clases: {base.clases_unicas}")
    print("=" * 60)

    return base


if __name__ == "__main__":
    generar_base_completa()
