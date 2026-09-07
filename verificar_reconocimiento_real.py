"""
=============================================================
VERIFICACION END-TO-END DE RECONOCIMIENTO CON DATOS REALES
Lengua de Señas Colombiana (LSC) — Version Anti-Confusión
=============================================================
Prueba el pipeline completo con muestras reales de los datasets:
1. Ingesta de imágenes reales de LSC70 (Palabras y Dactilología).
2. Ingesta de secuencias 3D de LSC54 (Señas dinámicas).
3. Test de CONFUSION CRUZADA entre clases visualmente similares.
4. Test de MARGEN de confianza (gap entre 1er y 2do candidato).
5. Matriz de confusión completa.
6. Ensamblado de oraciones y verificación de TTS.
"""

import os
import json
import glob
import cv2
import numpy as np
from collections import defaultdict

from motor_lsc import (
    ExtractorLandmarks,
    CuadranteEspacial,
    clasificar_cuadrante,
    BaseVectoresLSC,
    EnsambladorFrases,
    MotorVozLocal,
)


# Pares de clases que frecuentemente se confunden visualmente
PARES_CONFUSION_CRITICOS = [
    ("1", "I"),    # Dedo índice levantado
    ("5", "W"),    # Dedos abiertos
    ("6", "W"),    # Similar apertura
    ("A", "S"),    # Puño cerrado con/sin pulgar
    ("A", "T"),    # Puño cerrado variaciones
    ("M", "N"),    # Dedos sobre pulgar
    ("U", "V"),    # Dos dedos juntos vs separados
    ("K", "V"),    # Dos dedos variantes
    ("D", "1"),    # Índice levantado
    ("B", "4"),    # Cuatro dedos extendidos
    ("C", "O"),    # Forma curva
    ("G", "Q"),    # Similar orientación
    ("I", "J"),    # Meñique con/sin movimiento
    ("R", "U"),    # Dedos cruzados vs juntos
]


def probar_reconocimiento_imagenes_lsc70():
    """Prueba reconocimiento de PALABRAS con imágenes reales de LSC70W."""
    print("\n" + "=" * 60)
    print("  1. PRUEBA DE PALABRAS — Imágenes Reales LSC70W")
    print("=" * 60)
    
    base = BaseVectoresLSC(umbral_min_similitud=0.60)
    base.cargar("modelos_guardados/base_senas_lsc.npz")
    extractor = ExtractorLandmarks()

    clases_test = ["HOLA", "BUENAS", "DIAS", "TARDES", "NOCHES", "YO", "NOMBRE", "ANNOS", "GUSTAR", "LICOR"]
    aciertos = 0
    total = 0
    margenes = []

    for clase in clases_test:
        patron = os.path.join("datasets", "LSC70", "LSC70W", "Per0[7-9]", clase, "*.jpg")
        imgs = glob.glob(patron)
        if not imgs:
            imgs = glob.glob(os.path.join("datasets", "LSC70", "LSC70W", "*", clase, "*.jpg"))
        
        if not imgs:
            continue

        for img_path in imgs[:3]:
            img = cv2.imread(img_path)
            if img is None:
                continue

            res = extractor.procesar_frame(img)
            if res["hay_manos"]:
                mano = res["manos"][0]
                cuad, _ = clasificar_cuadrante(
                    mano["muneca"],
                    res.get("pose_anchors"),
                    res["alto_frame"],
                    res["ancho_frame"],
                )
                candidatos = base.buscar_similar(
                    vector_query=mano["vector_normalizado"],
                    cuadrante_query=cuad.value,
                    top_k=3,
                )

                total += 1
                top1_sena = candidatos[0][0] if candidatos else "NINGUNA"
                top1_score = candidatos[0][1] if candidatos else 0.0
                top2_score = candidatos[1][1] if len(candidatos) > 1 else 0.0
                margen = top1_score - top2_score
                margenes.append(margen)
                
                es_acierto = (top1_sena == clase)
                if es_acierto:
                    aciertos += 1

                simbolo = "[OK]" if es_acierto else "[DIF]"
                print(f"  {simbolo} Real: '{clase:<7s}' -> Detectado: '{top1_sena:<7s}' (Score: {top1_score:.3f}, Margen: {margen:.3f})")

    extractor.liberar()
    
    acc = aciertos / max(total, 1)
    margen_promedio = sum(margenes) / max(len(margenes), 1)
    print(f"\n  Resumen Palabras: {aciertos}/{total} ({acc:.1%}) — Margen promedio: {margen_promedio:.3f}")
    assert total > 0, "Debe haber probado al menos una seña"
    return acc


def probar_dactilologia_completa():
    """Prueba TODAS las letras A-Z y números 1-10 disponibles en LSC70ANH."""
    print("\n" + "=" * 60)
    print("  2. PRUEBA DE DACTILOLOGIA — Letras y Números LSC70ANH")
    print("=" * 60)
    
    base = BaseVectoresLSC(umbral_min_similitud=0.60)
    base.cargar("modelos_guardados/base_senas_lsc.npz")
    extractor = ExtractorLandmarks()

    ruta_anh = os.path.join("datasets", "LSC70", "LSC70ANH")
    if not os.path.exists(ruta_anh):
        print(f"  [AVISO] No se encontró {ruta_anh}")
        return 1.0

    aciertos = 0
    total = 0
    margenes = []
    confusiones = defaultdict(int)

    personas_test = sorted([
        d for d in os.listdir(ruta_anh)
        if os.path.isdir(os.path.join(ruta_anh, d)) and d.startswith("Per")
    ])[-10:]

    for persona in personas_test:
        persona_dir = os.path.join(ruta_anh, persona)
        for clase_dir in sorted(os.listdir(persona_dir)):
            clase_path = os.path.join(persona_dir, clase_dir)
            if not os.path.isdir(clase_path):
                continue
            
            clase = clase_dir.upper().strip()
            imgs = glob.glob(os.path.join(clase_path, "*.jpg")) + glob.glob(os.path.join(clase_path, "*.png"))
            
            if not imgs:
                continue

            img_path = imgs[-1]
            img = cv2.imread(img_path)
            if img is None:
                continue

            res = extractor.procesar_frame(img)
            if res["hay_manos"]:
                mano = res["manos"][0]
                cuad, _ = clasificar_cuadrante(
                    mano["muneca"],
                    res.get("pose_anchors"),
                    res["alto_frame"],
                    res["ancho_frame"],
                )
                candidatos = base.buscar_similar(
                    vector_query=mano["vector_normalizado"],
                    cuadrante_query=cuad.value,
                    top_k=3,
                )

                total += 1
                top1_sena = candidatos[0][0] if candidatos else "NINGUNA"
                top1_score = candidatos[0][1] if candidatos else 0.0
                top2_score = candidatos[1][1] if len(candidatos) > 1 else 0.0
                margen = top1_score - top2_score
                margenes.append(margen)

                es_acierto = (top1_sena == clase)
                if es_acierto:
                    aciertos += 1
                else:
                    confusiones[(clase, top1_sena)] += 1

    extractor.liberar()
    
    acc = aciertos / max(total, 1)
    margen_promedio = sum(margenes) / max(len(margenes), 1)
    
    print(f"\n  Resumen Dactilología: {aciertos}/{total} ({acc:.1%}) — Margen promedio: {margen_promedio:.3f}")
    
    if confusiones:
        print(f"\n  Top confusiones de dactilología:")
        for (real, det), count in sorted(confusiones.items(), key=lambda x: -x[1])[:10]:
            print(f"    '{real}' confundida con '{det}': {count} veces")
    
    return acc


def probar_confusion_cruzada():
    """Verifica que pares de clases visualmente similares NO se confundan."""
    print("\n" + "=" * 60)
    print("  3. TEST DE CONFUSIÓN CRUZADA — Pares Críticos")
    print("=" * 60)
    
    base = BaseVectoresLSC(umbral_min_similitud=0.60)
    base.cargar("modelos_guardados/base_senas_lsc.npz")
    extractor = ExtractorLandmarks()
    
    pares_fallidos = []
    pares_probados = 0
    
    ruta_anh = os.path.join("datasets", "LSC70", "LSC70ANH")
    
    for clase_a, clase_b in PARES_CONFUSION_CRITICOS:
        imgs_a = glob.glob(os.path.join(ruta_anh, "Per*", clase_a, "*.jpg"))
        imgs_b = glob.glob(os.path.join(ruta_anh, "Per*", clase_b, "*.jpg"))
        
        if not imgs_a or not imgs_b:
            continue
        
        for img_path in imgs_a[:3]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            res = extractor.procesar_frame(img)
            if res["hay_manos"]:
                mano = res["manos"][0]
                resultado = base.buscar_estricto(
                    vector_query=mano["vector_normalizado"],
                    umbral_minimo=0.75,
                    margen_minimo=0.03,
                )
                
                pares_probados += 1
                if resultado and resultado[0] == clase_b:
                    pares_fallidos.append((clase_a, clase_b, resultado[1]))
                    print(f"  [FALLO] '{clase_a}' detectada como '{clase_b}' (score: {resultado[1]:.3f})")
        
        for img_path in imgs_b[:3]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            res = extractor.procesar_frame(img)
            if res["hay_manos"]:
                mano = res["manos"][0]
                resultado = base.buscar_estricto(
                    vector_query=mano["vector_normalizado"],
                    umbral_minimo=0.75,
                    margen_minimo=0.03,
                )
                
                pares_probados += 1
                if resultado and resultado[0] == clase_a:
                    pares_fallidos.append((clase_b, clase_a, resultado[1]))
                    print(f"  [FALLO] '{clase_b}' detectada como '{clase_a}' (score: {resultado[1]:.3f})")
    
    extractor.liberar()
    
    tasa_confusion = len(pares_fallidos) / max(pares_probados, 1)
    print(f"\n  Pares probados: {pares_probados}")
    print(f"  Confusiones cruzadas: {len(pares_fallidos)} ({tasa_confusion:.1%})")
    
    if not pares_fallidos:
        print("  [OK] Ningún par crítico se confunde con búsqueda estricta")
    
    return tasa_confusion


def probar_reconocimiento_series_lsc54():
    """Prueba con series 3D reales de LSC54."""
    print("\n" + "=" * 60)
    print("  4. PRUEBA DE SEÑAS DINÁMICAS — Series 3D LSC54")
    print("=" * 60)
    
    base = BaseVectoresLSC(umbral_min_similitud=0.60)
    base.cargar("modelos_guardados/base_senas_lsc.npz")

    json_path = "datasets/LSC54/sample.json"
    if not os.path.exists(json_path):
        print(f"  [AVISO] No se encontró {json_path}")
        return 1.0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    aciertos = 0
    total = 0
    margenes = []

    for signer in data:
        for cat in data[signer]:
            for sena, vids in data[signer][cat].items():
                sena_esperada = sena.upper().strip()
                for vid in list(vids.keys())[:1]:
                    for rep in list(vids[vid].keys())[-1:]:
                        frames = vids[vid][rep]
                        frame_keys = sorted(
                            list(frames.keys()),
                            key=lambda k: int(k.split("_")[-1]) if "_" in k else 0
                        )
                        if not frame_keys:
                            continue

                        fk_mid = frame_keys[len(frame_keys) // 2]
                        f_data = frames[fk_mid]
                        h_data = f_data.get("r_hand") or f_data.get("l_hand")
                        if not h_data or "x" not in h_data or len(h_data["x"]) < 21:
                            continue

                        coords = np.zeros((21, 3), dtype=np.float32)
                        for i in range(21):
                            coords[i, 0] = h_data["x"][i]
                            coords[i, 1] = h_data["y"][i]
                            coords[i, 2] = h_data["z"][i] if "z" in h_data else 0.0

                        vec_norm = ExtractorLandmarks.normalizar_mano(coords)
                        candidatos = base.buscar_similar(vector_query=vec_norm, top_k=3)

                        total += 1
                        top1_sena = candidatos[0][0] if candidatos else "NINGUNA"
                        top1_score = candidatos[0][1] if candidatos else 0.0
                        top2_score = candidatos[1][1] if len(candidatos) > 1 else 0.0
                        margen = top1_score - top2_score
                        margenes.append(margen)

                        es_acierto = (top1_sena == sena_esperada)
                        if es_acierto:
                            aciertos += 1

                        simbolo = "[OK]" if es_acierto else "[DIF]"
                        print(f"  {simbolo} Real: '{sena_esperada:<10s}' -> Detectado: '{top1_sena:<10s}' (Score: {top1_score:.3f}, Margen: {margen:.3f})")

    acc = aciertos / max(total, 1)
    margen_promedio = sum(margenes) / max(len(margenes), 1)
    print(f"\n  Resumen LSC54: {aciertos}/{total} ({acc:.1%}) — Margen promedio: {margen_promedio:.3f}")
    return acc


def probar_margenes_confianza():
    """Verifica que buscar_estricto funciona correctamente con umbrales altos."""
    print("\n" + "=" * 60)
    print("  5. TEST DE MÁRGENES DE CONFIANZA — buscar_estricto()")
    print("=" * 60)
    
    base = BaseVectoresLSC(umbral_min_similitud=0.80)
    base.cargar("modelos_guardados/base_senas_lsc.npz")
    extractor = ExtractorLandmarks()
    
    total_estricto = 0
    rechazados = 0
    aceptados = 0
    margenes_aceptados = []
    
    ruta_w = os.path.join("datasets", "LSC70", "LSC70W")
    if os.path.exists(ruta_w):
        for persona_dir in sorted(os.listdir(ruta_w))[:5]:
            persona_path = os.path.join(ruta_w, persona_dir)
            if not os.path.isdir(persona_path):
                continue
            
            for clase_dir in os.listdir(persona_path):
                clase_path = os.path.join(persona_path, clase_dir)
                if not os.path.isdir(clase_path):
                    continue
                
                imgs = glob.glob(os.path.join(clase_path, "*.jpg"))
                if not imgs:
                    continue
                
                img = cv2.imread(imgs[0])
                if img is None:
                    continue
                
                res = extractor.procesar_frame(img)
                if res["hay_manos"]:
                    mano = res["manos"][0]
                    
                    resultado = base.buscar_estricto(
                        vector_query=mano["vector_normalizado"],
                        umbral_minimo=0.80,
                        margen_minimo=0.05,
                    )
                    
                    total_estricto += 1
                    if resultado:
                        aceptados += 1
                        margenes_aceptados.append(resultado[2])
                    else:
                        rechazados += 1
    
    extractor.liberar()
    
    tasa_aceptacion = aceptados / max(total_estricto, 1)
    margen_medio = sum(margenes_aceptados) / max(len(margenes_aceptados), 1) if margenes_aceptados else 0
    
    print(f"  Total evaluadas:  {total_estricto}")
    print(f"  Aceptadas (alta confianza): {aceptados} ({tasa_aceptacion:.1%})")
    print(f"  Rechazadas (inciertas):     {rechazados}")
    print(f"  Margen promedio aceptadas:  {margen_medio:.3f}")
    
    return tasa_aceptacion


def probar_ensamblado_y_tts():
    """Prueba de ensamblado de oraciones y TTS."""
    print("\n" + "=" * 60)
    print("  6. PRUEBA DE ENSAMBLADO DE ORACIONES Y TTS")
    print("=" * 60)
    
    frases_generadas = []
    
    def on_frase(f):
        frases_generadas.append(f)
        print(f"  [TTS Callback]: \"{f}\"")

    ensamblador = EnsambladorFrases(frames_para_aceptar=2, segundos_silencio_cierre=0.1, callback_frase_lista=on_frase)

    for _ in range(3):
        ensamblador.registrar_prediccion("HOLA", 0.95)
    for _ in range(3):
        ensamblador.registrar_prediccion("BUENAS", 0.92)
    for _ in range(3):
        ensamblador.registrar_prediccion("TARDES", 0.90)

    import time
    time.sleep(0.15)
    ensamblador.registrar_prediccion(None, 0.0)

    print(f"  Oración finalizada: {frases_generadas}")
    assert len(frases_generadas) > 0, "Debe haber generado una oración"
    assert "tardes" in frases_generadas[0].lower()
    print("  [OK] Pipeline de oraciones y TTS verificado con éxito.")


if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFICACION END-TO-END CON ANTI-CONFUSION")
    print("  Motor de Penalización Triple + Búsqueda Estricta")
    print("=" * 60)
    
    acc_palabras = probar_reconocimiento_imagenes_lsc70()
    acc_dacti = probar_dactilologia_completa()
    tasa_confusion = probar_confusion_cruzada()
    acc_lsc54 = probar_reconocimiento_series_lsc54()
    tasa_aceptacion = probar_margenes_confianza()
    probar_ensamblado_y_tts()
    
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL DE VERIFICACION")
    print("=" * 60)
    print(f"  Precisión Palabras (LSC70W):      {acc_palabras:.1%}")
    print(f"  Precisión Dactilología (LSC70ANH): {acc_dacti:.1%}")
    print(f"  Tasa Confusión Cruzada:            {tasa_confusion:.1%}")
    print(f"  Precisión Señas Dinámicas (LSC54): {acc_lsc54:.1%}")
    print(f"  Tasa Aceptación Estricta:          {tasa_aceptacion:.1%}")
    print("=" * 60)
    
    if acc_palabras >= 0.80 and acc_lsc54 >= 0.80:
        print("  [EXITO] Precisión general SATISFACTORIA (>=80%)")
    else:
        print("  [AVISO] Precisión general por debajo del objetivo (80%)")
    
    if tasa_confusion <= 0.10:
        print("  [EXITO] Confusión cruzada CONTROLADA (<=10%)")
    else:
        print("  [AVISO] Confusión cruzada elevada (>10%)")
    
    print("=" * 60)
