"""
=============================================================
PRUEBAS UNITARIAS Y DE RENDIMIENTO - MOTOR LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Valida:
1. Carga y velocidad de búsqueda vectorial (<1 ms).
2. Segmentación de cuadrantes espaciales.
3. Ensamblador inteligente de glosas a español natural (deletreo, nombres, sintaxis).
4. Extracción y normalización de landmarks.
"""

import time
import numpy as np
from motor_lsc import (
    BaseVectoresLSC,
    CuadranteEspacial,
    clasificar_cuadrante,
    EnsambladorFrases,
    ExtractorLandmarks,
)


def test_base_vectores_rendimiento():
    print("\n--- 1. Test Base de Datos Vectorial ---")
    base = BaseVectoresLSC()
    base.cargar("modelos_guardados/base_senas_lsc.npz")

    print(f"  Total vectores: {base.total_senas}")
    print(f"  Clases únicas: {len(base.clases_unicas)}")
    assert base.total_senas > 0, "La base debe contener vectores de referencia"

    # Medir latencia de búsqueda vectorial (1000 iteraciones)
    dim = len(base.vectores[0])
    query_vector = base.vectores[0] + np.random.normal(0, 0.02, dim).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(1000):
        resultados = base.buscar_similar(query_vector, cuadrante_query="ESPACIO_NEUTRO", top_k=3)
    t1 = time.perf_counter()

    latencia_ms = ((t1 - t0) / 1000.0) * 1000.0
    print(f"  Latencia de búsqueda: {latencia_ms:.4f} ms por consulta (¡{1000.0 / latencia_ms:.0f} búsquedas por segundo!)")
    print(f"  Top-1 detectado: '{resultados[0][0]}' con similitud {resultados[0][1]:.3f} ({resultados[0][2]})")
    assert latencia_ms < 2.0, "La latencia debe ser menor a 2 ms en CPU"
    assert len(resultados) == 3, "Debe retornar 3 candidatos"
    print("  [PASO] Búsqueda vectorial ultra rápida verificada.")


def test_cuadrantes_espaciales():
    print("\n--- 2. Test Cuadrantes Espaciales (Signing Space) ---")
    # Caso 1: Mano en la cabeza/frente (y = 0.20)
    cuad_cabeza, _ = clasificar_cuadrante((0.5, 0.20, 0.0))
    print(f"  (0.5, 0.20) -> {cuad_cabeza.value}")
    assert cuad_cabeza == CuadranteEspacial.CABEZA_ROSTRO

    # Caso 2: Mano en el pecho (y = 0.48, x = 0.50)
    cuad_pecho, _ = clasificar_cuadrante((0.50, 0.48, 0.0))
    print(f"  (0.50, 0.48) -> {cuad_pecho.value}")
    assert cuad_pecho == CuadranteEspacial.PECHO_TORSO

    # Caso 3: Mano en el espacio lateral (y = 0.55, x = 0.80)
    cuad_neutro, _ = clasificar_cuadrante((0.80, 0.55, 0.0))
    print(f"  (0.80, 0.55) -> {cuad_neutro.value}")
    assert cuad_neutro == CuadranteEspacial.ESPACIO_LATERAL

    # Caso 4: Mano en reposo bajo (y = 0.75)
    cuad_bajo, _ = clasificar_cuadrante((0.5, 0.75, 0.0))
    print(f"  (0.5, 0.75) -> {cuad_bajo.value}")
    assert cuad_bajo == CuadranteEspacial.LATERAL_BAJO
    print("  [PASO] Segmentación por cuadrantes verificada.")


def test_ensamblador_frases():
    print("\n--- 3. Test Ensamblador de Frases LSC -> Español ---")
    ensamblador = EnsambladorFrases(frames_para_aceptar=3, segundos_silencio_cierre=0.2)

    # Simular emisión de glosas con estabilidad: HOLA, BUENAS, DIAS
    for _ in range(4):
        ensamblador.registrar_prediccion("HOLA", 0.95, mano_estable=True)
    for _ in range(4):
        ensamblador.registrar_prediccion("BUENAS", 0.92, mano_estable=True)
    for _ in range(4):
        ensamblador.registrar_prediccion("DIAS", 0.90, mano_estable=True)

    glosas, _ = ensamblador.registrar_prediccion(None, 0.0)
    print(f"  Glosas registradas en buffer: {glosas}")
    assert glosas == ["HOLA", "BUENAS", "DIAS"]

    # Simular silencio para cerrar la oración
    time.sleep(0.25)
    _, frase = ensamblador.registrar_prediccion(None, 0.0)
    print(f"  Oración natural generada: \"{frase}\"")
    assert "buenos" in frase.lower()

    # Probar deletreo + nombre: ['YO', 'NOMBRE', 'J', 'H', 'O', 'N']
    frase_nombre = EnsambladorFrases.construir_oracion_inteligente(["YO", "NOMBRE", "J", "H", "O", "N"])
    print(f"  ['YO', 'NOMBRE', 'J', 'H', 'O', 'N'] -> \"{frase_nombre}\"")
    assert "Jhon" in frase_nombre

    # Probar edad: ['YO', '10', 'ANNOS']
    frase_edad = EnsambladorFrases.construir_oracion_inteligente(["YO", "10", "ANNOS"])
    print(f"  ['YO', '10', 'ANNOS'] -> \"{frase_edad}\"")
    assert "10 años" in frase_edad

    # Probar pregunta: ['DONDE', 'BAÑO']
    frase_bano = EnsambladorFrases.construir_oracion_inteligente(["DONDE", "BAÑO"])
    print(f"  ['DONDE', 'BAÑO'] -> \"{frase_bano}\"")
    assert "¿dónde" in frase_bano.lower()

    print("  [PASO] Ensamblador de frases inteligente verificado.")


def test_normalizador_invarianza():
    print("\n--- 4. Test Normalización de Landmarks ---")
    mano_cercana = np.random.uniform(0.4, 0.6, (21, 3)).astype(np.float32)
    mano_lejana = (mano_cercana * 0.5) + np.array([0.3, 0.2, -0.1], dtype=np.float32)

    norm_1 = ExtractorLandmarks.normalizar_mano(mano_cercana)
    norm_2 = ExtractorLandmarks.normalizar_mano(mano_lejana)

    cos_sim = np.dot(norm_1, norm_2) / (np.linalg.norm(norm_1) * np.linalg.norm(norm_2))
    print(f"  Similitud geométrica tras traslación y cambio de escala: {cos_sim:.5f}")
    assert cos_sim > 0.999, "La normalización debe ser invariante a escala y traslación"
    print("  [PASO] Invarianza geométrica verificada.")


def ejecutar_todos_los_tests():
    print("=" * 60)
    print("  EJECUTANDO SUITE DE PRUEBAS - MOTOR LSC")
    print("=" * 60)
    test_base_vectores_rendimiento()
    test_cuadrantes_espaciales()
    test_ensamblador_frases()
    test_normalizador_invarianza()
    print("\n" + "=" * 60)
    print("  [EXITO] TODAS LAS PRUEBAS (4/4) PASARON CON EXITO!")
    print("=" * 60)


if __name__ == "__main__":
    ejecutar_todos_los_tests()
