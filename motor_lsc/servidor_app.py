"""
=============================================================
SERVIDOR WEB Y WEBSOCKET DE RECONOCIMIENTO Y APRENDIZAJE LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Conecta la aplicación web/móvil con:
1. Motor de reconocimiento articular 3D (101D, 4842 vectores).
2. Ensamblador lingüístico de frases y gramática LSC -> Español.
3. Modo interactivo de Práctica y Aprendizaje con evaluación en vivo.
4. Traductor bidireccional Texto/Voz -> Secuencia de señas LSC.
5. Síntesis de voz local no bloqueante (pyttsx3).
"""

import os
import json
import base64
import time
import re
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .base_vectores import BaseVectoresLSC
from .extractor import ExtractorLandmarks
from .cuadrantes import clasificar_cuadrante, COLORES_CUADRANTE
from .ensamblador_frases import EnsambladorFrases, VentanaConsenso
from .tts_local import MotorVozLocal
from .catalogo_senas import DICCIONARIO_EDUCATIVO_LSC, obtener_catalogo_completo
from .exportador_poses_3d import ExportadorPoses3D


app = FastAPI(
    title="Seña LSC - Plataforma Integral",
    description="Motor de reconocimiento, práctica interactiva y asistencia en Lengua de Señas Colombiana.",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes globales
BASE_DB_PATH = os.path.join("modelos_guardados", "base_senas_lsc.npz")
MODELO_IA_PATH = os.path.join("modelos_guardados", "modelo_ia_lsc70.joblib")
MODELO_IA_NPZ = os.path.join("modelos_guardados", "modelo_ia_lsc70.npz")

from .clasificador_ia import ClasificadorIALSC

base_vectores = BaseVectoresLSC(umbral_min_similitud=0.72)
extractor = ExtractorLandmarks()
motor_tts = MotorVozLocal(rate=160)
ensamblador = EnsambladorFrases(callback_frase_lista=lambda frase: motor_tts.hablar(frase))
exportador_3d: ExportadorPoses3D = None  # Se inicializa después de cargar la base
clasificador_ia: Optional[ClasificadorIALSC] = None

# 1. Cargar modelo de IA entrenado (LSC70 109D)
if os.path.exists(MODELO_IA_PATH) or os.path.exists(MODELO_IA_NPZ):
    try:
        ruta_m = MODELO_IA_PATH if os.path.exists(MODELO_IA_PATH) else MODELO_IA_NPZ
        clasificador_ia = ClasificadorIALSC(ruta_m, umbral_confianza=0.70, margen_minimo=0.12)
        print(f"[Servidor LSC] Modelo de IA cargado exitosamente ({len(clasificador_ia.clases)} clases) desde {ruta_m}")
    except Exception as e:
        print(f"[Servidor LSC] Aviso al cargar modelo IA: {e}")

# 2. Cargar base vectorial como complemento/fallback
if os.path.exists(BASE_DB_PATH):
    try:
        base_vectores.cargar(BASE_DB_PATH)
        print(f"[Servidor LSC] Base de datos cargada: {base_vectores.total_senas} vectores de referencia.")
        exportador_3d = ExportadorPoses3D(base_vectores)
    except Exception as e:
        print(f"[Servidor LSC] Aviso base de vectores: {e}")


# Modelos Pydantic para API
class SolicitudTraduccion(BaseModel):
    glosas: List[str]


class SolicitudPronunciacion(BaseModel):
    texto: str


class SolicitudTextoALsc(BaseModel):
    texto: str


class SolicitudEvaluacion(BaseModel):
    sena_objetivo: str
    imagen_base64: str


class SolicitudCapturaMovil(BaseModel):
    sena: str
    imagenes_base64: List[str]


class SolicitudBenchmark(BaseModel):
    sena_objetivo: str
    imagen_base64: str


@app.get("/api/estado")
def obtener_estado():
    return {
        "estado": "activo",
        "total_senas_referencia": base_vectores.total_senas,
        "clases_disponibles": base_vectores.clases_unicas,
        "tts_activo": True,
        "total_catalogo_educativo": len(DICCIONARIO_EDUCATIVO_LSC),
    }


@app.get("/api/catalogo_educativo")
def api_catalogo_educativo():
    """Retorna todo el catálogo con instrucciones anatómicas, consejos y categorías."""
    return {
        "total": len(DICCIONARIO_EDUCATIVO_LSC),
        "senas": obtener_catalogo_completo(),
    }


@app.post("/api/texto_a_lsc")
def traducir_texto_a_lsc(req: SolicitudTextoALsc):
    """
    Convierte una oración en español a su secuencia de señas/glosas LSC equivalentes.
    """
    texto_limpio = req.texto.upper().strip()
    # Separar palabras
    palabras = re.findall(r'[A-ZÁÉÍÓÚÑ0-9]+', texto_limpio)
    
    secuencia_lsc = []
    
    # Lematización básica y normalización morfológica para LSC
    LEMAS_LSC = {
        "BIENVENIDOS": "BIENVENIDO", "BIENVENIDA": "BIENVENIDO", "BIENVENIDAS": "BIENVENIDO",
        "AYUDO": "AYUDAR", "AYUDAS": "AYUDAR", "AYUDAME": "AYUDAR", "AYUDANOS": "AYUDAR", "AYUDEN": "AYUDAR",
        "APOYO": "APOYAR", "APOYAS": "APOYAR", "APOYAME": "APOYAR", "APOYANOS": "APOYAR",
        "GUSTA": "GUSTAR", "GUSTAN": "GUSTAR", "GUSTARIA": "GUSTAR",
        "BUENO": "BUENAS", "BUENOS": "BUENAS", "BUENA": "BUENAS",
        "DIA": "DIAS", "TARDE": "TARDES", "NOCHE": "NOCHES",
        "AMIGOS": "AMIGO", "AMIGA": "AMIGO", "AMIGAS": "AMIGO",
        "FAMILIAS": "FAMILIA", "TRABAJO": "TRABAJAR", "TRABAJANDO": "TRABAJAR", "TRABAJAS": "TRABAJAR",
        "BAÑOS": "BAÑO", "BANOS": "BAÑO", "BANO": "BAÑO",
        "ESTOY": "YO", "SOMOS": "NOSOTROS",
    }

    # Partículas y nexos omitidos en la gramática visual de LSC (cuando hay más de una palabra)
    PARTICULAS_OMITIR = {"EL", "LA", "LOS", "LAS", "UN", "UNA", "UNOS", "UNAS", "DE", "DEL", "AL", "TE", "SE", "QUE"}

    frase_str = " ".join(palabras)
    if "HOLA" in frase_str and ("BUENOS DIAS" in frase_str or "BUEN DIA" in frase_str):
        secuencia_lsc = ["HOLA", "BUENAS", "DIAS"]
    elif "BUENOS DIAS" in frase_str or "BUEN DIA" in frase_str:
        secuencia_lsc = ["BUENAS", "DIAS"]
    elif "BUENAS TARDES" in frase_str:
        secuencia_lsc = ["BUENAS", "TARDES"]
    elif "BUENAS NOCHES" in frase_str:
        secuencia_lsc = ["BUENAS", "NOCHES"]
    elif "DONDE ESTA EL BANO" in frase_str or "DONDE QUEDA EL BANO" in frase_str or "BANO" in frase_str:
        secuencia_lsc = ["DONDE", "BAÑO"]
    elif "NECESITO AYUDA" in frase_str or "AYUDAME" in frase_str or "AYUDA" in frase_str:
        secuencia_lsc = ["YO", "AYUDAR"]
    elif "MI NOMBRE ES" in frase_str or "ME LLAMO" in frase_str:
        partes = frase_str.split("ES") if "ES" in frase_str else frase_str.split("LLAMO")
        nombre_detectado = partes[-1].strip() if len(partes) > 1 else ""
        secuencia_lsc = ["YO", "NOMBRE"] + list(nombre_detectado.replace(" ", ""))
    else:
        for p in palabras:
            p_clean = p.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
            # Aplicar lematización
            if p_clean in LEMAS_LSC:
                p_clean = LEMAS_LSC[p_clean]

            # En frases compuestas, omitir artículos y partículas gramaticales
            if len(palabras) > 1 and p_clean in PARTICULAS_OMITIR:
                continue

            if p_clean in DICCIONARIO_EDUCATIVO_LSC:
                secuencia_lsc.append(p_clean)
            elif p_clean.isdigit():
                secuencia_lsc.append(p_clean)
            else:
                # Deletreo dactilológico
                for letra in p_clean:
                    if letra in DICCIONARIO_EDUCATIVO_LSC:
                        secuencia_lsc.append(letra)

    # Enriquecer con metadatos para la vista de tarjetas
    detalles = []
    for g in secuencia_lsc:
        info = DICCIONARIO_EDUCATIVO_LSC.get(g, {
            "nombre": g,
            "categoria": "Letra" if len(g) == 1 else "Palabra",
            "cuadrante": "ESPACIO_LATERAL",
            "descripcion": f"Seña para {g}",
            "consejo": "Realiza la configuración manual en el espacio de señación.",
            "ejemplo_uso": "",
        })
        detalles.append({"id": g, **info})

    return {
        "texto_original": req.texto,
        "secuencia_glosas": secuencia_lsc,
        "tarjetas": detalles,
    }


@app.post("/api/texto_a_lsc_3d")
def traducir_texto_a_lsc_3d(req: SolicitudTextoALsc):
    """
    Convierte texto en español a una secuencia de poses 3D articulares
    para renderizar en un avatar Three.js.
    Cada seña incluye 21 landmarks 3D, extensión de dedos, normal de palma,
    cuadrante espacial y metadatos de animación.
    """
    global exportador_3d
    if exportador_3d is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Exportador 3D no inicializado. La base de datos vectorial no está cargada."},
        )

    # 1. Reutilizar la lógica de texto_a_lsc para obtener glosas
    resultado_glosas = traducir_texto_a_lsc(req)
    secuencia_glosas = resultado_glosas["secuencia_glosas"]

    if not secuencia_glosas:
        return {
            "texto_original": req.texto,
            "secuencia_glosas": [],
            "animacion": {"total_senas": 0, "duracion_total_ms": 0, "secuencia": []},
        }

    # 2. Generar secuencia de animación 3D
    animacion = exportador_3d.generar_secuencia_animacion(secuencia_glosas)

    return {
        "texto_original": req.texto,
        "secuencia_glosas": secuencia_glosas,
        "animacion": animacion,
    }


@app.get("/api/exportador_3d/estado")
def estado_exportador_3d():
    """Estado del exportador de poses 3D."""
    global exportador_3d
    if exportador_3d is None:
        return {"inicializado": False}
    return {
        "inicializado": True,
        **exportador_3d.obtener_estadisticas(),
    }


@app.post("/api/evaluar_sena")
def evaluar_sena_practica(req: SolicitudEvaluacion):
    """
    Evalúa la seña que el usuario está realizando frente a la cámara en el modo práctica.
    """
    sena_target = req.sena_objetivo.upper().strip()
    if not req.imagen_base64:
        return {"acierto": False, "confianza": 0.0, "mensaje": "No se recibió imagen"}

    img_bytes = base64.b64decode(req.imagen_base64.split(",")[-1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"acierto": False, "confianza": 0.0, "mensaje": "Imagen inválida"}

    res = extractor.procesar_frame(frame)
    if not res["hay_manos"]:
        return {
            "hay_manos": False,
            "acierto": False,
            "confianza": 0.0,
            "mensaje": "Coloca tu mano visible frente a la cámara",
            "dedos_detectados": [0, 0, 0, 0, 0],
        }

    mano = res["manos"][0]
    cuad, _ = clasificar_cuadrante(mano["muneca"], res.get("pose_anchors"), res["alto_frame"], res["ancho_frame"])
    
    candidatos = base_vectores.buscar_similar(mano["vector_normalizado"], cuadrante_query=cuad.value, top_k=5)
    
    confianza_target = 0.0
    for s, score, _, _ in candidatos:
        if s == sena_target:
            confianza_target = score
            break

    top_prediccion = candidatos[0][0] if candidatos else "DESCONOCIDO"
    top_score = candidatos[0][1] if candidatos else 0.0

    acierto = (top_prediccion == sena_target and top_score >= 0.72) or (confianza_target >= 0.76)
    
    exts = [round(float(e), 2) for e in mano.get("finger_extensions", [0]*5)]

    info_meta = DICCIONARIO_EDUCATIVO_LSC.get(sena_target, {})
    dedos_esperados = info_meta.get("dedos", [0]*5)

    consejo = info_meta.get("consejo", "Mantén la mano firme.")
    if not acierto:
        if dedos_esperados[1] == 1 and exts[1] < 0.5:
            consejo = "Extiende más el dedo índice hacia arriba."
        elif dedos_esperados[0] == 0 and exts[0] > 0.6:
            consejo = "Recoge el pulgar sobre la palma."
        elif top_prediccion != sena_target:
            consejo = f"Tu postura se parece a '{top_prediccion}'. Revisa la guía de dedos."

    return {
        "hay_manos": True,
        "sena_objetivo": sena_target,
        "sena_detectada": top_prediccion,
        "confianza": round(float(top_score), 3),
        "confianza_objetivo": round(float(confianza_target), 3),
        "acierto": acierto,
        "cuadrante": cuad.value,
        "dedos_detectados": exts,
        "dedos_esperados": dedos_esperados,
        "consejo": consejo,
    }


@app.post("/api/traducir")
def traducir_glosas(req: SolicitudTraduccion):
    oracion = EnsambladorFrases.construir_oracion_inteligente(req.glosas)
    return {"glosas": req.glosas, "oracion": oracion}


@app.post("/api/pronunciar")
def pronunciar_texto(req: SolicitudPronunciacion):
    motor_tts.hablar(req.texto, ignorar_cooldown=True)
    return {"status": "ok", "texto": req.texto}


@app.get("/manifest.json")
def pwa_manifest():
    ruta = os.path.join("estilo", "manifest.json")
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="application/manifest+json")
    return JSONResponse({})


@app.get("/sw.js")
def pwa_sw():
    ruta = os.path.join("estilo", "sw.js")
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="application/javascript")
    return Response("", media_type="application/javascript")


@app.get("/api/metricas")
def api_metricas():
    """Retorna las métricas del último entrenamiento, reporte y URLs de gráficos."""
    historial = []
    ruta_hist = os.path.join("resultados", "historial_entrenamientos.json")
    if os.path.exists(ruta_hist):
        try:
            with open(ruta_hist, "r", encoding="utf-8") as f:
                historial = json.load(f)
        except Exception:
            pass

    reporte_txt = ""
    ruta_rep = os.path.join("resultados", "reporte_clasificacion.txt")
    if os.path.exists(ruta_rep):
        try:
            with open(ruta_rep, "r", encoding="utf-8") as f:
                reporte_txt = f.read()
        except Exception:
            pass

    clases_modelo = []
    if clasificador_ia is not None:
        clases_modelo = [str(c) for c in clasificador_ia.clases]

    return {
        "historial": historial,
        "ultimo_entrenamiento": historial[-1] if historial else None,
        "reporte_txt": reporte_txt,
        "imagenes": {
            "matriz_confusion": "/resultados/matriz_confusion.png",
            "curvas_aprendizaje": "/resultados/curvas_aprendizaje.png",
            "metricas_por_clase": "/resultados/metricas_por_clase.png",
            "comparativa_historica": "/resultados/comparativa_historica.png",
        },
        "mejor_modelo": {
            "dimensiones": 109,
            "tipo": "Multimodal 109D (Mano Canónica 105D + Hombros Invariantes 4D)",
            "clases": clases_modelo,
        },
    }


@app.post("/api/benchmark/evaluar")
def api_benchmark_evaluar(req: SolicitudBenchmark):
    """
    Evalúa un fotograma para una seña específica, calculando no solo la clase
    predicha sino la precisión anatómica (coordenadas de hombros, postura de dedos y distancia).
    """
    try:
        b64_clean = req.imagen_base64.split(",")[-1]
        img_bytes = base64.b64decode(b64_clean)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"hay_manos": False, "acierto": False, "mensaje": "Fotograma no válido"}

        res = extractor.procesar_frame(frame)
        if not res["hay_manos"] or len(res["manos"]) == 0:
            return {
                "hay_manos": False,
                "acierto": False,
                "mensaje": "No se detecta mano frente a la cámara. Posiciónate con luz suficiente.",
            }

        mano = res["manos"][0]
        vec_109 = mano["vector_109d"]
        spatial = mano.get("spatial_coords", {})
        dy = spatial.get("dy", 0.0)

        sena_target = req.sena_objetivo.upper().strip()

        # Inferencia con IA 109D
        sena_predicha = "DESCONOCIDO"
        confianza = 0.0
        top_cands = []
        if clasificador_ia is not None:
            sena_predicha, confianza, top_cands = clasificador_ia.clasificar(vec_109)

        es_acierto = (sena_predicha == sena_target)

        # Diagnóstico anatómico
        diagnostico = "Postura correcta."
        zona = "PECHO"
        if dy < -0.30:
            zona = "CABEZA_ROSTRO"
        elif dy > 0.45:
            zona = "ABDOMEN_CADERA"

        if sena_target == "HOLA":
            if dy > -0.20:
                diagnostico = "Eleva la mano más alto, junto a la sien/oreja (evita bajarla al pecho para no confundir con GUSTAR)."
            elif es_acierto:
                diagnostico = f"¡Excelente! Elevación sobre hombros perfecta (dy={dy:.2f}) y mano abierta."
        elif sena_target == "GUSTAR":
            if dy < -0.30:
                diagnostico = "Baja la mano al pecho/esternón (está muy alta cerca de la cara)."
            elif dy > 0.60:
                diagnostico = "Sube la mano hacia el pecho (está muy abajo)."
            elif es_acierto:
                diagnostico = f"¡Excelente! Posición centrada sobre el pecho (dy={dy:.2f})."
        elif sena_target == "BUENAS":
            if dy < 0.10:
                diagnostico = "Coloca la mano en la zona media del tronco/abdomen."
            elif es_acierto:
                diagnostico = "¡Muy bien! Plano corporal medio correcto."

        return {
            "hay_manos": True,
            "sena_objetivo": sena_target,
            "sena_predicha": sena_predicha,
            "confianza": round(float(confianza), 3),
            "acierto": es_acierto,
            "zona_detectada": zona,
            "dy_hombros": round(float(dy), 3),
            "diagnostico": diagnostico,
            "top_candidatos": top_cands[:3],
            "landmarks": mano["landmarks_raw"][:, :2].tolist(),
        }
    except Exception as e:
        return {"hay_manos": False, "acierto": False, "mensaje": str(e)}


@app.post("/api/captura_movil")
def api_captura_movil(req: SolicitudCapturaMovil):
    """
    Guarda muestras HD tomadas con la cámara del celular en datasets/muestras_movil/{sena}/
    """
    try:
        sena_limpia = req.sena.upper().strip().replace(" ", "_")
        carpeta = os.path.join("datasets", "muestras_movil", sena_limpia)
        os.makedirs(carpeta, exist_ok=True)

        guardadas = 0
        ts = int(time.time())
        for idx, b64_img in enumerate(req.imagenes_base64):
            b64_clean = b64_img.split(",")[-1]
            img_bytes = base64.b64decode(b64_clean)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                fname = f"movil_{ts}_{idx+1:02d}.jpg"
                ruta_completa = os.path.join(carpeta, fname)
                cv2.imwrite(ruta_completa, frame)
                guardadas += 1

        total_en_carpeta = len(os.listdir(carpeta))
        return {
            "status": "ok",
            "sena": sena_limpia,
            "guardadas": guardadas,
            "total_carpeta": total_en_carpeta,
            "mensaje": f"Se guardaron {guardadas} muestras HD para {sena_limpia}. Total: {total_en_carpeta}.",
        }
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}


@app.websocket("/ws/reconocimiento")
async def websocket_reconocimiento(websocket: WebSocket):
    await websocket.accept()
    ventana = VentanaConsenso(tamano_ventana=10, min_consenso_pct=0.70, umbral_score_promedio=0.78, frames_histeresis=3)
    try:
        while True:
            mensaje_raw = await websocket.receive_text()
            data = json.loads(mensaje_raw)

            if "imagen_base64" in data:
                img_bytes = base64.b64decode(data["imagen_base64"].split(",")[-1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    res = extractor.procesar_frame(frame)

                    sena_detectada = None
                    score_max = 0.0
                    cuadrante_str = "DESCONOCIDO"
                    top_candidatos = []
                    dedos_exts = [0.0] * 5
                    es_estable = False
                    consenso_pct = 0.0

                    estado_ia = "ESPERANDO_MANO"
                    landmarks_norm = []

                    if res["hay_manos"]:
                        mano = res["manos"][0]
                        es_estable = mano.get("es_estable", True)
                        dedos_exts = [round(float(e), 2) for e in mano.get("finger_extensions", [0]*5)]
                        landmarks_norm = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in mano["landmarks_raw"]]

                        cuad_obj, _ = clasificar_cuadrante(
                            mano.get("muneca", (0.5, 0.5, 0)),
                            res.get("pose_anchors", {}),
                            res["alto_frame"],
                            res["ancho_frame"],
                        )
                        cuadrante_str = cuad_obj.value if hasattr(cuad_obj, "value") else str(cuad_obj)

                        if clasificador_ia and clasificador_ia.esta_cargado:
                            res_ia = clasificador_ia.predecir(
                                vector_105d=mano["vector_normalizado"],
                                cuadrante=cuadrante_str,
                                mano_estable=es_estable,
                                top_k=3,
                            )
                            top_candidatos = [
                                {"sena": str(c[0]), "similitud": round(float(c[1]), 3), "cuadrante": cuadrante_str}
                                for c in res_ia["candidatos"]
                            ]
                            estado_ia = res_ia["estado"]

                            if res_ia["es_valida"]:
                                s_cons, sc_cons, pct = ventana.alimentar(res_ia["etiqueta"], res_ia["confianza"])
                                consenso_pct = pct
                                if s_cons:
                                    sena_detectada = s_cons
                                    score_max = sc_cons
                            else:
                                ventana.alimentar(None, 0.0)
                        elif es_estable:
                            candidatos = base_vectores.buscar_similar(
                                vector_query=mano["vector_normalizado"],
                                cuadrante_query=cuadrante_str,
                                top_k=3,
                            )
                            top_candidatos = [
                                {"sena": c[0], "similitud": round(c[1], 3), "cuadrante": c[2]}
                                for c in candidatos
                            ]
                            res_estricto = base_vectores.buscar_estricto(
                                vector_query=mano["vector_normalizado"],
                                cuadrante_query=cuadrante_str,
                                umbral_minimo=base_vectores.umbral_min_similitud,
                                margen_minimo=0.04,
                            )
                            s_frame = res_estricto[0] if res_estricto else None
                            sc_frame = res_estricto[1] if res_estricto else 0.0
                            s_cons, sc_cons, pct = ventana.alimentar(s_frame, sc_frame)
                            consenso_pct = pct
                            if s_cons:
                                sena_detectada = s_cons
                                score_max = sc_cons
                                estado_ia = "SEÑA_DETECTADA"
                            else:
                                estado_ia = "INCIERTO"
                        else:
                            estado_ia = "TRANSICION"
                            ventana.alimentar(None, 0.0)
                    else:
                        estado_ia = "ESPERANDO_MANO"
                        ventana.alimentar(None, 0.0)

                    glosas_buffer, frase_lista = ensamblador.registrar_prediccion(
                        sena_detectada, score_max, mano_estable=es_estable
                    )

                    respuesta = {
                        "hay_manos": res["hay_manos"],
                        "es_estable": es_estable,
                        "dedos": dedos_exts,
                        "cuadrante": cuadrante_str,
                        "sena_detectada": sena_detectada,
                        "estado": estado_ia,
                        "confianza": round(score_max, 3),
                        "consenso_pct": round(consenso_pct, 2),
                        "top_candidatos": top_candidatos,
                        "glosas_acumuladas": glosas_buffer,
                        "frase_generada": frase_lista,
                        "landmarks": landmarks_norm,
                    }
                    await websocket.send_text(json.dumps(respuesta))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error]: {e}")


# Servir la aplicación Web en la raíz
@app.get("/", response_class=HTMLResponse)
def index():
    ruta_html = os.path.join("estilo", "index.html")
    if os.path.exists(ruta_html):
        with open(ruta_html, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Seña LSC - Servidor Activo</h1><p>Visita /docs para la API interactiva.</p>")


@app.get("/avatar", response_class=HTMLResponse)
def avatar_3d():
    """Sirve la interfaz del avatar 3D animado para texto → señas LSC."""
    ruta_html = os.path.join("estilo", "avatar_lsc.html")
    if os.path.exists(ruta_html):
        with open(ruta_html, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Avatar LSC - Página no encontrada</h1><p>Genera estilo/avatar_lsc.html primero.</p>")


# Montar archivos estáticos si existen
os.makedirs("estilo", exist_ok=True)
app.mount("/estilo", StaticFiles(directory="estilo"), name="estilo")

os.makedirs("resultados", exist_ok=True)
app.mount("/resultados", StaticFiles(directory="resultados"), name="resultados")


def iniciar_servidor(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    print(f"Iniciando servidor en http://localhost:{port} ...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    iniciar_servidor()
