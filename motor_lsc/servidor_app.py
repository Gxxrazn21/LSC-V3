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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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
base_vectores = BaseVectoresLSC(umbral_min_similitud=0.72)
extractor = ExtractorLandmarks()
motor_tts = MotorVozLocal(rate=160)
ensamblador = EnsambladorFrases(callback_frase_lista=lambda frase: motor_tts.hablar(frase))
exportador_3d: ExportadorPoses3D = None  # Se inicializa después de cargar la base

# Cargar base si existe
if os.path.exists(BASE_DB_PATH):
    try:
        base_vectores.cargar(BASE_DB_PATH)
        print(f"[Servidor LSC] Base de datos cargada: {base_vectores.total_senas} vectores de referencia.")
        exportador_3d = ExportadorPoses3D(base_vectores)
        print(f"[Servidor LSC] Exportador 3D inicializado: {len(exportador_3d.obtener_senas_disponibles())} señas con poses 3D.")
    except Exception as e:
        print(f"[Servidor LSC] Error al cargar base de vectores: {e}")


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
    
    # Mapeos directos de frases y modismos
    frase_str = " ".join(palabras)
    if "HOLA" in frase_str and "BUENOS DIAS" in frase_str:
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

                    if res["hay_manos"]:
                        mano = res["manos"][0]
                        es_estable = mano.get("es_estable", True)
                        dedos_exts = [round(float(e), 2) for e in mano.get("finger_extensions", [0]*5)]

                        cuad, _ = clasificar_cuadrante(
                            mano["muneca"],
                            res.get("pose_anchors"),
                            res["alto_frame"],
                            res["ancho_frame"],
                        )
                        cuadrante_str = cuad.value

                        if es_estable:
                            candidatos = base_vectores.buscar_similar(
                                vector_query=mano["vector_normalizado"],
                                cuadrante_query=cuadrante_str,
                                top_k=3,
                            )
                            top_candidatos = [
                                {"sena": c[0], "similitud": round(c[1], 3), "cuadrante": c[2]}
                                for c in candidatos
                            ]

                            # Búsqueda estricta anti-adivinanza
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
                        else:
                            ventana.alimentar(None, 0.0)
                    else:
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
                        "confianza": round(score_max, 3),
                        "consenso_pct": round(consenso_pct, 2),
                        "top_candidatos": top_candidatos,
                        "glosas_acumuladas": glosas_buffer,
                        "frase_generada": frase_lista,
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


def iniciar_servidor(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    print(f"Iniciando servidor en http://localhost:{port} ...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    iniciar_servidor()
