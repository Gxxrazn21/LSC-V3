"""
=============================================================
SERVIDOR WEB Y WEBSOCKET DE RECONOCIMIENTO LSC
Lengua de Señas Colombiana (LSC)
=============================================================
Conecta la interfaz gráfica ('Seña Android') con el motor de
reconocimiento vectorial en tiempo real, cuadrantes y TTS local.
"""

import os
import json
import base64
import time
from typing import List, Optional
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
from .ensamblador_frases import EnsambladorFrases
from .tts_local import MotorVozLocal


app = FastAPI(
    title="Seña LSC - Servidor de Reconocimiento",
    description="Motor de reconocimiento de Lengua de Señas Colombiana basado en vectores, cuadrantes espaciales y NLP local.",
    version="2.0.0",
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
base_vectores = BaseVectoresLSC()
extractor = ExtractorLandmarks()
motor_tts = MotorVozLocal(rate=160)
ensamblador = EnsambladorFrases(callback_frase_lista=lambda frase: motor_tts.hablar(frase))

# Cargar base si existe
if os.path.exists(BASE_DB_PATH):
    try:
        base_vectores.cargar(BASE_DB_PATH)
        print(f"[Servidor LSC] Base de datos cargada: {base_vectores.total_senas} vectores.")
    except Exception as e:
        print(f"[Servidor LSC] Error al cargar base de vectores: {e}")


# Modelos Pydantic para API
class SolicitudTraduccion(BaseModel):
    glosas: List[str]


class SolicitudPronunciacion(BaseModel):
    texto: str


@app.get("/api/estado")
def obtener_estado():
    return {
        "estado": "activo",
        "total_senas_referencia": base_vectores.total_senas,
        "clases_disponibles": base_vectores.clases_unicas,
        "tts_activo": True,
    }


@app.get("/api/catalogo")
def obtener_catalogo():
    return {
        "total": base_vectores.total_senas,
        "clases": base_vectores.clases_unicas,
        "senas_dinamicas": list(base_vectores.plantillas_dinamicas.keys()),
    }


@app.post("/api/traducir")
def traducir_glosas(req: SolicitudTraduccion):
    oracion = EnsambladorFrases._construir_oracion(req.glosas)
    return {"glosas": req.glosas, "oracion": oracion}


@app.post("/api/pronunciar")
def pronunciar_texto(req: SolicitudPronunciacion):
    motor_tts.hablar(req.texto, ignorar_cooldown=True)
    return {"status": "ok", "texto": req.texto}


@app.get("/", response_class=HTMLResponse)
def index():
    # Servir la interfaz de Seña Android si existe
    ruta_html = os.path.join("estilo", "Se_a LSC Android (2).html")
    if os.path.exists(ruta_html):
        with open(ruta_html, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Seña LSC - Servidor Activo</h1><p>Visita /docs para la API interactiva.</p>")


@app.websocket("/ws/reconocimiento")
async def websocket_reconocimiento(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Recibir frame codificado en base64 o paquete JSON
            mensaje_raw = await websocket.receive_text()
            data = json.loads(mensaje_raw)

            if "imagen_base64" in data:
                # Decodificar JPEG base64
                img_bytes = base64.b64decode(data["imagen_base64"].split(",")[-1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # 1. Extracción de landmarks
                    res = extractor.procesar_frame(frame)

                    sena_detectada = None
                    score_max = 0.0
                    cuadrante_str = "DESCONOCIDO"
                    top_candidatos = []

                    if res["hay_manos"]:
                        mano_principal = res["manos"][0]
                        cuad, _ = clasificar_cuadrante(
                            mano_principal["muneca"],
                            res.get("pose_anchors"),
                            res["alto_frame"],
                            res["ancho_frame"],
                        )
                        cuadrante_str = cuad.value

                        # 2. Búsqueda vectorial
                        candidatos = base_vectores.buscar_similar(
                            vector_query=mano_principal["vector_normalizado"],
                            cuadrante_query=cuadrante_str,
                            top_k=3,
                        )
                        if candidatos and candidatos[0][1] >= base_vectores.umbral_min_similitud:
                            sena_detectada = candidatos[0][0]
                            score_max = candidatos[0][1]
                            top_candidatos = [
                                {"sena": c[0], "similitud": round(c[1], 3), "cuadrante": c[2]}
                                for c in candidatos
                            ]

                    # 3. Ensamblado de glosas y oraciones
                    glosas_buffer, frase_lista = ensamblador.registrar_prediccion(
                        sena_detectada, score_max
                    )

                    # 4. Responder al frontend
                    respuesta = {
                        "hay_manos": res["hay_manos"],
                        "cuadrante": cuadrante_str,
                        "sena_detectada": sena_detectada,
                        "confianza": round(score_max, 3),
                        "top_candidatos": top_candidatos,
                        "glosas_acumuladas": glosas_buffer,
                        "frase_generada": frase_lista,
                    }
                    await websocket.send_text(json.dumps(respuesta))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error]: {e}")


def iniciar_servidor(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    print(f"Iniciando servidor en http://localhost:{port} ...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    iniciar_servidor()
