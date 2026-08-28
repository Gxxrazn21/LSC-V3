# Seña LSC v3.0 — Asistente y Academia de Lengua de Señas Colombiana con IA 🇨🇴🤟

Plataforma integral, accesible e interactiva de **Reconocimiento, Traducción y Enseñanza de la Lengua de Señas Colombiana (LSC)** en tiempo real, impulsada por un **Motor Articular Canónico 3D (101 Dimensiones)** y una **Base Vectorial Ultrarrápida (<1ms en CPU)**.

---

## 🌟 Características Principales

### 1. 📷 Traductor en Vivo (LSC ➔ Español y Voz)
* **Cámara en tiempo real a 30 FPS:** Integrada con MediaPipe Hands y Pose para captura biométrica precisa.
* **HUD con Telemetría Articular:** Monitor en vivo del estado de los 5 dedos (extendido vs flexionado) y visualización del esqueleto manual.
* **Segmentación por Cuadrantes Anatómicos (*Signing Space*):** Subdivide el espacio corporal en 6 zonas clave (*Cabeza/Rostro, Cuello/Garganta, Pecho/Torso, Espacio Central, Espacio Lateral, Reposo*) para eliminar ambigüedades entre señas de configuración manual parecida.
* **Ensamblador Gramatical y Síntesis de Voz (TTS):** Transforma secuencias de glosas LSC en español natural y las pronuncia en voz alta.

### 2. 🎓 Academia & Práctica Interactiva con Evaluación por IA
* **Catálogo de 52 señas:**
  * **Números:** `1`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `MIL`, `MILLÓN`.
  * **Alfabeto Dactilológico:** `A` hasta la `Z`.
  * **Saludos y Cortesía:** `HOLA`, `BUENAS`, `DÍAS`, `TARDES`, `NOCHES`, `BIENVENIDO`, `GRACIAS`, `BIEN`.
  * **Vida Diaria y Asistencia:** `AYUDAR`, `APOYAR`, `BAÑO`, `YO`, `NOMBRE`, `AÑOS`, `GUSTAR`, `LICOR`.
* **Modo Desafío con Cámara:** El usuario selecciona cualquier seña, activa la cámara de práctica y la IA evalúa su postura articular en tiempo real, otorgando retroalimentación correctiva (*"Extiende más el dedo índice", "Recoge el pulgar"*), porcentaje de acierto y celebración visual.

### 3. 💬 Traductor Bidireccional (Voz/Texto ➔ Señas LSC)
* Diseñado para facilitar la comunicación de **personas oyentes hacia personas sordas**.
* Permite hablar por el micrófono o escribir texto en español; el sistema analiza la sintaxis y presenta la secuencia de tarjetas ilustradas y descriptivas en LSC paso a paso.

### 4. ⚡ Frases Rápidas SOS y Vida Cotidiana
* Tarjetas de alta accesibilidad con un solo toque para situaciones de emergencia, transporte y atención al público con síntesis de voz instantánea:
  * *"¿Dónde queda el baño?"*
  * *"Necesito ayuda por favor."*
  * *"No entiendo, ¿puedes escribirlo o hablar despacio?"*
  * *"Muchas gracias por tu apoyo."*

---

## 🧠 Arquitectura del Motor Articular (101 Dimensiones)

El sistema no depende de redes neuronales pesadas en inferencia ni requiere GPUs costosas:

```
 Cámara / Imagen
       │
       ▼
 MediaPipe Hands (21 Landmarks 3D)
       │
       ▼
 Extractor Canónico Ortonormal (101D):
 ├─ 63 dims: Coordenadas proyectadas en base local de la palma
 ├─  5 dims: Estados continuos de extensión de dedos [0.0 - 1.0]
 ├─ 15 dims: Cosenos de ángulos de flexión articular (MCP, PIP, DIP)
 ├─ 10 dims: Distancias interdigitales entre puntas
 ├─  5 dims: Distancias al centro de la palma
 └─  3 dims: Vector normal de orientación de la palma
       │
       ▼
 Búsqueda Vectorial Coseno con Penalización Digital (<0.98 ms)
 (Catálogo precalculado de 4,842 vectores multi-sujeto)
       │
       ▼
 Ponderación por Cuadrante Anatómico + Filtro Cinético
       │
       ▼
 Ensamblador Lingüístico LSC ➔ Español + Síntesis de Voz
```

---

## 🚀 Instalación y Requisitos

### Requisitos Previos
* **Python 3.10 o 3.11** (Recomendado 3.11).
* Cámara web estándar o cámara de dispositivo móvil.

### Paso a Paso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Gxxrazn21/LSC-V3.git
   cd LSC-V3
   ```

2. **Crear y activar un entorno virtual:**
   * En Windows (PowerShell):
     ```powershell
     python -m venv venv_lsc
     .\venv_lsc\Scripts\Activate.ps1
     ```
   * En Linux / macOS:
     ```bash
     python3 -m venv venv_lsc
     source venv_lsc/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Modos de Ejecución

### 1. Iniciar la Plataforma Web Completa (Recomendado)
Inicia el servidor backend con WebSockets y abre la interfaz interactiva:

```bash
python -m motor_lsc.servidor_app
```
Luego abre tu navegador en: **`http://localhost:8000`**

### 2. Reconocimiento Directo en Ventana OpenCV con HUD
Ideal para pruebas rápidas de cámara y telemetría articular:

```bash
python predecir_vivo.py
```
* **Controles:**
  * `Q`: Salir.
  * `C`: Limpiar buffer de glosas.
  * `Espacio`: Forzar pronunciación inmediata de la frase acumulada.

### 3. Ejecutar Suite de Pruebas Unitarias
Valida latencia, invarianza geométrica, cuadrantes espaciales y ensamblador sintáctico:

```bash
python test_motor_lsc.py
```

### 4. Recompilar la Base de Datos Vectorial (Opcional)
Si agregas nuevas muestras a los datasets en `datasets/`:

```bash
python -m motor_lsc.generador_referencias
```

---

## 📂 Estructura del Proyecto

```
LSC-V3/
├── motor_lsc/                     # Motor central de IA y procesamiento
│   ├── __init__.py
│   ├── base_vectores.py           # Búsqueda vectorial y penalización articular
│   ├── catalogo_senas.py          # Diccionario pedagógico y metadatos de las 52 señas
│   ├── cuadrantes.py              # Subdivisión espacial anatómica (Signing Space)
│   ├── ensamblador_frases.py      # NLP gramatical LSC -> Español natural
│   ├── extractor.py               # Extractor articular canónico 3D de 101D
│   ├── generador_referencias.py   # Compilador de vectores multi-sujeto
│   ├── servidor_app.py            # Servidor FastAPI, API REST y WebSockets
│   └── tts_local.py               # Síntesis de voz local offline (pyttsx3)
├── estilo/                        # Interfaz gráfica web y móvil
│   ├── index.html                 # Aplicación interactiva moderna (Material/Glassmorphic)
│   └── Se_a LSC Android (2).html  # Prototipo Android M3
├── modelos_guardados/
│   ├── base_senas_lsc.npz         # Catálogo de 4,842 vectores precompilados (1.6 MB)
│   └── clases_lsc.json            # Clases y etiquetas del sistema
├── predecir_vivo.py               # Script de captura en vivo por cámara con HUD
├── test_motor_lsc.py              # Pruebas de rendimiento y precisión
├── verificar_reconocimiento_real.py # Verificación end-to-end con datos reales
├── requirements.txt               # Dependencias de Python
├── .gitignore                     # Configuración de exclusión de datos pesados/privados
└── README.md                      # Documentación del proyecto
```

---

## 🤝 Inclusión y Comunidad
Este proyecto fue concebido para romper barreras de comunicación y brindar una herramienta educativa y asistiva accesible a personas con discapacidad auditiva y oyentes en Colombia y Latinoamérica.

---

## 📄 Licencia
Distribuido bajo la Licencia MIT. Consulta `LICENSE` para más información.
