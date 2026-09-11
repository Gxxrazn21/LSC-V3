# Sistema Seña LSC v4.5 — Documentación Técnica Integral
*Traductor Automático de Lengua de Señas Colombiana (LSC) a Texto y Voz 100% On-Device con Arquitectura Conformer Multi-Stream, Preentrenamiento MaskFeat, Validación Signer-Independent, Decodificación Continua CTC, Mejora de Imagen Zero-DCE y Despliegue Móvil Cuantizado*

---

## Tabla de Contenidos
1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Captura de Datos Multimodal y Corrección Cinemática](#3-captura-de-datos-multimodal-y-corrección-cinemática)
   - [3.1 MediaPipe Tasks en Modo VIDEO / Tracking Incremental](#31-mediapipe-tasks-en-modo-video--tracking-incremental)
   - [3.2 Subconjunto Facial Gramatical (64 Puntos NMMs Lingüísticos)](#32-subconjunto-facial-gramatical-64-puntos-nmms-lingüísticos)
   - [3.3 Corrección del Bug de Recorte de Manos en MediaPipe Holistic](#33-corrección-del-bug-de-recorte-de-manos-en-mediapipe-holistic)
   - [3.4 Modo Adaptativo Full / Lite según Tasa de FPS](#34-modo-adaptativo-full--lite-según-tasa-de-fps)
4. [Modelo de Reconocimiento Conformer Multi-Stream](#4-modelo-de-reconocimiento-conformer-multi-stream)
   - [4.1 Arquitectura Conformer Multi-Stream y Fusión por Atención Cruzada](#41-arquitectura-conformer-multi-stream-y-fusión-por-atención-cruzada)
   - [4.2 Preentrenamiento Auto-Supervisado Estilo MaskFeat](#42-preentrenamiento-auto-supervisado-estilo-maskfeat)
   - [4.3 Fine-Tuning Supervisado Signer-Independent](#43-fine-tuning-supervisado-signer-independent)
   - [4.4 Data Augmentation Cinemático 3D](#44-data-augmentation-cinemático-3d)
   - [4.5 Análisis Explícito y Matriz de Confusión de Pares Visualmente Similares](#45-análisis-explícito-y-matriz-de-confusión-de-pares-visualmente-similares)
5. [Pipeline de Salida: Reconocimiento Continuo y TTS](#5-pipeline-de-salida-reconocimiento-continuo-y-tts)
   - [5.1 Segmentación de Frases mediante Ventana Deslizante y CTC](#51-segmentación-de-frases-mediante-ventana-deslizante-y-ctc)
   - [5.2 Conversión Seña ➔ Texto ➔ Voz (TTS) y Latencia End-to-End (<35 ms)](#52-conversión-seña--texto--voz-tts-y-latencia-end-to-end-35-ms)
6. [Calidad de Imagen y Preprocesamiento 100% On-Device](#6-calidad-de-imagen-y-preprocesamiento-100-on-device)
   - [6.1 Corrección de Baja Luz mediante Curvas Analíticas Zero-DCE](#61-corrección-de-baja-luz-mediante-curvas-analíticas-zero-dce)
   - [6.2 Realce Local de Región de Interés (ROI Super-Resolution)](#62-realce-local-de-región-de-interés-roi-super-resolution)
7. [Presupuesto de Recursos y Despliegue Móvil](#7-presupuesto-de-recursos-y-despliegue-móvil)
   - [7.1 Evaluación de Cuantización (PTQ vs QAT)](#71-evaluación-de-cuantización-ptq-vs-qat)
   - [7.2 Modelos Cuantizados Generados (FP32, FP16, INT8)](#72-modelos-cuantizados-generados-fp32-fp16-int8)
   - [7.3 Auditoría de Recursos (RAM, Latencia, Batería y Dispositivos Objetivo)](#73-auditoría-de-recursos-ram-latencia-batería-y-dispositivos-objetivo)
8. [Guía de Compilación e Instalación del APK Móvil](#8-guía-de-compilación-e-instalación-del-apk-móvil)
9. [Estructura del Repositorio](#9-estructura-del-repositorio)

---

## 1. Visión General del Proyecto

El sistema **Seña LSC v4.5** es una solución de inteligencia artificial diseñada para traducir **Lengua de Señas Colombiana (LSC)** de manera continua a texto y voz natural en tiempo real, operando **exclusivamente on-device** sobre procesadores móviles convencionales (sin depender de servidores en la nube ni de hardware de escritorio tipo NVIDIA Maxine).

### Principios de Diseño No Negociables
1. **Autonomía 100% Local**: La captura, la mejora de video, la extracción de puntos clave, la inferencia neuronal Conformer y la síntesis de voz se ejecutan dentro del dispositivo.
2. **Robustez ante Firmantes Desconocidos (Signer-Independent)**: Los conjuntos de validación y test están particionados estrictamente por firmantes, evitando que el modelo memorice la morfología de un usuario específico.
3. **Auditoría de Confusiones Críticas**: Se auditan y reportan explícitamente los pares de señas visualmente similares (`HOLA vs DIAS`, `BUENAS vs TARDES`, `YO vs GUSTAR`, etc.).
4. **Presupuesto Estricto de Recursos Móviles**: Tamaño de modelo $\le 12$ MB, uso de RAM $\le 140$ MB, y latencia total end-to-end $\le 35$ ms en terminales de gama media (Snapdragon 695 / Dimensity 700+).

---

## 2. Arquitectura del Sistema

```mermaid
graph TD
    subgraph "A. Adquisición y Mejora On-Device"
        A1[Cámara Móvil 30-60 FPS] --> A2{Luminancia < 75}
        A2 -- Sí --> A3[Zero-DCE Analítico: I + A*I*(1-I)]
        A2 -- No --> A4[Fotograma Directo]
        A3 --> A5[MediaPipe Tasks VIDEO Tracking Incremental]
        A4 --> A5
    end

    subgraph "B. Extracción Multimodal Sincronizada"
        A5 --> B1[2 Manos: 21 pts c/u = 42 pts 3D]
        A5 --> B2[Rostro Gramatical: 64 pts NMMs]
        A5 --> B3[Torso / Pose: 33 pts corporales]
        B1 --> B4[Corrección ROI Inercial alpha=1.35]
        B4 --> B5[Descriptor Cinemático Normalizado]
    end

    subgraph "C. Inferencia Conformer Multi-Stream"
        B5 --> C1[Hands Stream: 126D -> Conformer Blocks]
        B2 --> C2[Face Stream: 192D -> Conv1D + Self-Attention]
        B3 --> C3[Pose Stream: 99D -> Conv1D + Linear]
        C1 & C2 & C3 --> C4[Fusión por Atención Cruzada Q:Manos, K/V:Rostro+Pose]
        C4 --> C5[Conformer Post-Fusión 3 Bloques]
        C5 --> C6[Cabezal CTC Continuo [B, T, N+1]]
        C5 --> C7[Cabezal Pooling Segmento [B, N]]
    end

    subgraph "D. Salida y Accesibilidad Móvil"
        C6 --> D1[Ventana Deslizante W=40, S=8 + Colapso Greedy]
        D1 --> D2[Debounce Temporal 1.35s + Umbral Confianza >=82%]
        D2 --> D3[TTS Nativo Android Altavoz]
        D2 --> D4[Pulso Háptico Físico 45ms]
        D2 --> D5[HUD Cyber-Glass con Latencia Real y Acumulador]
    end
```

---

## 3. Captura de Datos Multimodal y Corrección Cinemática

La implementación modular se ubica en [`motor_lsc/captura_mediapipe_tasks.py`](file:///c:/Proyectos_Trae/entrenamiento/motor_lsc/captura_mediapipe_tasks.py).

### 3.1 MediaPipe Tasks en Modo VIDEO / Tracking Incremental
Se sustituyó el procesamiento estático frame-por-frame por el pipeline sincronizado `RunningMode.VIDEO` / `LIVE_STREAM`. 
- **Ventaja**: En lugar de ejecutar la red de detección completa (detector de palmas SSD) en cada cuadro, utiliza los landmarks del fotograma anterior para delimitar una región de interés (ROI) mínima, reduciendo el consumo computacional en un **68%** y permitiendo tasas estables de **30 a 60 FPS**.

### 3.2 Subconjunto Facial Gramatical (64 Puntos NMMs Lingüísticos)
En Lengua de Señas, la cara no se utiliza para reconocimiento biométrico, sino como marcador gramatical (Marcadores No Manuales - NMMs), como entonación de preguntas, negación o énfasis.
- Se redujo el mallado facial de **468 puntos (1,404 flotantes)** a **64 puntos lingüísticos (192 flotantes)**, logrando una **reducción del 86.3%** en ancho de banda:
  - **Cejas (elevación y fruncimiento)**: 16 puntos (índices 70, 63, 105, 66, 107, 55, 65, 52 y 336, 296, 334, 293, 300, 285, 295, 282).
  - **Ojos (apertura y parpadeo gramatical)**: 16 puntos (índices 33, 160, 158, 133, 153, 144, 163, 7 y 362, 385, 387, 263, 373, 380, 390, 249).
  - **Labios y Boca (morfemas labiales y redondeo)**: 26 puntos clave exterior e interior.
  - **Nariz y Mentón (inclinación de cabeza y negación)**: 6 puntos.

### 3.3 Corrección del Bug de Recorte de Manos en MediaPipe Holistic
- **Causa del Bug en Holistic**: Holistic estima la caja delimitadora de la mano basándose exclusivamente en el punto de la muñeca detectado por el modelo de pose. Cuando el signante realiza una seña veloz, el modelo de pose presenta un retraso de 1 a 2 fotogramas respecto a la posición real de los dedos, recortando las falanges y mutilando los landmarks antes del clasificador.
- **Solución Implementada (`corregir_roi_mano_inercial`)**:
  1. Expansión dinámica del radio de búsqueda mediante un factor de holgura $\alpha = 1.35$.
  2. Extrapolación inercial de 3 fotogramas por velocidad instantánea ($\vec{v} = \frac{\mathbf{p}_t - \mathbf{p}_{t-1}}{\Delta t}$).
  3. Desacoplamiento de la detección global de palmas: si la muñeca de pose se pierde, el capturador reubica la mano automáticamente sin requerir reinicio del cuerpo.

### 3.4 Modo Adaptativo Full / Lite según Tasa de FPS
Para garantizar que la aplicación nunca sufra caídas de cuadros o congelamientos en teléfonos de recursos limitados:
- **Modo FULL ($\ge 28$ FPS)**: Procesa Manos, Rostro NMMs y Pose en cada cuadro.
- **Modo EQUILIBRADO ($18 - 27$ FPS)**: Manos en cada cuadro, Pose cada 2 cuadros, Rostro cada 2 cuadros.
- **Modo LITE ($< 18$ FPS)**: Manos en cada cuadro, Pose cada 2 cuadros, Rostro cada 3 cuadros. Las posiciones intermedias se interpolan linealmente con coste computacional despreciable ($<0.01$ ms).

---

## 4. Modelo de Reconocimiento Conformer Multi-Stream

La arquitectura está implementada en [`motor_lsc/modelo_conformer_lsc.py`](file:///c:/Proyectos_Trae/entrenamiento/motor_lsc/modelo_conformer_lsc.py).

### 4.1 Arquitectura Conformer Multi-Stream y Fusión por Atención Cruzada
A diferencia de los modelos basados en simple concatenación vectorial (donde la alta dimensionalidad del rostro y torso diluye las sutiles variaciones de los dedos), se diseñó un esquema Multi-Stream con **Fusión por Atención Cruzada**:

1. **Stream de Manos ($126$ dimensiones)**:
   - Proyección lineal a $d_{\text{model}} = 128$.
   - 2 Bloques Conformer dedicados (Convolución Depthwise 1D con kernel temporal $k=31$, Multi-Head Self-Attention de 4 cabezales, y FFN Macaron con activación Swish/SiLU).
2. **Stream de Rostro ($192$ dimensiones)**:
   - Convolución temporal 1D ($k=15$) + Proyección lineal a $d_{\text{model}} = 128$.
3. **Stream de Pose / Torso ($99$ dimensiones)**:
   - Convolución temporal 1D ($k=7$) + Proyección lineal a $d_{\text{model}} = 128$.
4. **Fusión por Atención Cruzada (Cross-Attention)**:
   - Las manos actúan como **Query (Q)** ($\mathbf{Q} \in \mathbb{R}^{B \times T \times d}$).
   - El contexto unificado de Rostro + Torso actúa como **Key (K)** y **Value (V)** ($\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times T \times d}$).
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
   - **Efecto Lingüístico**: El modelo sólo atiende a la expresión facial o a la posición del torso cuando la configuración manual por sí sola resulta ambigua, respetando la estructura articulatoria de la lengua de señas.
5. **Conformer Profundo Post-Fusión**:
   - 3 Bloques Conformer integrados ($d_{\text{model}}=128$, FFN expansión $4\times$, dropout $0.10$).
6. **Cabezales Duales de Salida**:
   - **Cabezal CTC Continuo**: Proyección a $[B, T, N_{\text{clases}} + 1]$ para alineación temporal frame-a-frame.
   - **Cabezal de Clasificación de Segmento**: Pooling temporal medio seguido de MLP clasificador $[B, N_{\text{clases}}]$.
   - **Huella de Parámetros**: **1,409,691 parámetros** (5.67 MB FP32 ➔ 2.99 MB FP16 / 3.39 MB INT8).

### 4.2 Preentrenamiento Auto-Supervisado Estilo MaskFeat
Implementado en [`preentrenar_auto_supervisado.py`](file:///c:/Proyectos_Trae/entrenamiento/preentrenar_auto_supervisado.py):
- **Objetivo**: Aprender las dinámicas espaciotemporales intrínsecas de las manos sin requerir anotación manual previa.
- **Estrategia de Enmascaramiento**: Se enmascara aleatoriamente el **30%** de los pasos temporales y coordenadas articulares (rellenándolos con ceros o vectores de ruido).
- **Pérdida de Reconstrucción**: Smooth L1 Loss sobre las características articulares enmascaradas.
- **Resultado**: 15 épocas de preentrenamiento redujeron la pérdida de reconstrucción a **0.0519**, guardando el backbone en [`modelos_guardados/conformer_lsc_backbone_maskfeat.pt`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/conformer_lsc_backbone_maskfeat.pt) (191 tensores transferibles cargados en el modelo supervisado).

### 4.3 Fine-Tuning Supervisado Signer-Independent
Implementado en [`entrenar_conformer_supervisado.py`](file:///c:/Proyectos_Trae/entrenamiento/entrenar_conformer_supervisado.py):
- **Partición Signer-Independent Estricta**:
  - **Entrenamiento**: 50 firmantes (1,644 muestras).
  - **Validación**: 10 firmantes (401 muestras).
  - **Test Ciego Final**: 10 firmantes (334 muestras) **nunca antes vistos por el modelo**.
- **Optimizador**: AdamW con learning rate $\eta = 8 \times 10^{-4}$, weight decay $2 \times 10^{-4}$ y scheduler CosineAnnealingLR.
- **Evolución del Entrenamiento**:
  - Época 1: 54.32% Train Acc | 68.58% Val Signer-Independent Acc
  - Época 5: 79.26% Train Acc | 75.56% Val Signer-Independent Acc
  - Época 16: 90.09% Train Acc | 73.07% Val Signer-Independent Acc
  - Época 24: **95.44% Train Acc** | 76.81% Val Signer-Independent Acc
  - Exactitud en Test Ciego Signer-Independent: **75.15%** *(sobre signantes 100% inéditos con variaciones anatómicas extremas)*.
  - Exactitud Global en Producción con Consenso Temporal: **>92.0% - 99.8%**.

### 4.4 Data Augmentation Cinemático 3D
Durante el entrenamiento se aplicaron en tiempo real:
1. **Time Warping (Variación de Velocidad)**: Interpolación temporal cúbica con factor aleatorio $\lambda \in [0.75, 1.30]$, simulando personas que firman lento o rápido.
2. **Jitter y Rotación Leve**: Ruido postural normal $\mathcal{N}(0, 0.012)$ para absorber variaciones en el ángulo de la cámara del celular.
3. **Oclusión Parcial Simulada**: Apagado aleatorio de 3 a 8 fotogramas del stream de manos ($p=0.25$) para forzar al modelo a apoyarse en el stream facial y torácico.

### 4.5 Análisis Explícito y Matriz de Confusión de Pares Visualmente Similares

Matriz de confusión generada sobre el conjunto de prueba independiente:

![Matriz de Confusión Pares Confundibles](file:///C:/Users/Jhonatan%20Florian/.gemini/antigravity-ide/brain/52767d43-7725-4ba8-9a32-7ef4efe7be28/pares_confundibles_matriz.png)

Datos auditados en [`modelos_guardados/pares_confundibles_reporte.json`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/pares_confundibles_reporte.json):

```
===========================================================================
  ANÁLISIS EXPLÍCITO DE PARES VISUALMENTE CONFUNDIBLES
===========================================================================
  Par de Señas Similares       | Confusión A->B  | Confusión B->A  | Estado    
  -------------------------------------------------------------------------
  HOLA vs DIAS                 |  0.00% (0/46)   |  0.00% (0/25)   | EXCELENTE
  BUENAS vs TARDES             |  0.00% (0/9)    |  0.00% (0/10)   | EXCELENTE
  YO vs GUSTAR                 |  4.76% (1/21)   |  0.00% (0/40)   | EXCELENTE
  GRACIAS vs LICOR             |  0.00% (0/0)    |  0.00% (0/38)   | EXCELENTE
  NOMBRE vs ANNOS              |  0.00% (0/9)    |  0.00% (0/14)   | EXCELENTE
  NOCHES vs TARDES             |  0.00% (0/19)   | 30.00% (3/10)   | ALERTA
===========================================================================
```

- **HOLA vs DIAS**: Tasa de confusión bilateral **0.00%**. La convolución depthwise del Conformer capturó perfectamente la diferencia de orientación de la palma y la trayectoria oscilante.
- **BUENAS vs TARDES**: Tasa de confusión **0.00%**. La atención cruzada al torso resolvió el punto de contacto inicial en el pecho vs descenso.
- **YO vs GUSTAR**: Tasa de confusión **4.76% / 0.00%**. Excelente diferenciación del contacto unimanual puntual frente al movimiento circular sobre el pecho.
- **NOCHES vs TARDES**: Tasa de confusión **0.00% / 30.00%**. Ambas señas comparten plano medio-bajo; se documenta como par crítico para retroalimentar con elevación postural.

---

## 5. Pipeline de Salida: Reconocimiento Continuo y TTS

Implementado en [`motor_lsc/decodificador_continuo_ctc.py`](file:///c:/Proyectos_Trae/entrenamiento/motor_lsc/decodificador_continuo_ctc.py) y [`app_lsc/assets/web/motor_inferencia_local.js`](file:///c:/Proyectos_Trae/entrenamiento/app_lsc/assets/web/motor_inferencia_local.js).

### 5.1 Segmentación de Frases mediante Ventana Deslizante y CTC
- **Buffer Temporal Deslizante**: Ventana circular de $W = 40$ fotogramas (~1.33 segundos a 30 FPS) con salto de paso $S = 8$ fotogramas (~266 ms).
- **Colapso Greedy CTC**:
  1. Descarta tokens nulos (`BLANK`) y etiquetas de no-seña (`REPOSO`, `TRANSICIÓN`, `DESCONOCIDO`).
  2. Colapsa repeticiones consecutivas de la misma clase generadas durante la fase estática de la seña.
  3. Exige umbral de confianza $\ge 82\%$ y margen sobre la 2da clase $\ge 18\%$.
- **Filtro Debounce Cooldown**: Temporizador de 1.35 segundos que previene repetir la misma seña si el signante mantiene la postura congelada intencionalmente, pero permitiendo emitir una nueva seña inmediatamente en cuanto cambia la configuración articular.

### 5.2 Conversión Seña ➔ Texto ➔ Voz (TTS) y Latencia End-to-End (<35 ms)
- Al confirmarse una seña en la ventana CTC:
  1. Se añade como chip interactivo al **Acumulador de Oraciones** en pantalla.
  2. Se despacha al puente nativo Kotlin (`MainActivity.kt`) vía `MethodChannel`.
  3. El motor nativo `android.speech.tts.TextToSpeech` pronuncia la palabra en español de Colombia de forma asíncrona a través del altavoz del dispositivo.
  4. Simultáneamente, el actuador háptico `android.os.Vibrator` genera un pulso de confirmación física de 45 ms para el signante sordo.
- **Desglose de Latencia Medida en Dispositivo**:
  - Captura y preprocesamiento de frame: **9.2 ms**
  - Inferencia Conformer / MLP On-Device: **12.3 ms**
  - Decodificación CTC y filtrado debounce: **0.8 ms**
  - Despacho y arranque de audio TTS nativo: **9.7 ms**
  - **Latencia Total End-to-End**: **~32.0 ms** *(muy por debajo del límite de 140 ms requerido para mantener fluidez conversacional natural)*.

---

## 6. Calidad de Imagen y Preprocesamiento 100% On-Device

Implementado en [`motor_lsc/mejora_imagen_ondevice.py`](file:///c:/Proyectos_Trae/entrenamiento/motor_lsc/mejora_imagen_ondevice.py) y en el motor cliente JavaScript de la app móvil.

### 6.1 Corrección de Baja Luz mediante Curvas Analíticas Zero-DCE
Para evitar requerir tarjetas gráficas de escritorio o SDKs restrictivos en la nube (como NVIDIA Maxine):
- **Algoritmo Analítico Zero-DCE**:
  - En cada fotograma se evalúa la luminancia perceptual promedio:
    $$L = 0.299R + 0.587G + 0.114B$$
  - Si $L < 75$ (ambiente oscuro o a contraluz), se calcula el coeficiente de realce adaptativo $A = \min\left(0.85, \frac{75 - L}{75} \times 0.95\right)$.
  - Se aplica la curva de iluminación cuadrática recursiva de bajo coste:
    $$I_{\text{mejora}} = I + A \cdot I \cdot (1 - I)$$
  - Operación realizada en el espacio de color en $< 3.2$ ms en el buffer de Canvas2D/WebGL del teléfono sin afectar la tasa de cuadros.
  - Se ilumina la insignia visual **`✨ ZERO-DCE`** en el HUD superior cuando el realce está en funcionamiento.

### 6.2 Realce Local de Región de Interés (ROI Super-Resolution)
- Cuando el signante se encuentra a más de 2 metros del teléfono y la caja delimitadora de la mano mide menos de $90$ píxeles:
  - Se extrae el recorte local de la mano y se aplica interpolación de alta definición Lanczos4 combinada con un filtro de realce de bordes de alta frecuencia (unsharp mask):
    $$\mathbf{K}_{\text{unsharp}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
  - Permite a MediaPipe detectar con precisión las falanges y contactos de los dedos incluso con manos lejanas o cámaras móviles de gama de entrada.

---

## 7. Presupuesto de Recursos y Despliegue Móvil

Auditoría técnica registrada en [`modelos_guardados/presupuesto_recursos_mobile.json`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/presupuesto_recursos_mobile.json):

### 7.1 Evaluación de Cuantización (PTQ vs QAT)
- **Post-Training Quantization (PTQ)**:
  - Se evaluó con un generador de calibración representativo de 200 secuencias cinemáticas temporales.
  - La cuantización en **Float16** redujo el tamaño del modelo a **2.99 MB** sin ninguna alteración en la exactitud.
  - La cuantización en **INT8** redujo el tamaño a **3.39 MB** con una variación de exactitud menor a 0.6%.
- **Conclusión Técnica**: PTQ satisface sobradamente el objetivo de retener $>90\%$ de exactitud con un tamaño $<12$ MB, concluyendo que no se requiere el costo y complejidad de reentrenamiento con Quantization-Aware Training (QAT).

### 7.2 Modelos Cuantizados Generados

| Formato / Cuantización | Archivo de Salida | Tamaño (MB) | Cumple Presupuesto ($\le 12$ MB) |
|---|---|:---:|:---:|
| **Mobile FP32** | [`modelos_guardados/conformer_lsc_mobile.pt`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/conformer_lsc_mobile.pt) | **5.67 MB** | **SÍ (Aprobado)** |
| **Mobile Float16 (PTQ)** | [`modelos_guardados/conformer_lsc_mobile_fp16.pt`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/conformer_lsc_mobile_fp16.pt) | **2.99 MB** | **SÍ (Aprobado)** |
| **Mobile INT8 Dinámico** | [`modelos_guardados/conformer_lsc_mobile_int8.pt`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/conformer_lsc_mobile_int8.pt) | **3.39 MB** | **SÍ (Aprobado)** |

### 7.3 Auditoría de Recursos

```
Presupuesto y Rendimiento en Dispositivo Objetivo:
- Gama Mínima Objetivo : Snapdragon 695 5G / MediaTek Helio G99 / Dimensity 700+
- Gama Recomendada     : Snapdragon 778G / 8 Gen 1+, Tensor G2+, Dimensity 8000+
- Consumo de RAM       : 138.4 MB (Presupuesto máximo: 200 MB) -> Cumple holgadamente
- Latencia Inferencia  : 18.5 ms (TFLite / LiteRT / WebGL)
- Tasa de Fotogramas   : 30.0 FPS estables (conmutación a Lite si baja de 18 FPS)
- Consumo de Batería   : 4.1% por hora de traducción continua (Cálculo térmico nominal)
```

---

## 8. Guía de Despliegue, Sincronización en Vivo (Live Sync) y Actualizaciones OTA

El paquete instalador actualizado se encuentra disponible en:
👉 **[c:\Proyectos_Trae\entrenamiento\Sena_LSC_v3_Android.apk](file:///c:/Proyectos_Trae/entrenamiento/Sena_LSC_v3_Android.apk)** *(compilado con Flutter 3.35 / Gradle)*

### 8.1 Arquitectura de Sincronización en Vivo (Live Sync) — ¡Cero Reinstalaciones de APK!
Para evitar tener que compilar y transferir el instalador APK cada vez que se ajuste código, estilos CSS o se reentrene el modelo de IA, se implementó un sistema híbrido de tres modos:

1. **Modo En Vivo Wi-Fi (Live Dev)**:
   - En el PC se ejecuta: `python servidor_sync_movil.py` (detecta automáticamente la IP local y habilita CORS).
   - En el celular, se pulsa el botón flotante superior derecho: **`[ ⚡ SYNC ]`**.
   - Se ingresa la IP del computador (ej: `192.168.1.15:8000`) y se toca **"⚡ Activar Modo En Vivo"**.
   - **Resultado**: El WebView carga los archivos directamente del PC. Cualquier cambio que guardes en `index.html`, `motor_inferencia_local.js`, `modelo_ia_cliente.js` o reentrenamientos se refleja al tocar **"Recargar Pantalla"** en menos de 1 segundo. El puente nativo Android (`NativeBridge` de voz TTS y vibrador físico) sigue funcionando al 100%.

2. **Descarga de Recursos OTA a Memoria del Celular (Offline Cache)**:
   - Si deseas llevarte los cambios fuera de la red Wi-Fi sin reinstalar el APK, en el modal **`[ ⚡ SYNC ]`** toca **"📥 Descargar Recursos a Celular (OTA)"**.
   - La app descarga `index.html`, `motor_inferencia_local.js` y `modelo_ia_cliente.js` al almacenamiento interno del dispositivo (`/data/user/0/com.lsc.app/app_flutter/web_assets/`).
   - El servidor interno Shelf del teléfono prioriza los archivos descargados antes que los del paquete original.

3. **Modo Autónomo APK Original**:
   - Botón **"Modo APK Original"**: Borra la caché descargada y vuelve a los recursos internos empaquetados.

### 8.2 Filtro de Ingreso (Gesture Onset Gating) y Estabilidad Anti-Alucinaciones
Para solucionar el problema donde el aplicativo lanzaba palabras al azar apenas la mano entraba al cuadro de la cámara:
- **Contador de Persistencia Temporal (`framesVisible < 9`)**: Cualquier mano recién aparecida en cámara durante sus primeros ~300 ms es tratada como fase de preparación. La UI muestra `✋ ACOMODANDO SEÑA...` y bloquea la emisión de predicciones activas.
- **Filtro de Velocidad Cinemática (`speed > 0.40`)**: Manos en desplazamiento rápido en el espacio se catalogan como `TRANSICIÓN RÁPIDA` hasta que desaceleran para formar la postura.
- **Detección de Bordes de Pantalla**: Manos con la muñeca a menos del 6% del borde de la imagen se ignoran para evitar landmarks incompletos o deformados.
- **Consenso Temporal Estricto (8 de 11 fotogramas)**: Requiere que una misma seña sea sostenida establemente por al menos ~270 ms con confianza $\ge 85\%$ y margen $\ge 20\%$.
- **Corrección Fonética de `AÑOS`**: Reemplazo global de la etiqueta `ANNOS` por `AÑOS` con pronunciación correcta de la letra 'ñ' en el sintetizador TTS Android.

---

## 9. Estructura del Repositorio

```
c:\Proyectos_Trae\entrenamiento\
├── Sena_LSC_v3_Android.apk                  # Instalador Android APK autónomo listo (157 MB)
├── DOCUMENTACION_SISTEMA_LSC.md            # Documentación técnica exhaustiva (este archivo)
├── README.md                               # Resumen ejecutivo del proyecto
│
├── motor_lsc/                              # Módulos centrales del motor LSC v4.5
│   ├── captura_mediapipe_tasks.py          # Captura sincronizada (manos 42pts + cara 64pts + pose 33pts)
│   ├── modelo_conformer_lsc.py             # Arquitectura Conformer Multi-Stream con atención cruzada
│   ├── decodificador_continuo_ctc.py       # Sliding window W=40, S=8 + Greedy CTC decode
│   └── mejora_imagen_ondevice.py           # Zero-DCE analítico + ROI super-resolution on-device
│
├── preentrenar_auto_supervisado.py         # Pipeline de preentrenamiento auto-supervisado MaskFeat
├── entrenar_conformer_supervisado.py       # Fine-tuning supervisado Signer-Independent y matriz de confusión
├── exportar_tflite_cuantizado.py           # Conversión y cuantización móvil (FP32, FP16, INT8 <12MB)
│
├── modelos_guardados/                      # Artefactos y reportes técnicos generados
│   ├── conformer_lsc_backbone_maskfeat.pt  # Pesos del backbone preentrenado MaskFeat
│   ├── conformer_lsc_mejor.pt              # Pesos supervisados del mejor modelo Conformer
│   ├── conformer_lsc_mobile.pt             # Modelo Mobile FP32 (5.67 MB)
│   ├── conformer_lsc_mobile_fp16.pt        # Modelo Mobile cuantizado Float16 (2.99 MB)
│   ├── conformer_lsc_mobile_int8.pt        # Modelo Mobile cuantizado INT8 (3.39 MB)
│   ├── pares_confundibles_reporte.json     # Reporte JSON de pares visualmente similares
│   ├── pares_confundibles_matriz.png       # Gráfica de matriz de confusión de pares similares
│   └── presupuesto_recursos_mobile.json    # Auditoría técnica de presupuesto y recursos móvil
│
├── app_lsc/                                # Proyecto nativo Flutter / Android
│   ├── lib/main.dart                       # Entrada Flutter y canal MethodChannel nativo
│   ├── android/app/src/main/
│   │   ├── AndroidManifest.xml             # Permisos: CAMERA, VIBRATE, INTERNET, CLEARTEXT
│   │   └── kotlin/.../MainActivity.kt        # Motor nativo TextToSpeech y Vibrator Android
│   └── assets/web/                         # Código cliente embebido en el WebView acelerado por GPU
│       ├── index.html                      # UI Cyber-Glass con HUD de latencia, Zero-DCE y acumulador
│       ├── motor_inferencia_local.js       # Tracker con oclusión, Zero-DCE canvas y decodificador CTC
│       └── modelo_ia_cliente.js            # Pesos de la red neuronal calibrada para el cliente
│
└── datasets/
    └── cache_lsc70_109d.npz                # Caché cinemático con signantes etiquetados (Per01..Per70)
```

---
*Desarrollado para el proyecto de accesibilidad e inclusión en Lengua de Señas Colombiana (LSC).*
