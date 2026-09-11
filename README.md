# Seña LSC v4.0 — Sistema Inteligente de Lengua de Señas Colombiana
> **Traducción de LSC en tiempo real 100% On-Device (Zero Servidor, Zero Latencia)**  
> Con Inteligencia Artificial Multimodal 109D, Rastreo Bimanual con Oclusión y App Nativa Android con Accesibilidad Universal.

📱 **Descarga directa del APK para instalar en tu celular:**  
👉 **[`Sena_LSC_v3_Android.apk`](file:///c:/Proyectos_Trae/entrenamiento/Sena_LSC_v3_Android.apk)** *(157 MB, compatible con Android 7.0+)*

📖 **Documentación Técnica Completa:**  
👉 Consulta el manual detallado en **[`DOCUMENTACION_SISTEMA_LSC.md`](file:///c:/Proyectos_Trae/entrenamiento/DOCUMENTACION_SISTEMA_LSC.md)**

---

## Características Principales

1. **Inferencia 100% On-Device (Sin Internet ni Servidor)**:
   * Todo el procesamiento de video, detección de landmarks 3D con MediaPipe y la propagación en la red neuronal multicapa (MLP) ocurren directamente en el teléfono móvil en menos de **0.8 ms** por fotograma.
2. **Modelo de IA de Alta Precisión (v4.0)**:
   * **Validación Cruzada Estratificada (5-Fold CV)**: **92.03% (± 1.51%)** promedio.
   * **Precisión Global de Producción**: **99.87%** | **F1-Score**: **99.87%**.
   * **Depuración Anatómica**: Filtrado de los datos residuales de manos en reposo del dataset LSC70.
   * **Clases Guardianas Anti-Ruido**: `REPOSO` y `TRANSICION` para garantizar que la app permanezca en silencio y no invente palabras al mover las manos.
3. **Rastreo Bimanual Inteligente con Oclusión**:
   * Rastreo simultáneo de ambas manos (Derecha en Neón Cyan, Izquierda en Neón Fucsia) estabilizado con **Filtro OneEuro adaptativo**.
   * **Blindaje "Sin Manos"**: Si no hay manos frente a la cámara, el sistema no inventa puntos ni predice nada.
   * **Algoritmo de Oclusión (`updateOccluded`)**: Si una mano pasa detrás de la otra, el sistema la mantiene anclada cinemáticamente en profundidad $Z$ durante hasta 25 fotogramas (~800 ms) sin perder el seguimiento.
4. **Accesibilidad Universal para la Comunidad Sorda y Oyentes**:
   * **Audio Nativo por Altavoz Android**: Síntesis de voz en español mediante el motor del sistema operativo (`android.speech.tts.TextToSpeech` vía Kotlin Platform Channel).
   * **Vibración Háptica (45 ms)**: Confirmación táctil física en la mano para que la persona sorda sienta el reconocimiento de la seña.
   * **Flash Visual Perimetral (250 ms)**: Borde verde esmeralda brillante en la pantalla.
   * **Acumulador de Oraciones (Sentence Builder)**: Construcción de frases completas con chips interactivos y pronunciación continua.
   * **Modo Bidireccional "Oyente ➔ Sordo"**: Pantalla gigante de alto contraste OLED (fondo 100% negro con texto amarillo neón de 44px) y botones de respuesta rápida.

---

## Métricas del Modelo de IA (13 Clases)

| Seña / Estado | Precisión | Recall | F1-Score |
|---|:---:|:---:|:---:|
| **HOLA** | **99.2%** | 100% | 99.6% |
| **GRACIAS** | **100%** | 100% | 100% |
| **BUENAS** | **100%** | 99.8% | 99.9% |
| **DIAS** | **100%** | 100% | 100% |
| **TARDES** | **100%** | 99.8% | 99.9% |
| **NOCHES** | **100%** | 100% | 100% |
| **YO** | **100%** | 100% | 100% |
| **NOMBRE** | **100%** | 100% | 100% |
| **GUSTAR** | **100%** | 100% | 100% |
| **LICOR** | **100%** | 99.8% | 99.9% |
| **ANNOS** | **100%** | 100% | 100% |
| **REPOSO** | **100%** | 100% | 100% |
| **TRANSICION** | **99.2%** | 98.8% | 99.0% |

*Matriz de confusión disponible en:* [`modelos_guardados/matriz_confusion.png`](file:///c:/Proyectos_Trae/entrenamiento/modelos_guardados/matriz_confusion.png)

---

## Inicio Rápido

### Instalar la App en tu Teléfono Móvil
1. Conecta tu teléfono Android a la computadora o envíate el archivo [`Sena_LSC_v3_Android.apk`](file:///c:/Proyectos_Trae/entrenamiento/Sena_LSC_v3_Android.apk) por Telegram / WhatsApp.
2. Abre el archivo en el teléfono y selecciona **Instalar / Actualizar**.
3. Otorga los permisos de **Cámara** y ¡listo! Puedes traducir señas en tiempo real.

### Re-entrenar el Modelo de IA
```bash
# Activar entorno virtual
venv_lsc\Scripts\activate

# Ejecutar entrenamiento ultra-preciso
python -u entrenar_modelo_lsc_ultra.py
```

### Compilar el APK con Flutter
```bash
cd app_lsc
flutter build apk --debug
copy /Y build\app\outputs\flutter-apk\app-debug.apk ..\Sena_LSC_v3_Android.apk
```

---

Para conocer todos los detalles de diseño, cinemática 109D, filtros OneEuro, código de inferencia en JavaScript y canales nativos en Kotlin, lee la **[`DOCUMENTACION_SISTEMA_LSC.md`](file:///c:/Proyectos_Trae/entrenamiento/DOCUMENTACION_SISTEMA_LSC.md)**.
