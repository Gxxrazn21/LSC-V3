# Sistema de Reconocimiento de Lengua de Señas Colombiana (LSC)

## Arquitectura Ligera Basada en Vectores, Cuadrantes y NLP Offline

Este sistema permite el reconocimiento fluido de señas LSC en tiempo real sobre cualquier computador o dispositivo móvil, **sin requerir GPU dedicada ni APIs en la nube**.

---

## Flujo del Pipeline en Tiempo Real

```
                    Cámara en Vivo (OpenCV / Navegador)
                                  │
                                  ▼
                    Volteo horizontal tipo espejo
                                  │
                                  ▼
                     MediaPipe Hands + Pose
             (21 puntos 3D por mano + hombros / nariz)
                                  │
                                  ▼
                          ¿Hay mano detectada?
                          ┌── Sí ───────────────┐
                          │                     │
                          ▼                     ▼
          Normalización Geométrica 3D    Clasificación de Cuadrante
             (Invariante a escala,          (CABEZA_ROSTRO, PECHO_TORSO,
           traslación y desenfoque)          ESPACIO_NEUTRO, LATERAL)
                          │                     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                      Base de Datos Vectorial LSC
                       (Búsqueda Similitud Coseno
                        + Ponderación de Cuadrante)
                                 < 1 ms
                                     │
                                     ▼
                         ¿Similitud >= Umbral (70%)?
                          ┌── Sí ───────────────┐
                          │                     │
                          ▼                     ▼
              Acumulador de Glosas       Candidatos Top-3
              (Filtro de estabilidad)    (Para HUD y feedback)
                          │
                          ▼
            ¿Manos en reposo / silencio (1.2s)?
            ┌── Sí ─────────────────────────────┐
            │                                   │
            ▼                                   ▼
  Ensamblador de Frases NLP               Continuar acumulando
  (LSC -> Español Natural)                      glosas
            │
            ▼
   Motor TTS Local (pyttsx3)
   Pronunciación no bloqueante
```

---

## Cuadrantes Espaciales (*Signing Space*)

En LSC, el punto de articulación define el significado de la seña. El sistema segmenta el espacio en 4 zonas:

| Cuadrante | Región Corporal | Ejemplos de Señas | Color HUD |
|---|---|---|---|
| **CABEZA_ROSTRO** | Frente, mejillas, mentón, orejas | *Saber, Pensar, Amarillo, Licor* | Naranja |
| **PECHO_TORSO** | Pecho, hombros, esternón | *Yo, Sentir, Amor, Familia, Nombre* | Rosa |
| **ESPACIO_NEUTRO** | Frente al cuerpo en el centro | *Hola, Bienvenido, Ayudar, Casa* | Verde |
| **LATERAL_BAJO** | Fuera del eje central o reposo | Reposo / Transición | Gris |

---

## Comandos Principales

### 1. Reconocimiento en Vivo por Cámara con HUD
```bash
python predecir_vivo.py

# Opciones personalizadas:
python predecir_vivo.py --camera 1 --umbral 0.75
```

### 2. Iniciar Servidor Web / API para la App 'Seña Android'
```bash
python -m motor_lsc.servidor_app
# Abre en el navegador: http://localhost:8000
```

### 3. Recompilar / Actualizar Base de Vectores
```bash
python -m motor_lsc.generador_referencias
```

### 4. Ejecutar Suite de Pruebas
```bash
python test_motor_lsc.py
```
