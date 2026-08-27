# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sistema de Reconocimiento de Lengua de Señas Colombiana (LSC) — Motor Ligero basado en **Vectores de Landmarks 3D**, **Cuadrantes Espaciales de Signación**, **Búsqueda Vectorial por Similitud (<1ms en CPU)**, **Ensamblado de Frases LSC a Español Offline** y **Síntesis de Voz Local**.

## Environment Setup

```bash
# Instalar dependencias
pip install -r requirements.txt
```

Compatible con Python 3.10 - 3.12 (optimizado para Python 3.11).

## Common Commands

```bash
# 1. Ejecutar reconocimiento en vivo por cámara (OpenCV + HUD + TTS)
python predecir_vivo.py

# 2. Iniciar servidor FastAPI / WebSockets para la app 'Seña Android'
python -m motor_lsc.servidor_app

# 3. Compilar / Actualizar la base de vectores de referencia
python -m motor_lsc.generador_referencias

# 4. Ejecutar pruebas unitarias y de rendimiento
python test_motor_lsc.py

# 5. Ver configuración activa
python config.py
```

## Architecture

All configuration lives in `config.py` (which loads overrides from `.env`).

- `motor_lsc/extractor.py`: MediaPipe Hands + Pose landmark extractor with scale/translation/blur-invariant normalization.
- `motor_lsc/cuadrantes.py`: Spatial signing space classifier (`CABEZA_ROSTRO`, `PECHO_TORSO`, `ESPACIO_NEUTRO`, `LATERAL_BAJO`).
- `motor_lsc/base_vectores.py`: Vector search engine with Cosine Similarity and DTW sequence matching.
- `motor_lsc/generador_referencias.py`: Ingests LSC54, LSC70, and LSC50 to generate `modelos_guardados/base_senas_lsc.npz`.
- `motor_lsc/ensamblador_frases.py`: Offline LSC gloss to natural Spanish grammar mapper.
- `motor_lsc/tts_local.py`: Asynchronous local text-to-speech engine using pyttsx3.
- `motor_lsc/servidor_app.py`: FastAPI & WebSocket server for the web/mobile interface.
- `predecir_vivo.py`: Standalone camera live demo with real-time HUD.
