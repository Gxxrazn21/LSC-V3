# Arquitectura operativa LSC v7

Este documento es la referencia de producción. El modelo vigente en la app es
el MLP 109D exportado a JavaScript; los documentos y prototipos Conformer se
mantienen únicamente como investigación histórica, no como descripción de la
APK actual.

## Flujo de datos y entrenamiento

```text
Firmante (seudónimo + consentimiento)
  -> capturar_senas.py
  -> datasets/capturado/manifest.jsonl + .npy 109D
  -> preparar_dataset_lsc.py (checksum, forma, nitidez, consentimiento)
  -> datasets/curated/*.npz (X, y, signer_id, session_id)
  -> entrenar_modelo_lsc_reproducible.py
  -> métricas OOF por firmante + artefactos candidatos v7
  -> revisión humana y pruebas en teléfonos
  -> web/modelo_ia_cliente.js + publish_web_release.py
```

Las métricas que deciden una publicación son `oof_accuracy` y `oof_macro_f1`
con protocolo `stratified_group_by_signer`. La exactitud sobre el conjunto de
entrenamiento no es una métrica de publicación.

## Captura de datos

Cada sesión exige consentimiento y usa un seudónimo estable del firmante. Se
registran sesión, dispositivo, iluminación, nitidez, versión del extractor y
checksum. Nunca se deben almacenar nombres, documentos de identidad o vídeo
sin una política de consentimiento separada.

Ejemplo:

```powershell
python capturar_senas.py --signer-id s014 --session-id s014-interior-01 `
  --device-id pixel-7 --lighting interior-led --consent --auto --muestras 30
python preparar_dataset_lsc.py
python entrenar_modelo_lsc_reproducible.py
```

El conjunto de test debe contener firmantes completos que no aparezcan en
entrenamiento. Para cada nueva clase, capturar variación de mano dominante,
distancia, ángulo, iluminación y al menos dos sesiones por firmante.

## Frontend y release móvil

`web/` es la única fuente editable del frontend. `docs/`, `estilo/` y
`app_lsc/assets/web/` son destinos generados por:

```powershell
python scripts/publish_web_release.py --version 7.0.0 --min-shell-version 6.2.0
```

El comando produce `release-manifest.json` con hashes SHA-256. La aplicación
descarga el release en staging, verifica cada archivo y cambia el puntero de
la versión activa solo si el conjunto completo es válido. La opción
**Restaurar APK** elimina ese puntero y vuelve a los assets embebidos.

## Criterios para publicar una versión

1. Validación por firmante sin fugas y revisión de matriz de confusión.
2. Prueba manual en al menos tres teléfonos Android de distintas gamas.
3. Prueba de cámara: permiso denegado, cámara ocupada, baja luz y orientación.
4. Prueba OTA: descarga correcta, red interrumpida y restauración a APK.
5. `scripts/publish_web_release.py` ejecutado y cambios revisados antes del
   commit.
