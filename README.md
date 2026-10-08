# Gestual Vision — Traductor de Lengua de Señas Colombiana (LSC)

App Android (Flutter + WebView) que reconoce señas de LSC con la cámara del
celular. La detección de manos (MediaPipe) y la red neuronal corren en el
propio teléfono.

- **APK:** `Gestual_Vision_v8.3.0.apk` (Android)
- **Versión:** 8.3.0

## Qué reconoce

47 señas, agrupadas en los modos de la app:

| Modo | Señas |
|---|---|
| Palabras (10) | AÑOS, BUENAS, DIAS, GUSTAR, HOLA, LICOR, NOCHES, NOMBRE, TARDES, YO |
| Abecedario (27) | A–Z y Ñ |
| Números (10) | 1, 4, 5, 6, 7, 8, 9, 10, MIL, MILLON |

Además está la clase REPOSO (manos abajo o sin intención de señar). El
constructor de frases une BUENAS + DIAS/TARDES/NOCHES en "Buenos días",
"Buenas tardes" y "Buenas noches", y junta las letras deletreadas en una
palabra.

## Avatar que seña

En el catálogo de señas hay un avatar 3D (`web/avatar_lsc.glb`, generado desde
`Hombreblender/`). Al tocar una seña de la lista, el avatar la ejecuta. También
puedes escribir un texto: si la palabra está en el vocabulario, la seña
completa; si no, la deletrea letra por letra (las letras o números sin seña se
avisan).

El avatar no "sabe" las señas por el clasificador, que solo reconoce. Los
movimientos salen del dataset LSC70 (`scripts/generar_senas_avatar.py`):

- **Medición.** Las grabaciones se pasan por MediaPipe en 3D y todo se expresa
  en el marco del torso de la persona, no de la cámara.
- **Altura y lado** de codo y muñeca: se toman de la imagen 2D, porque el
  modelo 3D baja la muñeca de forma sistemática.
- **Profundidad:** sale del largo de los huesos; del 3D solo se usa hacia qué
  lado va.
- **Letras y números estáticos:** la forma de la mano es la mediana de 12
  personas y queda fija, sin temblor.
- **Señas con movimiento:** usan la trayectoria de una persona, suavizada.
- **Dedos:** se guardan como ángulos por articulación, con zona muerta para
  que los dedos estirados queden rectos, límites anatómicos y la punta
  acoplada a la articulación media.

`web/avatar_lsc.js` lo ejecuta así:

- **Brazo:** IK de dos huesos. El codo se elige para quedar cerca del dato,
  con la muñeca lo más recta posible y fuera del cuerpo.
- **Muñeca:** con límites (70°); el giro se reparte entre antebrazo y mano.
- **Colisiones:** manos y antebrazos no atraviesan el torso ni la cabeza
  (volúmenes medidos de la propia malla), y las dos manos no se cruzan.
- **Movimiento:** resortes críticamente amortiguados, que dan inercia natural
  sin rebotes. Al deletrear, la mano no baja entre letras.
- **Vida:** parpadeo y respiración.

HOLA usa la animación hecha a mano del `.glb`.

- **Corrección manual:** solo el puño de la A, que MediaPipe no resuelve.
  Antes de corregir otras se verificó con las fotos (por ejemplo, en LSC la U
  es índice + meñique, no índice + medio).
- **Verificación automática:** `node tests/e2e/diagnostico_avatar.js` mide, en
  las 47 señas:
  - **Penetración en el cuerpo:** máximo 2,5 cm, y es la animación hecha a mano.
  - **Temblor de dedos en señas sostenidas:** 0°.
  - **Flexión de muñeca:** como máximo 70°.
- **Revisión visual:** `tests/e2e/capturas_avatar.js` y `tira_avatar.js`
  comparan con la foto de referencia y muestran la seña cuadro a cuadro.
  Conviene que una persona que sepa LSC lo revise.
- **Rendimiento:** simular un cuadro cuesta unos 0,2 ms. Con el catálogo
  abierto se pausan la cámara y el reconocimiento.
- **Párpados:** el modelo los trae sin color de piel; la app los pinta al
  cargar (el `.glb` original no se modifica).

## Qué tan bien funciona (medido honestamente)

El modelo se evaluó con **personas que nunca vio durante el entrenamiento**
(GroupKFold por persona, 70 personas del dataset LSC70, sin datos sintéticos):

| Escenario (solo señas, sin contar REPOSO) | Acierto por cuadro |
|---|---|
| Modo Palabras | 80,4 % |
| Modo Abecedario | 84,3 % |
| Modo Números | 86,8 % |
| Modo Todo (47 señas a la vez) | 78,6 % |

En la app la seña solo se confirma cuando se sostiene varios cuadros, así que la
experiencia real suele ser mejor que el acierto por cuadro. Si los hombros no
salen en cámara, el acierto baja unos 10 puntos; por eso la app pide alejarse
cuando no los ve.

### Validador de señas parecidas

Algunas señas solo se diferencian por el movimiento (N/Ñ, I/J, DÍAS/NOCHES) o
por un detalle fino de la mano (1/6, 4/9, donde el 6 y el 9 son el 1 y el 4
con las puntas dobladas). Cuando la app confirma una seña de uno de estos
grupos, una segunda etapa revisa los últimos 1,5 s: combina las
probabilidades medias del grupo con el movimiento de la muñeca, el índice y el
meñique, y elige dentro del grupo.

Los grupos salen de la matriz de confusión. Un validador solo se activa si
mejora la precisión con personas no vistas (por secuencia completa):

| Grupo | Sin validador | Con validador |
|---|---|---|
| N / Ñ | 65,9 % | 94,2 % |
| 1 / 6 | 77,9 % | 91,4 % |
| AÑOS / DÍAS / NOCHES / TARDES | 82,2 % | 93,5 % |
| I / J / Y | 88,1 % | 92,4 % |
| G / H / R | 91,9 % | 95,2 % |
| 4 / 9 | 90,0 % | 92,1 % |

F/L, S/Z y C/E se descartaron porque el validador no mejoraba. W/8 y V/7 solo
se confunden en el modo Todo, porque una es letra y la otra número; en su
propio modo no compiten.

**Limitaciones conocidas.** Los rasgos de movimiento se aprendieron con
secuencias de 6 cuadros del dataset; en la app se calculan sobre la ventana de
1,5 s anterior a la confirmación. Conviene validarlo con usuarios reales.
Señar lento y sostener el final de la seña ayuda.
Señas como GRACIAS o BIEN no están incluidas porque no hay datos de varias
personas. Para añadirlas, ver "Agregar señas nuevas".

**Conexión.** Los modelos de MediaPipe (Hands y Pose) se descargan del CDN
jsdelivr la primera vez que abres la app, así que esa primera apertura necesita
internet. Después la inferencia ocurre en el teléfono.

Las métricas completas por clase y la matriz de confusión están en
`resultados/` (`metricas_actuales.json`, `reporte_clasificacion.txt` y
`matriz_confusion.png`).

## Estructura

```
web/                     Fuente única del frontend (index.html, motor, modelo)
  motor_inferencia_local.js   Vector 109D, rasgos de movimiento, red neuronal, validador y filtros
  diseno_v8.css               Capa visual (paleta, tipografía Atkinson Hyperlegible, modo oscuro)
  avatar_lsc.js               Avatar 3D: carga el .glb y convierte las señas en rotaciones de huesos
  avatar_lsc.glb              Avatar optimizado (texturas WebP 1024 px, 4.3 MB)
  senas_avatar.json           Movimientos de las 47 señas para el avatar (generado)
  three.module.min.js, GLTFLoader.js, BufferGeometryUtils.js   three.js r169 local (sin internet)
Hombreblender/           Fuente del avatar en Blender (.blend, .glb original, scripts)
  modelo_ia_cliente.js        Pesos del modelo (generado, no editar)
docs/ estilo/            Copias publicadas (GitHub Pages y OTA). No editar a mano.
app_lsc/                 App Flutter; assets/web es copia publicada de web/
scripts/
  extraer_landmarks_lsc70.py  Imágenes LSC70 -> landmarks crudos (MediaPipe)
  vectorizar_landmarks.js     Landmarks -> vectores 109D con el MISMO código de la app
  publish_web_release.py      Copia web/ a docs/, estilo/ y app_lsc/assets/web + manifiesto OTA
  analizar_confusiones.py     Compara el modelo con expertos por grupo de señas parecidas
  generar_senas_avatar.py     Dataset LSC70 -> movimientos del avatar (web/senas_avatar.json)
entrenar_modelo_lsc.py   Entrenamiento con validación por persona y exportación
capturar_senas.py        Captura de señas nuevas con la webcam
servidor_sync_movil.py   Sirve web/ en la red local para probar en el celular
tests/e2e/               Prueba de punta a punta en Chrome (MediaPipe JS + modelo)
```

## Reentrenar el modelo

Requiere `datasets/LSC70/` (LSC70W y LSC70AN), Python con
`requirements.txt` y Node.js.

```bash
python scripts/extraer_landmarks_lsc70.py      # ~15 min, genera datasets/landmarks_lsc70.jsonl
node scripts/vectorizar_landmarks.js           # genera datasets/vectores_lsc70_109d.json
python entrenar_modelo_lsc.py                  # valida por persona y exporta web/modelo_ia_cliente.js
python scripts/publish_web_release.py --version 8.1.1
cd app_lsc && flutter build apk --release
```

Los vectores se calculan con `web/motor_inferencia_local.js`, la misma función
que ejecuta la app. Así el modelo ve en el entrenamiento exactamente lo mismo
que verá en el celular. Si cambias la extracción de características en el
motor, vuelve a correr los pasos 2 y 3.

## Agregar señas nuevas

```bash
python capturar_senas.py --persona P01 --senas GRACIAS BIEN --muestras 30
python capturar_senas.py --persona P02 --senas GRACIAS BIEN --muestras 30
# ... repite con al menos 8–10 personas distintas
node scripts/vectorizar_landmarks.js
python entrenar_modelo_lsc.py
```

Las capturas van a `datasets/capturas_propias.jsonl`, que el vectorizador
incluye automáticamente. Para que la seña aparezca en un modo, agrégala a la
lista correspondiente (`PALABRAS`, `ABECEDARIO` o `NUMEROS`) en
`entrenar_modelo_lsc.py` y descríbela en `INFO_SENAS` en `web/index.html`.

## Prueba de punta a punta

```bash
python -m http.server 8766                     # desde la raíz, en otra terminal
npm --prefix tests/e2e install
node tests/e2e/probar_pipeline_navegador.js Per03,Per30   # señas por cuadro
node tests/e2e/probar_validador_navegador.js              # validador de señas parecidas
node tests/e2e/capturas_ui.js <carpeta> 360               # interfaz a 360 px, textos y avatar
PRIMER_PLANO=1 node tests/e2e/capturas_avatar.js <carpeta> A L Y   # primer plano de la mano del avatar
node tests/e2e/diagnostico_avatar.js                     # penetración, muñeca y temblor en las 47 señas
PASOS=8 node tests/e2e/tira_avatar.js <carpeta> NOCHES M   # la seña cuadro a cuadro
```

## Actualizaciones OTA

La app descarga `release-manifest.json` y los archivos de `web/` (con `estilo/`
como respaldo) desde la rama `main` de GitHub. Después de publicar con
`publish_web_release.py` y hacer push a `main`, las apps instaladas se
actualizan sin reinstalar el APK.
