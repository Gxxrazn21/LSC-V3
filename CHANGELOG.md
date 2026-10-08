# CHANGELOG — Gestual Vision (Traductor LSC)

## [8.2.0] — 2026-10-07
### Avatar que seña
- El catálogo incluye un avatar 3D (Hombreblender/avatar_lsc.glb, optimizado de 13,5 a 4,3 MB) que ejecuta cualquiera de las 47 señas al tocarla.
- Campo "Escribe para señar": seña las palabras del vocabulario y deletrea el resto.
- Movimientos generados desde LSC70 (`scripts/generar_senas_avatar.py`): la grabación más representativa de cada seña, pasada por MediaPipe 3D y trasladada a los 44 huesos Mixamo. HOLA usa la animación hecha a mano. La A tiene una corrección manual del puño.
- three.js r169 incluido localmente, así que el avatar funciona sin internet. Mientras el catálogo está abierto se pausan la cámara y el reconocimiento.

### Textos
- Ninguna palabra se parte: la seña principal y la pantalla completa ajustan el tamaño de letra hasta que la palabra cabe entera (antes BUENAS se veía como "BUENA" + "S").
- Tamaño de texto del WebView fijo al 100 %, así la "letra grande" de Android no desborda la interfaz.
- Filtros Palabras / Letras / Números en el catálogo.


## [8.1.0] — 2026-10-06
### Modelo: validador de señas parecidas
- Nueva segunda etapa: cuando se confirma una seña de un grupo confundible, una regresión logística decide dentro del grupo con las probabilidades medias de los últimos 1,5 s y rasgos de movimiento (muñeca, índice y meñique, normalizados por los hombros).
- Los grupos se detectan solos desde la matriz de confusión y un validador solo se activa si mejora con personas no vistas: N/Ñ 65,9 → 94,2 %, 1/6 77,9 → 91,4 %, AÑOS/DÍAS/NOCHES/TARDES 82,2 → 93,5 %, I/J/Y 88,1 → 92,4 %, G/H/R 91,9 → 95,2 %, 4/9 90,0 → 92,1 %.
- Los rasgos de movimiento los calcula `extraerRasgosMovimiento` en el motor JS, la misma función en entrenamiento y en la app.

### Interfaz
- Rediseño: paleta Papel/Tinta/Cobalto con Maíz reservado para la confirmación, modo oscuro automático y tipografía Atkinson Hyperlegible empaquetada (funciona sin internet).
- La seña reconocida va en una tarjeta flotante sobre la cámara; al confirmar, la palabra "se estampa" con un subrayado maíz.
- Íconos SVG en lugar de emojis, control segmentado para las pestañas, hojas inferiores para el catálogo y los ajustes, textos en minúscula legible y nombres con tilde (DÍAS, Ñ, MILLÓN).
- La sincronización en la nube pasa a Ajustes, así la cabecera queda limpia.


## [8.0.0] — 2026-10-06
### Modelo
- Modelo nuevo entrenado con 70 personas reales del dataset LSC70: 47 señas (10 palabras, 27 letras y 10 números) más REPOSO.
- Validación honesta por persona (GroupKFold), solo señas y con personas nunca vistas: Palabras 80,4 %, Abecedario 84,3 %, Números 86,8 %, Todo 78,6 %. Se quitan las métricas anteriores (88 % / 99 %), que medían datos ya vistos en el entrenamiento.
- REPOSO se acota a 2.500 cuadros para que no infle la métrica ni sesgue la red.
- Los vectores de entrenamiento se calculan con el mismo `motor_inferencia_local.js` de la app (paridad exacta), y cada muestra se usa con ambas manos (espejo).
- Se retiran GRACIAS, BIEN y los saludos compuestos sintéticos, que tenían datos de una sola persona. Los saludos ahora se arman en el constructor de frases (BUENAS + DIAS → "Buenos días").

### App
- Con dos manos visibles, la mano en reposo ya no opaca a la que está señando.
- El detector de mano neutra ya no bloquea señas con la mano abierta (B, 4, 5) en Abecedario y Números.
- El filtro de "mano en regazo" ya no depende de la posición en el cuadro (solo de la distancia a los hombros).
- Aviso "Aléjate un poco: que se vean tus hombros" cuando la pose no detecta los hombros.
- OTA: un release descargado más viejo que el APK ya no lo tapa, y la nube solo instala versiones más nuevas que la activa.
- Los números de una cifra ya no se pegan como letras deletreadas.
- Contadores de modos, diccionario y versión salen del modelo cargado. Se elimina el aviso falso de actualización en instalaciones nuevas.
- Accesibilidad: zoom permitido, `aria-label` en botones de icono, región `aria-live` para la seña detectada y respuestas rápidas como botones reales. La fila de acciones ya no se desborda en pantallas angostas.

### Proyecto
- Pipeline reproducible: `scripts/extraer_landmarks_lsc70.py` → `scripts/vectorizar_landmarks.js` → `entrenar_modelo_lsc.py` → `scripts/publish_web_release.py`.
- Nuevo `capturar_senas.py` (landmarks crudos, por persona) y prueba E2E en `tests/e2e/`.
- Se elimina el motor Python legado (`motor_lsc/`, servidor FastAPI, Docker), los scripts de auditoría, los modelos y cachés viejos y la documentación desactualizada.

---

Registro cronológico de todas las mejoras de interfaz, animaciones, rendimiento y accesibilidad aplicadas.

---

## [Motion-Release] — 2026-09-18
### Objetivo
UI más dinámica, profesional y usable; 100% funcionalidad original intacta; 60fps, WCAG 2.1 `prefers-reduced-motion`, cross-browser Chrome/Firefox/Safari/Edge, y sin dependencias externas nuevas.

---

## 1. Sistema Motion — Tokens, Accesibilidad y Promoción GPU (CSS)
**Archivos afectados** (3 copias idénticas, MD5 `7DD69E3BB957C5106BECF014B7B10032`):
- `app_lsc/assets/web/index.html` — APK (servidor shelf local)
- `estilo/index.html`              — Origen OTA cloud (GitHub raw)
- `docs/index.html`                — GitHub Pages / fuente autoritativa

**Ubicación código**: `docs/index.html:L70-L120`
- Tokens de duración/easing uniformes para toda la app:
  - `--dur-micro: 150ms`, `--dur-fast: 220ms`, `--dur-std: 320ms`, `--dur-slow: 500ms`, `--dur-xl: 700ms`
  - `--stagger-step: 45ms`
  - `--ease-spring`, `--ease-bounce`, `--ease-in-out`, `--ease-out`
  - `--gpu-off` para reducir opcionalmente `will-change` en equipos antiguos
- **WCAG 2.1 prefers-reduced-motion global** `@media (prefers-reduced-motion: reduce)` en `L94-L104`:
  - Todas las `animation-duration: 0.001ms !important; animation-iteration-count: 1 !important`
  - Todas las `transition-duration: 0.001ms !important`
  - `scroll-behavior: auto`
- **Promoción de capa compositora** `will-change: transform, opacity` y `transform: translateZ(0)` + `backdrop-filter + -webkit-backdrop-filter` en `L112-L120`
- Correcciones keyframes GPU-safe que **antes animaban box-shadow** (trigger repaint por frame):
  - `logo3DFloat` (`L295-L305`) — ahora solo animar `transform: rotate3d() translate3d()`
  - `pulseConfirmed` (`L528-L552`) — split en `.viewport-card { transform: scale(1.025) }` + pseudo `::after { animation: pulseRing }` único paint por ciclo

---

## 2. 17 Nuevos @keyframes + Transiciones vistas/modales/hover (CSS)
**Ubicación código**: `docs/index.html:L1234-L1705`

### 2.1 Keyframes nuevos (solo `transform` + `opacity` — compositor-only)
| Keyframe | Uso | Líneas |
|---|---|---|
| `fadeInUp` / `fadeOutDown` | Entrada/salida vertical genérica | L1241-L1272 |
| `viewInLeft` / `viewInRight` | Cambio de tabs Traductor↔Bidi | L1275-L1308 |
| `viewOutLeft` / `viewOutRight` | Salida tabs opuesta | L1311-L1342 |
| `chipIn` | Entrada chip de oración con stagger | L1345-L1367 |
| `rippleExpand` | Efecto clic ripple (GPU scale+opacity) | L1370-L1392 |
| `orbitA` / `orbitB` | Loader orbit tricolor Colombia | L1395-L1444 |
| `successPop` | Micro-feedback bounce en acciones OK | L1447-L1469 |
| `shakeX` | Micro-feedback shake en errores | L1472-L1490 |
| `fullscreenZoomIn` | Zoom al abrir pantalla completa | L1493-L1510 |
| `pulseRing` | Glow anillo exterior confirmación | L1513-L1531 |
| `drawerInUp` / `drawerOutDown` | Modal bottom sheet traductor | L1534-L1566 |
| `modalBackdropIn` / `modalBackdropOut` | Overlay backdrop drawer+modal | L1569-L1591 |

### 2.2 Transiciones entre vistas + modales
- Tabs `setActiveTab`: `#tab-traductor` / `#tab-bidi` salen con `viewOut*` y entran con `viewIn*` + clase `.transitioning` visibility lock → `L1684-L1704`
- `.modal-backdrop`, `.drawer-modal`, `.fullscreen-modal` regla global: `display:flex` siempre + `visibility:hidden + opacity:0 + pointer-events:none` → transitions funcionan correctamente (antes `display:none` impedía animar) → `L1621-L1653`

### 2.3 Hover mejorado en 12 categorías de componentes
Cada hover usa `transform: translateY(-2px/translate3d) scale(1.02/1.03) opacity(0.92)` + cursor pointer + easing spring:
1. `.btn-header` (L1614)
2. `.btn-cloud-sync` (L1609)
3. `.tab-btn` incluidos indicadores subrayado slide (L1594-L1607)
4. `.cat-pill` (L1597)
5. `.hero-card` (L1656)
6. `.sentence-card` (L1659)
7. `.viewport` + `.viewport-card` (L1661-L1668)
8. `.word-chip` + `.chip-speak` + `.chip-remove` (L1671)
9. `.btn-action` (btnSpeakAll, btnCopy, btnBackspace, btnClearAll) (L1674-L1680)
10. `.quick-card` (tarjetas rápidas bidi) (L1617)
11. `.btn-cam-circle` (círculo grabación cámara) (L1665)
12. `.custom-input` (TextFields focus) + `.btn-sign-play` (dicc) (L1683)

---

## 3. Motion Helpers JS + IntersectionObserver (lazy animate-on-enter)
**Ubicación código**: `docs/index.html:L2016-L2148`

- `PREFERS_REDUCED_MOTION` (`matchMedia`) — guard JS global honrado en `motionDur`, `flashSuccess/Error`, `addClassTimed`
- `motionDur(ms)` → si reduced-motion, retorna `min(ms, 40)`
- `addClassTimed(el, cls, ms)` / `removeClassTimed` → cleanup seguro con Set timestamps por elemento
- `flashSuccess(el)` y `flashError(el)` → micro WAAPI-like vía clases + duration scale
- `createRipple(e, target)` + `installRipple()` → ripple clic 100% compositing (`transform: scale()` + `opacity`)
- `initLazyAnimate()` → `IntersectionObserver threshold 0.12, rootMargin -40px` que añade `.lazy-in` solo cuando el elemento entra a viewport; `unobserve` tras visible para ahorrar CPU
- `dismissSplashAnimated()` → stagger secuencial splash interior: logo → title → badge → loader bar → loader txt → backdrop fade-out (L2110-L2140)

### 3.1 Uso de motion helpers en UI crítica
- **Renderizado chips oración** (`L2227-L2322`): cada chip nace `.chip-enter` con `--stagger-index = idx`, ripple en speak/remove, `flashSuccess` al quitar/copiar/hablar
- **Botones constructores** `btnSpeakAll / btnCopySentence / btnBackspace / btnClearAll`: `flashSuccess` (acción válida) o `flashError` (oración vacía)
- **WAAPI** (Web Animations API nativo) en `btnSpeakAll.onpointerenter`: pulsos `brightness 1.0↔1.4` con duration escalado reduced-motion
- **Tabs setActiveTab** (`L2331-L2413`): view-exit clase + `animationend` listener luego view-enter dirección left/right según posición tabs[0..1]
- **Quick cards Bidi** + **Fullscreen texto gigante**: stagger `[data-animate]` al mostrar + flashSuccess al clicar / flashError si texto vacío
- **Diccionario** `sign-item` (`L2523-L2543`): lazy animate stagger index + ripple + flashSuccess al reproducir
- **Inicialización DOMContentLoaded** (`L2952-L2985`): marca `.animate-on-enter` en 8 elementos estáticos (header, nav, hero, sent-card, vp-card, footer, splash-children), `installRipple(document)`, `initLazyAnimate()`, `transitionend` drawer re-observe IO, startCamera/initMediaPipe

### 3.2 Comentarios en línea añadidos (Fase 4 req2)
- `emitirFeedback` (L2150): doc de orquestación multimodal (háptico/voz/chime/flash) + reduced-motion
- `activarFlashVisual` (L2210): compositor-only opacity
- `actualizarUI` (L2840): máquina de estados PREPARANDO → INTERPRETANDO → CONFIRMADO → REPOSO + pulse-confirmed trigger
- `dibujarFeedbackVisual` (L2924): loop canvas ~60fps, sin animaciones CSS dentro

---

## 4. Splash nativo Flutter + OrbitLoader Tricolor
**Archivo**: `app_lsc/lib/main.dart`
**Ubicación código**: `L47-L811`

### 4.1 Splash nativo animado (antes `CircularProgressIndicator` genérico)
- `L51`: `bool _splashVisible = true;`
- `L141-L151`: `onPageFinished` del WebView → `setState _splashVisible = false` tras delay 380ms (deja splash HTML interior visible durante stagger dismiss)
- `L565-L704`: Nuevo `build()` → `Scaffold → SafeArea → Stack = [ WebViewWidget, AnimatedOpacity→AnimatedContainer→IgnorePointer→splashContent ]`
  - AnimatedOpacity (duration 420ms easeOutCubic)
  - AnimatedContainer transform scale 1.0 → 1.04 (sutil zoom salida)
  - `TweenAnimationBuilder<double>` logo 0.82 → 1.0 curve elasticOut 850ms
  - `TweenAnimationBuilder<double>` título translate y -24→0 + fade 700ms
  - `TweenAnimationBuilder<double>` subtítulo 950ms delay 150ms
  - `TweenAnimationBuilder<double>` status 1150ms delay 250ms
  - **`_TricolorOrbitLoader`** widget nuevo

### 4.2 Widget `_TricolorOrbitLoader extends StatefulWidget`
- `L707-L811`: `State with SingleTickerProviderStateMixin`
- `AnimationController(1350ms, repeat)`: 3 dots 12px en grados 0° (amarillo F59E0B) / 120° (azul 0033A0) / 240° (rojo CE1126) + centro 00E5FF 8px
- Cada `_dot` → `Transform.rotate(angle + animValue * 2π) + Transform.translate(offset radiusX/radiusY)` + pulse `scale = 1.0 + 0.35 * sin(animValue*2π + phase*2π/3)` → `Curves.linear`

---

## 5. Modal Sync Flutter: AnimatedCrossFade + SnackBars Floating
**Archivo**: `app_lsc/lib/main.dart`
**Ubicación código**: `L279-L708`

### 5.1 Colapsable sección avanzada con AnimatedCrossFade (antes `if (showAdvanced) …[ ]` sin transición)
- `L577-L708`:
  - Icon expand expand_less/expand_more envuelto en `AnimatedSwitcher(220ms) → FadeTransition` con `ValueKey<bool>(showAdvanced)`
  - `AnimatedCrossFade(320ms / reverse 220ms, sizeCurve easeInOutCubic)`:
    - showFirst: `SizedBox.shrink()`
    - showSecond: Column `TextField IP:Port + TextButton "Conectar Wi-Fi Local"`
- SnackBar conectar host `behavior: floating + shape radius14 + margin 14/14`

### 5.2 SnackBars floating + radius en 4 acciones de sync
- Descarga cloud OK (verde, 4s): `L440-L458`
- Descarga cloud FAIL (rojo): `L460-L482`
- Restaurar APK (ámbar): `L553-L570`
- Conectar WiFi local (cian default): `L677-L694`

---

## 6. Cumplimiento Fase 3 (Rendimiento 60fps / A11y / Cross-browser)
✅ **GPU-only**: todos los `@keyframes` usan exclusivamente `transform` y `opacity` (verificado corregidos los 2 que usaban `box-shadow`).
✅ **`prefers-reduced-motion`**: regla CSS global `@media` + guard JS `PREFERS_REDUCED_MOTION` + `motionDur()` escalando todas las duraciones.
✅ **Cross-browser**: todos los `backdrop-filter` tienen prefijo `-webkit-backdrop-filter`; `-webkit-animation`, `-webkit-transform` incluidos en reglas críticas; keyframes con prefijos `-webkit-@keyframes`.
✅ **Lazy load animaciones**: `IntersectionObserver` threshold 0.12 + `unobserve` tras primer visible; `.animate-on-enter` solo para elementos above-the-fold.

---

## 7. Fase 4 — Dependencias / Verificaciones
- `skills-lock.json`: `git diff skills-lock.json` sin diferencias (0 líneas cambiadas). **No requiere actualización**.
- No se añadió paquete npm/pubspec nuevo; 0 nuevas dependencias externas.
- Stack 100% intacto: Flutter webview_flutter 4.13.1 + shelf 1.4.2 + MediaPipe CDN JS + CSS puro + WAAPI.

---

## 8. Fase 4 — Pruebas pendientes (por ejecutar en entorno real)
| Item | Estado | Comentario |
|---|---|---|
| Chrome DevTools Performance trace ≥58fps, 0 forced layout | Pendiente | Recomendado grabación 10s: cambiar tabs, pulsar 5 botones, confirmar 1 seña, abrir drawer, cerrar. |
| Smoke test Chrome 128+ / Firefox 129+ / Safari 17+ / Edge 128+ | Pendiente | Checklist: tabs transición, ripple clic, pulse orbit loader, drawer modal entrada/salida, fullscreen zoom, chips stagger, flashSuccess/Error, prefers-reduced-motion Settings→Accesibilidad. |
| Compilación APK debug `flutter run` | Pendiente | Validar splash nativo, AnimatedCrossFade showAdvanced, TricolorOrbitLoader no causa jank, SnackBars flotantes, fade-out splash nativo → HTML splash interior transición limpia. |
| `dart analyze` / `flutter analyze` | Pendiente | Confirmar 0 errors / warnings en main.dart. |
| A11y audit `@axe-core/nvd3` sobre WebView | Pendiente | Alto contraste, focus visibles, prefers-reduced-motion honrado. |

---

## MD5 de integridad de las 3 copias HTML
```
docs/index.html            MD5=7DD69E3BB957C5106BECF014B7B10032  size=112970 B
estilo/index.html          MD5=7DD69E3BB957C5106BECF014B7B10032  ✓ 1:1
app_lsc/assets/web/index.html  MD5=7DD69E3BB957C5106BECF014B7B10032  ✓ 1:1
```
