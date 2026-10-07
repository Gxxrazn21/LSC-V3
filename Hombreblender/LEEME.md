# Avatar LSC — v1.0 (tu modelo riggeado)

## Archivos
- **avatar_lsc.blend** — escena de Blender (5.2) con el modelo, el esqueleto, las expresiones y la animación HOLA. Texturas empaquetadas.
- **avatar_lsc.glb** — para la app (Three.js / Unity / React Native). Incluye la animación "HOLA" y los morph targets.
- **avatar_hola.mp4** — render de la seña HOLA (720 px, 30 fps).
- **lamina_vistas.png / lamina_expresiones.png** — hojas de referencia.
- **scripts/** — el proceso completo, reproducible: prep -> rig_v4 -> face_v4 -> anim_v4 -> export4 / render4.

## Esqueleto (nombres Mixamo, 44 huesos)
Hips, Spine, Spine1, Spine2, Neck, Head;
por lado: Shoulder, Arm, ForeArm, Hand y 5 dedos × 3 falanges
(mixamorig:LeftHandIndex1..3, Middle, Ring, Pinky, Thumb). Prefijo: `mixamorig:`.
Pose de reposo: T-pose. Medio cuerpo (corte en la cadera, z = 0.90 m).

## Expresiones (morph targets, nombres ARKit)
browInnerUp, browOuterUpLeft/Right, browDownLeft/Right, eyeWideLeft/Right, eyeSquintLeft/Right,
eyeBlinkLeft/Right (en las mallas ParpadoLeft/ParpadoRight), mouthSmileLeft/Right, mouthFrownLeft/Right,
mouthStretchLeft/Right, mouthPucker, mouthFunnel, mouthPress, jawOpen, noseSneerLeft/Right.

## Notas
- Los ojos están pintados en la textura (así salió del generador): el parpadeo usa párpados añadidos;
  la mirada sigue a la cabeza.
- La boca también es pintada: jawOpen abre la línea de la boca, pero no hay interior (dientes/lengua).
