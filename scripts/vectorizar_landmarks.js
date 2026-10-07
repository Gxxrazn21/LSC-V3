/**
 * Convierte landmarks crudos (datasets/landmarks_lsc70.jsonl) en vectores 109D
 * usando exactamente la misma función que ejecuta la app
 * (web/motor_inferencia_local.js -> extraerDescriptorMultimodal).
 *
 * - Mano activa (la más elevada respecto a los hombros) -> etiqueta de la seña.
 * - Mano pasiva colgando por debajo del torso -> muestra real de REPOSO.
 * - Cada muestra se emite en sus dos quiralidades (isLeft false/true) para que el
 *   modelo sea invariante a qué mano usa la persona y a errores de lateralidad.
 *
 * Uso: node scripts/vectorizar_landmarks.js
 * Entradas: datasets/landmarks_lsc70.jsonl y, si existe, datasets/capturas_propias.jsonl
 */
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..');
const motor = require(path.join(ROOT, 'web', 'motor_inferencia_local.js'));

const entradas = ['landmarks_lsc70.jsonl', 'capturas_propias.jsonl']
  .map(f => path.join(ROOT, 'datasets', f))
  .filter(f => fs.existsSync(f));
const salida = path.join(ROOT, 'datasets', 'vectores_lsc70_109d.json');

// dy relativo a hombros (en anchos de hombro) por encima del cual la mano se considera en reposo
const DY_REPOSO = 1.6;

function dyRelativo(coords, pose) {
  const z = motor.extraerZonasCorporalesPose(coords, pose, false);
  return z.dy;
}

async function main() {
  const X = [], y = [], personas = [], quiralidad = [];
  let sinMano = 0, sinPose = 0, total = 0;

  // Secuencias (cuadros de una misma seña de una persona) para el validador de señas parecidas
  const secuencias = new Map();
  const claveSecuencia = (r) => {
    if (r.subconjunto === 'PROPIAS') {
      // Capturas propias: bloques de ~1 s (8 cuadros a 0.12 s) en orden de captura
      const k = `${r.persona}|${r.clase}`;
      const n = (contadorPropias.get(k) || 0);
      contadorPropias.set(k, n + 1);
      return { clave: `PROPIAS|${k}|${Math.floor(n / 8)}`, orden: n };
    }
    const m = /_(\d+)_?\.\w+$/.exec(r.archivo);
    return { clave: `${r.subconjunto}|${r.persona}|${r.clase}`, orden: m ? +m[1] : 0 };
  };
  const contadorPropias = new Map();

  const emitir = (coords, pose, etiqueta, persona, seq) => {
    for (const isLeft of [false, true]) {
      const { vec109 } = motor.extraerDescriptorMultimodal(coords, pose, isLeft);
      if (vec109.length !== 109 || vec109.some(v => !Number.isFinite(v))) return;
      if (seq && !isLeft) {
        if (!secuencias.has(seq.clave)) secuencias.set(seq.clave, { persona, clase: etiqueta, cuadros: [] });
        secuencias.get(seq.clave).cuadros.push({ orden: seq.orden, fila: X.length, coords, pose });
      }
      X.push(vec109.map(v => Math.round(v * 1e6) / 1e6));
      y.push(etiqueta);
      personas.push(persona);
      quiralidad.push(isLeft ? 1 : 0);
    }
  };

  for (const entrada of entradas) for await (const linea of readline.createInterface({ input: fs.createReadStream(entrada), crlfDelay: Infinity })) {
    if (!linea.trim()) continue;
    const r = JSON.parse(linea);
    total++;
    if (!r.manos.length) { sinMano++; continue; }
    // Sin hombros no hay anclaje espacial fiable: la app también los necesita.
    if (!r.pose) { sinPose++; continue; }

    const manos = r.manos.map(m => ({ ...m, dy: dyRelativo(m.coords, r.pose) }));
    manos.sort((a, b) => a.dy - b.dy);
    const activa = manos[0];
    if (activa.dy < DY_REPOSO) {
      emitir(activa.coords, r.pose, r.clase, r.persona, claveSecuencia(r));
    }
    for (const pasiva of manos.slice(activa.dy < DY_REPOSO ? 1 : 0)) {
      if (pasiva.dy >= DY_REPOSO) emitir(pasiva.coords, r.pose, 'REPOSO', r.persona);
    }
  }

  const seqs = [];
  for (const s of secuencias.values()) {
    s.cuadros.sort((a, b) => a.orden - b.orden);
    const rasgos = motor.extraerRasgosMovimiento(s.cuadros);
    if (!rasgos) continue;
    seqs.push({ persona: s.persona, clase: s.clase, filas: s.cuadros.map(c => c.fila), rasgos: rasgos.map(v => Math.round(v * 1e5) / 1e5) });
  }
  fs.writeFileSync(salida, JSON.stringify({ X, y, personas, quiralidad, secuencias: seqs }));
  console.log(`Secuencias con movimiento: ${seqs.length}`);
  console.log(`Registros: ${total} | sin mano: ${sinMano} | sin pose: ${sinPose} | vectores: ${X.length} -> ${salida}`);
}

main();
