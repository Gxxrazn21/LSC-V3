/**
 * Diagnóstico numérico de las señas del avatar (simulado a 30 cuadros/s):
 *   penetración máxima de manos/antebrazos en el cuerpo (cm), flexión máxima de muñeca,
 *   giro máximo del antebrazo, error medio de orientación de la palma frente al dato,
 *   y temblor de los dedos en señas estáticas (variación de ángulos mientras se sostiene).
 * Uso: python -m http.server 8766 (raíz) ; node tests/e2e/diagnostico_avatar.js [SEÑAS...]
 */
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const pedidas = process.argv.slice(2);

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  page.on('pageerror', e => console.error('ERROR', e.message));
  await page.goto('http://localhost:8766/tests/e2e/avatar.html', { waitUntil: 'load' });
  await page.evaluate(() => window.listo);
  const filas = await page.evaluate((pedidas) => {
    const av = window.av; av.detener();
    const nombres = pedidas.length ? pedidas : Object.keys(av.senas.senas);
    const out = [];
    for (const s of nombres) {
      av.cancelar(); av.poseDescanso(); av.senar(s);
      const total = av.duracionTotal(s);
      let pen = 0, mun = 0, giro = 0, err = 0, n = 0;
      const dedosSosten = [];
      const sena = av.senas.senas[s];
      for (let t = 0; t < total; t += 1 / 30) {
        av.paso(1 / 30);
        for (const lado of ['Right', 'Left']) {
          // Penetración real (tras la resolución): muñeca, palma, yemas y antebrazo
          const pts = [av._w(`${lado}Hand`), av._centroPalma(lado), av._w(`${lado}ForeArm`).lerp(av._w(`${lado}Hand`), 0.5)];
          for (const d of ['Index', 'Middle', 'Pinky']) pts.push(av._w(`${lado}Hand${d}3`));
          for (const p of pts) pen = Math.max(pen, av._empuje(p, 0).length());
          const dg = av.diag && av.diag[lado];
          if (dg && (lado === 'Right' || sena.cuadros.some(c => c.Left))) {
            mun = Math.max(mun, dg.muneca); giro = Math.max(giro, Math.abs(dg.giro));
            if (t > 0.6) { err += dg.errorPalma; n++; }
          }
        }
        if (sena.estatica && t > 0.75) dedosSosten.push(Array.from(av.estado.Right.dedos.x));
      }
      let temblor = 0;
      if (dedosSosten.length > 2) {
        for (let k = 0; k < dedosSosten[0].length; k++) {
          const vals = dedosSosten.map(v => v[k]);
          temblor = Math.max(temblor, Math.max(...vals) - Math.min(...vals));
        }
      }
      out.push({ s, pen: +(pen * 100).toFixed(1), mun: Math.round(mun), giro: Math.round(giro), errPalma: n ? Math.round(err / n) : null, temblor: +temblor.toFixed(1) });
    }
    return out;
  }, pedidas);
  console.log('seña     penetr(cm) muñeca(°) giro(°) errorPalma(°) temblorDedos(°)');
  for (const f of filas) console.log(`${f.s.padEnd(8)} ${String(f.pen).padStart(9)} ${String(f.mun).padStart(9)} ${String(f.giro).padStart(7)} ${String(f.errPalma).padStart(13)} ${String(f.temblor).padStart(15)}`);
  const peor = k => filas.reduce((m, f) => Math.max(m, f[k] || 0), 0);
  console.log(`MÁXIMOS: penetración ${peor('pen')} cm | muñeca ${peor('mun')}° | giro ${peor('giro')}° | error palma ${peor('errPalma')}° | temblor ${peor('temblor')}°`);
  await browser.close();
})();
