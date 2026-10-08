/**
 * Tira de cuadros del avatar a lo largo de una seña (para ver naturalidad, temblor y colisiones).
 * Uso: python -m http.server 8766 (raíz) ; node tests/e2e/tira_avatar.js <salida> NOCHES M N NN
 * Con PRIMER_PLANO=1 la cámara sigue la mano derecha.
 */
const path = require('path');
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const [salida, ...senas] = process.argv.slice(2);
const PRIMER_PLANO = process.env.PRIMER_PLANO === '1';
const PASOS = +(process.env.PASOS || 8);

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 300, height: 380 });
  const errores = [];
  page.on('pageerror', e => errores.push(e.message));
  await page.goto('http://localhost:8766/tests/e2e/avatar.html', { waitUntil: 'load' });
  await page.evaluate(() => window.listo);
  await page.evaluate(() => window.av.detener());
  for (const s of senas) {
    const total = await page.evaluate(s => window.av.duracionTotal ? window.av.duracionTotal(s) : (window.av.senas.senas[s] ? window.av.senas.senas[s].duracion + 0.5 : 2), s);
    await page.evaluate(s => { window.av.senar(s); }, s);
    for (let k = 0; k < PASOS; k++) {
      await page.evaluate((dt, pp) => {
        const av = window.av;
        const n = Math.max(1, Math.round(dt * 30)); for (let i = 0; i < n; i++) av.paso(dt / n);
        if (pp) {
          const m = av.huesos.RightHandMiddle1.getWorldPosition(av.camara.position.clone());
          const prev = av.camara.position.clone();
          av.camara.position.set(m.x, m.y, m.z + 0.45); av.camara.lookAt(m);
          av._dibujar(); av.camara.position.copy(prev); av._colocarCamara();
        } else av._dibujar();
      }, k === 0 ? 0.05 : total / (PASOS - 1), PRIMER_PLANO);
      await page.screenshot({ path: path.join(salida, `tira_${s}_${k}.png`) });
    }
    await page.evaluate(() => window.av.cancelar());
  }
  console.log(errores.join('\n') || 'sin errores');
  await browser.close();
})();
