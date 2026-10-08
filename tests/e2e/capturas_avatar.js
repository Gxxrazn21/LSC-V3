/**
 * Capturas del avatar en el momento clave de cada seña (para compararlas con la referencia).
 * Simula a 30 cuadros/s como en el teléfono (resortes y colisiones incluidos).
 * Uso: python -m http.server 8766 (raíz) ; node tests/e2e/capturas_avatar.js <salida> HOLA A L Y
 * Con PRIMER_PLANO=1 la cámara se acerca a la mano derecha.
 */
const path = require('path');
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const [salida, ...senas] = process.argv.slice(2);
const PRIMER_PLANO = process.env.PRIMER_PLANO === '1';

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 420, height: 520 });
  const errores = [];
  page.on('pageerror', e => errores.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !/404/.test(m.text())) errores.push(m.text()); });
  await page.goto('http://localhost:8766/tests/e2e/avatar.html', { waitUntil: 'load' });
  const huesos = await page.evaluate(() => window.listo);
  console.log('huesos:', huesos.length);
  await page.evaluate(() => window.av.detener());
  await page.screenshot({ path: path.join(salida, 'avatar_descanso.png') });
  for (const s of senas) {
    const ok = await page.evaluate(async (s, pp) => {
      const av = window.av;
      if (!av.senasDisponibles().includes(s)) return false;
      av.cancelar(); av.poseDescanso();
      av.senar(s);
      const sena = av.senas.senas[s];
      // Momento clave: seña estática sostenida, o 60 % de la trayectoria
      const t = av.clips[s] ? av.clips[s].duration * 0.45
        : (sena.estatica ? 0.35 + sena.duracion * 0.8 : 0.35 + sena.duracion * 0.6);
      for (let x = 0; x < t; x += 1 / 30) av.paso(1 / 30);
      if (pp) {
        const m = av.huesos.RightHandMiddle1.getWorldPosition(av.camara.position.clone());
        const prev = av.camara.position.clone();
        av.camara.position.set(m.x, m.y, m.z + 0.42); av.camara.lookAt(m);
        av._dibujar(); av.camara.position.copy(prev); av._colocarCamara();
      } else av._dibujar();
      return true;
    }, s, PRIMER_PLANO);
    await page.screenshot({ path: path.join(salida, `avatar_${s}.png`) });
    if (!ok) console.log('sin datos para', s);
  }
  console.log(errores.join('\n') || 'sin errores');
  await browser.close();
})();
