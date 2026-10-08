/**
 * Capturas del avatar haciendo señas (para revisarlas junto a la foto de referencia).
 * Uso: python -m http.server 8766 (raíz) ; node tests/e2e/capturas_avatar.js <salida> HOLA A L Y 5 YO
 */
const path = require('path');
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const [salida, ...senas] = process.argv.slice(2);
const PRIMER_PLANO = process.env.PRIMER_PLANO === '1';

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 420, height: 520 });
  const errores = [];
  page.on('pageerror', e => errores.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errores.push(m.text()); });
  await page.goto('http://localhost:8766/tests/e2e/avatar.html', { waitUntil: 'load' });
  const huesos = await page.evaluate(() => window.listo);
  await page.evaluate(p => { window.PRIMER_PLANO = p; }, PRIMER_PLANO);
  console.log('huesos:', huesos.length, huesos.slice(0, 6).join(','), '...');
  await new Promise(r => setTimeout(r, 800));
  await page.screenshot({ path: path.join(salida, 'avatar_descanso.png') });
  for (const s of senas) {
    // Avanza la seña hasta la mitad y la congela para la captura
    const ok = await page.evaluate(async (s) => {
      const av = window.av;
      av.detener();
      const p = av.senar(s);
      const dur = av.senas && av.senas.senas[s] ? 0.25 + av.senas.senas[s].duracion * 0.7 : (av.clips[s] ? av.clips[s].duration * 0.45 : 0);
      if (!dur) return false;
      if (av.accionClip) av.mixer.update(dur); else av._avanzar(dur);
      if (window.PRIMER_PLANO) {
        // Cámara cerca de la mano derecha para revisar los dedos
        const m = av.huesos.RightHandMiddle1.getWorldPosition(new av.camara.position.constructor());
        const prev = av.camara.position.clone();
        av.camara.position.set(m.x, m.y, m.z + 0.42); av.camara.lookAt(m);
        av._dibujar();
        av.camara.position.copy(prev); av._colocarCamara();
      } else av._dibujar();
      av.cancelar();
      return true;
    }, s);
    await page.screenshot({ path: path.join(salida, `avatar_${s}.png`) });
    if (!ok) console.log('sin datos para', s);
  }
  console.log(errores.join('\n') || 'sin errores');
  await browser.close();
})();
