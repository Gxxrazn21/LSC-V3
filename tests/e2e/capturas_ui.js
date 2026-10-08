/**
 * Capturas de la interfaz (web/index.html) a ancho de celular para revisión visual:
 * traductor, textos largos, catálogo con el avatar señando, oyente -> sordo y modo oscuro.
 * Uso: python -m http.server 8765 --directory web ; node tests/e2e/capturas_ui.js <salida> [ancho]
 */
const path = require('path');
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const salida = process.argv[2] || '.';
const ancho = +(process.argv[3] || 360);
const espera = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
           '--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  await page.setViewport({ width: ancho, height: 800, deviceScaleFactor: 1 });
  const errores = [];
  page.on('pageerror', e => errores.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !/404/.test(m.text())) errores.push(m.text()); });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  await page.goto('http://localhost:8765/index.html', { waitUntil: 'load' });
  await espera(5000);
  const foto = nombre => page.screenshot({ path: path.join(salida, `${nombre}.png`) });

  // Congelar la visión para poder escribir en la tarjeta sin que el bucle la borre
  await page.evaluate(() => { window.VISION_PAUSADA = true; });
  const desbordes = [];
  for (const palabra of ['BUENAS', 'MILLÓN', 'NOMBRE', 'TARDES']) {
    const r = await page.evaluate(async (p) => {
      const t = document.getElementById('predTitle');
      t.textContent = p; t.classList.add('confirmed');
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      return { p, cabe: t.scrollWidth <= t.clientWidth + 1, alto: t.getBoundingClientRect().height, px: getComputedStyle(t).fontSize };
    }, palabra);
    desbordes.push(r);
  }
  console.log('Seña principal:', JSON.stringify(desbordes));
  await foto('1_traductor_buenas');

  // Pantalla completa con una frase larga
  const fs = await page.evaluate(async () => {
    document.querySelector('.quick-card').click();
    await new Promise(r => setTimeout(r, 400));
    const t = document.getElementById('fullscreenText');
    return { cabe: t.scrollWidth <= t.clientWidth + 1, px: getComputedStyle(t).fontSize };
  });
  console.log('Pantalla completa:', JSON.stringify(fs));
  await foto('2_pantalla_completa');
  await page.click('#btnCloseFullscreen'); await espera(400);

  // Catálogo con el avatar
  await page.click('#btnOpenDictionary');
  await page.waitForFunction(() => document.getElementById('avatarEstado').hidden, { timeout: 60000 });
  await espera(600);
  await foto('3_catalogo_avatar');
  await page.evaluate(() => {
    const fila = [...document.querySelectorAll('#signsListContainer .sign-item')].find(r => r.dataset.sena === 'BUENAS');
    fila.click();
  });
  await espera(1100);
  await foto('4_avatar_buenas');
  await page.type('#inputDeletrear', 'Hola Ana');
  await page.click('#formDeletrear button');
  await espera(2600);
  await foto('5_avatar_deletreo');
  const pausada = await page.evaluate(() => window.VISION_PAUSADA);
  await page.click('#btnCloseDictionary'); await espera(500);
  const reanudada = await page.evaluate(() => window.VISION_PAUSADA);
  console.log(`Visión pausada con catálogo abierto: ${pausada} | reanudada al cerrar: ${!reanudada}`);

  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  await page.click('#btnOpenDictionary'); await espera(1200);
  await foto('6_catalogo_oscuro');
  console.log(errores.join('\n') || 'sin errores');
  await browser.close();
})();
