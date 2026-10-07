/**
 * Capturas de la interfaz (web/index.html) a ancho de celular para revisión visual.
 * Uso: python -m http.server 8765 --directory web ; node tests/e2e/capturas_ui.js <carpeta_salida>
 */
const path = require('path');
const puppeteer = require('puppeteer-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const salida = process.argv[2] || '.';

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 412, height: 915, deviceScaleFactor: 1 });
  const errores = [];
  page.on('pageerror', e => errores.push(e.message));
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  await page.goto('http://localhost:8765/index.html', { waitUntil: 'load' });
  await new Promise(r => setTimeout(r, 6000));
  const foto = async (nombre) => page.screenshot({ path: path.join(salida, `${nombre}.png`) });
  await foto('1_traductor');
  // Simula una seña confirmada para ver el estado principal
  await page.evaluate(() => {
    const t = document.getElementById('predTitle'); t.textContent = 'HOLA'; t.classList.add('confirmed');
    document.getElementById('predConfidence').textContent = '92%';
    document.getElementById('progressBar').style.width = '92%';
  });
  await foto('2_confirmado');
  await page.click('#btnOpenDictionary'); await new Promise(r => setTimeout(r, 700)); await foto('3_diccionario');
  await page.click('#btnCloseDictionary'); await new Promise(r => setTimeout(r, 500));
  await page.click('#btnOpenSettings'); await new Promise(r => setTimeout(r, 700)); await foto('4_ajustes');
  await page.reload({ waitUntil: 'load' }); await new Promise(r => setTimeout(r, 3000));
  await page.click('#tabBidi'); await new Promise(r => setTimeout(r, 700)); await foto('5_oyente');
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  await new Promise(r => setTimeout(r, 400)); await foto('6_oscuro');
  console.log(errores.join('\n') || 'sin errores');
  await browser.close();
})();
