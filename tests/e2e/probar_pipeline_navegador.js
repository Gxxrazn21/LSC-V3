/**
 * Prueba de punta a punta en Chrome headless: imágenes LSC70 -> MediaPipe JS ->
 * motor_inferencia_local.js -> modelo_ia_cliente.js (lo mismo que corre la app).
 *
 * Uso (desde la raíz del repo):
 *   python -m http.server 8766          (en otra terminal)
 *   npm --prefix tests/e2e install
 *   node tests/e2e/probar_pipeline_navegador.js [Per03,Per30,Per60]
 * Con SIN_POSE=1 simula que no se ven los hombros (persona muy cerca de la cámara).
 * Se cuenta como acierto lo que la app MOSTRARÍA (pred.sena), no el top-1 crudo.
 */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const ROOT = path.resolve(__dirname, '..', '..');
const BASE = 'http://localhost:8766';
const PERSONAS = (process.argv[2] || 'Per03,Per30,Per60').split(',');
const SIN_POSE = process.env.SIN_POSE === '1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const modelo = require(path.join(ROOT, 'web', 'modelo_ia_cliente.js'));
const modoDe = c => Object.entries(modelo.categorias).find(([m, l]) => m !== 'todo' && l.includes(c))?.[0] || 'todo';

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
  const page = await browser.newPage();
  page.on('pageerror', e => console.error('PAGEERROR', e.message));
  await page.goto(`${BASE}/tests/e2e/pipeline.html`, { waitUntil: 'load' });
  await page.evaluate(() => window.listo);

  let total = 0, okTodo = 0, okModo = 0, sinMano = 0;
  const fallos = {};
  for (const sub of ['LSC70W', 'LSC70AN']) {
    for (const per of PERSONAS) {
      const dirPer = path.join(ROOT, 'datasets', 'LSC70', sub, per);
      if (!fs.existsSync(dirPer)) continue;
      for (const claseDir of fs.readdirSync(dirPer)) {
        const clase = claseDir === 'ANNOS' ? 'AÑOS' : claseDir;
        const imgs = fs.readdirSync(path.join(dirPer, claseDir)).filter(f => /\.(jpg|png)$/i.test(f)).sort();
        if (!imgs.length) continue;
        const archivo = imgs[Math.floor(imgs.length / 2)];
        const url = `${BASE}/datasets/LSC70/${sub}/${per}/${claseDir}/${archivo}`;
        let todo, enModo;
        try {
          todo = await page.evaluate((u, s) => window.probar(u, 'todo', s), url, SIN_POSE);
          enModo = await page.evaluate((u, m, s) => window.probar(u, m, s), url, modoDe(clase), SIN_POSE);
        } catch (e) {
          console.error(`No se pudo procesar ${url}: ${e.message}`);
          continue;
        }
        total++;
        if (!todo.length) { sinMano++; continue; }
        // Igual que la app: la mano que no está en REPOSO y con mayor confianza
        const elegir = preds => preds.filter(p => p.raw !== 'REPOSO').sort((a, b) => b.conf - a.conf)[0] || preds[0];
        const a = elegir(todo), b = elegir(enModo);
        if (a.sena === clase) okTodo++;
        if (b.sena === clase) okModo++; else (fallos[clase] = fallos[clase] || []).push(b.sena);
      }
    }
  }
  await browser.close();
  console.log(`Imágenes: ${total} | sin mano: ${sinMano}`);
  console.log(`Acierto (modo Todo): ${(100 * okTodo / total).toFixed(1)}%`);
  console.log(`Acierto (modo de su categoría): ${(100 * okModo / total).toFixed(1)}%`);
  console.log('Fallos (modo categoría):', JSON.stringify(fallos));
})();
