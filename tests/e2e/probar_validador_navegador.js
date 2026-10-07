/**
 * Prueba del validador de señas parecidas en Chrome headless con MediaPipe JS:
 * pasa cada secuencia LSC70 (6 cuadros) por el mismo pipeline de la app y
 * compara la decisión por probabilidades medias contra la del validador.
 *
 * Uso (desde la raíz, con `python -m http.server 8766` corriendo):
 *   node tests/e2e/probar_validador_navegador.js [Per05,Per25,Per45,Per65]
 */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const ROOT = path.resolve(__dirname, '..', '..');
const BASE = 'http://localhost:8766';
const PERSONAS = (process.argv[2] || 'Per05,Per25,Per45,Per65').split(',');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const modelo = require(path.join(ROOT, 'web', 'modelo_ia_cliente.js'));
const CARPETA = { ANNOS: 'AÑOS' };

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
  const page = await browser.newPage();
  await page.goto(`${BASE}/tests/e2e/pipeline.html`, { waitUntil: 'load' });
  await page.evaluate(() => window.listo);

  const resumen = {};
  for (const v of modelo.validadores) {
    const clave = v.clases.join('/');
    resumen[clave] = { n: 0, base: 0, validador: 0 };
    for (const clase of v.clases) {
      const carpetaClase = Object.keys(CARPETA).find(k => CARPETA[k] === clase) || clase;
      for (const sub of ['LSC70W', 'LSC70AN']) {
        for (const per of PERSONAS) {
          const dir = path.join(ROOT, 'datasets', 'LSC70', sub, per, carpetaClase);
          if (!fs.existsSync(dir)) continue;
          const urls = fs.readdirSync(dir).filter(f => /\.(jpg|png)$/i.test(f)).sort()
            .map(f => `${BASE}/datasets/LSC70/${sub}/${per}/${carpetaClase}/${f}`);
          const r = await page.evaluate(async (urls, grupo) => {
            const cuadros = [];
            for (const u of urls) {
              const manos = await window.probarCuadro(u);
              if (manos) cuadros.push(manos);
            }
            if (cuadros.length < 3) return null;
            const m = window.MODELO_LSC;
            const medias = new Array(m.clases.length).fill(0);
            cuadros.forEach(c => c.probs.forEach((p, j) => { medias[j] += p / cuadros.length; }));
            const idx = grupo.map(c => m.clases.indexOf(c));
            const base = grupo[idx.map(i => medias[i]).reduce((b, p, k, a) => (p > a[b] ? k : b), 0)];
            const rasgos = MotorLSCLocal.extraerRasgosMovimiento(cuadros);
            return { base, validada: MotorLSCLocal.validarSenaParecida(base, medias, rasgos, m, null) };
          }, urls, v.clases);
          if (!r) continue;
          resumen[clave].n++;
          if (r.base === clase) resumen[clave].base++;
          if (r.validada === clase) resumen[clave].validador++;
        }
      }
    }
  }
  await browser.close();
  for (const [g, r] of Object.entries(resumen)) {
    console.log(`${g.padEnd(26)} secuencias ${String(r.n).padStart(3)} | sin validador ${(100 * r.base / r.n).toFixed(0)}% | con validador ${(100 * r.validador / r.n).toFixed(0)}%`);
  }
})();
