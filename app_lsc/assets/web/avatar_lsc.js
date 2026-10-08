/**
 * Avatar LSC: carga avatar_lsc.glb con three.js y ejecuta señas de forma natural.
 *
 * Las señas vienen de senas_avatar.json (scripts/generar_senas_avatar.py, a partir
 * de las grabaciones reales de LSC70), expresadas en el marco del torso:
 *   muneca/codo  posición relativa al hombro, normalizada por el largo del brazo
 *   mano         dirección y normal de la palma
 *   dedos        ángulos por articulación (base, apertura, medio, punta)
 *   pulgar       dirección de sus falanges en el marco de la mano
 *
 * Cómo se mueve el cuerpo:
 *   - Brazo por IK de dos huesos. El codo se elige entre las posiciones posibles
 *     para quedar cerca del dato, con la muñeca lo más recta posible y fuera del cuerpo.
 *   - Muñeca con límites anatómicos; el giro se reparte entre antebrazo y mano.
 *   - Dedos rotados sobre su eje de flexión real (calibrado desde el esqueleto).
 *   - Colisiones: manos y antebrazos no atraviesan el torso ni la cabeza, y las
 *     dos manos no se atraviesan entre sí.
 *   - Resortes críticamente amortiguados: inercia natural sin rebotes.
 *   - Parpadeo y respiración.
 * HOLA usa la animación hecha a mano del .glb.
 *
 * Solo simula y dibuja mientras la sección está visible (iniciar/detener).
 */
import * as THREE from 'three';
import { GLTFLoader } from './GLTFLoader.js';

const PREFIJO = 'mixamorig';  // GLTFLoader quita los ':' de los nombres
const DEDOS = ['Index', 'Middle', 'Ring', 'Pinky'];
const LADOS = ['Right', 'Left'];
const RAD = Math.PI / 180;

// Límites y repartos anatómicos
const MAX_FLEXION_MUNECA = 70 * RAD;
const MAX_GIRO_ANTEBRAZO = 170 * RAD;
const FRACCION_GIRO_ANTEBRAZO = 0.5;
// Frecuencias de los resortes (rad/s): más alto = responde más rápido
const W_BRAZO = 10, W_MANO = 12, W_DEDOS = 15;
// Tiempos de la seña
const ENTRADA = 0.35, SOSTEN_DINAMICA = 0.35;

const COLOR_PARPADO = '#B98E7A';

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);

/** Resorte críticamente amortiguado para vectores o listas de números. */
class Resorte {
  constructor(valor) { this.x = Float64Array.from(valor); this.v = new Float64Array(valor.length); }
  fijar(valor) { this.x.set(valor); this.v.fill(0); }
  paso(objetivo, w, dt) {
    const sub = Math.max(1, Math.ceil(dt / (1 / 90)));
    const h = dt / sub;
    for (let s = 0; s < sub; s++) {
      for (let i = 0; i < this.x.length; i++) {
        const a = w * w * (objetivo[i] - this.x[i]) - 2 * w * this.v[i];
        this.v[i] += a * h;
        this.x[i] += this.v[i] * h;
      }
    }
    return this.x;
  }
}

/** Rotación que lleva el par (a0, b0) al par (a1, b1): a exacto, b lo más cerca posible. */
function alinear(a0, b0, a1, b1) {
  const base = (a, b) => {
    const x = a.clone().normalize();
    const y = b.clone().sub(x.clone().multiplyScalar(b.dot(x))).normalize();
    const z = new THREE.Vector3().crossVectors(x, y);
    return new THREE.Matrix4().makeBasis(x, y, z);
  };
  const m0 = base(a0, b0), m1 = base(a1, b1);
  const r = m1.multiply(m0.transpose());
  return new THREE.Quaternion().setFromRotationMatrix(r);
}

/** Separa q en giro alrededor de `eje` (unitario) y el resto (columpio): q = columpio * giro. */
function separarGiro(q, eje) {
  const p = eje.clone().multiplyScalar(eje.dot(new THREE.Vector3(q.x, q.y, q.z)));
  const giro = new THREE.Quaternion(p.x, p.y, p.z, q.w);
  if (giro.lengthSq() < 1e-12) giro.set(0, 0, 0, 1); else giro.normalize();
  const columpio = q.clone().multiply(giro.clone().invert());
  return { giro, columpio };
}

function anguloDe(q) { return 2 * Math.acos(Math.min(1, Math.abs(q.w))); }

function limitarAngulo(q, maximo) {
  const a = anguloDe(q);
  if (a <= maximo) return q;
  return new THREE.Quaternion().slerp(q, maximo / a);
}

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

export class AvatarLSC {
  constructor(contenedor) {
    this.contenedor = contenedor;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    contenedor.appendChild(this.renderer.domElement);

    this.escena = new THREE.Scene();
    this.camara = new THREE.PerspectiveCamera(28, 1, 0.05, 20);
    this.escena.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.6));
    const sol = new THREE.DirectionalLight(0xffffff, 1.4);
    sol.position.set(0.6, 1.8, 2.2);
    this.escena.add(sol);

    this.reloj = new THREE.Clock();
    this.activo = false;
    this.tiempo = 0;
    this.actual = null;     // seña en curso
    this._bucle = this._bucle.bind(this);
    new ResizeObserver(() => this._ajustarTamano()).observe(contenedor);
  }

  // ------------------------------------------------------------------ carga
  async cargar(urlModelo = 'avatar_lsc.glb', urlSenas = 'senas_avatar.json') {
    const [gltf, senas] = await Promise.all([
      new GLTFLoader().loadAsync(urlModelo),
      fetch(urlSenas).then(r => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    this.modelo = gltf.scene;
    this.escena.add(this.modelo);
    this.senas = senas;
    this.mixer = new THREE.AnimationMixer(this.modelo);
    this.clips = Object.fromEntries(gltf.animations.map(c => [c.name, c]));
    this.mixer.addEventListener('finished', () => this._finClip());

    this.huesos = {};
    this.parpados = [];
    this.modelo.traverse(o => {
      if (o.isBone) this.huesos[o.name.replace(PREFIJO, '')] = o;
      if (o.isMesh) {
        o.frustumCulled = false;
        const dic = o.morphTargetDictionary || {};
        if ('eyeBlinkLeft' in dic || 'eyeBlinkRight' in dic) {
          this.parpados.push(o);
          // Los párpados del modelo (zona gris de la textura y una banda de pestañas oscura que
          // al cerrar tapa el ojo como un disco) se pintan con tono de piel
          // (sus normales apuntan hacia adentro, por eso un material sin sombreado)
          o.material = new THREE.MeshBasicMaterial({ color: COLOR_PARPADO, side: THREE.DoubleSide });
        }
      }
      if (o.isSkinnedMesh && !this.malla) this.malla = o;
    });
    this.reposo = {};
    for (const [n, h] of Object.entries(this.huesos)) this.reposo[n] = h.quaternion.clone();
    this.modelo.updateMatrixWorld(true);

    this._calibrar();
    this._construirVolumenes();
    this._encuadrar();
    this.poseDescanso();
    return this;
  }

  nombresHuesos() { return Object.keys(this.huesos); }
  senasDisponibles() {
    return [...new Set([...Object.keys(this.senas ? this.senas.senas : {}), ...Object.keys(this.clips)])];
  }

  _w(nombre) { return this.huesos[nombre].getWorldPosition(new THREE.Vector3()); }
  _qw(nombre) { return this.huesos[nombre].getWorldQuaternion(new THREE.Quaternion()); }

  /** Mide el esqueleto en pose de descanso (T) y calibra ejes de codo, mano y dedos. */
  _calibrar() {
    const signoPalma = (this.senas && this.senas.signo_palma) || 1;
    this.cal = {};
    for (const lado of LADOS) {
      const lsigno = lado === 'Right' ? 1 : -1;
      const S = this._w(`${lado}Arm`), E = this._w(`${lado}ForeArm`), W = this._w(`${lado}Hand`);
      const c = {
        Lu: S.distanceTo(E), Lf: E.distanceTo(W),
        brazoDirReposo: E.clone().sub(S).normalize(),
        qBrazoReposo: this._qw(`${lado}Arm`),
        qManoReposo: this._qw(`${lado}Hand`),
        sx: lado === 'Right' ? -1 : 1,
      };
      // Eje del codo: el que, en positivo, lleva la mano hacia adelante (+z)
      c.ejeCodo = this._ejeQueMueve(`${lado}ForeArm`, `${lado}Hand`, new THREE.Vector3(0, 0, 1), [0, 2]);
      c.bisagraReposo = c.ejeCodo.eje.clone().multiplyScalar(c.ejeCodo.signo)
        .applyQuaternion(this._qw(`${lado}ForeArm`)).normalize();
      // Ejes largos (giro) de antebrazo y mano en su espacio local
      c.ejeLargoAntebrazo = this.huesos[`${lado}Hand`].position.clone().normalize();
      c.ejeLargoMano = this.huesos[`${lado}HandMiddle1`].position.clone().normalize();
      // Marco de la mano en reposo (mismo cálculo que el generador: muñeca, nudillos índice y meñique)
      const D = this._w(`${lado}HandMiddle1`).sub(W).normalize();
      const N = new THREE.Vector3().crossVectors(this._w(`${lado}HandIndex1`).sub(W), this._w(`${lado}HandPinky1`).sub(W))
        .normalize().multiplyScalar(signoPalma * lsigno);
      N.sub(D.clone().multiplyScalar(N.dot(D))).normalize();
      c.D = D; c.N = N; c.L = new THREE.Vector3().crossVectors(N, D);
      // Dedos: eje de flexión (hacia la palma) y de apertura (hacia L) por falange
      c.dedos = {};
      for (const dedo of DEDOS) {
        const f1 = this._ejesDedo(`${lado}Hand${dedo}1`, `${lado}Hand${dedo}2`, N, c.L);
        const f2 = this._ejesDedo(`${lado}Hand${dedo}2`, `${lado}Hand${dedo}3`, N, c.L);
        c.dedos[dedo] = [f1, f2, f2];
      }
      // Pulgar: direcciones de reposo de sus falanges
      const T1 = this._w(`${lado}HandThumb1`), T2 = this._w(`${lado}HandThumb2`), T3 = this._w(`${lado}HandThumb3`);
      c.pulgarReposo = [T2.clone().sub(T1).normalize(), T3.clone().sub(T2).normalize()];
      this.cal[lado] = c;
    }
  }

  /** Busca, entre los ejes locales indicados, el que mueve `hijo` hacia `dirMundo`. */
  _ejeQueMueve(hueso, hijo, dirMundo, indices) {
    const b = this.huesos[hueso];
    const p0 = this._w(hijo);
    let mejor = null;
    for (const i of indices) {
      const eje = new THREE.Vector3(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0);
      b.quaternion.copy(this.reposo[hueso]).multiply(new THREE.Quaternion().setFromAxisAngle(eje, 0.4));
      this.modelo.updateMatrixWorld(true);
      const d = this._w(hijo).sub(p0).dot(dirMundo);
      if (!mejor || Math.abs(d) > Math.abs(mejor.d)) mejor = { eje, signo: Math.sign(d) || 1, d };
    }
    b.quaternion.copy(this.reposo[hueso]);
    this.modelo.updateMatrixWorld(true);
    return mejor;
  }

  _ejesDedo(hueso, hijo, N, L) {
    const largo = this.huesos[hijo].position.clone().normalize();
    // Los dos ejes locales más perpendiculares al hueso
    const candidatos = [0, 1, 2].sort((a, b) => Math.abs(largo.getComponent(a)) - Math.abs(largo.getComponent(b))).slice(0, 2);
    const flex = this._ejeQueMueve(hueso, hijo, N, candidatos);
    const otro = candidatos.find(i => flex.eje.getComponent(i) === 0);
    const aper = this._ejeQueMueve(hueso, hijo, L, [otro]);
    return { flex: flex.eje, sFlex: flex.signo, aper: aper.eje, sAper: aper.signo };
  }

  /** Volumen del torso por rebanadas (desde la malla en pose de reposo) y esfera de la cabeza. */
  _construirVolumenes() {
    const pos = this.malla.geometry.attributes.position;
    const m = this.malla.matrixWorld;
    const hombroX = Math.abs(this._w('RightArm').x) * 0.95;
    const cuello = this._w('Neck'), cabeza = this._w('Head');
    const y0 = new THREE.Box3().setFromObject(this.malla).min.y;
    const paso = 0.02;
    const n = Math.max(1, Math.ceil((cuello.y - y0) / paso));
    const rebanadas = Array.from({ length: n }, () => ({ a: 0, zmin: Infinity, zmax: -Infinity, k: 0 }));
    const cabezaPts = [];
    const p = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += 2) {
      p.fromBufferAttribute(pos, i).applyMatrix4(m);
      if (p.y > cuello.y + 0.04 && Math.abs(p.x) < 0.16) { cabezaPts.push(p.clone()); continue; }
      if (p.y < y0 || p.y >= cuello.y || Math.abs(p.x) > hombroX) continue;
      const r = rebanadas[Math.min(n - 1, Math.floor((p.y - y0) / paso))];
      r.a = Math.max(r.a, Math.abs(p.x)); r.zmin = Math.min(r.zmin, p.z); r.zmax = Math.max(r.zmax, p.z); r.k++;
    }
    // Rellenar rebanadas vacías y suavizar
    let ultima = rebanadas.find(r => r.k) || { a: 0.15, zmin: -0.1, zmax: 0.1 };
    for (const r of rebanadas) { if (!r.k) Object.assign(r, { a: ultima.a, zmin: ultima.zmin, zmax: ultima.zmax }); ultima = r; }
    this.torso = { y0, paso, rebanadas, ytop: cuello.y };
    const centro = cabezaPts.reduce((s, q) => s.add(q), new THREE.Vector3()).multiplyScalar(1 / Math.max(1, cabezaPts.length));
    // Radio típico de la cabeza (mediana), no el máximo: el máximo incluye pelo y orejas
    // y alejaría las manos de la cara en señas que la tocan (AÑOS, LICOR)
    const dists = cabezaPts.map(q => q.distanceTo(centro)).sort((a, b) => a - b);
    const radio = dists.length ? dists[Math.floor(dists.length * 0.5)] : 0.1;
    this.cabeza = { centro: cabezaPts.length ? centro : cabeza, radio };
    this.cuello = { a: cuello, b: cabeza, radio: 0.06 };
  }

  /** Vector para sacar el punto `p` del cuerpo (cero si está fuera). */
  _empuje(p, margen) {
    const out = new THREE.Vector3();
    const t = this.torso;
    if (p.y > t.y0 && p.y < t.ytop) {
      const r = t.rebanadas[Math.min(t.rebanadas.length - 1, Math.floor((p.y - t.y0) / t.paso))];
      const a = r.a + margen, zc = (r.zmax + r.zmin) / 2, b = (r.zmax - r.zmin) / 2 + margen;
      const nx = p.x / a, nz = (p.z - zc) / b;
      const d = Math.hypot(nx, nz);
      if (d < 1) {
        // Siempre hacia adelante: una mano nunca debe quedar dentro ni detrás del pecho
        const fx = Math.max(-0.999, Math.min(0.999, nx));
        const nzFuera = Math.sqrt(1 - fx * fx);
        out.set(fx * a - p.x, 0, zc + nzFuera * b - p.z);
      }
    }
    const c = this.cabeza;
    const dc = p.distanceTo(c.centro);
    if (dc < c.radio + margen) {
      const dir = p.clone().sub(c.centro);
      if (dir.lengthSq() < 1e-8) dir.set(0, 0, 1);
      dir.normalize();
      if (dir.z < 0.2) { dir.z = 0.2; dir.normalize(); }
      out.add(dir.multiplyScalar(c.radio + margen - dc));
    }
    return out;
  }

  // ---------------------------------------------------------------- cámara
  _encuadrar() {
    const caja = new THREE.Box3().setFromObject(this.modelo);
    const cabeza = this._w('Head');
    const techo = cabeza.y + 0.30, piso = caja.min.y - 0.06;
    this.encuadre = { yCentro: (techo + piso) / 2, alto: (techo - piso) * 1.08, ancho: 1.0,
                      z: caja.getCenter(new THREE.Vector3()).z };
    this._colocarCamara();
  }

  _colocarCamara() {
    const e = this.encuadre;
    if (!e) return;
    const t = Math.tan(THREE.MathUtils.degToRad(this.camara.fov / 2));
    const dist = Math.max(e.alto / (2 * t), e.ancho / (2 * t * this.camara.aspect));
    this.camara.position.set(0, e.yCentro, e.z + dist);
    this.camara.lookAt(0, e.yCentro, 0);
  }

  _ajustarTamano() {
    const w = this.contenedor.clientWidth, h = this.contenedor.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camara.aspect = w / h;
    this.camara.updateProjectionMatrix();
    this._colocarCamara();
    this._dibujar();
  }

  // ------------------------------------------------------------- bucle
  iniciar() {
    if (this.activo) return;
    this.activo = true;
    this.reloj.getDelta();
    requestAnimationFrame(this._bucle);
  }

  detener() { this.activo = false; }

  _bucle() {
    if (!this.activo) return;
    this.paso(Math.min(this.reloj.getDelta(), 0.05));
    this._dibujar();
    requestAnimationFrame(this._bucle);
  }

  _dibujar() { if (this.modelo) this.renderer.render(this.escena, this.camara); }

  /** Avanza la simulación `dt` segundos (sin dibujar). */
  paso(dt) {
    this.tiempo += dt;
    if (this.accionClip) {
      this.mixer.update(dt);
      this._idle(dt);
      this._aplicarMezcla(dt);
      return;
    }
    const objetivos = this._objetivos(dt);
    this._idle(dt);
    for (const lado of LADOS) this._seguir(lado, objetivos[lado], dt);
    this._resolverCuerpo();
    this._aplicarMezcla(dt);
  }

  // --------------------------------------------------------- objetivos
  _descanso(lado) {
    const c = this.cal[lado], sx = c.sx;
    const r = new THREE.Vector3(sx * 0.2, -0.95, 0.18).normalize().multiplyScalar(0.97);
    const pulgar = this._pulgarRelajado || [[0.55, 0.35, -0.75], [0.75, 0.3, -0.6], [0.85, 0.3, -0.4]];
    return {
      muneca: r.toArray(), codo: [sx * 0.35, -0.9, -0.2],
      mano: { dir: [0, -1, 0.12], normal: [-sx, 0, 0.1] },
      dedos: DEDOS.map(() => [14, 0, 20, 12]), pulgar,
    };
  }

  /** Objetivos de cada lado en este instante, según la seña en curso. */
  _objetivos(dt) {
    const out = { Right: this._descanso('Right'), Left: this._descanso('Left') };
    const a = this.actual;
    if (!a) return out;
    a.t += dt;
    const cuadros = a.sena.cuadros;
    const n = cuadros.length;
    const dur = a.sena.duracion;
    let u;  // posición en los cuadros [0, n-1]
    if (a.sena.estatica || n === 1) u = 0;
    else u = Math.min(Math.max(a.t - ENTRADA, 0) / dur, 1) * (n - 1);
    for (const lado of LADOS) {
      if (!cuadros.some(c => c[lado])) continue;
      out[lado] = this._muestrear(cuadros, lado, u);
    }
    const total = ENTRADA + dur + (a.sena.estatica ? 0 : SOSTEN_DINAMICA);
    if (a.t >= total) {
      const fin = a.fin;
      this.actual = null;
      fin(true);
    }
    return out;
  }

  _muestrear(cuadros, lado, u) {
    const n = cuadros.length;
    const i = Math.min(Math.floor(u), n - 1), t = u - i;
    const c = k => cuadros[Math.max(0, Math.min(n - 1, k))][lado] || cuadros[Math.max(0, Math.min(n - 1, i))][lado];
    const p0 = c(i - 1), p1 = c(i), p2 = c(i + 1), p3 = c(i + 2);
    const mezcla = (f) => {
      const a0 = f(p0), a1 = f(p1), a2 = f(p2), a3 = f(p3);
      return a1.map((_, k) => catmull(a0[k], a1[k], a2[k], a3[k], t));
    };
    return {
      muneca: mezcla(x => x.muneca), codo: mezcla(x => x.codo),
      mano: { dir: mezcla(x => x.mano.dir), normal: mezcla(x => x.mano.normal) },
      dedos: DEDOS.map((_, d) => mezcla(x => x.dedos[d])),
      pulgar: [0, 1, 2].map(k => mezcla(x => x.pulgar[k])),
    };
  }

  // ---------------------------------------------------------- resortes
  _estado(lado) {
    this.estado = this.estado || {};
    if (!this.estado[lado]) {
      const d = this._descanso(lado);
      this.estado[lado] = {
        muneca: new Resorte(d.muneca), codo: new Resorte(d.codo),
        dir: new Resorte(d.mano.dir), normal: new Resorte(d.mano.normal),
        dedos: new Resorte(d.dedos.flat()), pulgar: new Resorte(d.pulgar.flat()),
        thetaCodo: null,
      };
    }
    return this.estado[lado];
  }

  _seguir(lado, obj, dt) {
    const e = this._estado(lado);
    e.muneca.paso(obj.muneca, W_BRAZO, dt);
    e.codo.paso(obj.codo, W_BRAZO, dt);
    e.dir.paso(obj.mano.dir, W_MANO, dt);
    e.normal.paso(obj.mano.normal, W_MANO, dt);
    e.dedos.paso(obj.dedos.flat(), W_DEDOS, dt);
    e.pulgar.paso(obj.pulgar.flat(), W_DEDOS, dt);
  }

  // ----------------------------------------------------- cinemática
  _resolverCuerpo() {
    const pasoIK = (lado, extra) => {
      const e = this._estado(lado), c = this.cal[lado];
      const S = this._w(`${lado}Arm`);
      const r = v3(e.muneca.x);
      if (r.length() > 0.98) r.setLength(0.98);
      const T = S.clone().add(r.multiplyScalar(c.Lu + c.Lf)).add(extra);
      this._resolverBrazo(lado, T, v3(e.codo.x), v3(e.dir.x).normalize(), v3(e.normal.x));
      this._aplicarDedos(lado);
      return T;
    };
    for (const lado of LADOS) {
      const extra = new THREE.Vector3();
      for (let it = 0; it < 3; it++) {
        pasoIK(lado, extra);
        const empuje = this._empujeMano(lado);
        if (empuje.lengthSq() < 4e-6) break;
        extra.add(empuje.multiplyScalar(1.15));
      }
      this[`_extra${lado}`] = extra;
    }
    // Las dos manos no se atraviesan: la derecha pasa por delante
    const pR = this._centroPalma('Right'), pL = this._centroPalma('Left');
    const dist = pR.distanceTo(pL);
    if (dist < 0.09) {
      const extra = this._extraRight.clone().add(new THREE.Vector3(0, 0, 0.09 - dist));
      pasoIK('Right', extra);
    }
  }

  _centroPalma(lado) {
    return this._w(`${lado}Hand`).add(this._w(`${lado}HandMiddle1`)).multiplyScalar(0.5);
  }

  /** Mayor empuje necesario para sacar muñeca, palma y yemas del cuerpo. */
  _empujeMano(lado) {
    const pts = [[this._w(`${lado}Hand`), 0.035], [this._centroPalma(lado), 0.03]];
    for (const d of ['Index', 'Middle', 'Pinky', 'Thumb']) {
      const p2 = this._w(`${lado}Hand${d}2`), p3 = this._w(`${lado}Hand${d}3`);
      pts.push([p3.clone().add(p3.clone().sub(p2).multiplyScalar(0.85)), 0.012]);
    }
    let mayor = new THREE.Vector3();
    for (const [p, m] of pts) {
      const e = this._empuje(p, m);
      if (e.lengthSq() > mayor.lengthSq()) mayor = e;
    }
    return mayor;
  }

  /** IK de dos huesos con elección del codo, bisagra real del codo y muñeca limitada. */
  _resolverBrazo(lado, T, pista, dirMano, normalMano) {
    const c = this.cal[lado], e = this._estado(lado);
    const S = this._w(`${lado}Arm`);
    const d = T.clone().sub(S);
    const dist = Math.min(Math.max(d.length(), 0.05), (c.Lu + c.Lf) * 0.999);
    const dh = d.normalize();
    T = S.clone().add(dh.clone().multiplyScalar(dist));
    const cosA = (c.Lu * c.Lu + dist * dist - c.Lf * c.Lf) / (2 * c.Lu * dist);
    const A = Math.acos(Math.min(1, Math.max(-1, cosA)));
    const C = S.clone().add(dh.clone().multiplyScalar(c.Lu * Math.cos(A)));
    const radio = c.Lu * Math.sin(A);
    // Base del círculo del codo, orientada por la pista del dato
    let u = pista.clone().sub(dh.clone().multiplyScalar(pista.dot(dh)));
    if (u.lengthSq() < 1e-6) u.set(c.sx, -1, 0).sub(dh.clone().multiplyScalar(dh.y));
    u.normalize();
    const w = new THREE.Vector3().crossVectors(dh, u);
    const codoEn = th => C.clone().add(u.clone().multiplyScalar(radio * Math.cos(th))).add(w.clone().multiplyScalar(radio * Math.sin(th)));
    const costo = th => {
      const E = codoEn(th);
      const ante = T.clone().sub(E).normalize();
      const flexMuneca = Math.acos(Math.min(1, Math.max(-1, ante.dot(dirMano))));
      let k = th * th + 0.6 * flexMuneca * flexMuneca;
      if (e.thetaCodo !== null) { const dd = th - e.thetaCodo; k += 0.5 * dd * dd; }
      for (const f of [0, 0.35, 0.7]) {
        const p = E.clone().lerp(T, f);
        k += 400 * this._empuje(p, 0.045).length();
      }
      if (E.y > S.y + 0.03) k += 8 * (E.y - S.y);
      if (E.z < S.z - 0.1) k += 8 * (S.z - 0.1 - E.z);
      return k;
    };
    let mejor = 0, mejorK = Infinity;
    for (let th = -Math.PI; th < Math.PI; th += Math.PI / 18) {
      const k = costo(th);
      if (k < mejorK) { mejorK = k; mejor = th; }
    }
    for (let th = mejor - 0.15; th <= mejor + 0.15; th += 0.03) {
      const k = costo(th);
      if (k < mejorK) { mejorK = k; mejor = th; }
    }
    e.thetaCodo = mejor;
    const E = codoEn(mejor);

    // Brazo: dirección al codo + bisagra del codo alineada con el plano del brazo
    const dirBrazo = E.clone().sub(S).normalize();
    const dirAnte = T.clone().sub(E).normalize();
    let bisagra = new THREE.Vector3().crossVectors(dirBrazo, dirAnte);
    if (bisagra.lengthSq() < 1e-6) bisagra = c.bisagraReposo.clone();
    bisagra.normalize();
    const qBrazo = alinear(c.brazoDirReposo, c.bisagraReposo, dirBrazo, bisagra).multiply(c.qBrazoReposo);
    this._fijarMundo(`${lado}Arm`, qBrazo);
    // Antebrazo: gira sobre la bisagra hasta apuntar a la muñeca
    this._apuntar(`${lado}ForeArm`, `${lado}Hand`, dirAnte);

    // Mano: orientación objetivo a partir de su dirección y normal
    const N = normalMano.clone().sub(dirMano.clone().multiplyScalar(normalMano.dot(dirMano))).normalize();
    const qManoMundo = alinear(c.D, c.N, dirMano, N).multiply(c.qManoReposo);
    const ante = this.huesos[`${lado}ForeArm`], mano = this.huesos[`${lado}Hand`];
    ante.updateMatrixWorld(true);
    const qAnte = ante.getWorldQuaternion(new THREE.Quaternion());
    const local = qAnte.clone().invert().multiply(qManoMundo);
    const desvio = this.reposo[`${lado}Hand`].clone().invert().multiply(local);
    let { giro, columpio } = separarGiro(desvio, c.ejeLargoMano);
    columpio = limitarAngulo(columpio, MAX_FLEXION_MUNECA);
    // Ángulo de giro con signo, limitado y continuo respecto al cuadro anterior
    const gv = new THREE.Vector3(giro.x, giro.y, giro.z);
    let phi = 2 * Math.atan2(gv.dot(c.ejeLargoMano), giro.w);
    if (phi > Math.PI) phi -= 2 * Math.PI;
    if (phi < -Math.PI) phi += 2 * Math.PI;
    phi = Math.max(-MAX_GIRO_ANTEBRAZO, Math.min(MAX_GIRO_ANTEBRAZO, phi));
    const parteAnte = new THREE.Quaternion().setFromAxisAngle(c.ejeLargoAntebrazo, phi * FRACCION_GIRO_ANTEBRAZO);
    ante.quaternion.multiply(parteAnte);
    const giroLim = new THREE.Quaternion().setFromAxisAngle(c.ejeLargoMano, phi);
    const localFinal = this.reposo[`${lado}Hand`].clone().multiply(columpio).multiply(giroLim);
    mano.quaternion.copy(parteAnte.clone().invert().multiply(localFinal));
    ante.updateMatrixWorld(true);
    // Diagnóstico: cuánto se dobló la muñeca, cuánto giró y cuánto se alejó la palma del dato
    const real = mano.getWorldQuaternion(new THREE.Quaternion());
    this.diag = this.diag || {};
    this.diag[lado] = {
      muneca: anguloDe(columpio) / RAD, giro: phi / RAD,
      errorPalma: anguloDe(real.clone().invert().multiply(qManoMundo)) / RAD,
    };
  }

  _fijarMundo(nombre, qMundo) {
    const b = this.huesos[nombre];
    const qPadre = b.parent.getWorldQuaternion(new THREE.Quaternion());
    b.quaternion.copy(qPadre.invert().multiply(qMundo));
    b.updateMatrixWorld(true);
  }

  /** Rota `hueso` desde su reposo para que el segmento hacia `hijo` apunte a `dirMundo`. */
  _apuntar(hueso, hijo, dirMundo) {
    const b = this.huesos[hueso], c = this.huesos[hijo];
    if (!b || !c) return;
    b.quaternion.copy(this.reposo[hueso]);
    b.updateMatrixWorld(true);
    const qPadreInv = b.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
    const restDir = c.position.clone().normalize().applyQuaternion(this.reposo[hueso]);
    const objetivo = dirMundo.clone().applyQuaternion(qPadreInv).normalize();
    b.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(restDir, objetivo).multiply(this.reposo[hueso]));
    b.updateMatrixWorld(true);
  }

  _aplicarDedos(lado) {
    const e = this._estado(lado), c = this.cal[lado];
    const ang = e.dedos.x;
    const q = new THREE.Quaternion();
    DEDOS.forEach((dedo, d) => {
      const [base, apertura, medio, punta] = [ang[d * 4], ang[d * 4 + 1], ang[d * 4 + 2], ang[d * 4 + 3]];
      const flex = [base, medio, punta];
      for (let k = 0; k < 3; k++) {
        const nombre = `${lado}Hand${dedo}${k + 1}`;
        const ej = c.dedos[dedo][k];
        const b = this.huesos[nombre];
        b.quaternion.copy(this.reposo[nombre]).multiply(q.setFromAxisAngle(ej.flex, ej.sFlex * flex[k] * RAD));
        if (k === 0) b.quaternion.multiply(q.setFromAxisAngle(ej.aper, ej.sAper * apertura * RAD));
      }
    });
    // Pulgar: direcciones en el marco actual de la mano
    const mano = this.huesos[`${lado}Hand`];
    mano.updateMatrixWorld(true);
    const qDelta = mano.getWorldQuaternion(new THREE.Quaternion()).multiply(c.qManoReposo.clone().invert());
    const D = c.D.clone().applyQuaternion(qDelta), N = c.N.clone().applyQuaternion(qDelta), L = c.L.clone().applyQuaternion(qDelta);
    const pul = e.pulgar.x;
    const dir = k => D.clone().multiplyScalar(pul[k * 3]).add(N.clone().multiplyScalar(pul[k * 3 + 1]))
      .add(L.clone().multiplyScalar(pul[k * 3 + 2])).normalize();
    this._apuntar(`${lado}HandThumb1`, `${lado}HandThumb2`, dir(0));
    this._apuntar(`${lado}HandThumb2`, `${lado}HandThumb3`, dir(1));
    // La última falange no tiene hijo: se usa la dirección de la anterior como reposo
    const b3 = this.huesos[`${lado}HandThumb3`];
    b3.quaternion.copy(this.reposo[`${lado}HandThumb3`]);
    b3.updateMatrixWorld(true);
    const ref = this._w(`${lado}HandThumb3`).sub(this._w(`${lado}HandThumb2`)).normalize();
    const qp = b3.getWorldQuaternion(new THREE.Quaternion());
    const giro = new THREE.Quaternion().setFromUnitVectors(ref, dir(2));
    this._fijarMundo(`${lado}HandThumb3`, giro.multiply(qp));
  }

  // ----------------------------------------------------- vida
  _idle(dt) {
    // Respiración: leve balanceo del pecho
    const r = Math.sin(this.tiempo * 2 * Math.PI / 4.2) * 0.012;
    for (const n of ['Spine1', 'Spine2']) {
      if (this.huesos[n]) this.huesos[n].quaternion.copy(this.reposo[n]).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), r));
    }
    // Parpadeo cada 2.5–5 s
    this._proxParpadeo = this._proxParpadeo ?? this.tiempo + 2;
    let v = 0;
    const dtp = this.tiempo - this._proxParpadeo;
    if (dtp > 0) {
      v = dtp < 0.08 ? dtp / 0.08 : dtp < 0.16 ? 1 - (dtp - 0.08) / 0.08 : 0;
      if (dtp > 0.16) this._proxParpadeo = this.tiempo + 2.5 + Math.random() * 2.5;
    }
    for (const m of this.parpados) {
      for (const k of ['eyeBlinkLeft', 'eyeBlinkRight']) {
        const i = m.morphTargetDictionary[k];
        if (i !== undefined) m.morphTargetInfluences[i] = v;
      }
    }
    this.modelo.updateMatrixWorld(true);
  }

  /** Transición suave entre la animación hecha a mano y el control por datos. */
  _instantanea() {
    const s = {};
    for (const [n, h] of Object.entries(this.huesos)) s[n] = h.quaternion.clone();
    return s;
  }

  _mezclarDesde(instantanea, dur = 0.3) { this._mezcla = { desde: instantanea, t: 0, dur }; }

  _aplicarMezcla(dt) {
    const m = this._mezcla;
    if (!m) return;
    m.t += dt;
    const k = Math.min(m.t / m.dur, 1);
    const s = k * k * (3 - 2 * k);
    for (const [n, h] of Object.entries(this.huesos)) {
      const destino = h.quaternion.clone();
      h.quaternion.copy(m.desde[n]).slerp(destino, s);
    }
    this.modelo.updateMatrixWorld(true);
    if (k >= 1) this._mezcla = null;
  }

  // ---------------------------------------------------------- API
  /** Pose de descanso inmediata (sin transición). */
  poseDescanso() {
    this.actual = null;
    for (const lado of LADOS) {
      const e = this._estado(lado), d = this._descanso(lado);
      e.muneca.fijar(d.muneca); e.codo.fijar(d.codo);
      e.dir.fijar(d.mano.dir); e.normal.fijar(d.mano.normal);
      e.dedos.fijar(d.dedos.flat()); e.pulgar.fijar(d.pulgar.flat());
      e.thetaCodo = null;
    }
    if (this.senas && this.senas.senas['5'] && !this._pulgarRelajado) {
      this._pulgarRelajado = this.senas.senas['5'].cuadros[0].Right.pulgar;
    }
    this._idle(0);
    this._resolverCuerpo();
  }

  duracionTotal(nombre) {
    if (this.clips[nombre]) return this.clips[nombre].duration;
    const s = this.senas && this.senas.senas[nombre];
    return s ? ENTRADA + s.duracion + (s.estatica ? 0 : SOSTEN_DINAMICA) : 0;
  }

  /** Reproduce una seña. Se resuelve al terminar (la mano queda en el espacio de señas). */
  senar(nombre) {
    return new Promise(resolve => {
      this._detenerClip(false);
      if (this.actual) { const f = this.actual.fin; this.actual = null; f(false); }
      if (this.clips[nombre]) {
        const antes = this._instantanea();
        this.accionClip = this.mixer.clipAction(this.clips[nombre]);
        this.accionClip.reset().setLoop(THREE.LoopOnce, 1).play();
        this.accionClip.clampWhenFinished = true;
        this._mezclarDesde(antes, 0.25);
        this._resolverClip = resolve;
      } else if (this.senas && this.senas.senas[nombre]) {
        this.actual = { sena: this.senas.senas[nombre], t: 0, fin: resolve };
      } else {
        resolve(false);
      }
    });
  }

  /** Seña una secuencia sin bajar las manos entre señas; al final vuelve a descanso. */
  async senarSecuencia(nombres, alCambiar) {
    this._secuenciaId = (this._secuenciaId || 0) + 1;
    const id = this._secuenciaId;
    for (const n of nombres) {
      if (id !== this._secuenciaId) return;
      if (alCambiar) alCambiar(n);
      await this.senar(n);
    }
    if (id === this._secuenciaId && alCambiar) alCambiar(null);
  }

  cancelar() {
    this._secuenciaId = (this._secuenciaId || 0) + 1;
    this._detenerClip(false);
    if (this.actual) { const f = this.actual.fin; this.actual = null; f(false); }
  }

  _detenerClip(mezclar = true) {
    if (!this.accionClip) return;
    const antes = this._instantanea();
    this.accionClip.stop();
    this.accionClip = null;
    for (const [n, q] of Object.entries(this.reposo)) this.huesos[n].quaternion.copy(q);
    if (mezclar) this._mezclarDesde(antes, 0.3);
    for (const lado of LADOS) this._estado(lado).thetaCodo = null;
    if (this._resolverClip) { const r = this._resolverClip; this._resolverClip = null; r(false); }
  }

  _finClip() {
    const r = this._resolverClip;
    this._resolverClip = null;
    this._detenerClip(true);
    if (r) r(true);
  }
}
