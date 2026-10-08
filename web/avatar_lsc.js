/**
 * Avatar LSC: carga avatar_lsc.glb con three.js y reproduce señas.
 *
 * Las señas salen de señas_avatar.json (generado por scripts/generar_senas_avatar.py
 * a partir de las grabaciones reales de LSC70): por cada cuadro trae la dirección
 * de cada segmento del brazo y de cada falange. Aquí se convierten en rotaciones
 * de los huesos Mixamo del avatar. HOLA usa la animación hecha a mano del .glb.
 *
 * Solo renderiza mientras la sección del avatar está visible (iniciar/detener),
 * para no competir con la cámara y MediaPipe.
 */
import * as THREE from 'three';
import { GLTFLoader } from './GLTFLoader.js';

const PREFIJO = 'mixamorig';  // GLTFLoader quita los ':' de los nombres
const FALANGES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];

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
    this.senas = null;
    this.reproduccion = null;  // { frames, t, duracion, alTerminar }
    this.cola = [];
    this._bucle = this._bucle.bind(this);
    new ResizeObserver(() => this._ajustarTamano()).observe(contenedor);
  }

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
    this.modelo.traverse(o => {
      if (o.isBone) this.huesos[o.name.replace(PREFIJO, '')] = o;
      if (o.isMesh) o.frustumCulled = false;
    });
    // Pose de descanso guardada para volver a ella y como referencia del retarget
    this.reposo = {};
    for (const [n, h] of Object.entries(this.huesos)) this.reposo[n] = h.quaternion.clone();

    this._encuadrar();
    this.poseDescanso();
    return this;
  }

  nombresHuesos() { return Object.keys(this.huesos); }
  senasDisponibles() {
    return [...Object.keys(this.senas ? this.senas.senas : {}), ...Object.keys(this.clips)]
      .filter((v, i, a) => a.indexOf(v) === i);
  }

  _encuadrar() {
    const caja = new THREE.Box3().setFromObject(this.modelo);
    const cabeza = this.huesos.Head ? this.huesos.Head.getWorldPosition(new THREE.Vector3()) : caja.max;
    // Plano medio: de la cintura a un palmo sobre la cabeza (manos levantadas incluidas)
    const techo = cabeza.y + 0.32;
    const piso = caja.min.y + 0.08;
    this.encuadre = { yCentro: (techo + piso) / 2, alto: (techo - piso) * 1.08, ancho: 1.0,
                      z: caja.getCenter(new THREE.Vector3()).z };
    this._colocarCamara();
  }

  _colocarCamara() {
    const e = this.encuadre;
    if (!e) return;
    const t = Math.tan(THREE.MathUtils.degToRad(this.camara.fov / 2));
    // Lo que limite más: el alto (cabeza a cintura) o el ancho (brazos abiertos)
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

  iniciar() {
    if (this.activo) return;
    this.activo = true;
    this.reloj.getDelta();
    requestAnimationFrame(this._bucle);
  }

  detener() { this.activo = false; }

  _bucle() {
    if (!this.activo) return;
    const dt = Math.min(this.reloj.getDelta(), 0.05);
    if (this.accionClip) this.mixer.update(dt);
    else if (this.reproduccion) this._avanzar(dt);
    this._dibujar();
    requestAnimationFrame(this._bucle);
  }

  _dibujar() { if (this.modelo) this.renderer.render(this.escena, this.camara); }

  /** Brazos abajo y manos relajadas (la T-pose del .glb no sirve para señar). */
  poseDescanso() {
    for (const [n, q] of Object.entries(this.reposo)) this.huesos[n].quaternion.copy(q);
    const abajo = new THREE.Vector3(0, -1, 0.08).normalize();
    for (const lado of ['Left', 'Right']) {
      const x = lado === 'Right' ? -1 : 1;
      this._apuntar(`${lado}Arm`, `${lado}ForeArm`, new THREE.Vector3(0.18 * x, -1, 0.05).normalize());
      this._apuntar(`${lado}ForeArm`, `${lado}Hand`, new THREE.Vector3(0.08 * x, -1, 0.25).normalize());
      this._apuntar(`${lado}Hand`, `${lado}HandMiddle1`, abajo);
    }
  }

  /**
   * Rota `hueso` para que el segmento hacia `hijo` apunte a `dirMundo`.
   * Si se da `lateralMundo`, también fija el giro sobre ese eje (palma).
   */
  _apuntar(hueso, hijo, dirMundo, lateralMundo = null, hijoLateral = null) {
    const b = this.huesos[hueso], c = this.huesos[hijo];
    if (!b || !c) return;
    b.quaternion.copy(this.reposo[hueso]);
    b.updateMatrixWorld(true);
    const qPadre = b.parent.getWorldQuaternion(new THREE.Quaternion());
    const qPadreInv = qPadre.clone().invert();
    // Dirección de reposo del segmento en el espacio del padre
    const restDir = c.position.clone().normalize().applyQuaternion(this.reposo[hueso]);
    const objetivo = dirMundo.clone().applyQuaternion(qPadreInv).normalize();
    let q = new THREE.Quaternion().setFromUnitVectors(restDir, objetivo);
    if (lateralMundo && hijoLateral && this.huesos[hijoLateral]) {
      // Alinear también el eje lateral (p. ej. índice -> meñique) proyectado sobre el plano perpendicular
      const lat0 = this.huesos[hijoLateral].position.clone().sub(c.position)
        .applyQuaternion(this.reposo[hueso]).applyQuaternion(q);
      const lat1 = lateralMundo.clone().applyQuaternion(qPadreInv);
      const proyectar = v => v.sub(objetivo.clone().multiplyScalar(v.dot(objetivo))).normalize();
      proyectar(lat0); proyectar(lat1);
      if (lat0.lengthSq() > 0.5 && lat1.lengthSq() > 0.5) {
        const ang = Math.atan2(objetivo.dot(new THREE.Vector3().crossVectors(lat0, lat1)), lat0.dot(lat1));
        q = new THREE.Quaternion().setFromAxisAngle(objetivo, ang).multiply(q);
      }
    }
    b.quaternion.copy(q.multiply(this.reposo[hueso]));
    b.updateMatrixWorld(true);
  }

  /** Aplica un cuadro de seña: direcciones en el mundo del avatar (x der. del espectador, y arriba, z hacia la cámara). */
  _aplicarCuadro(f) {
    const V = a => new THREE.Vector3(a[0], a[1], a[2]).normalize();
    for (const lado of ['Right', 'Left']) {
      const L = f[lado];
      if (!L) continue;
      if (L.brazo) this._apuntar(`${lado}Arm`, `${lado}ForeArm`, V(L.brazo));
      if (L.antebrazo) this._apuntar(`${lado}ForeArm`, `${lado}Hand`, V(L.antebrazo));
      if (L.mano) {
        this._apuntar(`${lado}Hand`, `${lado}HandMiddle1`, V(L.mano.dir), V(L.mano.lateral),
          `${lado}HandPinky1`);
      }
      if (L.dedos) {
        FALANGES.forEach((dedo, i) => {
          const segs = L.dedos[i];
          if (!segs) return;
          for (let k = 0; k < 3; k++) {
            const h = `${lado}Hand${dedo}${k + 1}`;
            const hijo = k < 2 ? `${lado}Hand${dedo}${k + 2}` : null;
            if (hijo) this._apuntar(h, hijo, V(segs[k]));
            else this._apuntarPunta(h, V(segs[k]));
          }
        });
      }
    }
  }

  /** La última falange no tiene hijo: se usa la dirección de la falange anterior como reposo. */
  _apuntarPunta(hueso, dirMundo) {
    const b = this.huesos[hueso];
    if (!b) return;
    b.quaternion.copy(this.reposo[hueso]);
    const restDir = b.position.clone().normalize().applyQuaternion(this.reposo[hueso]);
    const qPadreInv = b.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
    const objetivo = dirMundo.clone().applyQuaternion(qPadreInv).normalize();
    b.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(restDir, objetivo).multiply(this.reposo[hueso]));
    b.updateMatrixWorld(true);
  }

  /** Reproduce una seña por nombre. Devuelve una promesa que se resuelve al terminar. */
  senar(nombre) {
    return new Promise(resolve => {
      this._detenerClip();
      // Las animaciones hechas a mano en el .glb (HOLA) tienen prioridad sobre las del dataset
      if (this.clips[nombre]) {
        this.poseDescanso();
        this.accionClip = this.mixer.clipAction(this.clips[nombre]);
        this.accionClip.reset().setLoop(THREE.LoopOnce, 1).play();
        this.accionClip.clampWhenFinished = true;
        this._resolverClip = resolve;
      } else if (this.senas && this.senas.senas[nombre]) {
        const s = this.senas.senas[nombre];
        this.reproduccion = { frames: s.cuadros, t: 0, duracion: s.duracion, alTerminar: resolve };
      } else {
        resolve(false);
      }
    });
  }

  /** Seña una secuencia (palabras del vocabulario o letras a deletrear). */
  async senarSecuencia(nombres, alCambiar) {
    this._secuenciaId = (this._secuenciaId || 0) + 1;
    const id = this._secuenciaId;
    for (const n of nombres) {
      if (id !== this._secuenciaId) return;
      if (alCambiar) alCambiar(n);
      await this.senar(n);
    }
    if (id === this._secuenciaId) { this.poseDescanso(); if (alCambiar) alCambiar(null); }
  }

  cancelar() {
    this._secuenciaId = (this._secuenciaId || 0) + 1;
    this._detenerClip();
    if (this.reproduccion) { const r = this.reproduccion; this.reproduccion = null; r.alTerminar(false); }
    this.poseDescanso();
  }

  _detenerClip() {
    if (this.accionClip) { this.accionClip.stop(); this.accionClip = null; }
  }

  _finClip() {
    this._detenerClip();
    this.poseDescanso();
    if (this._resolverClip) { const r = this._resolverClip; this._resolverClip = null; r(true); }
  }

  _avanzar(dt) {
    const r = this.reproduccion;
    r.t += dt;
    const n = r.frames.length;
    // Entrada desde descanso (0.25 s), seña, y sostén final (0.35 s)
    const entrada = 0.25, sosten = 0.35;
    const tSena = Math.max(0, r.t - entrada);
    const u = Math.min(tSena / r.duracion, 1) * (n - 1);
    const i = Math.min(Math.floor(u), n - 2 >= 0 ? n - 2 : 0);
    const a = r.frames[i], b = r.frames[Math.min(i + 1, n - 1)];
    const k = n > 1 ? u - i : 0;
    this.poseDescanso();
    const cuadro = interpolarCuadro(a, b, k);
    if (r.t < entrada) {
      // Mezcla desde la pose de descanso hacia el primer cuadro
      this._mezclarDesdeDescanso(r.frames[0], r.t / entrada);
    } else {
      this._aplicarCuadro(cuadro);
    }
    if (r.t >= entrada + r.duracion + sosten) {
      this.reproduccion = null;
      this.poseDescanso();
      r.alTerminar(true);
    }
  }

  _mezclarDesdeDescanso(cuadro, k) {
    const guardado = {};
    for (const [n, h] of Object.entries(this.huesos)) guardado[n] = h.quaternion.clone();
    this._aplicarCuadro(cuadro);
    const s = k * k * (3 - 2 * k);
    for (const [n, h] of Object.entries(this.huesos)) {
      const destino = h.quaternion.clone();
      h.quaternion.copy(guardado[n]).slerp(destino, s);
    }
  }
}

function interpolarVec(a, b, k) {
  if (!a) return b; if (!b) return a;
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

function interpolarCuadro(a, b, k) {
  const out = {};
  for (const lado of ['Right', 'Left']) {
    const A = a[lado], B = b[lado];
    if (!A && !B) continue;
    const X = A || B, Y = B || A;
    out[lado] = {
      brazo: interpolarVec(X.brazo, Y.brazo, k),
      antebrazo: interpolarVec(X.antebrazo, Y.antebrazo, k),
      mano: X.mano && Y.mano ? { dir: interpolarVec(X.mano.dir, Y.mano.dir, k), lateral: interpolarVec(X.mano.lateral, Y.mano.lateral, k) } : (X.mano || Y.mano),
      dedos: X.dedos && Y.dedos ? X.dedos.map((d, i) => d && Y.dedos[i] ? d.map((s, j) => interpolarVec(s, Y.dedos[i][j], k)) : d) : (X.dedos || Y.dedos),
    };
  }
  return out;
}
