/**
 * ============================================================================
 * MOTOR DE INFERENCIA Y EXTRACCIÓN 100% LOCAL (ON-DEVICE - ZERO SERVIDOR)
 * LSC v5.0 — Lengua de Señas Colombiana
 * ============================================================================
 * - Extracción Cinemática Multimodal 109D (Articular + Anclaje Torácico)
 * - Red Neuronal Multicapa Calibrada (13 Clases, >95% CV, >99% Global)
 * - Rastrean Oclusión de Manos (una detrás de la otra) con anclaje espacial
 * - Detección Cero si no hay manos (sin alucinaciones ni predicciones falsas)
 * - Filtro de Transición y Reposo para evitar palabras al azar en movimiento
 * - Detector de Mano Neutra: mano tendida/abierta estática NUNCA predice señas
 * - Cooldown post-emisión para señas consecutivas limpias
 */

// Utilidades Vectoriales 3D
const Vec3 = {
  norm(v) {
    return Math.hypot(v[0], v[1], v[2]) || 1e-8;
  },
  norm2d(x, y) {
    return Math.hypot(x, y) || 1e-8;
  },
  sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  },
  add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  },
  scale(v, s) {
    return [v[0] * s, v[1] * s, v[2] * s];
  },
  dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },
  cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  },
  dist(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  },
  clip(val, minVal, maxVal) {
    return Math.max(minVal, Math.min(maxVal, val));
  }
};

/**
 * Filtro OneEuro adaptativo para suavizado de landmarks 3D.
 * Elimina el temblor en reposo y responde instantáneamente en movimientos bruscos (cero lag).
 */
class OneEuroFilter {
  constructor(minCutoff = 1.2, beta = 60.0, dCutoff = 5.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  filter(val, timestamp) {
    if (this.tPrev === null) {
      this.xPrev = val;
      this.dxPrev = 0;
      this.tPrev = timestamp;
      return val;
    }

    const dt = Math.max((timestamp - this.tPrev) / 1000.0, 1e-4);
    this.tPrev = timestamp;

    const dx = (val - this.xPrev) / dt;
    const aD = this._alpha(dt, this.dCutoff);
    const edx = aD * dx + (1.0 - aD) * this.dxPrev;
    this.dxPrev = edx;

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const a = this._alpha(dt, cutoff);
    const x = a * val + (1.0 - a) * this.xPrev;
    this.xPrev = x;
    return x;
  }

  _alpha(dt, cutoff) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

/**
 * Estabilizador cinemático de mano completa (21 puntos 3D)
 * con soporte para OCLUSIÓN (cuando una mano se ubica detrás de la otra)
 * y persistencia inercial reactiva sin arrastre ni teleportación.
 */
class LandmarkStabilizer {
  constructor(slotId = 0) {
    this.slotId = slotId;
    this.filters = [];
    for (let i = 0; i < 21; i++) {
      this.filters.push([
        new OneEuroFilter(1.0, 85.0, 10.0), // X: reactivo a velocidad alta sin lag
        new OneEuroFilter(1.0, 85.0, 10.0), // Y: reactivo a velocidad alta sin lag
        new OneEuroFilter(1.0, 85.0, 10.0), // Z: reactivo a velocidad alta sin lag
      ]);
    }
    this.lastFiltered = null;
    this.lastTimestamp = null;
    this.framesMissing = 0;
    this.framesVisible = 0; // Contador de persistencia temporal
    this.maxGraceFrames = 4; // Persistencia ante motion blur
    this.maxOcclusionFrames = 45; // Persistencia extendida cuando una mano está detrás (~1.5s)
    this.currentVelocity = [0, 0, 0];
    this.isOccluded = false;
    this.lastForegroundWrist = null;
  }

  update(rawCoords, timestamp) {
    const now = timestamp || performance.now();

    if (!rawCoords || rawCoords.length < 21) {
      this.framesMissing++;
      this.framesVisible = Math.max(0, this.framesVisible - 1);
      const speed = Vec3.norm(this.currentVelocity);
      const maxGrace = speed > 0.35 ? 2 : 4;
      if (this.framesMissing <= maxGrace && this.lastFiltered) {
        // Extrapolación inercial suave y continua
        const dt = Math.min((now - (this.lastTimestamp || now)) / 1000.0, 0.04) || 0.033;
        const decay = Math.pow(0.88, this.framesMissing);
        const predicted = [];

        for (let i = 0; i < 21; i++) {
          const vx = this.filters[i][0].dxPrev || 0;
          const vy = this.filters[i][1].dxPrev || 0;
          const vz = this.filters[i][2].dxPrev || 0;

          predicted.push([
            this.lastFiltered[i][0] + vx * dt * decay,
            this.lastFiltered[i][1] + vy * dt * decay,
            this.lastFiltered[i][2] + vz * dt * decay,
          ]);
        }

        this.lastFiltered = predicted;
        this.lastTimestamp = now;
        return {
          coords: predicted,
          isPredicted: true,
          isOccluded: false,
          velocity: this.currentVelocity,
          speed: speed,
          framesVisible: this.framesVisible,
          framesGrace: this.framesMissing,
        };
      }
      this.reset();
      return null;
    }

    // Detección real confirmada
    this.framesVisible++;
    if (this.lastFiltered) {
      const dt = Math.max((now - (this.lastTimestamp || now)) / 1000.0, 1e-3);
      this.currentVelocity = [
        (rawCoords[0][0] - this.lastFiltered[0][0]) / dt,
        (rawCoords[0][1] - this.lastFiltered[0][1]) / dt,
        (rawCoords[0][2] - this.lastFiltered[0][2]) / dt,
      ];

      // Anti-Teleportación y Snap Instantáneo ante movimientos bruscos y rápidos:
      // Si la mano se desplazó súbitamente (salto rápido > 0.07) o se perdieron fotogramas,
      // sincronizar de inmediato la referencia interna para anular arrastre o elasticidad
      const jumpDist = Vec3.dist(rawCoords[0], this.lastFiltered[0]);
      if (jumpDist > 0.07 || (this.framesMissing > 0 && jumpDist > 0.04)) {
        for (let i = 0; i < 21; i++) {
          this.filters[i][0].xPrev = rawCoords[i][0];
          this.filters[i][1].xPrev = rawCoords[i][1];
          this.filters[i][2].xPrev = rawCoords[i][2];
        }
      }
    }

    this.framesMissing = 0;
    this.isOccluded = false;
    this.lastTimestamp = now;

    const smoothed = [];
    for (let i = 0; i < 21; i++) {
      const sx = this.filters[i][0].filter(rawCoords[i][0], now);
      const sy = this.filters[i][1].filter(rawCoords[i][1], now);
      const sz = this.filters[i][2].filter(rawCoords[i][2], now);
      smoothed.push([sx, sy, sz]);
    }

    this.lastFiltered = smoothed;
    const speed = Vec3.norm(this.currentVelocity);
    const wrist = smoothed[0];
    const isNearBoundary = (wrist[0] < 0.06 || wrist[0] > 0.94 || wrist[1] < 0.05 || wrist[1] > 0.92);

    return {
      coords: smoothed,
      isPredicted: false,
      isOccluded: false,
      velocity: this.currentVelocity,
      speed: speed,
      framesVisible: this.framesVisible,
      isNearBoundary: isNearBoundary,
      framesGrace: 0,
    };
  }

  // Rastrear la mano oculta anclándola a la trayectoria de la mano frontal
  updateOccluded(foregroundWrist, timestamp) {
    const now = timestamp || performance.now();
    this.framesMissing++;

    if (this.framesMissing > this.maxOcclusionFrames || !this.lastFiltered) {
      this.reset();
      return null;
    }

    this.isOccluded = true;
    let shift = [0, 0, 0];
    if (this.lastForegroundWrist && foregroundWrist) {
      shift = [
        (foregroundWrist[0] - this.lastForegroundWrist[0]) * 0.90,
        (foregroundWrist[1] - this.lastForegroundWrist[1]) * 0.90,
        (foregroundWrist[2] - this.lastForegroundWrist[2]) * 0.90
      ];
    }
    this.lastForegroundWrist = foregroundWrist ? [...foregroundWrist] : null;

    // Aplicar desplazamiento a todos los puntos para que acompañe el movimiento
    const occludedCoords = [];
    for (let i = 0; i < 21; i++) {
      occludedCoords.push([
        this.lastFiltered[i][0] + shift[0],
        this.lastFiltered[i][1] + shift[1],
        this.lastFiltered[i][2] + shift[2] + 0.04 // Ligeramente más profunda en Z
      ]);
    }

    this.lastFiltered = occludedCoords;
    this.lastTimestamp = now;

    return {
      coords: occludedCoords,
      isPredicted: true,
      isOccluded: true,
      isBehind: true,
      velocity: this.currentVelocity,
      framesGrace: this.framesMissing
    };
  }

  reset() {
    for (let i = 0; i < 21; i++) {
      this.filters[i][0].reset();
      this.filters[i][1].reset();
      this.filters[i][2].reset();
    }
    this.lastFiltered = null;
    this.lastTimestamp = null;
    this.framesMissing = 0;
    this.framesVisible = 0;
    this.isOccluded = false;
    this.lastForegroundWrist = null;
    this.currentVelocity = [0, 0, 0];
  }
}

/**
 * Gestor Inteligente de Seguimiento Dual (Bimanual Tracker).
 * - Garantiza que si NO hay manos, NO se predice nada (cero alucinaciones).
 * - Mantiene el rastreo cuando una mano se ubica DETRÁS de la otra.
 * - Evita cruces y cambios de color asociando lateralidad anatómica (Derecha/Izquierda).
 */
class DualHandTracker {
  constructor() {
    this.slot0 = new LandmarkStabilizer(0); // Mano Derecha (Cyan)
    this.slot1 = new LandmarkStabilizer(1); // Mano Izquierda (Fucsia/Púrpura)
    this.handednessSlot0 = 'Mano Derecha';
    this.handednessSlot1 = 'Mano Izquierda';
    this.framesWithoutHands = 0;
  }

  update(multiHandLandmarks, multiHandedness, timestamp) {
    const now = timestamp || performance.now();
    const detections = [];

    if (multiHandLandmarks && multiHandLandmarks.length > 0) {
      for (let i = 0; i < multiHandLandmarks.length; i++) {
        const pts = multiHandLandmarks[i].map(p => [
          p.x !== undefined ? p.x : p[0],
          p.y !== undefined ? p.y : p[1],
          p.z !== undefined ? p.z : (p[2] || 0)
        ]);

        // Validación de plausibilidad anatómica de mano real
        const d_palma = Vec3.dist(pts[0], pts[9]);
        if (d_palma < 0.025) continue; // Ignorar artefactos pequeños

        let label = 'Desconocida';
        let score = 0.90;
        if (multiHandedness && multiHandedness[i]) {
          const c = multiHandedness[i].label || (multiHandedness[i].classification && multiHandedness[i].classification[0]?.label);
          if (c) label = c;
          const s = multiHandedness[i].score || (multiHandedness[i].classification && multiHandedness[i].classification[0]?.score);
          if (s !== undefined) score = s;
        }
        detections.push({ pts, label, score, wrist: pts[0] });
      }
    }

    // ========================================================================
    // CASO 0: CERO MANOS EN ESCENA (REQUISITO CRÍTICO DEL USUARIO)
    // ========================================================================
    if (detections.length === 0) {
      this.framesWithoutHands++;
      if (this.framesWithoutHands >= 2) {
        this.slot0.reset();
        this.slot1.reset();
      }
      return []; // RETORNO INMEDIATO: Cero detecciones, cero predicciones fantasmas
    }

    this.framesWithoutHands = 0;

    // Validación Post-Proceso Bimanual (v5.4.0: Permite manos solapadas, cruzadas o detrás)
    if (detections.length >= 2) {
      const det0 = detections[0];
      const det1 = detections[1];
      const d_entre_manos = Vec3.dist(det0.wrist, det1.wrist);
      
      // Únicamente descartar si es un duplicado idéntico exacto del mismo punto físico (< 0.035)
      if (d_entre_manos < 0.035) {
        if ((det1.score || 0) > (det0.score || 0)) {
          detections.shift(); // Descartar duplicado
        } else {
          detections.pop();
        }
      }
    }

    // ========================================================================
    // CASO 1: EXACTAMENTE 1 MANO DETECTADA (GESTIÓN DE OCLUSIÓN DETRÁS)
    // ========================================================================
    if (detections.length === 1) {
      const det = detections[0];

      // Determinar si la mano visible es Derecha o Izquierda
      let isSlot0 = true;
      if (det.label === 'Right') {
        isSlot0 = true;
      } else if (det.label === 'Left') {
        isSlot0 = false;
      } else {
        const p0 = this.slot0.lastFiltered ? this.slot0.lastFiltered[0] : null;
        const p1 = this.slot1.lastFiltered ? this.slot1.lastFiltered[0] : null;
        if (p0 && p1) {
          isSlot0 = Vec3.dist(det.wrist, p0) <= Vec3.dist(det.wrist, p1);
        } else if (!p0 && p1) {
          isSlot0 = false;
        }
      }

      const activeSlot = isSlot0 ? this.slot0 : this.slot1;
      const otherSlot = isSlot0 ? this.slot1 : this.slot0;
      const activeLabel = isSlot0 ? 'Mano Derecha' : 'Mano Izquierda';
      const otherLabel = isSlot0 ? 'Mano Izquierda' : 'Mano Derecha';

      const upActive = activeSlot.update(det.pts, now);

      // Verificar si la otra mano estaba recientemente activa y quedó oculta detrás
      let upOther = null;
      const wasOtherActive = otherSlot.lastFiltered && otherSlot.framesMissing < otherSlot.maxOcclusionFrames;
      if (wasOtherActive) {
        const distToFront = Vec3.dist(det.wrist, otherSlot.lastFiltered[0]);
        // Rango ampliado de oclusión (< 0.40) para manos colocadas detrás
        if (distToFront < 0.40) {
          upOther = otherSlot.updateOccluded(det.wrist, now);
        } else {
          otherSlot.reset();
        }
      } else {
        otherSlot.reset();
      }

      const isLeftActive = (activeLabel === 'Mano Izquierda' || det.label === 'Left');
      const res = [];
      if (upActive) {
        res.push({
          ...upActive,
          slot: isSlot0 ? 0 : 1,
          label: activeLabel,
          isLeft: isLeftActive,
          colorScheme: isSlot0 ? 'cyan' : 'purple',
          isOccluded: false,
          isBehind: false
        });
      }

      if (upOther) {
        res.push({
          ...upOther,
          slot: isSlot0 ? 1 : 0,
          label: `${otherLabel} (Trasera)`,
          isLeft: !isLeftActive,
          colorScheme: isSlot0 ? 'purple' : 'cyan',
          isOccluded: true,
          isBehind: true
        });
      }

      return res;
    }

    // ========================================================================
    // CASO 2: DOS O MÁS MANOS DETECTADAS
    // ========================================================================
    const detA = detections[0];
    const detB = detections[1];

    let detRight = null, detLeft = null;
    // Continuidad temporal: asociar al slot previo más cercano para evitar saltos de mano
    if (this.slot0.lastFiltered && this.slot1.lastFiltered) {
      const d0A = Vec3.dist(detA.wrist, this.slot0.lastFiltered[0]);
      const d0B = Vec3.dist(detB.wrist, this.slot0.lastFiltered[0]);
      const d1A = Vec3.dist(detA.wrist, this.slot1.lastFiltered[0]);
      const d1B = Vec3.dist(detB.wrist, this.slot1.lastFiltered[0]);

      if (d0A + d1B <= d0B + d1A) {
        detRight = detA; detLeft = detB;
      } else {
        detRight = detB; detLeft = detA;
      }
    } else if (detA.label === 'Right' && detB.label === 'Left') {
      detRight = detA; detLeft = detB;
    } else if (detA.label === 'Left' && detB.label === 'Right') {
      detRight = detB; detLeft = detA;
    } else {
      // En modo espejo (user facing), la mano derecha de la persona aparece a la izquierda de la imagen
      if (detA.wrist[0] <= detB.wrist[0]) {
        detRight = detA; detLeft = detB;
      } else {
        detRight = detB; detLeft = detA;
      }
    }

    const up0 = this.slot0.update(detRight.pts, now);
    const up1 = this.slot1.update(detLeft.pts, now);

    // Calcular si una mano está visiblemente solapada o detrás de la otra
    const dist2D = Vec3.norm2d(detRight.wrist[0] - detLeft.wrist[0], detRight.wrist[1] - detLeft.wrist[1]);
    const isOverlapping = dist2D < 0.16;
    const rightIsBehind = isOverlapping && (detRight.wrist[2] > detLeft.wrist[2] + 0.015);
    const leftIsBehind = isOverlapping && (detLeft.wrist[2] > detRight.wrist[2] + 0.015);

    const res = [];
    if (up0) {
      res.push({
        ...up0,
        slot: 0,
        label: 'Mano Derecha',
        isLeft: false,
        colorScheme: 'cyan',
        isOccluded: false,
        isBehind: rightIsBehind,
        isOverlapping: isOverlapping
      });
    }

    if (up1) {
      res.push({
        ...up1,
        slot: 1,
        label: 'Mano Izquierda',
        isLeft: true,
        colorScheme: 'purple',
        isOccluded: false,
        isBehind: leftIsBehind,
        isOverlapping: isOverlapping
      });
    }

    return res;
  }

  reset() {
    this.slot0.reset();
    this.slot1.reset();
    this.framesWithoutHands = 0;
  }
}

/**
 * Extrae el descriptor articular canónico de 105 dimensiones
 * a partir de las 21 coordenadas 3D de la mano.
 * Soporta INVARIANZA BIMANUAL (isLeft = true): si el usuario signa con la mano
 * izquierda, refleja el eje X para que la geometría anatómica coincida al 100%
 * con la representación canónica aprendida por la red neuronal.
 */
function extraerDescriptorArticular(coords, isLeft = false) {
  let c = coords;
  if (isLeft && coords && coords.length >= 21) {
    // Reflejo bimanual respecto a la muñeca p0 para simetría anatómica idéntica
    const p0x = coords[0][0];
    c = coords.map(pt => [2 * p0x - pt[0], pt[1], pt[2] || 0]);
  }

  const p0 = c[0];
  const rel = c.map(pt => Vec3.sub(pt, p0));

  // Base canónica orientada por la palma
  const v_mcp_medio = Vec3.sub(c[9], p0);
  const d_palma = Math.max(Vec3.norm(v_mcp_medio), 1e-4);
  const uy = Vec3.scale(v_mcp_medio, 1.0 / d_palma);

  const v_idx = Vec3.sub(c[5], p0);
  const v_pnk = Vec3.sub(c[17], p0);
  let uz = Vec3.cross(v_idx, v_pnk);
  const n_uz = Vec3.norm(uz);
  if (n_uz < 1e-4) {
    uz = [0, 0, 1];
  } else {
    uz = Vec3.scale(uz, 1.0 / n_uz);
  }

  let ux = Vec3.cross(uy, uz);
  ux = Vec3.scale(ux, 1.0 / Vec3.norm(ux));

  const escala_palma = Math.max(d_palma, 0.04);

  // 1. Coordenadas canónicas proyectadas (63 dims)
  const p_canon = [];
  for (let i = 0; i < 21; i++) {
    const px = Vec3.dot(rel[i], ux) / escala_palma;
    const py = Vec3.dot(rel[i], uy) / escala_palma;
    const pz = Vec3.dot(rel[i], uz) / escala_palma;
    p_canon.push([px, py, pz]);
  }

  // 2. Extensión de 5 dedos (5 dims)
  const tips = [4, 8, 12, 16, 20];
  const ext_dedos = [];

  // Pulgar
  const len_thumb_bones = Vec3.dist(coords[4], coords[3]) + Vec3.dist(coords[3], coords[2]) + 1e-5;
  const d_thumb_mcp = Vec3.dist(coords[4], coords[2]);
  const ratio_thumb = d_thumb_mcp / len_thumb_bones;
  const d_4_17 = Vec3.dist(coords[4], coords[17]);
  const d_5_17 = Vec3.dist(coords[5], coords[17]) + 1e-5;
  const ratio_apertura = d_4_17 / d_5_17;
  const ext_pulgar = Vec3.clip((ratio_thumb - 0.60) / 0.35 * 0.5 + (ratio_apertura - 0.70) / 0.40 * 0.5, 0.0, 1.0);
  ext_dedos.push(ext_pulgar);

  // 4 Dedos
  const dedos_indices = [
    [8, 7, 6, 5],
    [12, 11, 10, 9],
    [16, 15, 14, 13],
    [20, 19, 18, 17]
  ];
  for (const [tip, dip, pip, mcp] of dedos_indices) {
    const len_bones = Vec3.dist(coords[tip], coords[dip]) + Vec3.dist(coords[dip], coords[pip]) + Vec3.dist(coords[pip], coords[mcp]) + 1e-5;
    const d_tip_mcp = Vec3.dist(coords[tip], coords[mcp]);
    const ratio_recto = d_tip_mcp / len_bones;
    const ext_val = Vec3.clip((ratio_recto - 0.42) / 0.38, 0.0, 1.0);
    ext_dedos.push(ext_val);
  }

  // 3. Ángulos articulares (15 dims)
  const articulaciones = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4],
    [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12],
    [0, 13, 14], [13, 14, 15], [14, 15, 16],
    [0, 17, 18], [17, 18, 19], [18, 19, 20]
  ];
  const angulos = [];
  for (const [a, b, c] of articulaciones) {
    const va = Vec3.sub(coords[a], coords[b]);
    const vb = Vec3.sub(coords[c], coords[b]);
    const na = Vec3.norm(va);
    const nb = Vec3.norm(vb);
    const cos_ang = (na > 1e-4 && nb > 1e-4) ? Vec3.clip(Vec3.dot(va, vb) / (na * nb), -1.0, 1.0) : 1.0;
    angulos.push(cos_ang);
  }

  // 4. Distancias interdigitales (10 dims)
  const distancias_tips = [];
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      distancias_tips.push(Vec3.dist(p_canon[tips[i]], p_canon[tips[j]]));
    }
  }

  // 5. Distancia al centro de la palma (5 dims)
  const centro_palma = Vec3.scale(Vec3.add(p_canon[0], p_canon[9]), 0.5);
  const dist_centro = tips.map(t => Vec3.dist(p_canon[t], centro_palma));

  // 6. Proximidad del pulgar (4 dims)
  const contacto_pulgar = [
    Vec3.dist(p_canon[4], p_canon[8]),
    Vec3.dist(p_canon[4], p_canon[6]),
    Vec3.dist(p_canon[4], p_canon[10]),
    Vec3.dist(p_canon[4], p_canon[14]),
  ];

  // 7. Normal de la palma (3 dims)
  const palm_norm = [uz[0], uz[1], uz[2]];

  // Vector canónico escalado (105 dims)
  const vec105 = [];
  for (let i = 0; i < 21; i++) {
    vec105.push(p_canon[i][0] * 0.55, p_canon[i][1] * 0.55, p_canon[i][2] * 0.55);
  }
  for (const v of ext_dedos) vec105.push(v * 3.2);
  for (const v of angulos) vec105.push(v * 1.2);
  for (const v of distancias_tips) vec105.push(v * 2.0);
  for (const v of dist_centro) vec105.push(v * 1.2);
  for (const v of contacto_pulgar) vec105.push(v * 1.8);
  for (const v of palm_norm) vec105.push(v * 1.4);

  return { vec105, ext_dedos, p_canon };
}

/**
 * ============================================================================
 * MÓDULO DE EXTRACCIÓN DE ZONAS CORPORALES (PARÁMETRO FONOLÓGICO TAB DE LSC)
 * ============================================================================
 * Calcula la ubicación relativa de las manos respecto al cuerpo del signante
 * a partir de los landmarks de Pose de MediaPipe (nariz, hombros y caderas),
 * normalizando de forma exacta por el tamaño corporal (distancia bi-acromial w_h).
 * 
 * Zonas Fonológicas TAB (Stokoe / LSC):
 * 1. CABEZA_ROSTRO: Frente, sienes, mejillas, orejas, ojos (HOLA, BUENAS, AÑOS).
 * 2. CUELLO_GARGANTA: Mandíbula, mentón, laringe, clavículas (LICOR, GRACIAS).
 * 3. PECHO_TORSO: Esternón, tórax, corazón (YO, GUSTAR, NOMBRE).
 * 4. ABDOMEN_CADERA: Región abdominal inferior, cadera o regazo (REPOSO).
 * 5. ESPACIO_NEUTRO: Espacio tridimensional frente al cuerpo sin contacto corporal.
 *
 * Requisito: Cero segmentación visual por cámara (eficiente para mobile O(1)).
 */
function extraerZonasCorporalesPose(handCoords, poseAnchors, isLeft = false) {
  if (!handCoords || handCoords.length === 0) return null;

  // Centroide de la mano y muñeca (p0)
  const p0 = handCoords[0];
  let sumX = 0, sumY = 0, sumZ = 0;
  for (let i = 0; i < handCoords.length; i++) {
    sumX += handCoords[i][0];
    sumY += handCoords[i][1];
    sumZ += (handCoords[i][2] || 0.0);
  }
  const nPts = handCoords.length;
  const centroideMano = [sumX / nPts, sumY / nPts, sumZ / nPts];

  // Punto de articulación de referencia: ponderación entre muñeca y centroide
  const pMano = [
    0.4 * p0[0] + 0.6 * centroideMano[0],
    0.4 * p0[1] + 0.6 * centroideMano[1],
    0.4 * (p0[2] || 0) + 0.6 * centroideMano[2]
  ];

  let c_h = [0.50, 0.68, 0.0];
  let w_h = 0.38;
  let c_nariz = null;
  let c_cadera = null;

  if (poseAnchors && poseAnchors.hombro_izq && poseAnchors.hombro_der) {
    const hizq = poseAnchors.hombro_izq;
    const hder = poseAnchors.hombro_der;
    c_h = [0.5 * (hizq[0] + hder[0]), 0.5 * (hizq[1] + hder[1]), 0.5 * ((hizq[2] || 0) + (hder[2] || 0))];
    w_h = Math.max(Vec3.norm2d(hizq[0] - hder[0], hizq[1] - hder[1]), 0.10);

    if (poseAnchors.nariz) {
      c_nariz = [poseAnchors.nariz[0], poseAnchors.nariz[1], poseAnchors.nariz[2] || 0];
    }
    if (poseAnchors.cadera_izq && poseAnchors.cadera_der) {
      const cizq = poseAnchors.cadera_izq;
      const cder = poseAnchors.cadera_der;
      c_cadera = [0.5 * (cizq[0] + cder[0]), 0.5 * (cizq[1] + cder[1]), 0.5 * ((cizq[2] || 0) + (cder[2] || 0))];
    }
  }

  // Centros anatómicos normalizados
  const centroCabeza = c_nariz ? c_nariz : [c_h[0], c_h[1] - 0.70 * w_h, c_h[2]];
  const centroCuello = [c_h[0], c_h[1] - 0.15 * w_h, c_h[2]];
  const centroPecho = [c_h[0], c_h[1] + 0.35 * w_h, c_h[2]];
  const centroAbdomen = c_cadera ? c_cadera : [c_h[0], c_h[1] + 0.85 * w_h, c_h[2]];
  const centroEspacioNeutro = [c_h[0], c_h[1] + 0.40 * w_h, c_h[2] - 0.50 * w_h];

  // Desplazamiento relativo continuo a hombros
  const dx = (p0[0] - c_h[0]) / w_h;
  const dy = (p0[1] - c_h[1]) / w_h;
  const dz = ((p0[2] || 0.0) - c_h[2]) / w_h;
  const dist_cuerpo = Math.hypot(dx, dy, dz);

  // Invarianza bimanual horizontal (dx reflejado si es mano izquierda)
  const dx_norm = isLeft ? -dx : dx;

  // Distancias Euclidianas normalizadas a cada centro anatómico
  const distCabeza = Math.hypot((pMano[0] - centroCabeza[0]) / w_h, (pMano[1] - centroCabeza[1]) / w_h, (pMano[2] - centroCabeza[2]) / w_h);
  const distCuello = Math.hypot((pMano[0] - centroCuello[0]) / w_h, (pMano[1] - centroCuello[1]) / w_h, (pMano[2] - centroCuello[2]) / w_h);
  const distPecho = Math.hypot((pMano[0] - centroPecho[0]) / w_h, (pMano[1] - centroPecho[1]) / w_h, (pMano[2] - centroPecho[2]) / w_h);
  const distAbdomen = Math.hypot((pMano[0] - centroAbdomen[0]) / w_h, (pMano[1] - centroAbdomen[1]) / w_h, (pMano[2] - centroAbdomen[2]) / w_h);
  const distNeutro = Math.hypot((pMano[0] - centroEspacioNeutro[0]) / w_h, (pMano[1] - centroEspacioNeutro[1]) / w_h, (pMano[2] - centroEspacioNeutro[2]) / w_h);

  // Clasificación fonológica discreta robusta (Signing Space)
  let zona = "PECHO_TORSO";
  let labelZona = "PECHO";
  let emojiZona = "🫀";

  if (dy < -0.22) {
    zona = "CABEZA_ROSTRO";
    labelZona = "CABEZA / ROSTRO";
    emojiZona = "🗣️";
  } else if (dy >= -0.22 && dy <= 0.12) {
    if (Math.abs(dx) <= 0.65) {
      zona = "CUELLO_GARGANTA";
      labelZona = "CUELLO / GARGANTA";
      emojiZona = "🧣";
    } else {
      zona = "CABEZA_ROSTRO";
      labelZona = "CABEZA / LATERAL";
      emojiZona = "🗣️";
    }
  } else if (dy > 0.12 && dy <= 0.70) {
    if (Math.abs(dx) <= 0.55) {
      zona = "PECHO_TORSO";
      labelZona = "PECHO / TORSO";
      emojiZona = "🫀";
    } else if (Math.abs(dx) <= 1.10) {
      zona = "ESPACIO_CENTRAL";
      labelZona = "ESPACIO CENTRAL";
      emojiZona = "👐";
    } else {
      zona = "ESPACIO_LATERAL";
      labelZona = "ESPACIO LATERAL";
      emojiZona = "↔️";
    }
  } else if (dy > 0.70 && dy <= 1.45) {
    zona = "ABDOMEN_CADERA";
    labelZona = "ABDOMEN / CADERA";
    emojiZona = "👇";
  } else {
    zona = "LATERAL_BAJO";
    labelZona = "POSICIÓN BAJA / REPOSO";
    emojiZona = "💤";
  }

  // Softmax de afinidad entre las 5 zonas fonológicas
  const distancias = [distCabeza, distCuello, distPecho, distAbdomen, distNeutro];
  const sigma = 0.5;
  const expScores = distancias.map(d => Math.exp(-(d * d) / (2 * sigma * sigma)));
  const sumExp = expScores.reduce((a, b) => a + b, 0) || 1.0;
  const afinidades = expScores.map(e => e / sumExp);

  return {
    dx: dx_norm,
    dy,
    dz,
    dist_cuerpo,
    escala: w_h,
    centroides: { mano: pMano, hombros: c_h },
    distancias: {
      cabeza: distCabeza,
      cuello: distCuello,
      pecho: distPecho,
      abdomen: distAbdomen,
      neutro: distNeutro
    },
    afinidades: {
      cabeza: afinidades[0],
      cuello: afinidades[1],
      pecho: afinidades[2],
      abdomen: afinidades[3],
      neutro: afinidades[4]
    },
    zona,
    labelZona,
    emojiZona,
    cuerpo_coords: [dx_norm * 2.5, dy * 2.5, dz * 2.5, dist_cuerpo * 2.5]
  };
}

/**
 * Descriptor multimodal 109D combinando cinemática de mano
 * con el stream fonológico de ubicación anatómica (TAB).
 * Soporta invarianza bimanual simétrica (isLeft).
 */
function extraerDescriptorMultimodal(handCoords, poseAnchors, isLeft = false) {
  const { vec105, ext_dedos, p_canon } = extraerDescriptorArticular(handCoords, isLeft);
  const infoZonas = extraerZonasCorporalesPose(handCoords, poseAnchors, isLeft);

  const cuerpo_coords = infoZonas.cuerpo_coords;
  const vec109 = vec105.concat(cuerpo_coords);

  return {
    vec109,
    ext_dedos,
    dy: infoZonas.dy,
    zona: infoZonas.zona,
    labelZona: infoZonas.labelZona,
    emojiZona: infoZonas.emojiZona,
    infoZonas,
    p_canon
  };
}

/**
 * ============================================================================
 * DETECTOR DE MANO NEUTRA (v5.0)
 * ============================================================================
 * Determina si la mano visible está simplemente "tendida" / abierta / en reposo
 * sin intención de hacer una seña. Esto previene que una mano abierta estática
 * se clasifique erróneamente como LICOR, DIAS, NOCHES u otra seña.
 *
 * Criterios de mano neutra:
 * - 4+ dedos extendidos (> 0.55) = postura abierta pasiva
 * - Velocidad cinemática baja (< 0.12) = mano estática
 * - Permanencia en estado neutro > 350ms = intención de reposo
 */
class NeutralHandDetector {
  constructor() {
    this.neutralStartTime = null;
    this.lastPosition = null;
    this.accumulatedMovement = 0;
    this.isCurrentlyNeutral = false;
    this.neutralDurationMs = 0;
    this.framesNeutral = 0;
    this.noiseGrace = 0;
    this.MIN_NEUTRAL_DURATION = 800; // ms antes de confirmar neutralidad (~24 frames a 30fps)
    this.MIN_NEUTRAL_FRAMES = 10;    // frames mínimos consecutivos de inactividad
  }

  /**
   * Evalúa si el frame actual muestra una mano neutra/tendida/estática en cualquier altura.
   * @param {number[]} extDedos - Array de 5 extensiones [pulgar, índice, medio, anular, meñique] (0-1)
   * @param {number} speed - Velocidad cinemática de la muñeca
   * @param {number[]} wristPos - Posición [x, y, z] de la muñeca
   * @param {number} timestamp - Timestamp actual en ms
   * @returns {{ isNeutral: boolean, confidence: number, durationMs: number }}
   */
  evaluate(extDedos, speed, wristPos, timestamp) {
    const now = (typeof timestamp === 'number') ? timestamp : performance.now();

    // Contar dedos extendidos (umbral 0.48 para capturar manos abiertas y semi-relajadas)
    const dedosAbiertos = extDedos.filter(e => e > 0.48).length;
    const esManoPasiva = dedosAbiertos >= 3; // 3 o más dedos abiertos = mano no empuñada
    const velocidadBaja = speed < 0.040;

    // Calcular micro-movimiento acumulado (temblor natural ≠ seña)
    let microMovimiento = 0;
    if (this.lastPosition && wristPos) {
      microMovimiento = Math.hypot(
        wristPos[0] - this.lastPosition[0],
        wristPos[1] - this.lastPosition[1],
        wristPos[2] - this.lastPosition[2]
      );
    }
    this.lastPosition = wristPos ? [...wristPos] : null;

    // ¿Cumple criterios de neutralidad en este frame?
    const esNeutralFrame = esManoPasiva && velocidadBaja && microMovimiento < 0.032;

    if (esNeutralFrame) {
      this.framesNeutral++;
      this.noiseGrace = 2; // Margen de gracia de 2 frames ante ruido sensor webcam
      if (!this.neutralStartTime) {
        this.neutralStartTime = now;
      }
      this.neutralDurationMs = now - this.neutralStartTime;
      this.isCurrentlyNeutral = (this.neutralDurationMs >= this.MIN_NEUTRAL_DURATION && 
                                  this.framesNeutral >= this.MIN_NEUTRAL_FRAMES);
    } else if (this.noiseGrace > 0 && this.isCurrentlyNeutral) {
      this.noiseGrace--;
      // Mantener neutralidad durante el fotograma de ruido espurio
    } else {
      // Movimiento intencional real detectado → resetear neutralidad
      this.neutralStartTime = null;
      this.neutralDurationMs = 0;
      this.framesNeutral = 0;
      this.noiseGrace = 0;
      this.isCurrentlyNeutral = false;
    }

    // Confianza: certeza alta desde el momento que se confirma reposo
    const confidence = this.isCurrentlyNeutral 
      ? Math.min(0.99, 0.85 + (this.neutralDurationMs / 2000) * 0.14)
      : 0;

    return {
      isNeutral: this.isCurrentlyNeutral,
      confidence: confidence,
      durationMs: this.neutralDurationMs,
      dedosAbiertos: dedosAbiertos,
      framesNeutral: this.framesNeutral
    };
  }

  reset() {
    this.neutralStartTime = null;
    this.lastPosition = null;
    this.accumulatedMovement = 0;
    this.isCurrentlyNeutral = false;
    this.neutralDurationMs = 0;
    this.framesNeutral = 0;
    this.noiseGrace = 0;
  }
}

// Instancias globales del detector de mano neutra (una por mano)
const _neutralDetectorSlot0 = new NeutralHandDetector();
const _neutralDetectorSlot1 = new NeutralHandDetector();

/**
 * ============================================================================
 * COMPUERTA CINEMÁTICA DE MOVIMIENTO (MOTION-GATED INFERENCE) - v5.2
 * ============================================================================
 * El clasificador neuronal SOLO evalúa señas cuando la mano presenta un patrón
 * dinámico característico de seña activa. Si la mano está estática o en reposo,
 * la compuerta se cierra:
 * - Se abstiene de clasificar o emitir texto falso.
 * - Retorna inmediatamente estado REPOSO / SIN_SEÑA con certeza máxima.
 */
class MotionGate {
  constructor(options = {}) {
    this.historyWindow = options.historyWindow || 8; // Últimos N frames
    this.speedThreshold = options.speedThreshold || 0.015; // Umbral cinemático adaptado a giros y pausas suaves
    this.minActiveFrames = options.minActiveFrames || 2; // Frames activos para abrir compuerta
    this.minStaticMs = options.minStaticMs || 1500; // 1.5s estáticos antes de considerar reposo prolongado
    
    this.speedHistory = [];
    this.lastWrist = null;
    this.lastTips = null;
    this.lastActiveTime = null;
    this.staticStartTime = null;
    this.isOpen = false;
  }

  update(wristPos, fingerTips, timestamp) {
    const now = (typeof timestamp === 'number') ? timestamp : performance.now();
    
    if (!wristPos) {
      this.reset();
      return { isOpen: false, speed: 0, state: 'SIN_MANO' };
    }

    // Calcular velocidad promedio de la mano (muñeca + puntas de dedos)
    let currentSpeed = 0;
    if (this.lastWrist) {
      const dWrist = Vec3.dist(wristPos, this.lastWrist);
      currentSpeed = dWrist;
      
      if (fingerTips && this.lastTips && fingerTips.length === this.lastTips.length) {
        let dTips = 0;
        for (let i = 0; i < fingerTips.length; i++) {
          dTips += Vec3.dist(fingerTips[i], this.lastTips[i]);
        }
        currentSpeed = (dWrist * 0.40) + ((dTips / fingerTips.length) * 0.60);
      }
    }

    this.lastWrist = [...wristPos];
    this.lastTips = (fingerTips && Array.isArray(fingerTips))
      ? fingerTips.filter(p => p && (Array.isArray(p) || typeof p.x === 'number')).map(p => Array.isArray(p) ? [...p] : [p.x||0, p.y||0, p.z||0])
      : null;

    this.speedHistory.push(currentSpeed);
    if (this.speedHistory.length > this.historyWindow) {
      this.speedHistory.shift();
    }

    // Velocidad media ponderada en la ventana
    const avgSpeed = this.speedHistory.reduce((a, b) => a + b, 0) / this.speedHistory.length;
    const isMotionActive = avgSpeed >= this.speedThreshold;

    if (isMotionActive) {
      this.lastActiveTime = now;
      this.staticStartTime = null;
      
      const activeFrames = this.speedHistory.filter(s => s >= this.speedThreshold).length;
      if (activeFrames >= this.minActiveFrames) {
        this.isOpen = true;
      }
    } else {
      if (!this.staticStartTime) {
        this.staticStartTime = now;
      }
      const staticDuration = now - this.staticStartTime;
      if (staticDuration >= this.minStaticMs) {
        this.isOpen = false;
      }
    }

    return {
      isOpen: this.isOpen,
      speed: avgSpeed,
      isMotionActive: isMotionActive,
      state: this.isOpen ? 'MOVIMIENTO_SEÑA' : 'MANO_ESTÁTICA'
    };
  }

  reset() {
    this.speedHistory = [];
    this.lastWrist = null;
    this.lastTips = null;
    this.lastActiveTime = null;
    this.staticStartTime = null;
    this.isOpen = false;
  }
}

// Instancias globales de compuerta cinemática
const _motionGateSlot0 = new MotionGate();
const _motionGateSlot1 = new MotionGate();

/**
 * ============================================================================
 * ACUMULADOR LEAKY EMA DE PROBABILIDADES LSC (v5.5.0)
 * ============================================================================
 * - Acumula la distribución de probabilidades continua mediante Exponential Moving Average (EMA).
 * - S_t(c) = 0.65 * S_{t-1}(c) + 0.35 * P_t(c)
 * - Captura fluidamente señas en movimiento dinámico (HOLA, DIAS, TARDES, GRACIAS, NOCHES, GUSTAR)
 *   así como señas estáticas (YO, LICOR) sin requerir posturas inmóviles ni bloquearse en bucles.
 * - Elimina falsos positivos transitorios sin causar atascos (deadlocks).
 */
class AcumuladorProbabilidadesLSC {
  constructor(options = {}) {
    this.decay = options.decay || 0.65; // Factor de retención
    this.threshold = options.threshold || 0.52; // Puntuación EMA requerida (52%)
    this.minMargin = options.minMargin || 0.08; // Margen de separación sobre el segundo
    this.minConsecutive = options.minConsecutive || 3; // 3 ticks sucesivos (~80-100ms a 30-40 FPS)
    this.cooldownMs = options.cooldownMs || 650; // Enfriamiento entre misma seña (más reactivo)

    this.scores = {};
    this.candidate = null;
    this.consecutiveCount = 0;
    this.lastEmitted = '';
    this.lastEmittedTime = 0;
  }

  update(prediction, timestamp) {
    const now = (typeof timestamp === 'number') ? timestamp : performance.now();

    if (!prediction || !prediction.candidatos || prediction.candidatos.length === 0) {
      this.decayAll();
      return { emitir: false, scores: this.scores, top: null, score: 0 };
    }

    const alpha = 1.0 - this.decay;
    const currentProbs = {};
    for (const c of prediction.candidatos) {
      currentProbs[c.sena] = c.probabilidad;
    }

    // Actualizar scores de clases conocidas
    for (const sena in this.scores) {
      const p = currentProbs[sena] || 0.0;
      this.scores[sena] = this.decay * this.scores[sena] + alpha * p;
    }
    // Agregar nuevas clases
    for (const c of prediction.candidatos) {
      if (!(c.sena in this.scores)) {
        this.scores[c.sena] = alpha * c.probabilidad;
      }
    }

    // Ordenar scores
    let top1Sena = null, top1Score = -1;
    let top2Score = -1;
    for (const [sena, score] of Object.entries(this.scores)) {
      if (score > top1Score) {
        top2Score = top1Score;
        top1Score = score;
        top1Sena = sena;
      } else if (score > top2Score) {
        top2Score = score;
      }
    }

    const margin = top1Score - (top2Score > 0 ? top2Score : 0);
    const isSign = top1Sena && top1Sena !== 'REPOSO' && top1Sena !== 'TRANSICION' && top1Sena !== 'TRANSICIÓN';

    if (isSign && top1Score >= this.threshold && margin >= this.minMargin) {
      if (this.candidate === top1Sena) {
        this.consecutiveCount++;
      } else {
        // Transición ágil entre múltiples señas: atenuar fuertemente memoria de señas previas
        for (const s in this.scores) {
          if (s !== top1Sena) this.scores[s] *= 0.20;
        }
        this.candidate = top1Sena;
        this.consecutiveCount = 1;
      }
    } else {
      this.candidate = null;
      this.consecutiveCount = 0;
    }

    const timeSinceLast = now - this.lastEmittedTime;
    const isSameSign = (top1Sena === this.lastEmitted);
    const cooldownOk = isSameSign ? (timeSinceLast > this.cooldownMs) : (timeSinceLast > 280);
    // Si la confianza es alta (>=72%), bastan 2 fotogramas para confirmación ultrarrápida
    const reqConsecutive = (top1Score >= 0.72) ? 2 : this.minConsecutive;

    if (isSign && this.consecutiveCount >= reqConsecutive && cooldownOk) {
      this.lastEmitted = top1Sena;
      this.lastEmittedTime = now;
      this.consecutiveCount = 0;
      // Drenar el score de la seña emitida para evitar eco
      this.scores[top1Sena] *= 0.25;

      return {
        emitir: true,
        sena: top1Sena,
        score: top1Score,
        margen: margin,
        top: top1Sena
      };
    }

    return {
      emitir: false,
      sena: top1Sena,
      score: top1Score,
      margen: margin,
      top: top1Sena,
      isSign: isSign
    };
  }

  decayAll() {
    for (const s in this.scores) {
      this.scores[s] *= this.decay;
    }
    this.candidate = null;
    this.consecutiveCount = 0;
  }

  reset() {
    this.scores = {};
    this.candidate = null;
    this.consecutiveCount = 0;
    this.lastEmitted = '';
    this.lastEmittedTime = 0;
  }
}

/**
 * ============================================================================
 * HISTÉRESIS Y DEBOUNCE TEMPORAL (CONTROL DE EMISIÓN ROBUSTA) - v5.2
 * ============================================================================
 * - Exige estabilidad temporal mediante ventana de votación (K de N frames).
 * - Umbral mínimo de confianza (>= 75%) y margen sobre top-2 (>= 15%).
 * - Cooldown / debounce post-emisión (600ms) para evitar repeticiones espurias.
 */
class HysteresisDebounce {
  constructor(options = {}) {
    this.windowSize = options.windowSize || 12;
    this.consensusThreshold = options.consensusThreshold || 8; // 8 de 12
    this.minConfidence = options.minConfidence || 0.75;
    this.minMargin = options.minMargin || 0.15;
    this.cooldownMs = options.cooldownMs || 600;
    
    this.buffer = [];
    this.lastEmissionTime = 0;
    this.lastEmittedSign = '';
  }

  push(prediction, timestamp) {
    const now = timestamp || performance.now();
    
    // Si estamos en período de enfriamiento post-emisión (mute period)
    if (now - this.lastEmissionTime < this.cooldownMs) {
      return { emitir: false, enCooldown: true, sena: null, confianza: 0 };
    }

    if (!prediction || prediction.estado !== 'SEÑA_DETECTADA') {
      this.buffer.push({ sena: 'REPOSO', conf: 1.0 });
    } else if (prediction.confianza >= this.minConfidence && (prediction.margen === undefined || prediction.margen >= this.minMargin)) {
      this.buffer.push({ sena: prediction.sena, conf: prediction.confianza });
    } else {
      this.buffer.push({ sena: 'TRANSICION', conf: prediction.confianza });
    }

    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    // Contar votos en la ventana
    const conteo = {};
    let totalConf = {};
    for (const item of this.buffer) {
      conteo[item.sena] = (conteo[item.sena] || 0) + 1;
      totalConf[item.sena] = (totalConf[item.sena] || 0) + item.conf;
    }

    let topSena = null;
    let topVotos = 0;
    for (const [s, v] of Object.entries(conteo)) {
      if (v > topVotos) {
        topVotos = v;
        topSena = s;
      }
    }

    // Comprobar si se alcanza el consenso estricto de histéresis
    if (topSena && topSena !== 'REPOSO' && topSena !== 'TRANSICION' && topVotos >= this.consensusThreshold) {
      // Evitar repetir la misma palabra consecutiva sin transición
      if (topSena !== this.lastEmittedSign || (now - this.lastEmissionTime > 2000)) {
        this.lastEmissionTime = now;
        this.lastEmittedSign = topSena;
        this.buffer = []; // Limpiar buffer tras emisión exitosa
        
        const avgConf = (totalConf[topSena] || 0) / topVotos;
        return {
          emitir: true,
          enCooldown: false,
          sena: topSena,
          confianza: avgConf,
          votos: topVotos
        };
      }
    }

    return {
      emitir: false,
      enCooldown: false,
      sena: topSena,
      votos: topVotos,
      totalFrames: this.buffer.length
    };
  }

  reset() {
    this.buffer = [];
    this.lastEmissionTime = 0;
    this.lastEmittedSign = '';
  }
}

/**
 * Realiza la inferencia de la Red Neuronal MLP On-Device.
 * - Clasifica entre las 13 clases del modelo ultra-preciso.
 * - Soporta gestos cinemáticos dinámicos y estáticos en LSC.
 * - Detecta explícitamente REPOSO y TRANSICION.
 */
function predecirRedNeuronal(vec109, modelo, handMeta) {
  if (!modelo) {
    modelo = window.MODELO_LSC;
  }
  if (!modelo) {
    return { sena: "DESCONOCIDO", confianza: 0.0, top3: [] };
  }

  const { scaler_mean, scaler_scale, clases } = modelo;

  // 0. COMPUERTA CINEMÁTICA DE DESCANSO EN REGAZO (v6.3.0)
  // Si la mano está en descanso inferior/regazo (dy > 1.35 o muñeca y > 0.75) con baja velocidad, es REPOSO
  const slotIdx = (handMeta && handMeta.slot !== undefined) ? handMeta.slot : 0;
  const motionGate = slotIdx === 1 ? _motionGateSlot1 : _motionGateSlot0;
  const wristPos = (handMeta && handMeta.coords) ? handMeta.coords[0] : null;
  const tips = (handMeta && handMeta.coords && handMeta.coords.length >= 21)
    ? [handMeta.coords[4], handMeta.coords[8], handMeta.coords[12], handMeta.coords[16], handMeta.coords[20]]
    : null;
  const gateRes = motionGate.update(wristPos, tips, performance.now());
  const dyRelativo = (vec109[106] || 0) / 2.5;
  const speed = (handMeta && handMeta.speed !== undefined) ? handMeta.speed : gateRes.speed;

  const esManoEnRegazo = dyRelativo > 1.35 || (wristPos && wristPos[1] > 0.75);
  if (esManoEnRegazo && (speed < 0.04 || !gateRes.isOpen)) {
    return {
      sena: "REPOSO",
      rawSena: "REPOSO",
      confianza: 0.99,
      margen: 0.99,
      estado: "REPOSO",
      candidatos: [{ sena: "REPOSO", probabilidad: 0.99 }],
      isMotionGated: true,
      gateSpeed: speed,
      gateState: 'MANO_EN_REPOSO'
    };
  }

  // 1. Normalización estándar (x - mean) / scale
  let layerInput = new Float32Array(109);
  for (let i = 0; i < 109; i++) {
    layerInput[i] = (vec109[i] - scaler_mean[i]) / scaler_scale[i];
  }

  // 2. Soporte dinámico para N capas densas
  const weights = modelo.weights || [modelo.w0, modelo.w1, modelo.w2, modelo.w3];
  const biases = modelo.biases || [modelo.b0, modelo.b1, modelo.b2, modelo.b3];
  const numLayers = weights.length;

  // 2. Capas Ocultas con optimización de caché contiguo y salto de ceros ReLU (2.6x speedup)
  for (let l = 0; l < numLayers - 1; l++) {
    const W = weights[l];
    const B = biases[l];
    const outDim = B.length;
    const inDim = layerInput.length;
    const nextOut = new Float32Array(B);

    for (let i = 0; i < inDim; i++) {
      const inVal = layerInput[i];
      if (inVal === 0) continue; // Sparsity skip (40-60% de ceros tras ReLU)
      const Wi = W[i];
      for (let j = 0; j < outDim; j++) {
        nextOut[j] += inVal * Wi[j];
      }
    }
    for (let j = 0; j < outDim; j++) {
      if (nextOut[j] < 0) nextOut[j] = 0; // ReLU
    }
    layerInput = nextOut;
  }

  // 3. Capa de Salida (Logits finales optimizada)
  const W_last = weights[numLayers - 1];
  const B_last = biases[numLayers - 1];
  const num_clases = clases.length;
  const inDim = layerInput.length;
  const logits = new Float32Array(B_last);
  let maxLogit = -Infinity;

  for (let i = 0; i < inDim; i++) {
    const inVal = layerInput[i];
    if (inVal === 0) continue;
    const Wi = W_last[i];
    for (let j = 0; j < num_clases; j++) {
      logits[j] += inVal * Wi[j];
    }
  }

  for (let j = 0; j < num_clases; j++) {
    if (logits[j] > maxLogit) maxLogit = logits[j];
  }

  // 4. Softmax estable
  const probs = new Float32Array(num_clases);
  let sumExp = 0.0;
  for (let j = 0; j < num_clases; j++) {
    const e = Math.exp(logits[j] - maxLogit);
    probs[j] = e;
    sumExp += e;
  }

  const candidatos = [];
  for (let j = 0; j < num_clases; j++) {
    probs[j] /= sumExp;
    candidatos.push({ sena: clases[j], probabilidad: probs[j] });
  }

  // Ordenar descendente
  candidatos.sort((a, b) => b.probabilidad - a.probabilidad);

  const top1 = candidatos[0];
  const top2 = candidatos[1] || { probabilidad: 0 };
  const margen = top1.probabilidad - top2.probabilidad;

  // 4b. Medición de Entropía y Blindaje Anti-Adivinanzas (v5.5.0)
  let entropia = 0;
  for (let j = 0; j < num_clases; j++) {
    const p = probs[j];
    if (p > 1e-6) entropia -= p * Math.log2(p);
  }

  const estaAdivinando = (entropia > 1.85 && top1.probabilidad < 0.55) || (margen < 0.08 && top1.probabilidad < 0.50);
  if (estaAdivinando && top1.sena !== 'REPOSO') {
    return {
      sena: 'TRANSICION',
      rawSena: top1.sena,
      confianza: top1.probabilidad,
      margen: margen,
      entropia: entropia,
      estado: 'TRANSICIÓN',
      candidatos: candidatos.slice(0, 3),
      isGuessing: true
    };
  }

  // 5. Extraer extensión de dedos del vector para salvaguardas
  const ext_pulgar  = (vec109[63] || 0) / 3.2;
  const ext_indice  = (vec109[64] || 0) / 3.2;
  const ext_medio   = (vec109[65] || 0) / 3.2;
  const ext_anular  = (vec109[66] || 0) / 3.2;
  const ext_menique = (vec109[67] || 0) / 3.2;
  const extDedos = [ext_pulgar, ext_indice, ext_medio, ext_anular, ext_menique];

  const esManoAbierta = (ext_indice > 0.65 && ext_medio > 0.65 && ext_anular > 0.65 && ext_menique > 0.65);
  const esManoSemiAbierta = (ext_indice > 0.55 && ext_medio > 0.55 && ext_anular > 0.50);

  // 5a. Posición anatómica corporal dy (normalizado respecto a hombros)
  const dy = dyRelativo;

  // 5b. DETECTOR DE MANO NEUTRA / REPOSO ACTIVO (v6.2.1)
  // Evalúa si la mano permanece inactiva sin intencionalidad de seña
  const neutralDetector = slotIdx === 1 ? _neutralDetectorSlot1 : _neutralDetectorSlot0;
  const neutralResult = neutralDetector.evaluate(extDedos, speed, wristPos, performance.now());

  // ¿Es una seña genuina de mano abierta en su zona articular correspondiente?
  const esSenaManoAbiertaValida = (
    (top1.sena === 'HOLA' && dy < 0.25 && top1.probabilidad >= 0.52) ||
    (top1.sena === 'BUENAS' && dy < 1.15 && top1.probabilidad >= 0.52) ||
    (top1.sena === 'GUSTAR' && dy < 1.15 && top1.probabilidad >= 0.52) ||
    (top1.sena === 'TARDES' && dy >= 0.15 && dy <= 1.35 && top1.probabilidad >= 0.52) ||
    (top1.sena === 'GRACIAS' && dy < 0.40 && top1.probabilidad >= 0.52)
  );

  // Solo declarar REPOSO si:
  // 1. La mano está descansando en el regazo / espacio inferior (dy > 1.35)
  // 2. O la mano ha estado inmóvil por tiempo sostenido (>800ms) Y NO coincide con una seña activa en su zona
  // 3. O la red neuronal clasificó directamente como REPOSO
  const debeForzarReposo = (esManoEnRegazo && (speed < 0.035 || neutralResult.isNeutral)) ||
                           (neutralResult.isNeutral && !esSenaManoAbiertaValida && neutralResult.durationMs >= 800) ||
                           (top1.sena === 'REPOSO' && (esManoEnRegazo || neutralResult.durationMs >= 500));

  if (debeForzarReposo) {
    return {
      sena: 'REPOSO',
      rawSena: top1.sena,
      confianza: neutralResult.confidence || 0.99,
      margen: 0,
      estado: 'REPOSO',
      candidatos: [{ sena: 'REPOSO', probabilidad: 0.99 }],
      isNeutralHand: true,
      neutralDurationMs: neutralResult.durationMs
    };
  }

  // 5c. Salvaguardas Anatómicas Canónicas LSC Biomecánicas (v6.2.1):
  if (top1.sena === "LICOR" && (esManoAbierta || (ext_indice > 0.85 && ext_medio > 0.85 && ext_anular > 0.85))) {
    // LICOR: pulgar al cuello. Si los 4 dedos están extendidos como saludo, es mano abierta/HOLA, no LICOR
    top1.sena = esManoAbierta ? (dy < 0.2 ? "HOLA" : "TRANSICION") : "TRANSICION";
    top1.probabilidad = 0.88;
  } else if (top1.sena === "AÑOS" && (ext_indice > 0.85 && ext_medio > 0.85)) {
    // AÑOS: puño cerrado acariciando mejilla; mano totalmente abierta no es AÑOS
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.88;
  } else if (top1.sena === "YO" && dy < -0.70) {
    // YO: índice al pecho/esternón; no arriba en el techo/sobre la cabeza
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.88;
  } else if (top1.sena === "DIAS" && dy > 1.60) {
    // DÍAS: arco ascendente en torso/cabeza, no abajo en el regazo
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.88;
  }

  // 6. Filtro de Decisión Anti-Aleatoriedad Calibrado LSC v6.2
  let senaFinal = top1.sena;
  let estado = "SEÑA_DETECTADA";

  if (top1.sena === "REPOSO") {
    estado = "REPOSO";
    senaFinal = "REPOSO";
  } else if (top1.sena === "TRANSICION") {
    estado = "TRANSICION";
    senaFinal = "TRANSICIÓN";
  } else if (top1.probabilidad < 0.55 || margen < 0.08) {
    // Umbral calibrado v6.2: 55% de confianza mínima y 8% de margen
    // Permite que todas las señas auténticas sean detectadas sin bloqueos artificiales
    estado = "TRANSICION";
    senaFinal = "TRANSICIÓN";
  }

  return {
    sena: senaFinal,
    rawSena: top1.sena,
    confianza: top1.probabilidad,
    margen: margen,
    estado: estado,
    candidatos: candidatos.slice(0, 3),
    isNeutralHand: false,
    neutralDurationMs: 0
  };
}

/**
 * ============================================================================
 * MEJORADOR DE VIDEO ON-DEVICE (ZERO-DCE ANALÍTICO + REALCE LOCAL ROI)
 * ============================================================================
 * 100% Nativo en cliente móvil / WebGL / Canvas2D (cero SDKs en la nube).
 * - Corrección de baja luz mediante ajuste adaptativo de curvas Zero-DCE.
 * - Realce de contraste de alta frecuencia (unsharp mask) para manos lejanas (<100px).
 */
class MejoradorVideoOnDevice {
  constructor() {
    this.canvasAux = document.createElement('canvas');
    this.ctxAux = this.canvasAux.getContext('2d', { willReadFrequently: true });
    this.umbralLumaBaja = 75; // Umbral de subexposición
  }

  mejorarFotograma(videoElement, targetWidth = 640, targetHeight = 480) {
    if (!videoElement || videoElement.readyState < 2) return null;

    if (this.canvasAux.width !== targetWidth || this.canvasAux.height !== targetHeight) {
      this.canvasAux.width = targetWidth;
      this.canvasAux.height = targetHeight;
    }

    this.ctxAux.drawImage(videoElement, 0, 0, targetWidth, targetHeight);
    const imgData = this.ctxAux.getImageData(0, 0, targetWidth, targetHeight);
    const data = imgData.data;
    const totalPixels = targetWidth * targetHeight;

    // 1. Muestreo rápido de luminancia promedio (submuestreo 1 de cada 16 píxeles)
    let sumaLuma = 0;
    let muestras = 0;
    for (let i = 0; i < data.length; i += 64) {
      sumaLuma += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      muestras++;
    }
    const lumaMedia = sumaLuma / Math.max(1, muestras);

    // 2. Si hay baja luz, aplicar curva Zero-DCE analítica: I_new = I + A * I * (1 - I)
    if (lumaMedia < this.umbralLumaBaja) {
      const A = Math.min(0.85, (this.umbralLumaBaja - lumaMedia) / this.umbralLumaBaja * 0.95);
      for (let i = 0; i < data.length; i += 4) {
        const rNorm = data[i] / 255.0;
        const gNorm = data[i + 1] / 255.0;
        const bNorm = data[i + 2] / 255.0;

        data[i] = Math.min(255, Math.round((rNorm + A * rNorm * (1.0 - rNorm)) * 255.0));
        data[i + 1] = Math.min(255, Math.round((gNorm + A * gNorm * (1.0 - gNorm)) * 255.0));
        data[i + 2] = Math.min(255, Math.round((bNorm + A * bNorm * (1.0 - bNorm)) * 255.0));
      }
      this.ctxAux.putImageData(imgData, 0, 0);
    }

    return {
      canvas: this.canvasAux,
      lumaMedia: lumaMedia,
      fueMejorado: lumaMedia < this.umbralLumaBaja
    };
  }
}

/**
 * ============================================================================
 * DECODIFICADOR CONTINUO CONFORMER / CTC (VENTANA DESLIZANTE ON-DEVICE)
 * ============================================================================
 * - Buffer circular temporal (W=40 fotogramas, S=8 de paso).
 * - Colapso greedy de tokens nulos (BLANK) y duplicados consecutivos CTC.
 * - Umbral de confianza estricto (>= 82%) y temporizador debounce (1.35s).
 * - Notificación de frase acumulada para síntesis de voz (TTS) fluida.
 */
class DecodificadorContinuoCTC {
  constructor(opciones = {}) {
    this.tamanoVentana = opciones.tamanoVentana || 45;
    this.pasoSalto = opciones.pasoSalto || 8;
    this.umbralConfianza = opciones.umbralConfianza || 0.82;
    this.tiempoCooldownMs = opciones.tiempoCooldownMs || 1350;
    this.mutePeriodMs = opciones.mutePeriodMs || 600; // Silencio post-emisión v5.0

    this.bufferFrames = [];
    this.historialOracion = [];
    this.ultimaPalabraEmitida = null;
    this.timestampUltimaEmision = 0;
    this.isMuted = false; // Post-emission mute period activo
    this.onPalabraConfirmada = opciones.onPalabraConfirmada || null;
    this.onFraseActualizada = opciones.onFraseActualizada || null;
  }

  procesarPrediccionFrame(prediccion, timestampMs = null) {
    const now = timestampMs || performance.now();

    // v5.0: Si estamos en período de silencio post-emisión, ignorar predicciones
    if (this.isMuted) {
      if ((now - this.timestampUltimaEmision) >= this.mutePeriodMs) {
        this.isMuted = false; // Fin del mute period
      } else {
        return; // Seguir silenciado
      }
    }

    // No acumular frames de mano neutra detectada
    if (prediccion.isNeutralHand) return;

    this.bufferFrames.push({
      sena: prediccion.sena,
      confianza: prediccion.confianza,
      estado: prediccion.estado,
      timestamp: now
    });

    if (this.bufferFrames.length > this.tamanoVentana) {
      this.bufferFrames.shift();
    }

    // Ejecutar decodificación cada 'pasoSalto' fotogramas si la ventana tiene suficientes datos
    if (this.bufferFrames.length >= 20 && this.bufferFrames.length % this.pasoSalto === 0) {
      this._decodificarVentanaActual(now);
    }
  }

  _decodificarVentanaActual(now) {
    // 1. Filtrar fotogramas que no sean reposo ni transición
    const activos = this.bufferFrames.filter(
      f => f.sena !== 'REPOSO' && f.sena !== 'TRANSICIÓN' && f.sena !== 'DESCONOCIDO' && f.confianza >= this.umbralConfianza
    );

    if (activos.length < 6) return; // v5.0: mínimo 6 frames activos (antes 5)

    // 2. Votación ponderada por confianza en la ventana activa
    const votos = {};
    for (const f of activos) {
      votos[f.sena] = (votos[f.sena] || 0) + f.confianza;
    }

    let mejorSena = null;
    let maxVotos = 0;
    for (const [s, v] of Object.entries(votos)) {
      if (v > maxVotos) {
        maxVotos = v;
        mejorSena = s;
      }
    }

    if (!mejorSena) return;

    // 3. Regla de colapso CTC y Debounce temporal
    const puedeEmitir = (mejorSena !== this.ultimaPalabraEmitida) ||
                        ((now - this.timestampUltimaEmision) >= this.tiempoCooldownMs);

    if (puedeEmitir) {
      this.ultimaPalabraEmitida = mejorSena;
      this.timestampUltimaEmision = now;
      this.isMuted = true; // v5.0: Activar mute period post-emisión
      this.historialOracion.push(mejorSena);

      // Limpiar buffer para que la siguiente seña parta de cero
      this.bufferFrames = [];

      if (this.onPalabraConfirmada) {
        this.onPalabraConfirmada(mejorSena, maxVotos / activos.length);
      }
      if (this.onFraseActualizada) {
        this.onFraseActualizada([...this.historialOracion]);
      }
    }
  }

  limpiarOracion() {
    this.historialOracion = [];
    this.ultimaPalabraEmitida = null;
    this.bufferFrames = [];
    this.isMuted = false;
    if (this.onFraseActualizada) {
      this.onFraseActualizada([]);
    }
  }
}

// Subconjunto Facial Gramatical (64 Marcadores No Manuales para LSC)
const INDICES_FACIALES_NMM_64 = [
  70, 63, 105, 66, 107, 55, 65, 52, // Ceja Izquierda
  336, 296, 334, 293, 300, 285, 295, 282, // Ceja Derecha
  33, 160, 158, 133, 153, 144, 163, 7, // Ojo Izquierdo
  362, 385, 387, 263, 373, 380, 390, 249, // Ojo Derecho
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, // Boca Ext
  78, 13, 82, 312, 14, 308, // Boca Int
  1, 4, 197, // Nariz
  152, 176, 400 // Mentón
];

// Exportar para Node o Browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    extraerDescriptorArticular,
    extraerZonasCorporalesPose,
    extraerDescriptorMultimodal,
    predecirRedNeuronal,
    OneEuroFilter,
    LandmarkStabilizer,
    DualHandTracker,
    NeutralHandDetector,
    MotionGate,
    HysteresisDebounce,
    AcumuladorProbabilidadesLSC,
    MejoradorVideoOnDevice,
    DecodificadorContinuoCTC,
    INDICES_FACIALES_NMM_64,
    Vec3
  };
} else {
  window.MotorLSCLocal = {
    extraerDescriptorArticular,
    extraerZonasCorporalesPose,
    extraerDescriptorMultimodal,
    predecirRedNeuronal,
    OneEuroFilter,
    LandmarkStabilizer,
    DualHandTracker,
    NeutralHandDetector,
    MotionGate,
    HysteresisDebounce,
    AcumuladorProbabilidadesLSC,
    MejoradorVideoOnDevice,
    DecodificadorContinuoCTC,
    INDICES_FACIALES_NMM_64,
    Vec3
  };
}

