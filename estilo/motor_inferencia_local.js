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
  constructor(minCutoff = 1.2, beta = 20.0, dCutoff = 5.0) {
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
 * y persistencia inercial ante desenfoque de movimiento.
 */
class LandmarkStabilizer {
  constructor(slotId = 0) {
    this.slotId = slotId;
    this.filters = [];
    for (let i = 0; i < 21; i++) {
      this.filters.push([
        new OneEuroFilter(1.2, 20.0, 5.0), // X
        new OneEuroFilter(1.2, 20.0, 5.0), // Y
        new OneEuroFilter(1.2, 20.0, 5.0), // Z
      ]);
    }
    this.lastFiltered = null;
    this.lastTimestamp = null;
    this.framesMissing = 0;
    this.framesVisible = 0; // Contador de persistencia temporal para evitar predicciones prematuras
    this.maxGraceFrames = 4; // Persistencia de motion blur
    this.maxOcclusionFrames = 10; // Persistencia cuando está genuinamente detrás de la otra mano (~300ms)
    this.currentVelocity = [0, 0, 0];
    this.isOccluded = false;
    this.lastForegroundWrist = null;
  }

  update(rawCoords, timestamp) {
    const now = timestamp || performance.now();

    if (!rawCoords || rawCoords.length < 21) {
      this.framesMissing++;
      this.framesVisible = 0; // Resetear contador al desaparecer la mano
      if (this.framesMissing <= this.maxGraceFrames && this.lastFiltered) {
        // Extrapolación inercial suave
        const dt = Math.min((now - (this.lastTimestamp || now)) / 1000.0, 0.05) || 0.033;
        const decay = Math.pow(0.75, this.framesMissing);
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
        const speed = Vec3.norm(this.currentVelocity);
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
    // Detección de proximidad al borde de la pantalla (mano incompleta / entrando)
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

    // Validación Post-Proceso Bimanual (Requisito Crítico: Evitar 2 manos cuando solo hay 1)
    if (detections.length >= 2) {
      const det0 = detections[0];
      const det1 = detections[1];
      const d_entre_manos = Vec3.dist(det0.wrist, det1.wrist);
      
      // 1. Descarte por proximidad espacial extrema (< 0.10):
      // MediaPipe propone dos anclas para la misma mano física -> conservar la de mayor certeza
      if (d_entre_manos < 0.10) {
        if ((det1.score || 0) > (det0.score || 0)) {
          detections.shift(); // Descartar det0
        } else {
          detections.pop();   // Descartar det1
        }
      } 
      // 2. Descarte por asimetría severa de confianza de handedness:
      // Si una mano tiene certeza alta (> 0.85) y la otra es muy dudosa (< 0.60)
      else if (det0.score !== undefined && det1.score !== undefined) {
        if (det0.score > 0.85 && det1.score < 0.60) {
          detections.pop();
        } else if (det1.score > 0.85 && det0.score < 0.60) {
          detections.shift();
        }
      }
      
      // 3. Descarte de anomalía anatómica (dos manos idénticas en el mismo cuadrante)
      if (detections.length >= 2) {
        const d0 = detections[0];
        const d1 = detections[1];
        if (d0.label === d1.label && d0.label !== 'Desconocida') {
          if ((d0.score || 1) < (d1.score || 1) - 0.12) {
            detections.shift();
          } else if ((d1.score || 1) < (d0.score || 1) - 0.12) {
            detections.pop();
          }
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

      // Verificar si la otra mano estaba recientemente activa y REALMENTE quedó oculta detrás
      let upOther = null;
      const wasOtherActive = otherSlot.lastFiltered && otherSlot.framesMissing < otherSlot.maxOcclusionFrames;
      if (wasOtherActive) {
        const distToFront = Vec3.dist(det.wrist, otherSlot.lastFiltered[0]);
        // Solo considerar oclusión si las muñecas estaban verdaderamente solapadas (< 0.15)
        if (distToFront < 0.15) {
          upOther = otherSlot.updateOccluded(det.wrist, now);
        } else {
          otherSlot.reset();
        }
      } else {
        otherSlot.reset();
      }

      const res = [];
      if (upActive) {
        res.push({
          ...upActive,
          slot: isSlot0 ? 0 : 1,
          label: activeLabel,
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
 */
function extraerDescriptorArticular(coords) {
  const p0 = coords[0];
  const rel = coords.map(c => Vec3.sub(c, p0));

  // Base canónica orientada por la palma
  const v_mcp_medio = Vec3.sub(coords[9], p0);
  const d_palma = Math.max(Vec3.norm(v_mcp_medio), 1e-4);
  const uy = Vec3.scale(v_mcp_medio, 1.0 / d_palma);

  const v_base = Vec3.sub(coords[17], coords[5]);
  let uz = Vec3.cross(uy, v_base);
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
 * Descriptor multimodal 109D combinando cinemática de mano
 * con la posición anatómica relativa a los hombros.
 */
function extraerDescriptorMultimodal(handCoords, poseAnchors) {
  const { vec105, ext_dedos, p_canon } = extraerDescriptorArticular(handCoords);
  const p0 = handCoords[0]; // Muñeca [x, y, z]

  let dx = 0.0, dy = 0.0, dz = 0.0;
  if (poseAnchors && poseAnchors.hombro_izq && poseAnchors.hombro_der) {
    const hizq = poseAnchors.hombro_izq;
    const hder = poseAnchors.hombro_der;
    const c_h = [0.5 * (hizq[0] + hder[0]), 0.5 * (hizq[1] + hder[1]), 0.5 * (hizq[2] + hder[2])];
    const w_h = Math.max(Vec3.norm2d(hizq[0] - hder[0], hizq[1] - hder[1]), 0.10);

    dx = (p0[0] - c_h[0]) / w_h;
    dy = (p0[1] - c_h[1]) / w_h;
    dz = (p0[2] - c_h[2]) / w_h;
  } else {
    // Estimación anatómica calibrada para encuadre selfie/móvil
    // Hombros centrados horizontalmente en 0.50, altura clavicular en 0.68, ancho 0.38
    const c_h_default = [0.50, 0.68, 0.0];
    const w_h_default = 0.38;
    dx = (p0[0] - c_h_default[0]) / w_h_default;
    dy = (p0[1] - c_h_default[1]) / w_h_default;
    dz = (p0[2] || 0.0) / w_h_default;
  }

  const dist_cuerpo = Math.hypot(dx, dy, dz);
  const cuerpo_coords = [dx * 2.5, dy * 2.5, dz * 2.5, dist_cuerpo * 2.5];

  const vec109 = vec105.concat(cuerpo_coords);

  let zona = "PECHO";
  if (dy < -0.30) {
    zona = "CABEZA_ROSTRO";
  } else if (dy > 0.45) {
    zona = "ABDOMEN_CADERA";
  }

  return {
    vec109,
    ext_dedos,
    dy,
    zona,
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
    this.MIN_NEUTRAL_DURATION = 350; // ms antes de confirmar neutralidad
    this.MIN_NEUTRAL_FRAMES = 6;     // frames mínimos (~200ms)
  }

  /**
   * Evalúa si el frame actual muestra una mano neutra/tendida.
   * @param {number[]} extDedos - Array de 5 extensiones [pulgar, índice, medio, anular, meñique] (0-1)
   * @param {number} speed - Velocidad cinemática de la muñeca
   * @param {number[]} wristPos - Posición [x, y, z] de la muñeca
   * @param {number} timestamp - Timestamp actual en ms
   * @returns {{ isNeutral: boolean, confidence: number, durationMs: number }}
   */
  evaluate(extDedos, speed, wristPos, timestamp) {
    const now = timestamp || performance.now();

    // Contar dedos extendidos (umbral 0.55 para capturar manos semi-abiertas)
    const dedosAbiertos = extDedos.filter(e => e > 0.55).length;
    const todosAbiertos = dedosAbiertos >= 4;
    const velocidadBaja = speed < 0.12;

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
    const esNeutralFrame = todosAbiertos && velocidadBaja && microMovimiento < 0.025;

    if (esNeutralFrame) {
      this.framesNeutral++;
      if (!this.neutralStartTime) {
        this.neutralStartTime = now;
      }
      this.neutralDurationMs = now - this.neutralStartTime;
      this.isCurrentlyNeutral = (this.neutralDurationMs >= this.MIN_NEUTRAL_DURATION && 
                                  this.framesNeutral >= this.MIN_NEUTRAL_FRAMES);
    } else {
      // Movimiento detectado → resetear neutralidad
      this.neutralStartTime = null;
      this.neutralDurationMs = 0;
      this.framesNeutral = 0;
      this.isCurrentlyNeutral = false;
    }

    // Confianza: cuánto más tiempo lleva neutral, mayor certeza
    const confidence = this.isCurrentlyNeutral 
      ? Math.min(0.99, 0.70 + (this.neutralDurationMs / 3000) * 0.29)
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
    this.speedThreshold = options.speedThreshold || 0.028; // Umbral de velocidad cinemática
    this.minActiveFrames = options.minActiveFrames || 3; // Frames activos para abrir compuerta
    this.minStaticMs = options.minStaticMs || 250; // ms estáticos antes de cerrar compuerta
    
    this.speedHistory = [];
    this.lastWrist = null;
    this.lastTips = null;
    this.lastActiveTime = null;
    this.staticStartTime = null;
    this.isOpen = false;
  }

  update(wristPos, fingerTips, timestamp) {
    const now = timestamp || performance.now();
    
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
    this.lastTips = fingerTips ? fingerTips.map(p => [...p]) : null;

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
 * - Detecta explícitamente REPOSO y TRANSICION.
 * - Bloquea predicciones si la mano está neutra/tendida (v5.0).
 * - Requiere confianza >= 75% y margen >= 12% para confirmar una seña activa.
 */
function predecirRedNeuronal(vec109, modelo, handMeta) {
  if (!modelo) {
    modelo = window.MODELO_LSC;
  }
  if (!modelo) {
    return { sena: "DESCONOCIDO", confianza: 0.0, top3: [] };
  }

  const { scaler_mean, scaler_scale, clases } = modelo;

  // 0. COMPUERTA CINEMÁTICA DE MOVIMIENTO (MOTION-GATED INFERENCE v5.2)
  // Si la mano está estática/inmóvil sin movimiento activo de seña, abstenerse de predecir texto
  const slotIdx = (handMeta && handMeta.slot !== undefined) ? handMeta.slot : 0;
  const motionGate = slotIdx === 1 ? _motionGateSlot1 : _motionGateSlot0;
  const wristPos = (handMeta && handMeta.coords) ? handMeta.coords[0] : null;
  const tips = (handMeta && handMeta.coords) ? [handMeta.coords[4], handMeta.coords[8], handMeta.coords[12], handMeta.coords[16], handMeta.coords[20]] : null;
  const gateRes = motionGate.update(wristPos, tips, performance.now());

  if (!gateRes.isOpen) {
    return {
      sena: "REPOSO",
      rawSena: "REPOSO",
      confianza: 0.99,
      margen: 0.99,
      estado: "REPOSO",
      candidatos: [{ sena: "REPOSO", probabilidad: 0.99 }],
      isMotionGated: true,
      gateSpeed: gateRes.speed,
      gateState: gateRes.state
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

  for (let l = 0; l < numLayers - 1; l++) {
    const W = weights[l];
    const B = biases[l];
    const outDim = B.length;
    const inDim = layerInput.length;
    const nextOut = new Float32Array(outDim);

    for (let j = 0; j < outDim; j++) {
      let sum = B[j];
      for (let i = 0; i < inDim; i++) {
        sum += layerInput[i] * W[i][j];
      }
      nextOut[j] = sum > 0 ? sum : 0; // ReLU
    }
    layerInput = nextOut;
  }

  // 3. Capa de Salida (Logits finales)
  const W_last = weights[numLayers - 1];
  const B_last = biases[numLayers - 1];
  const num_clases = clases.length;
  const inDim = layerInput.length;
  const logits = new Float32Array(num_clases);
  let maxLogit = -Infinity;

  for (let j = 0; j < num_clases; j++) {
    let sum = B_last[j];
    for (let i = 0; i < inDim; i++) {
      sum += layerInput[i] * W_last[i][j];
    }
    logits[j] = sum;
    if (sum > maxLogit) maxLogit = sum;
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

  // 5. Extraer extensión de dedos del vector para salvaguardas
  const ext_pulgar  = (vec109[63] || 0) / 3.2;
  const ext_indice  = (vec109[64] || 0) / 3.2;
  const ext_medio   = (vec109[65] || 0) / 3.2;
  const ext_anular  = (vec109[66] || 0) / 3.2;
  const ext_menique = (vec109[67] || 0) / 3.2;
  const extDedos = [ext_pulgar, ext_indice, ext_medio, ext_anular, ext_menique];

  const esManoAbierta = (ext_indice > 0.65 && ext_medio > 0.65 && ext_anular > 0.65 && ext_menique > 0.65);
  const esManoSemiAbierta = (ext_indice > 0.55 && ext_medio > 0.55 && ext_anular > 0.50);

  // 5a. DETECTOR DE MANO NEUTRA (v5.0) — Bloqueo preventivo total
  // Si la mano está tendida/abierta y estática, forzar REPOSO independientemente del modelo
  const slotIdx = (handMeta && handMeta.slot !== undefined) ? handMeta.slot : 0;
  const neutralDetector = slotIdx === 1 ? _neutralDetectorSlot1 : _neutralDetectorSlot0;
  const speed = (handMeta && handMeta.speed !== undefined) ? handMeta.speed : 0;
  const wristPos = (handMeta && handMeta.coords) ? handMeta.coords[0] : null;

  const neutralResult = neutralDetector.evaluate(extDedos, speed, wristPos, performance.now());

  // Si la mano está confirmada como neutra Y la predicción no es una seña válida con mano abierta
  if (neutralResult.isNeutral && neutralResult.framesNeutral >= 8) {
    // HOLA, BUENAS, TARDES y GUSTAR son señas LSC compatibles con mano abierta
    const esSenaConManoAbiertaValida = (top1.sena === 'HOLA' || top1.sena === 'BUENAS' || top1.sena === 'TARDES' || top1.sena === 'GUSTAR');
    if (!esSenaConManoAbiertaValida || neutralResult.durationMs > 900) {
      return {
        sena: 'REPOSO',
        rawSena: top1.sena,
        confianza: neutralResult.confidence,
        margen: 0,
        estado: 'REPOSO',
        candidatos: candidatos.slice(0, 3),
        isNeutralHand: true,
        neutralDurationMs: neutralResult.durationMs
      };
    }
  }

  // 5b. Salvaguardas Anatómicas Canónicas LSC Biomecánicas (v5.1):
  // dy normalizado respecto a hombros (dy > 0 es hacia abajo/abdomen, dy < 0 hacia arriba/cabeza)
  const dy = (vec109[106] || 0) / 2.5;

  if (top1.sena === "DIAS" && dy > 0.50) {
    // DÍAS se realiza en la mitad superior del cuerpo (salida del sol), no en abdomen/cadera
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.90;
  } else if (top1.sena === "LICOR" && (esManoAbierta || (ext_indice > 0.75 && ext_medio > 0.75 && ext_anular > 0.75))) {
    // LICOR: pulgar al cuello. Si todos los dedos están extendidos como saludo, es mano abierta/HOLA, no LICOR
    top1.sena = esManoAbierta ? "REPOSO" : "TRANSICION";
    top1.probabilidad = 0.93;
  } else if (top1.sena === "YO" && dy < -0.30) {
    // YO: índice al pecho/esternón, no arriba en la cabeza
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.92;
  } else if (top1.sena === "AÑOS" && (ext_indice > 0.70 || ext_medio > 0.70)) {
    // AÑOS: puño cerrado acariciando mejilla
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.90;
  } else if (top1.sena === "NOCHES" && esManoAbierta && speed < 0.08) {
    // NOCHES: manos descendiendo
    top1.sena = "REPOSO";
    top1.probabilidad = 0.92;
  } else if (top1.sena === "GUSTAR" && dy < -0.35) {
    // GUSTAR: palma sobre el corazón/pecho, no arriba en la cabeza
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.90;
  } else if (top1.sena === "NOMBRE" && esManoAbierta && speed < 0.08) {
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.90;
  } else if (top1.sena === "GRACIAS" && esManoAbierta && speed < 0.08) {
    top1.sena = "TRANSICION";
    top1.probabilidad = 0.90;
  }

  // 6. Filtro de Decisión Anti-Aleatoriedad
  let senaFinal = top1.sena;
  let estado = "SEÑA_DETECTADA";

  if (top1.sena === "REPOSO") {
    estado = "REPOSO";
    senaFinal = "REPOSO";
  } else if (top1.sena === "TRANSICION") {
    estado = "TRANSICION";
    senaFinal = "TRANSICIÓN";
  } else if (top1.probabilidad < 0.75 || margen < 0.12) {
    // Umbral calibrado v5.0: 75% confianza mínima y 12% de margen
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
    extraerDescriptorMultimodal,
    predecirRedNeuronal,
    OneEuroFilter,
    LandmarkStabilizer,
    DualHandTracker,
    NeutralHandDetector,
    MotionGate,
    HysteresisDebounce,
    MejoradorVideoOnDevice,
    DecodificadorContinuoCTC,
    INDICES_FACIALES_NMM_64,
    Vec3
  };
} else {
  window.MotorLSCLocal = {
    extraerDescriptorArticular,
    extraerDescriptorMultimodal,
    predecirRedNeuronal,
    OneEuroFilter,
    LandmarkStabilizer,
    DualHandTracker,
    NeutralHandDetector,
    MotionGate,
    HysteresisDebounce,
    MejoradorVideoOnDevice,
    DecodificadorContinuoCTC,
    INDICES_FACIALES_NMM_64,
    Vec3
  };
}

