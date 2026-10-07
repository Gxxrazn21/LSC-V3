"""
Rostro del avatar v4:
 - blendshapes ARKit en la malla (cejas, ojos, boca, nariz) por campos de deformación suaves
   centrados en los rasgos detectados en la textura
 - párpados reales (mallas aparte) para parpadear: en reposo quedan plegados e invisibles
Salida: avatar_v4.blend
"""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = "/home/claude/avatar_lsc/v4"
P = "mixamorig:"
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "rigged.blend"))
ob = bpy.data.objects["Avatar"]; me = ob.data
rig = bpy.data.objects["AvatarLSC_Rig"]
V = np.array([v.co[:] for v in me.vertices]); N = len(V)
x, y, z = V[:, 0], V[:, 1], V[:, 2]

# ---- rasgos (detectados en la textura, ver detección en el chat)
EYE = {1: np.array([0.0466, 1.6556]), -1: np.array([-0.0472, 1.6556])}
EYE_A, EYE_B = 0.0205, 0.0165
BROW_Z = 1.6885
MOUTH_Z, MOUTH_X = 1.5745, 0.031
NOSE = np.array([0.0, 1.603])

front = smooth_front = np.clip((-y - 0.02) / 0.04, 0, 1)            # solo cara frontal
face_zone = front * (z > 1.53) * (z < 1.75) * (np.abs(x) < 0.11)

def g2(cx, cz, sx, sz):
    return np.exp(-(((x - cx) / sx) ** 2) - (((z - cz) / sz) ** 2)) * face_zone

def smoothstep(a, b, t):
    t = np.clip((t - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

ob.shape_key_add(name="Basis")
KEYS = {}
def add(name, D):
    kb = ob.shape_key_add(name=name, from_mix=False)
    kb.data.foreach_set("co", (V + D).ravel()); kb.value = 0.0
    KEYS[name] = D

def D0(): return np.zeros((N, 3))

# --- cejas
d = D0(); w = sum(g2(s * 0.030, BROW_Z, 0.016, 0.013) for s in (1, -1)); d[:, 2] = 0.0060 * w; d[:, 0] = -0.0008 * np.sign(x) * w
add("browInnerUp", d)
for s, nm in ((1, "Left"), (-1, "Right")):
    d = D0(); w = g2(s * 0.064, BROW_Z, 0.018, 0.013); d[:, 2] = 0.0055 * w
    add("browOuterUp" + nm, d)
    d = D0(); w = g2(s * 0.044, BROW_Z, 0.026, 0.012); d[:, 2] = -0.0045 * w; d[:, 0] = -s * 0.0015 * w; d[:, 1] = -0.0010 * w
    add("browDown" + nm, d)
# --- ojos (abrir / entrecerrar)
for s, nm in ((1, "Left"), (-1, "Right")):
    ex, ez = EYE[s]
    d = D0(); w = g2(ex, ez + EYE_B * 0.9, 0.022, 0.009); d[:, 2] = 0.0035 * w
    w2 = g2(ex, ez - EYE_B * 0.9, 0.022, 0.008); d[:, 2] -= 0.0018 * w2
    add("eyeWide" + nm, d)
    d = D0(); w2 = g2(ex, ez - EYE_B * 1.1, 0.024, 0.010); d[:, 2] = 0.0035 * w2; d[:, 1] = -0.0010 * w2
    w = g2(ex, ez + EYE_B * 1.1, 0.022, 0.008); d[:, 2] -= 0.0015 * w
    add("eyeSquint" + nm, d)
# --- boca
for s, nm in ((1, "Left"), (-1, "Right")):
    d = D0()
    w = g2(s * MOUTH_X, MOUTH_Z, 0.013, 0.010)
    d[:, 2] += 0.0055 * w; d[:, 0] += s * 0.0030 * w; d[:, 1] += 0.0015 * w
    wc = g2(s * 0.042, 1.597, 0.020, 0.016)                              # mejilla sube
    d[:, 2] += 0.0025 * wc; d[:, 1] -= 0.0015 * wc
    add("mouthSmile" + nm, d)
    d = D0(); w = g2(s * MOUTH_X, MOUTH_Z, 0.013, 0.010); d[:, 2] -= 0.0045 * w; d[:, 0] += s * 0.0008 * w
    add("mouthFrown" + nm, d)
    d = D0(); w = g2(s * MOUTH_X, MOUTH_Z, 0.012, 0.010); d[:, 0] += s * 0.0040 * w; d[:, 1] += 0.0010 * w
    add("mouthStretch" + nm, d)
    d = D0(); w = g2(s * 0.013, 1.604, 0.010, 0.008); d[:, 2] += 0.0022 * w; d[:, 1] -= 0.0006 * w
    add("noseSneer" + nm, d)
# mandíbula: todo lo que está bajo la línea de la boca baja (la línea pintada se abre)
d = D0()
lat = np.exp(-((x / 0.055) ** 2))
below = smoothstep(MOUTH_Z + 0.0005, MOUTH_Z - 0.004, z) * (z > 1.505) * front
fade = smoothstep(1.505, 1.53, z)
d[:, 2] = -0.0085 * below * fade * lat
d[:, 1] = 0.0015 * below * fade * lat
add("jawOpen", d)
d = D0(); w = g2(0, MOUTH_Z, 0.030, 0.012); d[:, 0] = -0.40 * x * w; d[:, 1] = -0.0045 * w
add("mouthPucker", d)
d = D0(); w = g2(0, MOUTH_Z, 0.028, 0.012); d[:, 0] = -0.20 * x * w; d[:, 1] = -0.0050 * w
d[:, 2] -= 0.0025 * g2(0, MOUTH_Z - 0.004, 0.02, 0.006)
add("mouthFunnel", d)
d = D0(); w = g2(0, MOUTH_Z, 0.028, 0.008); d[:, 2] = 0.0012 * np.sign(MOUTH_Z - z + 1e-6) * w; d[:, 1] = 0.0008 * w
add("mouthPress", d)

# ---------------------------------------------------------------- párpados
UVS = np.zeros((N, 2)); ul = me.uv_layers.active.data
for p_ in me.polygons:
    for li in p_.loop_indices: UVS[me.loops[li].vertex_index] = ul[li].uv[:]
bvh = BVHTree.FromPolygons([Vector(v) for v in V], [tuple(p.vertices) for p in me.polygons])
skin = bpy.data.materials.new("Parpado"); skin.use_nodes = True
b = skin.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.72, 0.43, 0.32, 1); b.inputs["Roughness"].default_value = 0.55
lash = bpy.data.materials.new("Pestanas"); lash.use_nodes = True
lb = lash.node_tree.nodes["Principled BSDF"]
lb.inputs["Base Color"].default_value = (0.02, 0.015, 0.012, 1); lb.inputs["Roughness"].default_value = 0.7

def surf(px, pz, off=0.0009):
    hit = bvh.ray_cast(Vector((px, -0.5, pz)), Vector((0, 1, 0)))
    return hit[0] + Vector((0, -1, 0)) * off if hit[0] else Vector((px, -0.1, pz))

NU, NV = 26, 12
import json
PROF = {int(k): np.array(v) for k, v in json.load(open(os.path.join(HERE, "eye_profiles.json"))).items()}
lids = []
for s, nm in ((1, "Left"), (-1, "Right")):
    ex, ez = EYE[s]
    pr = PROF[s]; pr = pr[np.argsort(pr[:, 0])]
    # suavizar contornos y ampliar 0.8 mm
    from scipy.ndimage import uniform_filter1d
    top_s = uniform_filter1d(pr[:, 1], 5) + 0.0009
    bot_s = uniform_filter1d(pr[:, 2], 5) - 0.0007
    x0, x1 = pr[0, 0] - 0.0012, pr[-1, 0] + 0.0012
    bm = bmesh.new(); rows_rest, rows_closed = [], []
    for j in range(NV + 1):
        v = (j / (NV - 1)) * 0.90 if j < NV else 1.0
        rr, rc = [], []
        for i in range(NU + 1):
            u = i / NU
            px = x0 + (x1 - x0) * u
            top = float(np.interp(px, pr[:, 0], top_s)); bot = float(np.interp(px, pr[:, 0], bot_s))
            if u in (0.0, 1.0): mid = (top + bot) / 2; top = bot = mid
            zc = top + (bot - top) * v
            edge = math.sin(math.pi * u)
            closed = surf(px, zc, 0.00015 + (0.0009 * math.sin(math.pi * min(v * 0.6, 1.0)) + 0.0003 * v) * edge)
            rest = surf(px, top, 0.0001)
            rr.append(rest); rc.append(closed)
        rows_rest.append(rr); rows_closed.append(rc)
    verts = [[bm.verts.new(p) for p in row] for row in rows_rest]
    for j in range(NV):
        for i in range(NU):
            f = bm.faces.new([verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]])
            f.material_index = 1 if j >= NV - 1 else 0
    lm = bpy.data.meshes.new("Parpado" + nm); bm.to_mesh(lm); bm.free()
    lm.materials.append(me.materials[0]); lm.materials.append(lash)
    tgt = np.array([ex, 0, ez + 0.020]); cand = np.where(face_zone > 0.5)[0]
    k0 = cand[np.argmin(np.linalg.norm(V[cand][:, [0, 2]] - tgt[[0, 2]], axis=1))]
    uv_skin = UVS[k0]
    uvl = lm.uv_layers.new(name="UVMap")
    for l in uvl.data: l.uv = uv_skin
    for p in lm.polygons: p.use_smooth = True
    lo = bpy.data.objects.new("Parpado" + nm, lm); bpy.context.scene.collection.objects.link(lo)
    lo.shape_key_add(name="Basis")
    kb = lo.shape_key_add(name="eyeBlink" + nm, from_mix=False)
    flat = [p for row in rows_closed for p in row]
    for k, p in enumerate(flat): kb.data[k].co = p
    kb.value = 0.0
    # los párpados acompañan "eyeSquint"/"eyeWide" (el borde superior sigue la piel)
    for other in ("eyeWide" + nm, "eyeSquint" + nm, "browDown" + nm, "browInnerUp"):
        D = KEYS[other]
        kd_idx = []
        ko = lo.shape_key_add(name=other, from_mix=False)
        for k, vv in enumerate(lo.data.vertices):
            dd = np.linalg.norm(V - np.array(vv.co[:]), axis=1); n_ = np.argmin(dd)
            ko.data[k].co = vv.co + Vector(D[n_])
        ko.value = 0.0
    g = lo.vertex_groups.new(name=P + "Head"); g.add(list(range(len(lm.vertices))), 1.0, "REPLACE")
    lo.parent = rig; m_ = lo.modifiers.new("Armature", "ARMATURE"); m_.object = rig
    lids.append(lo)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "avatar_v4.blend"))
print("OK claves:", list(KEYS))
