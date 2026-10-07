"""
Riggea el modelo del usuario (medio cuerpo, T-pose):
 - detecta articulaciones (hombros, codos, muñecas, nudillos, falanges, pulgar, cuello, cabeza)
 - crea esqueleto con nombres Mixamo (mixamorig:*) y 3 falanges por dedo
 - pesos automáticos (bone heat) + limpieza
Salida: rigged.blend
"""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector

HERE = "/home/claude/avatar_lsc/v4"
P = "mixamorig:"
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "prep.blend"))
ob = bpy.data.objects["Avatar"]; me = ob.data
V = np.array([v.co[:] for v in me.vertices])

def slab(axis, val, w=0.003, extra=None):
    m = np.abs(V[:, axis] - val) < w
    if extra is not None: m &= extra
    return m

def centroid(mask):
    return V[mask].mean(0)

J = {}   # articulaciones por lado
for side, sx in (("Left", 1), ("Right", -1)):
    arm_band = (np.abs(V[:, 2] - 1.39) < 0.09)
    def c_at(x, w=0.004):
        m = slab(0, sx * x, w, arm_band)
        P_ = V[m]
        return np.array([sx * x, (P_[:, 1].min() + P_[:, 1].max()) / 2, (P_[:, 2].min() + P_[:, 2].max()) / 2])
    sh = c_at(0.17); sh[2] = c_at(0.24)[2]          # el hombro: altura del eje del brazo
    el = c_at(0.435)
    wr = c_at(0.645)
    J[side] = dict(clav=np.array([sx * 0.035, 0.03, 1.425]), sh=sh, el=el, wr=wr)
    # ---- dedos: separados para x >= 0.77; ordenados por z (arriba = índice, el pulgar está arriba)
    zc = {"Index": 1.465, "Middle": 1.436, "Ring": 1.405, "Pinky": 1.373}
    hand = (np.abs(V[:, 0]) > 0.64) & (np.sign(V[:, 0]) == sx)
    for fn, z0 in zc.items():
        pts = []
        z = z0
        for x in np.arange(0.77, 0.87, 0.004):
            m = slab(0, sx * x, 0.002, hand & (np.abs(V[:, 2] - z) < 0.012))
            if m.sum() < 4: break
            c = V[m]; cc = np.array([sx * x, (c[:, 1].min() + c[:, 1].max()) / 2, (c[:, 2].min() + c[:, 2].max()) / 2])
            pts.append(cc); z = cc[2]
        pts = np.array(pts)
        # punta: vértice más lejano de esa franja
        fm = hand & (np.abs(V[:, 2] - pts[-1, 2]) < 0.012) & (np.abs(V[:, 0]) > abs(pts[-1, 0]) - 0.01)
        tip = V[fm][np.argmax(np.abs(V[fm][:, 0]))].copy(); tip[1] = pts[-1, 1]; tip[2] = pts[-1, 2]
        base = pts[0] - np.array([sx * 0.022, 0, 0])                      # nudillo (MCP) dentro de la palma
        L = np.linalg.norm(tip - base)
        dirv = (tip - base) / L
        mcp = base; pip = base + dirv * L * 0.45; dip = base + dirv * L * 0.75
        # ajustar a la línea central real
        def snap(p):
            k = np.argmin(np.linalg.norm(pts - p, axis=1)); q = p.copy(); q[1:] = pts[k][1:] if abs(pts[k][0] - p[0]) < 0.01 else p[1:]; return q
        J[side][fn] = [mcp, snap(pip), snap(dip), tip]
    # ---- pulgar: apunta hacia arriba (+z); rebanadas en z
    th = hand & (np.abs(V[:, 0]) > 0.66) & (np.abs(V[:, 0]) < 0.79)
    tpts = []
    for z in np.arange(1.475, 1.52, 0.004):
        m = slab(2, z, 0.002, th & (V[:, 2] > 1.47))
        if m.sum() < 3: break
        c = V[m]
        # quedarse con el grupo más cercano al centro del pulgar (x ~ 0.73-0.76)
        tpts.append(np.array([(c[:, 0].min() + c[:, 0].max()) / 2, (c[:, 1].min() + c[:, 1].max()) / 2, z]))
    tpts = np.array(tpts)
    ttip = V[th][np.argmax(V[th][:, 2])].copy(); ttip[:2] = tpts[-1][:2]
    cmc = wr + np.array([sx * 0.030, -0.004, 0.035])
    t2 = tpts[0] if len(tpts) else cmc + (ttip - cmc) * 0.45
    t3 = t2 + (ttip - t2) * 0.5
    J[side]["Thumb"] = [cmc, t2, t3, ttip]
    J[side]["hand_tail"] = J[side]["Middle"][0]

# ---------------------------------------------------------------- esqueleto
arm = bpy.data.armatures.new("AvatarLSC")
rig = bpy.data.objects.new("AvatarLSC_Rig", arm)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones
def bone(n, h, t, parent=None, roll_to=(0, -1, 0)):
    b = eb.new(P + n); b.head = Vector(h); b.tail = Vector(t)
    if parent: b.parent = eb[P + parent]
    b.align_roll(Vector(roll_to))
    return b
yc = 0.025
bone("Hips", (0, yc, 0.92), (0, yc, 1.02))
bone("Spine", (0, yc, 1.02), (0, yc, 1.13), "Hips")
bone("Spine1", (0, yc, 1.13), (0, yc, 1.25), "Spine")
bone("Spine2", (0, yc, 1.25), (0, yc, 1.40), "Spine1")
bone("Neck", (0, 0.025, 1.455), (0, 0.015, 1.535), "Spine2")
bone("Head", (0, 0.015, 1.535), (0, 0.0, 1.80), "Neck")
for side in ("Left", "Right"):
    j = J[side]
    bone(side + "Shoulder", j["clav"], j["sh"], "Spine2", (0, 0, 1))
    bone(side + "Arm", j["sh"], j["el"], side + "Shoulder", (0, 0, 1))
    bone(side + "ForeArm", j["el"], j["wr"], side + "Arm", (0, 0, 1))
    bone(side + "Hand", j["wr"], j["hand_tail"], side + "ForeArm", (0, -1, 0))
    for fn in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        p = j[fn]
        for i in range(3):
            bone(f"{side}Hand{fn}{i+1}", p[i], p[i + 1],
                 side + "Hand" if i == 0 else f"{side}Hand{fn}{i}", (0, -1, 0))
for b in eb: b.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")

# ---------------------------------------------------------------- pesos automáticos
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True); rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
nz = sum(1 for v in me.vertices if sum(g.weight for g in v.groups) < 1e-4)
print("vértices sin peso:", nz)
np.save(os.path.join(HERE, "joints.npy"), J, allow_pickle=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "rigged.blend"))
print("OK", len(arm.bones), "huesos")
for side in ("Left",):
    for k, v in J[side].items():
        print(side, k, np.round(np.array(v), 3).tolist())
