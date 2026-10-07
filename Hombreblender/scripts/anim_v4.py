"""
Animación HOLA (LSC) para el avatar v2 — mano dominante derecha.
Mano en configuración B (plana) que sale de la sien hacia el interlocutor.
Movimiento natural: anticipación, arcos, giro distribuido antebrazo/muñeca,
dedos con curvatura escalonada, hombro y torso acompañan, respiración, parpadeo,
cabeza y expresión facial (sonrisa y cejas arriba = saludo amable).
"""
import bpy, math, os, sys
import numpy as np
from mathutils import Vector, Matrix, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
P = "mixamorig:"
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "avatar_v4.blend"))
rig = bpy.data.objects["AvatarLSC_Rig"]
body = bpy.data.objects["Avatar"]
brows = [o for o in bpy.data.objects if o.name.startswith("Parpado")]
pbs = rig.pose.bones
bones = rig.data.bones
for pb in pbs: pb.rotation_mode = "QUATERNION"
upd = bpy.context.view_layer.update

def B(n): return bones[P + n]
def PB(n): return pbs[P + n]

# ---------------------------------------------------------------- marcos de reposo de las manos
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
REST = {}
for side, sx in (("Left", 1), ("Right", -1)):
    h = B(side + "Hand").head_local
    f0 = (B(side + "HandMiddle1").head_local - h).normalized()
    across = (B(side + "HandIndex1").head_local - B(side + "HandPinky1").head_local).normalized()
    n0 = (f0.cross(across) * sx).normalized()            # normal de la palma
    across = n0.cross(f0).normalized()
    REST[side] = dict(f=f0, n=n0, M=Matrix((f0, -n0, f0.cross(-n0))).transposed())

def hframe(f, b):
    f = Vector(f).normalized(); b = Vector(b); b = (b - f * b.dot(f)).normalized()
    return Matrix((f, b, f.cross(b))).transposed()

# ---------------------------------------------------------------- utilidades de pose
def reset():
    for pb in pbs:
        pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
    upd()

def set_world_rot(pb, R3):
    M = R3.to_4x4(); M.translation = pb.matrix.translation
    pb.matrix = M; upd()

def rot_world(name, axis, deg):
    pb = PB(name)
    R = Matrix.Rotation(math.radians(deg), 3, Vector(axis))
    set_world_rot(pb, R @ pb.matrix.to_3x3())

def aim(name, target):
    pb = PB(name)
    M3 = pb.matrix.to_3x3()
    y = M3.col[1].normalized()
    d = (Vector(target) - pb.matrix.translation).normalized()
    set_world_rot(pb, y.rotation_difference(d).to_matrix() @ M3)

def two_bone(S, W, l1, l2, pole):
    u = W - S; d = min(u.length, (l1 + l2) * 0.999); u.normalize()
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = Vector(pole); p = (p - u * p.dot(u)).normalized()
    return S + u * a + p * h, S + u * d

def arm(side, wrist, fingers_dir, back_dir, pole, twist_share=0.55):
    S = PB(side + "Arm").head.copy()
    l1, l2 = B(side + "Arm").length, B(side + "ForeArm").length
    E, W = two_bone(S, Vector(wrist), l1, l2, pole)
    aim(side + "Arm", E); aim(side + "ForeArm", W)
    # orientación objetivo de la mano
    rest = REST[side]
    R = hframe(fingers_dir, back_dir) @ hframe(rest["f"], -rest["n"]).inverted()
    Rh = R @ B(side + "Hand").matrix_local.to_3x3()
    # repartir el giro entre antebrazo y muñeca (evita "torcer" la muñeca)
    fa = PB(side + "ForeArm")
    rel = fa.matrix.to_3x3().inverted() @ Rh
    q = rel.to_quaternion()
    tw = Quaternion((q.w, 0, q.y, 0)); tw.normalize()
    ang = 2 * math.atan2(tw.y, tw.w)
    ax = fa.matrix.to_3x3().col[1].normalized()
    set_world_rot(fa, Matrix.Rotation(ang * twist_share, 3, ax) @ fa.matrix.to_3x3())
    set_world_rot(PB(side + "Hand"), Rh)

def finger_axis(side, name):
    b = B(name)
    d = (b.tail_local - b.head_local).normalized()
    a = d.cross(REST[side]["n"]).normalized()
    return b.matrix_local.to_3x3().inverted() @ a

def curl(side, finger, a1, a2, a3, spread=0.0):
    for i, a in enumerate((a1, a2, a3), 1):
        nm = f"{side}Hand{finger}{i}"
        ax = finger_axis(side, nm)
        q = Quaternion(ax, math.radians(a))
        if i == 1 and spread:
            nl = B(nm).matrix_local.to_3x3().inverted() @ REST[side]["n"]
            q = Quaternion(nl, math.radians(spread * (1 if side == "Left" else -1))) @ q
        PB(nm).rotation_quaternion = q
    upd()

SHAPES = {   # (base, medio, punta) por dedo + separación
    "relajada": dict(Thumb=(8, 12, 10, 4), Index=(14, 18, 10, -3), Middle=(18, 24, 12, 0),
                     Ring=(22, 28, 14, 3), Pinky=(26, 32, 16, 6)),
    "B":        dict(Thumb=(22, 18, 8, -10), Index=(2, 3, 2, 2), Middle=(2, 2, 2, 0),
                     Ring=(3, 3, 2, -2), Pinky=(4, 4, 3, -4)),
}
def hand_shape(side, name, mix=None, t=0.0):
    s = SHAPES[name]
    for fn, (a1, a2, a3, sp) in s.items():
        if mix:
            b1, b2, b3, bs = SHAPES[mix][fn]
            a1, a2, a3, sp = [x + (y - x) * t for x, y in zip((a1, a2, a3, sp), (b1, b2, b3, bs))]
        curl(side, fn, a1, a2, a3, sp)

# ---------------------------------------------------------------- referencias del cuerpo
eyeR = Vector((-0.047, -0.09, 1.6556)); eyeL = Vector((0.047, -0.09, 1.6556))
head_h = B("Head").head_local
sh_R = B("RightArm").head_local; sh_L = B("LeftArm").head_local
L_arm = B("RightArm").length + B("RightForeArm").length

def rest_arm(side, sway=0.0):
    """reposo de intérprete: manos frente al abdomen, dedos hacia el centro"""
    sx = 1 if side == "Left" else -1
    W = Vector((sx * (0.125 + sway), -0.125, 1.075 + (0.012 if side == "Left" else 0.0)))
    return dict(wrist=W, fingers_dir=(-sx * 0.85, -0.35, -0.12), back_dir=(sx * 0.15, -0.25, 1.0),
                pole=(sx * 0.6, 0.5, -0.7))

temple = Vector((-0.100, -0.065, 1.700))
def right_temple():
    f = Vector((0.55, -0.05, 0.85)).normalized()
    return dict(wrist=temple - f * 0.185, fingers_dir=f, back_dir=(-0.75, 0.6, 0.0),
                pole=(-0.6, 0.35, -1.0))
def right_out(wave=0.0):
    f = Vector((-0.12, -0.10, 1.0)).normalized()
    b = Vector((0.0, 1.0, 0.05))
    f = Quaternion(b, math.radians(wave)) @ f
    W = Vector((sh_R.x - 0.14, sh_R.y - 0.24, eyeR.z - 0.06))
    return dict(wrist=W, fingers_dir=f, back_dir=b, pole=(-0.6, 0.3, -1.0))

def breathe(fr):
    a = 0.9 * math.sin(2 * math.pi * fr / 95.0)
    rot_world("Spine1", (1, 0, 0), a)
    rot_world("Spine2", (1, 0, 0), -0.6 * a)

def body_motion(lean=0.0, turn=0.0, shrug=0.0):
    rot_world("Spine", (1, 0, 0), lean)
    rot_world("Spine1", (0, 0, 1), turn)
    if shrug: rot_world("RightShoulder", (0, 1, 0), shrug)

def head(pitch=0, yaw=0, roll=0):
    for n, k in (("Neck", 0.4), ("Head", 0.6)):
        rot_world(n, (1, 0, 0), pitch * k)
        rot_world(n, (0, 0, 1), yaw * k)
        rot_world(n, (0, 1, 0), roll * k)

def gaze(dx=0.0, dz=0.0):
    pass   # ojos pintados: la mirada acompaña a la cabeza

# ---------------------------------------------------------------- guion (frame, mano der., forma, cabeza, cuerpo, cara)
FACE0 = dict(mouthSmileLeft=0.22, mouthSmileRight=0.22)
KEYS = [
    (1,  "rest",   ("relajada", None, 0), (0, 0, 0),    (0, 0, 0),    FACE0),
    (10, "rest",   ("relajada", None, 0), (0, 0, 0),    (0, 0, 0),    FACE0),
    (16, "antic",  ("relajada", "B", 0.4), (1, 2, 0),   (0.5, 2, 0),  dict(FACE0, browInnerUp=0.15)),
    (26, "temple", ("B", None, 0),        (-3, -6, -4), (-1, 4, 4),  dict(mouthSmileLeft=0.45, mouthSmileRight=0.45, browInnerUp=0.35)),
    (31, "temple", ("B", None, 0),        (-3, -7, -5), (-1, 5, 4),  dict(mouthSmileLeft=0.5, mouthSmileRight=0.5, browInnerUp=0.45, browOuterUpLeft=0.2, browOuterUpRight=0.2)),
    (44, "out0",   ("B", None, 0),        (3, -3, 2),   (1, 2, 2),   dict(mouthSmileLeft=0.85, mouthSmileRight=0.85, browInnerUp=0.55, browOuterUpLeft=0.35, browOuterUpRight=0.35, eyeSquintLeft=0.2, eyeSquintRight=0.2)),
    (50, "outW+",  ("B", None, 0),        (4, -2, 3),   (1, 2, 2),   dict(mouthSmileLeft=0.9, mouthSmileRight=0.9, browInnerUp=0.5, jawOpen=0.12, eyeSquintLeft=0.25, eyeSquintRight=0.25)),
    (56, "outW-",  ("B", None, 0),        (3, -2, 1),   (1, 2, 2),   dict(mouthSmileLeft=0.9, mouthSmileRight=0.9, browInnerUp=0.45, jawOpen=0.08, eyeSquintLeft=0.25, eyeSquintRight=0.25)),
    (63, "out0",   ("B", None, 0),        (2, -1, 1),   (0.5, 1, 1), dict(mouthSmileLeft=0.8, mouthSmileRight=0.8, browInnerUp=0.35, eyeSquintLeft=0.15, eyeSquintRight=0.15)),
    (78, "rest",   ("B", "relajada", 0.6), (0, 0, 0),   (0, 0, 0),   dict(mouthSmileLeft=0.45, mouthSmileRight=0.45)),
    (86, "rest",   ("relajada", None, 0), (0.5, 0, 0), (0, 0, 0),   dict(mouthSmileLeft=0.35, mouthSmileRight=0.35)),
    (100, "rest",  ("relajada", None, 0), (0, 0, 0),    (0, 0, 0),   FACE0),
]
def right_pose(tag):
    if tag == "rest": return rest_arm("Right")
    if tag == "antic":
        r = rest_arm("Right"); r["wrist"] = r["wrist"] + Vector((-0.02, -0.06, 0.10)); return r
    if tag == "temple": return right_temple()
    if tag == "out0": return right_out(0)
    if tag == "outW+": return right_out(13)
    if tag == "outW-": return right_out(-9)

BLINKS = [(5, 0), (7, 1), (10, 0), (66, 0), (68, 1), (71, 0), (92, 0), (94, 1), (97, 0)]
FACE_KEYS = [k.name for k in body.data.shape_keys.key_blocks if k.name != "Basis"]

sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, 100
sc.render.fps = 30

def key_face(frame, fc):
    for ob in [body] + brows:
        for kn in FACE_KEYS:
            if kn.startswith("eyeBlink"): continue
            kb = ob.data.shape_keys.key_blocks.get(kn)
            if kb is None: continue
            kb.value = fc.get(kn, 0.0); kb.keyframe_insert("value", frame=frame)

PREVQ = {}
for frame, rtag, (shape, mix, mt), hd, bd, fc in KEYS:
    sc.frame_set(frame)
    reset()
    breathe(frame)
    body_motion(*bd)
    head(*hd)
    gaze(0, -1.0 if rtag == "temple" else 0)
    arm("Left", **rest_arm("Left", sway=0.004 * math.sin(frame / 9)))
    arm("Right", **right_pose(rtag))
    hand_shape("Right", shape, mix, mt); hand_shape("Left", "relajada")
    for pb in pbs:
        q = pb.rotation_quaternion.copy()
        pq = PREVQ.get(pb.name)
        if pq is not None and q.dot(pq) < 0:
            pb.rotation_quaternion = -q
        PREVQ[pb.name] = pb.rotation_quaternion.copy()
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    key_face(frame, fc)

for frame, v in BLINKS:
    for ob in [body] + brows:
        for kn in ("eyeBlinkLeft", "eyeBlinkRight"):
            kb = ob.data.shape_keys.key_blocks.get(kn)
            if kb: kb.value = v; kb.keyframe_insert("value", frame=frame)

def fcurves_of(idb):
    ad = idb.animation_data
    if not ad or not ad.action: return []
    act = ad.action
    out = list(getattr(act, "fcurves", []) or [])
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags: out += list(bag.fcurves)
    return out
for idb in [rig] + [o.data.shape_keys for o in [body] + brows]:
    for fcu in fcurves_of(idb):
        for kp in fcu.keyframe_points:
            kp.interpolation = "BEZIER"; kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "avatar_v4_hola.blend"))
print("OK animación")
