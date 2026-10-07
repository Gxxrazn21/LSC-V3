"""
Ajuste del esqueleto para que no se separe al posar a mano:
 1. Conecta las cadenas (columna, cuello-cabeza, hombro-brazo-antebrazo-mano, falanges).
 2. Bloquea traslación y escala en todos los huesos excepto la cadera (raíz).
 3. Agrega controles IK opcionales por brazo (mover la mano arrastra todo el brazo),
    apagados por defecto para no alterar la animación existente.
"""
import bpy, math
from mathutils import Vector

P = "mixamorig:"
rig = bpy.data.objects["AvatarLSC_Rig"]
arm = rig.data

if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
for pb in rig.pose.bones:
    pb.location = (0, 0, 0); pb.scale = (1, 1, 1)

bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones
connected = []
for b in eb:
    if b.parent and (b.head - b.parent.tail).length < 1e-4:
        b.use_connect = True; connected.append(b.name)

# ---- controles IK (no deforman)
ik_info = {}
for side, sx in (("Left", 1), ("Right", -1)):
    hand = eb[P + side + "Hand"]; fore = eb[P + side + "ForeArm"]; up = eb[P + side + "Arm"]
    t = eb.get("IK_Mano_" + side) or eb.new("IK_Mano_" + side)
    t.head = hand.head.copy(); t.tail = hand.head + (hand.tail - hand.head).normalized() * 0.08
    t.roll = hand.roll; t.parent = None; t.use_deform = False
    pole = eb.get("IK_Codo_" + side) or eb.new("IK_Codo_" + side)
    pole.head = Vector((sx * 0.40, 0.25, 0.95)); pole.tail = pole.head + Vector((0, 0.05, 0))
    pole.parent = None; pole.use_deform = False
bpy.ops.object.mode_set(mode="POSE")

for pb in rig.pose.bones:
    root = pb.name == P + "Hips" or pb.name.startswith("IK_")
    pb.lock_location = (not root,) * 3
    pb.lock_scale = (True, True, True)
    pb.rotation_mode = "QUATERNION"

# interruptores IK (0 = animación FK normal, 1 = posar arrastrando la mano)
for side in ("Left", "Right"):
    key = "IK_brazo_" + ("izq" if side == "Left" else "der")
    rig[key] = 0.0
    rig.id_properties_ui(key).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
                                     description="0 = animación normal, 1 = mover el brazo arrastrando la mano")
    fore = rig.pose.bones[P + side + "ForeArm"]
    for c in list(fore.constraints):
        if c.type == "IK": fore.constraints.remove(c)
    ik = fore.constraints.new("IK")
    ik.target = rig; ik.subtarget = "IK_Mano_" + side
    ik.pole_target = rig; ik.pole_subtarget = "IK_Codo_" + side
    ik.pole_angle = math.radians(145 if side == "Left" else 35); ik.chain_count = 2; ik.use_stretch = False
    d = ik.driver_add("influence").driver
    d.type = "AVERAGE"
    v = d.variables.new(); v.name = "ik"; v.type = "SINGLE_PROP"
    v.targets[0].id = rig; v.targets[0].data_path = f'["{key}"]'
    hand = rig.pose.bones[P + side + "Hand"]
    for c in list(hand.constraints):
        if c.type == "COPY_ROTATION": hand.constraints.remove(c)
    cr = hand.constraints.new("COPY_ROTATION"); cr.target = rig; cr.subtarget = "IK_Mano_" + side
    d2 = cr.driver_add("influence").driver; d2.type = "AVERAGE"
    v2 = d2.variables.new(); v2.name = "ik"; v2.type = "SINGLE_PROP"
    v2.targets[0].id = rig; v2.targets[0].data_path = f'["{key}"]'
    for n in ("IK_Mano_" + side, "IK_Codo_" + side):
        rig.pose.bones[n].custom_shape = None
        rig.data.bones[n].hide = False

bpy.ops.object.mode_set(mode="OBJECT")
result_info = {"conectados": len(connected)}
print("rig_fix", result_info)
