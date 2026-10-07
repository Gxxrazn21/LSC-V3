# Importa, fusiona vértices duplicados, escala y coloca la base en z=0.90; guarda prep.blend
import bpy, bmesh
from mathutils import Matrix, Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='/home/claude/avatar_lsc/v4/modelo.glb')
o=[o for o in bpy.data.objects if o.type=='MESH'][0]
o.name='Avatar'; o.data.name='Avatar'
bpy.context.view_layer.objects.active=o
S=0.9
o.data.transform(Matrix.Translation((0,0,0.90)) @ Matrix.Scale(S,4))
o.matrix_world=Matrix.Identity(4)
bm=bmesh.new(); bm.from_mesh(o.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
bm.to_mesh(o.data); bm.free()
for p in o.data.polygons: p.use_smooth=True
for ob in list(bpy.data.objects):
    if ob is not o: bpy.data.objects.remove(ob)
bpy.ops.wm.save_as_mainfile(filepath='/home/claude/avatar_lsc/v4/prep.blend')
print('OK', len(o.data.vertices), [round(x,3) for x in o.dimensions])
