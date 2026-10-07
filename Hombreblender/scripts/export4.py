import bpy, os, json, struct
import sys
SRC=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else '/home/claude/avatar_lsc/v4/avatar_v4_hola.blend'
bpy.ops.wm.open_mainfile(filepath=SRC)
for o in list(bpy.data.objects):
    if o.type in ('LIGHT','CAMERA'): bpy.data.objects.remove(o)
for o in bpy.data.objects:
    if o.type=='MESH': o.data.name=o.name
OUT='/home/claude/avatar_lsc/v4/avatar_lsc.glb'
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_animations=True, export_animation_mode='SCENE',
    export_morph=True, export_morph_normal=False, export_skins=True, export_def_bones=True, export_apply=False, export_yup=True)
d=bytearray(open(OUT,'rb').read())
n=struct.unpack('<I',d[12:16])[0]; j=json.loads(d[20:20+n]); rest=d[20+n:]
anims=j.get('animations',[])
if len(anims)>1:
    ch,sm=[],[]
    for a in anims:
        off=len(sm); sm+=a['samplers']
        for c in a['channels']: c=dict(c); c['sampler']+=off; ch.append(c)
    j['animations']=[{'name':'HOLA','channels':ch,'samplers':sm}]
elif anims: anims[0]['name']='HOLA'
js=json.dumps(j,separators=(',',':')).encode(); js+=b' '*((4-len(js)%4)%4)
out=bytearray(d[:12])+struct.pack('<I',len(js))+b'JSON'+js+rest; out[8:12]=struct.pack('<I',len(out))
open(OUT,'wb').write(out)
print('OK', os.path.getsize(OUT)//1024,'KB', [(a['name'],len(a['channels'])) for a in j['animations']], 'meshes',[m['name'] for m in j['meshes']], 'morph', [m.get('extras',{}).get('targetNames',[])[:3] for m in j['meshes']])
