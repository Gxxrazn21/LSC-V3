import sys, os, time; sys.path.insert(0,'/home/claude/avatar_lsc')
import bpy
from render_setup import setup, place
a=sys.argv[sys.argv.index('--')+1:]; mode=a[0]
O='/home/claude/avatar_lsc/v4/render/'; os.makedirs(O+'frames',exist_ok=True)
if mode=='video':
    bpy.ops.wm.open_mainfile(filepath='/home/claude/avatar_lsc/v4/avatar_v4_hola.blend')
    cam=setup(720,14); sc=bpy.context.scene; sc.view_settings.view_transform='Standard'; sc.view_settings.exposure=-0.85
    place(cam,(0,0,1.37),4,0,ortho=0.98)
    for f in range(int(a[1]),int(a[2])+1):
        sc.frame_set(f); sc.render.filepath=O+f'frames/hola_{f:03d}.png'
        t=time.time(); bpy.ops.render.render(write_still=True); print('frame',f,round(time.time()-t,1),flush=True)
else:
    bpy.ops.wm.open_mainfile(filepath='/home/claude/avatar_lsc/v4/avatar_v4.blend')
    cam=setup(800,24); sc=bpy.context.scene; sc.view_settings.view_transform='Standard'; sc.view_settings.exposure=-0.85
    sc.render.resolution_x=1000; sc.render.resolution_y=600
    for name,yaw in (('frente',0),('perfil',90),('espalda',180),('tresc',35)):
        place(cam,(0,0,1.36),4,yaw,ortho=1.8); sc.render.filepath=O+f'vista_{name}.png'; bpy.ops.render.render(write_still=True)
    sc.render.resolution_x=sc.render.resolution_y=600
    place(cam,(0,0,1.645),3,0,ortho=0.25)
    objs=[o for o in bpy.data.objects if o.type=='MESH' and o.data.shape_keys]
    ex={'neutral':{},'alegre':dict(mouthSmileLeft=1,mouthSmileRight=1,eyeSquintLeft=0.4,eyeSquintRight=0.4,browInnerUp=0.2),
        'enojado':dict(browDownLeft=1,browDownRight=1,mouthFrownLeft=0.8,mouthFrownRight=0.8,noseSneerLeft=0.5,noseSneerRight=0.5,mouthPress=0.6),
        'sorprendido':dict(browInnerUp=1,browOuterUpLeft=0.9,browOuterUpRight=0.9,eyeWideLeft=1,eyeWideRight=1,jawOpen=0.9),
        'pregunta':dict(browInnerUp=0.9,browOuterUpLeft=0.7,browOuterUpRight=0.7,eyeWideLeft=0.5,eyeWideRight=0.5),
        'parpadeo':dict(eyeBlinkLeft=1,eyeBlinkRight=1,mouthSmileLeft=0.3,mouthSmileRight=0.3)}
    for n,e in ex.items():
        for o in objs:
            for k in o.data.shape_keys.key_blocks:
                if k.name!='Basis': k.value=e.get(k.name,0.0)
        sc.render.filepath=O+f'expr_{n}.png'; bpy.ops.render.render(write_still=True)
