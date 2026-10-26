import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# Look-development stills for "every firefly that escapes brings back two friends". Positions are arranged by hand, not simulated.
import numpy as np, cv2, subprocess, math
from PIL import Image, ImageDraw, ImageFont
D=ZT+''
O=D+'jar/'
W,H=1080,1920
raw=subprocess.run(['ffmpeg','-v','error','-ss','3','-i',D+'escape/bgclip.mp4','-frames:v','1','-vf',f'scale={W}:{H}','-f','rawvideo','-pix_fmt','bgr24','-'],capture_output=True).stdout
BG=np.frombuffer(raw,np.uint8).reshape(H,W,3).astype(np.float32)
BG=cv2.GaussianBlur(BG,(0,0),9)*0.62
X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
BG*=(1-0.35*(X**2*0.6+Y**2*0.5))[...,None]

JX0,JX1,JY0,JY1=230,850,720,1480          # jar body
LIDY=JY0-10
def draw_scene(inside,outside,gapx,count,stage_note):
    img=BG.copy()
    # windowsill the jar stands on
    sill=np.zeros((H,W),np.float32); sill[JY1+8:JY1+60,:]=1
    img=img*(1-0.55*cv2.GaussianBlur(sill,(0,0),3)[...,None])+cv2.GaussianBlur(sill,(0,0),3)[...,None]*np.array([30,45,70],np.float32)*0.55
    # glass: faint body tint, edges, two long highlights
    body=np.zeros((H,W),np.float32)
    cv2.rectangle(body,(JX0,JY0),(JX1,JY1),1.0,-1); body=cv2.GaussianBlur(body,(0,0),6)
    img=img*(1-0.10*body[...,None])+body[...,None]*np.array([40,38,30],np.float32)*0.10
    edge=np.zeros((H,W),np.float32)
    cv2.rectangle(edge,(JX0,JY0),(JX1,JY1),1.0,5,cv2.LINE_AA)
    cv2.line(edge,(JX0+40,JY0+40),(JX0+40,JY1-60),0.55,10,cv2.LINE_AA)
    cv2.line(edge,(JX1-70,JY0+90),(JX1-70,JY0+330),0.35,6,cv2.LINE_AA)
    edge=cv2.GaussianBlur(edge,(0,0),2.0)
    img+=edge[...,None]*np.array([215,225,235],np.float32)*0.45
    # lid: a brass ring with an opening that slides along it
    lid=np.zeros((H,W),np.float32)
    cv2.rectangle(lid,(JX0-12,LIDY-22),(JX1+12,LIDY+8),1.0,-1)
    gw=120; cv2.rectangle(lid,(int(gapx-gw/2),LIDY-30),(int(gapx+gw/2),LIDY+12),0.0,-1)
    lid=cv2.GaussianBlur(lid,(0,0),1.2)
    img=img*(1-lid[...,None])+lid[...,None]*np.array([70,120,160],np.float32)
    # fireflies: warm cores with wide halos
    glow=np.zeros((H//2,W//2),np.float32); core=np.zeros((H,W),np.float32)
    for (x,y,b) in inside+outside:
        cv2.circle(glow,(int(x/2),int(y/2)),9,b*(0.6 if len(inside)>200 else 1),-1,cv2.LINE_AA)
        cv2.circle(core,(int(x),int(y)),9 if len(inside)<5 else 7,b,-1,cv2.LINE_AA)
    halo=cv2.resize(cv2.GaussianBlur(glow,(0,0),14),(W,H))*5.0
    halo=halo/(1+halo*0.9)
    soft=cv2.GaussianBlur(core,(0,0),2.2)
    soft=soft/(1+soft*0.35)
    FIRE=np.array([120,225,255],np.float32)
    img+=(halo*0.55)[...,None]*np.array([60,150,230],np.float32)+np.clip(soft,0,1.4)[...,None]*FIRE
    # text
    im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
    f1=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',48)
    f2=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',96)
    d.text((W/2,170),"every firefly that escapes",font=f1,fill=255,anchor='mm')
    d.text((W/2,232),"brings back two friends",font=f1,fill=255,anchor='mm')
    d.text((W/2,350),f"{count}",font=f2,fill=255,anchor='mm',features=['tnum'])
    t=np.asarray(im,np.float32)/255
    img=img*(1-t[...,None])+t[...,None]*np.array([235,240,248],np.float32)
    small=Image.fromarray(cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    dd=ImageDraw.Draw(small); f3=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',34)
    dd.text((W/2,H-90),stage_note,font=f3,fill=(255,210,120),anchor='mm')
    return small

r=np.random.RandomState(4)
def scatter(n,y0,y1,x0=JX0+20,x1=JX1-20):
    return [(r.uniform(x0,x1),r.uniform(y0,y1),r.uniform(0.6,1.0)) for _ in range(n)]
def packed(n,ytop,depth):
    pts=[]
    while len(pts)<n:
        x=r.uniform(JX0+14,JX1-14); y=ytop+abs(r.normal(0,depth))
        if y<JY1-14: pts.append((x,y,r.uniform(0.6,1.0)))
    return pts
def stream(n,gx):
    ys=np.sort(r.uniform(470,LIDY-30,n))
    return [(gx+r.normal(0,26+(LIDY-y)*0.22),y,r.uniform(0.4,0.9)*min(1,(y-430)/220)) for y in ys]

a=draw_scene([(430,1050,1.0),(690,1300,0.9)],[],330,2,"0:00  two fireflies drift about inside")
b=draw_scene(scatter(34,780,1440)+packed(10,JY0+16,40),stream(9,640),640,46,"0:25  escapes start to snowball")
c=draw_scene(packed(420,JY0+16,300),stream(70,520),520,1024,"0:55  the jar glows like a lantern")
canvas=Image.new('RGB',(W*3+40,H),(20,20,24))
for i,im in enumerate((a,b,c)): canvas.paste(im,(i*(W+20),0))
canvas.resize(((W*3+40)//2,H//2)).save(O+'jar_mock.jpg',quality=90)
