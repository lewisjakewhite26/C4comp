import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys
D=ZT+''
sys.path.insert(0,D+'lava')
import numpy as np, cv2, lava
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H=1280,720
FB='/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf'; FS='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
bg=lava.shade(75.0,1)
bg=cv2.resize(np.clip(bg,0,255).astype(np.uint8),(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
x=np.linspace(0,1,W)[None,:,None]; bg*=0.40+0.60*np.clip((x-0.15)/0.45,0,1)
bg=cv2.cvtColor(np.clip(bg*1.08,0,255).astype(np.uint8),cv2.COLOR_BGR2RGB)
def make(mins,out):
    im=Image.fromarray(bg).convert('RGBA')
    lay=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(lay)
    F1=ImageFont.truetype(FB,400); F2=ImageFont.truetype(FB,92); F3=ImageFont.truetype(FS,30)
    num=str(mins); x0=70; base=392
    bb=d.textbbox((0,0),num,font=F1,anchor='ls'); nw=bb[2]-bb[0]
    d.text((x0-bb[0],base),num,font=F1,fill=(255,255,255,255),anchor='ls')
    tx=x0+nw+28
    d.text((tx,base-118),'MINUTE' if mins==1 else 'MINUTES',font=F2,fill=(255,255,255,255),anchor='ls')
    d.text((tx,base),'TIMER',font=F2,fill=(255,170,235,255),anchor='ls')
    cx=x0+6; cy=base+92
    for ch in 'CALM LAVA LAMP  ·  NO ALARM':
        d.text((cx,cy),ch,font=F3,fill=(255,215,245,235),anchor='ls'); cx+=d.textlength(ch,font=F3)+4
    a=lay.split()[3]
    glow=Image.new('RGBA',(W,H),(230,80,220,0)); glow.putalpha(a.filter(ImageFilter.GaussianBlur(22)).point(lambda v:int(v*0.5)))
    sh=Image.new('RGBA',(W,H),(25,0,30,0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(10)).point(lambda v:int(v*0.85)))
    im=Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(im,sh),glow),lay)
    im.convert('RGB').save(out,quality=92)
for m in (2,3,5,10): make(m,D+f'lava/thumb/lava_lamp_timer_{m}min_thumb.jpg')
