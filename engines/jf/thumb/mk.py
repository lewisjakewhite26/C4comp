import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
D=ZT+'jf/'
W,H=1280,720
FB='/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf'
FS='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
FM='/usr/share/fonts/opentype/inter/Inter-Medium.otf'
GRADE="colorchannelmixer=rr=0.80:rb=0.05:gg=0.97:bb=1.0,eq=gamma=0.96:saturation=1.04"
raw=subprocess.run(['ffmpeg','-v','error','-i',D+'loopM.mp4','-vf',f"select=eq(n\\,1344),{GRADE}",'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
bg=np.frombuffer(raw,np.uint8).reshape(720,1280,3).astype(np.float32)
# shift the jellyfish a little right, lift its glow, darken the left for text
M=np.float32([[1.08,0,330],[0,1.08,-20]]); bg=cv2.warpAffine(bg,M,(W,H),borderMode=cv2.BORDER_CONSTANT,borderValue=(1,3,10))
small=cv2.resize(bg,(W//4,H//4)); bloom=cv2.resize(cv2.GaussianBlur(small,(0,0),6),(W,H))
bg=bg*1.12+bloom*0.45
x=np.linspace(0,1,W)[None,:,None]
bg*=0.30+0.70*np.clip((x-0.15)/0.45,0,1)**1.2
bg=np.clip(bg,0,255).astype(np.uint8)

def make(mins,out):
    im=Image.fromarray(bg).convert('RGBA')
    num=str(mins)
    F1=ImageFont.truetype(FB,400); F2=ImageFont.truetype(FB,92); F3=ImageFont.truetype(FS,30)
    lay=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(lay)
    x0,y0=70,92
    bb=d.textbbox((0,0),num,font=F1,anchor='ls')
    nw=bb[2]-bb[0]
    base=y0+300
    d.text((x0-bb[0],base),num,font=F1,fill=(255,255,255,255),anchor='ls')
    tx=x0+nw+28
    word='MINUTE' if mins==1 else 'MINUTES'
    d.text((tx,base-118),word,font=F2,fill=(255,255,255,255),anchor='ls')
    d.text((tx,base),'TIMER',font=F2,fill=(150,225,255,255),anchor='ls')
    # tag line, letter-spaced
    tag='CALM JELLYFISH  ·  NO ALARM'
    cx=x0+6; cy=base+92
    for ch in tag:
        d.text((cx,cy),ch,font=F3,fill=(190,230,255,235),anchor='ls')
        cx+=d.textlength(ch,font=F3)+4
    # soft cyan glow + dark shadow behind the type
    a=lay.split()[3]
    glow=Image.new('RGBA',(W,H),(60,170,255,0)); glow.putalpha(a.filter(ImageFilter.GaussianBlur(22)).point(lambda v:int(v*0.55)))
    shadow=Image.new('RGBA',(W,H),(0,6,20,0)); shadow.putalpha(a.filter(ImageFilter.GaussianBlur(10)).point(lambda v:int(v*0.8)))
    im=Image.alpha_composite(im,shadow); im=Image.alpha_composite(im,glow); im=Image.alpha_composite(im,lay)
    im.convert('RGB').save(out,quality=92)
for m in (1,2,5,10):
    make(m,D+f'thumb/jellyfish_timer_{m}min_thumb.jpg')
