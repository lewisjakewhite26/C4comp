import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess, os
D=ZT+''
sys.path.insert(0,D+'jf'); sys.path.insert(0,D+'lava'); sys.path.insert(0,D+'jelly')
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import lava
# ---- portrait lava: re-seed the blobs for a tall, narrow lamp
lava.W,lava.H=1080,1920; lava.SW,lava.SH=270,480
lava.sx=np.linspace(0,1080,270,dtype=np.float32)[None,:]; lava.sy=np.linspace(0,1920,480,dtype=np.float32)[:,None]
r=np.random.RandomState(31)
XS=[150,930,420,700,260,820,560,330,880]
lava.B=[dict(x0=XS[i]+r.uniform(-40,40),ax=r.uniform(25,60),fx=r.uniform(0.02,0.05),px=r.rand()*lava.TAU,
        per=r.uniform(36,56),ph=(i/9.0+r.uniform(-0.04,0.04))%1,rad=r.uniform(95,145),wob=r.rand()*lava.TAU) for i in range(9)]
src=open(D+'lava/lavatimer.py').read()
src=src.replace("W,H,FPS=1920,1080,24; CX,CY=W//2,H//2; R=330; TAIL=6.0","W,H,FPS=1080,1920,24; CX,CY=W//2,H//2-40; R=370; TAIL=6.0")
exec(src.split("def frame(")[0])
from lavatimer import audio
FNT=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',230)
DS=860; DX0,DY0=CX-DS//2,CY-DS//2
_dc={}
def digits(sec):
    if sec in _dc: return _dc[sec]
    m_,s_=divmod(sec,60); im=Image.new('L',(DS,DS),0); d=ImageDraw.Draw(im)
    d.text((DS/2,DS/2),f'{m_}:{s_:02d}',font=FNT,fill=255,anchor='mm',features=['tnum'])
    a=np.asarray(im,np.float32)/255
    halo=cv2.GaussianBlur(a,(0,0),18); mul=(1-0.65*np.clip(halo*2.2,0,1))[...,None]
    glow=cv2.GaussianBlur(a,(0,0),12)[...,None]*np.array([200,60,230],np.float32)
    _dc.clear(); _dc[sec]=(a[...,None],mul,glow); return _dc[sec]
def frame(t,total):
    img=lava.shade(t,2)
    if t>=total:
        e=math.exp(-((t-total)-0.7)**2/0.9)
        small=cv2.resize(img,(W//4,H//4)); img=img*(1+0.5*e)+cv2.resize(cv2.GaussianBlur(small,(0,0),10),(W,H))*0.7*e
    img[Y0:Y0+S,X0:X0+S]=ring(img[Y0:Y0+S,X0:X0+S],t,total)
    left=max(0,math.ceil(total-t-1e-6)) if t<total else 0
    a,mul,glow=digits(left); end=math.exp(-((t-total)-0.6)**2/0.8) if t>=total else 0.0
    roi=img[DY0:DY0+DS,DX0:DX0+DS]; roi=roi*mul+glow*(0.45+0.6*end); roi=roi*(1-a)+a*TXT
    img[DY0:DY0+DS,DX0:DX0+DS]=roi
    return np.clip(img,0,255).astype(np.uint8)
if __name__=='__main__':
    mode=sys.argv[1]; mins=int(sys.argv[2]); total=mins*60.0
    if mode=='still':
        for tt in sys.argv[3].split(','): cv2.imwrite(D+f'lava/p_{tt}.png',frame(float(tt),total))
    else:
        v=D+f'lava/pv_{mins}.mp4'; a=D+f'lava/pa_{mins}.wav'; out=D+f'lava/lava_lamp_timer_{mins}min_tiktok.mp4'
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
            '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',v],stdin=subprocess.PIPE)
        for f in range(int((total+TAIL)*FPS)): p.stdin.write(frame(f/FPS,total).tobytes())
        p.stdin.close(); p.wait()
        audio(total,a)
        subprocess.run(['ffmpeg','-y','-v','error','-i',v,'-i',a,'-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out])
        os.remove(v); os.remove(a); print('done',os.path.getsize(out))
