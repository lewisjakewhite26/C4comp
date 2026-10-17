import numpy as np, cv2, sys, subprocess, colorsys, math
from PIL import Image,ImageDraw,ImageFont,ImageFilter
cv2.setNumThreads(1)
a,b,OUT=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
POS=np.load('pos.npy',mmap_mode='r')
W=16/9.0; N=320; NC=4; FPS=30; TS=3.0
RW,RH=960,540; PPU=540.0
col=np.repeat(np.arange(NC),N//NC)
HUES=[0.90,0.76,0.56,0.30]
CB=np.array([colorsys.hsv_to_rgb(h,0.85,1.0) for h in HUES],np.float32)[:,::-1]
SIG=0.022*PPU
yy,xx=np.mgrid[0:RH,0:RW].astype(np.float32)
vg=1-0.45*(((xx/RW-0.5)*2)**2*0.7+((yy/RH-0.5)*2)**2*0.55)
BG=np.stack([0.060+0.03*(1-yy/RH),0.020+0.008*(1-yy/RH),0.020+0.01*(1-yy/RH)],2).astype(np.float32)*vg[...,None]
def render(pos):
    D=np.zeros((NC,RH,RW),np.float32)
    px=np.clip((pos[:,0]*PPU).astype(int),0,RW-1); py=np.clip((pos[:,1]*PPU).astype(int),0,RH-1)
    for c in range(NC):
        m=col==c; np.add.at(D[c],(py[m],px[m]),1.0)
        D[c]=cv2.GaussianBlur(D[c],(0,0),SIG)
    D*=0.55*2*np.pi*SIG*SIG
    T=D.sum(0); th=0.34; M=np.clip((T-th)/0.10,0,1); M=M*M*(3-2*M)
    D2=D**3.0
    colr=np.einsum('chw,cj->hwj',D2,CB)/np.maximum(D2.sum(0),1e-6)[...,None]
    colr=colr/np.maximum(colr.max(2,keepdims=True),1e-3)
    h=cv2.GaussianBlur(T,(0,0),2.5)
    gx=cv2.Sobel(h,cv2.CV_32F,1,0,ksize=3)*0.5; gy=cv2.Sobel(h,cv2.CV_32F,0,1,ksize=3)*0.5
    nl=np.sqrt(gx*gx+gy*gy+1.0); nx,ny,nzz=-gx/nl,-gy/nl,1.0/nl
    L=np.array([-0.45,-0.55,0.70],np.float32); L/=np.linalg.norm(L)
    diff=np.clip(nx*L[0]+ny*L[1]+nzz*L[2],0,1)
    Hh=L+np.array([0,0,1],np.float32); Hh/=np.linalg.norm(Hh)
    spec=np.clip(nx*Hh[0]+ny*Hh[1]+nzz*Hh[2],0,1)**40
    inner=np.clip((T-th)/2.6,0,1)
    body=colr*(0.50+0.55*diff[...,None])*(0.62+0.5*inner[...,None])
    rim=np.exp(-((T-th-0.06)/0.10)**2)[...,None]*colr*0.85
    glow=cv2.GaussianBlur((M[...,None]*colr),(0,0),RW*0.03)
    out=BG+glow*0.50
    out=out*(1-M[...,None])+(body+rim*0.5+spec[...,None]*0.75)*M[...,None]
    out=out+rim*0.30
    return np.clip(out,0,1)
FD='/usr/share/fonts/truetype/google-fonts/'
FT=ImageFont.truetype(FD+'Poppins-Bold.ttf',150); FS=ImageFont.truetype(FD+'Poppins-Medium.ttf',54); FN=ImageFont.truetype(FD+'Poppins-Bold.ttf',120)
def overlay(img,t):
    im=Image.fromarray((img*255).astype(np.uint8)[...,::-1]).convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
    if t<TS+0.4:
        a=min(1,t/0.4)*(1 if t<TS-0.4 else max(0,1-(t-(TS-0.4))/0.8))
        if a>0:
            bd=Image.new('RGBA',im.size,(0,0,0,0)); ImageDraw.Draw(bd).rounded_rectangle((300,360,1620,680),60,fill=(0,0,8,int(190*a))); im=Image.alpha_composite(im,bd.filter(ImageFilter.GaussianBlur(45)))
            for s,f,y,c in (("5 Minute Timer",FT,470,(255,255,255)),("calm neon liquid",FS,590,(255,190,245))):
                d.text((963,y+5),s,font=f,fill=(0,0,0,int(170*a)),anchor='mm'); d.text((960,y),s,font=f,fill=c+(int(255*a),),anchor='mm')
    rem=max(0.0,300-(t-TS)) if t>=TS else 300.0
    sec=int(math.ceil(rem-1e-9)); txt=f"{sec//60}:{sec%60:02d}"
    ta=min(1,max(0,(t-TS+0.6)/0.6)) if t<TS else 1.0
    if t>TS+300+4.0: ta=max(0,1-(t-TS-304)/2.0)*0.0+ta  # keep 0:00 shown
    al=int(235*ta)
    if al>0:
        d.text((963,988),txt,font=FN,fill=(0,0,0,int(150*ta)),anchor='mm'); d.text((960,985),txt,font=FN,fill=(255,255,255,al),anchor='mm')
    prog=min(1,max(0,(t-TS)/300.0))
    d.rectangle((160,1058,1760,1062),fill=(255,255,255,int(35*ta)))
    d.rectangle((160,1058,160+int(1600*prog),1062),fill=(255,150,235,int(150*ta)))
    return np.asarray(Image.alpha_composite(im,ov).convert('RGB'))[...,::-1]
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',OUT],stdin=subprocess.PIPE)
for f in range(a,b):
    img=cv2.resize(render(np.array(POS[f])),(1920,1080),interpolation=cv2.INTER_CUBIC)
    img=overlay(img,f/FPS)
    p.stdin.write(np.ascontiguousarray(img).tobytes())
p.stdin.close(); p.wait(); print('seg done',a,b)
