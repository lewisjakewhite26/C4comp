import numpy as np, cv2, sys, subprocess, colorsys, time
cv2.setNumThreads(2)
T=float(sys.argv[1]); DW=int(sys.argv[2]); OUTF=sys.argv[3]; ORI='h'; NFRAMES_ARG=int(sys.argv[4]) if len(sys.argv)>4 else 0
OUTW,OUTH=1920,1080
FPS=24; N=int(T*FPS); SUB=2; DT=1.0/(FPS*SUB)
DH=DW*OUTH//OUTW
SW,SH=320,180; KK=180/320.0
SC=DW/SW
gx,gy=np.meshgrid(np.arange(SW,dtype=np.float32),np.arange(SH,dtype=np.float32))
dgx,dgy=np.meshgrid(np.arange(DW,dtype=np.float32),np.arange(DH,dtype=np.float32))
K5=np.array([[0,1,0],[1,0,1],[0,1,0]],np.float32)
rng=np.random.default_rng(12)
PAL=[0.60,0.66,0.72,0.78,0.85,0.55,0.93,0.68]
drops=[]; EVENTS=[]
def gauss(w,h,cx,cy,r):
    xs=np.arange(w,dtype=np.float32)[None,:]; ys=np.arange(h,dtype=np.float32)[:,None]
    return np.exp(-((xs-cx)**2+(ys-cy)**2)/(r*r))
nz=np.random.default_rng(3)
def tilenoise(h,w,s,seed):
    r=np.random.default_rng(seed); a=r.standard_normal((h//s+2,w//s+2)).astype(np.float32)
    return cv2.resize(a,(w,h),interpolation=cv2.INTER_CUBIC)
NOISE=[tilenoise(DH,DW,max(3,DW//60),k) for k in range(4)]
def project(u,v,p):
    u[:,0]=0;u[:,-1]=0;v[0,:]=0;v[-1,:]=0
    d=np.zeros_like(u); d[:,1:-1]+=(u[:,2:]-u[:,:-2])*0.5; d[1:-1,:]+=(v[2:,:]-v[:-2,:])*0.5
    for _ in range(34):
        p=(cv2.filter2D(p,-1,K5,borderType=cv2.BORDER_REPLICATE)-d)*0.25
    u[:,1:-1]-=(p[:,2:]-p[:,:-2])*0.5; v[1:-1,:]-=(p[2:,:]-p[:-2,:])*0.5
    return p
def curl_of(u,v):
    c=np.zeros_like(u); c[1:-1,1:-1]=((v[1:-1,2:]-v[1:-1,:-2])-(u[2:,1:-1]-u[:-2,1:-1]))*0.5; return c
EPS=5.0*KK; VDAMP=0.35; DDECAY=0.01; BUOY=float(sys.argv[5]) if len(sys.argv)>5 else 90.0*KK
def step(S,splats):
    u,v,p,D=S['u'],S['v'],S['p'],S['D']
    mx=gx-DT*u; my=gy-DT*v
    u=cv2.remap(u,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE); v=cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    c=curl_of(u,v); ac=np.abs(c)
    ngx=np.zeros_like(c); ngy=np.zeros_like(c); ngx[:,1:-1]=(ac[:,2:]-ac[:,:-2])*0.5; ngy[1:-1,:]=(ac[2:,:]-ac[:-2,:])*0.5
    nm=np.sqrt(ngx**2+ngy**2)+1e-5
    u+=DT*EPS*(ngy/nm)*c; v+=-DT*EPS*(ngx/nm)*c
    # buoyancy: dense ink sinks
    dens=cv2.resize(D.max(2),(SW,SH),interpolation=cv2.INTER_AREA)
    v+=DT*BUOY*np.minimum(dens,1.5)
    for (sx,sy,vy,mag,r) in splats:
        g=gauss(SW,SH,sx*SW,sy*SH,r); v+=vy*mag*g
    p=project(u,v,p)
    damp=np.exp(-VDAMP*DT); u*=damp; v*=damp
    U=cv2.resize(u,(DW,DH),interpolation=cv2.INTER_LINEAR)*SC; V=cv2.resize(v,(DW,DH),interpolation=cv2.INTER_LINEAR)*SC
    D=cv2.remap(D,dgx-DT*U,dgy-DT*V,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    D=cv2.GaussianBlur(D,(0,0),0.55)
    np.maximum(D,0,out=D); D*=np.exp(-DDECAY*DT)
    m=D.max(2,keepdims=True); D*=1.0/(1.0+np.maximum(0,m-1.15))
    S.update(u=u,v=v,p=p,D=D)
def forcing(f,S):
    sp=[]
    for d in drops:
        k=f-d['f0']
        if k<0 or k>=d['L']: continue
        prog=k/d['L']; env=np.sin(np.pi*prog)
        r=10.5*KK*d['r']
        sp.append((d['x'],d['y'],1.0,70*KK*env,r))
        col=np.array(colorsys.hsv_to_rgb(d['hue'],0.85,1.0),np.float32)[::-1]
        g=gauss(DW,DH,d['x']*DW,d['y']*DH,r*SC*1.25)
        S['D']+=(g*(1+0.35*NOISE[d['k']%4])*0.34*d['amt']*env)[...,None]*col[None,None,:]
    return sp
def new_state(): return dict(u=np.zeros((SH,SW),np.float32),v=np.zeros((SH,SW),np.float32),p=np.zeros((SH,SW),np.float32),D=np.zeros((DH,DW,3),np.float32))
yy,xx=np.mgrid[0:DH,0:DW].astype(np.float32)
BG=np.zeros((DH,DW,3),np.float32)
gr=(yy/DH)
BG[...,0]=0.060+0.030*(1-gr); BG[...,1]=0.040+0.020*(1-gr); BG[...,2]=0.016+0.008*(1-gr)   # BGR deep blue
vg=1-0.35*(((xx/DW-0.5)*2)**2*0.7+((yy/DH-0.5)*2)**2*0.5)
def tone(D):
    img=1-np.exp(-1.6*D)
    gxm=cv2.Sobel(D.max(2),cv2.CV_32F,1,0,ksize=3); gym=cv2.Sobel(D.max(2),cv2.CV_32F,0,1,ksize=3)
    edge=cv2.GaussianBlur(np.sqrt(gxm**2+gym**2),(0,0),1.0)
    b=cv2.GaussianBlur(img,(0,0),DW/60.0)
    out=BG+img*1.0+0.30*b+edge[...,None]*0.25*np.array([1,0.95,0.9],np.float32)
    out=np.clip(out*vg[...,None],0,1)
    return out**0.9
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf',74) if __import__('os').path.exists('/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf') else ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',74)
def label(img,secs):
    m,s_=divmod(int(np.ceil(max(secs,0))),60); txt=f"{m}:{s_:02d}"
    im=Image.fromarray((img*255).astype(np.uint8)[...,::-1]); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
    d.text((OUTW//2+3,105+4),txt,font=FONT,fill=(0,0,0,110),anchor='mm'); d.text((OUTW//2,105),txt,font=FONT,fill=(255,255,255,205),anchor='mm')
    im=Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB'); return np.asarray(im)
import os,pickle,glob
SEG=480
os.makedirs('segw',exist_ok=True)
t0=time.time(); rngd=np.random.default_rng(5)
TOT=N+8*FPS
S=new_state(); last=-999; kcount=0; M=0.0; start=0
ck=sorted(glob.glob('segw/ck_*.pkl'))
if ck:
    st=pickle.load(open(ck[-1],'rb')); S=st['S']; drops[:]=st['drops']; EVENTS[:]=st['EVENTS']; last=st['last']; kcount=st['kcount']; M=st['M']; rngd=st['rngd']; start=st['f']; print('resume',start,flush=True)
import json
def open_seg(i):
    return subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OUTW}x{OUTH}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',f'segw/s_{i:03d}.mp4'],stdin=subprocess.PIPE)
p=None
for f in range(start,TOT):
    if f%SEG==0:
        if p: p.stdin.close(); p.wait()
        pickle.dump(dict(S=S,drops=drops,EVENTS=EVENTS,last=last,kcount=kcount,M=M,rngd=rngd,f=f),open(f'segw/ck_{f:06d}.pkl','wb'))
        p=open_seg(f//SEG)
    elif p is None: p=open_seg(f//SEG)
    tsec=f/FPS
    target=min(0.985,0.985*tsec/T)
    if f<N and f-last>=int(0.8*FPS) and (M<target or f<2*FPS) and f>=FPS//2:
        hue=PAL[kcount%len(PAL)]+rngd.uniform(-0.015,0.015)
        drops.append(dict(f0=f,x=rngd.uniform(0.12,0.88),y=rngd.uniform(0.09,0.2),hue=hue,r=rngd.uniform(0.85,1.25),L=int(0.6*FPS),amt=rngd.uniform(0.9,1.15),k=kcount))
        EVENTS.append(round(tsec,3)); last=f; kcount+=1
    sp=forcing(f,S)
    for k in range(SUB): step(S,sp if k==0 else [])
    if f%6==0:
        dm=cv2.resize(np.minimum(S['D'].max(2),1.0),(85,48),interpolation=cv2.INTER_AREA); M=float(dm.mean())
    img=tone(S['D'])
    if f>=N:
        q=(f-N)/FPS; img=np.clip(img*(1+0.18*np.exp(-q*1.2)*np.sin(np.pi*min(q,1.0))),0,1)
    img=cv2.resize(img,(OUTW,OUTH),interpolation=cv2.INTER_CUBIC)
    g=cv2.resize(np.random.default_rng(f%7).normal(0,0.006,(OUTH//2,OUTW//2)).astype(np.float32),(OUTW,OUTH))
    img=np.clip(img+g[...,None],0,1)
    fr=label(img[...,::-1].copy()[...,::-1],T-tsec)
    p.stdin.write(np.ascontiguousarray(fr).tobytes())
    if f%48==0: print(f,round(time.time()-t0),'M',round(M,3),'target',round(target,3),flush=True)
p.stdin.close(); p.wait()
json.dump(EVENTS,open('events_w.json','w')); print('done',time.time()-t0,len(EVENTS))
