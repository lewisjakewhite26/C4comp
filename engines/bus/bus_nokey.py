import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# One seamless loop cycle of the top-deck window: several street clips behind the keyed window, crossfaded,
# with rain on the glass. Cycle ends by fading into its own start, so it can be repeated for any length.
import sys, os, math, subprocess
import numpy as np, cv2
O=os.path.dirname(os.path.abspath(__file__))+'/'
W,H,FPS=1920,1080,30
XF=1.5                                                     # crossfade seconds
U='/mnt/user-data/uploads/'; A=ZT+'uploads/'
CLIPS=[(U+'3087318-uhd_3840_2160_30fps.mp4',0.4,14.0),
       (U+'4844567-uhd_3840_2160_30fps.mp4',0.4,23.3),
       (A+'c84ad424-2108672-hd_1280_720_30fps.mp4',0.4,21.6)]

# ---------------- no foreground: the whole frame is the real top-deck view through rainy glass
WY0,WY1,WX0,WX1=0,H,0,W
GH,GW=H,W
# ---------------- timing
LENS=[d-s for _,s,d in CLIPS]
CYCLE=sum(LENS)-XF*len(CLIPS)                              # every join (incl. last->first) overlaps by XF
NF=int(round(CYCLE*FPS)); CYCLE=NF/FPS

# ---------------- rain on glass (trickle speeds chosen so the cycle loops exactly)
rng=np.random.RandomState(3)
drops=[(rng.uniform(0,GW),rng.uniform(0,GH),r*rng.uniform(.8,1.2),rng.uniform(0.7,1.3),rng.uniform(0,6.3))
       for r in rng.choice([2.0,3.0,4.5,6.0,8.0,11.0,14.0],p=[.27,.24,.2,.14,.08,.05,.02],size=1700)]
def drop_layer(dl):
    off=np.zeros((GH,GW,2),np.float32); a=np.zeros((GH,GW),np.float32); hl=np.zeros((GH,GW),np.float32); rim=np.zeros((GH,GW),np.float32)
    for x,y,r,asp,ph in dl:
        x0,x1=int(max(0,x-r-2)),int(min(GW,x+r+3)); y0,y1=int(max(0,y-r*1.6-2)),int(min(GH,y+r*1.6+3))
        if x1<=x0 or y1<=y0: continue
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        dx=(xx-x)/r; dy=(yy-y)/(r*asp); ang=np.arctan2(dy,dx)
        d2=(dx*dx+dy*dy)*(1+0.18*np.sin(3*ang+ph)+0.10*np.sin(5*ang+2*ph))
        ins=np.clip((1-d2)*r,0,1)
        off[y0:y1,x0:x1,0]+=ins*(-dx*r*2.2); off[y0:y1,x0:x1,1]+=ins*(-dy*r*2.2-r*1.5)
        a[y0:y1,x0:x1]=np.maximum(a[y0:y1,x0:x1],ins)
        hl[y0:y1,x0:x1]+=ins*np.exp(-((dx+0.35)**2+(dy+0.45)**2)/0.03)
        rim[y0:y1,x0:x1]=np.maximum(rim[y0:y1,x0:x1],ins*np.clip((d2-0.55)/0.45,0,1))
    return off,a,hl,rim
STATIC=drop_layer(drops)
SPAN=GH*1.4
TR=[(rng.uniform(0,GW),rng.uniform(0,SPAN),rng.uniform(2.5,4.5),SPAN*rng.randint(1,4)/CYCLE) for _ in range(34)]
gx,gy=np.meshgrid(np.arange(GW,dtype=np.float32),np.arange(GH,dtype=np.float32))
FOG=(0.10+0.18*np.clip((gy/GH-0.55)/0.45,0,1))[...,None]
def glass(bg,t):
    extra=[]; streak=np.zeros((GH,GW),np.float32)
    for x0,y0,r,v in TR:
        y=(y0+v*t)%SPAN-GH*0.2; x=x0+3*math.sin(y*0.02+x0)
        extra.append((x,y,r,1.25,0.0))
        cv2.line(streak,(int(x),int(max(0,y-v*2.5))),(int(x),int(y)),0.35,max(1,int(r*0.6)),cv2.LINE_AA)
    off,a,hl,rim=STATIC
    o2,a2,h2,r2=drop_layer(extra); off=off+o2; a=np.maximum(a,a2); hl=hl+h2; rim=np.maximum(rim,r2)
    ref=cv2.remap(cv2.GaussianBlur(bg,(0,0),1.2),gx+off[...,0],gy+off[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    out=bg*(1-a[...,None])+(ref*1.05+6)*a[...,None]
    out*=(1-0.28*rim)[...,None]
    out+=np.clip(hl,0,1)[...,None]*np.array([235,235,240],np.float32)*0.22
    out=out*(1-streak[...,None]*0.5)+cv2.GaussianBlur(bg,(0,0),3)*streak[...,None]*0.5
    return out*(1-FOG)+cv2.GaussianBlur(bg,(0,0),10)*FOG+8

# ---------------- plates
def reader(path,ss,dur):
    vf=f'fps={FPS},crop=iw:iw*{GH}/{GW}:0:(ih-iw*{GH}/{GW})/2,scale={GW}:{GH}:flags=lanczos'
    p=subprocess.Popen(['ffmpeg','-v','error','-ss',str(ss),'-i',path,'-t',str(dur),'-vf',vf,'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    n=GW*GH*3
    while True:
        b=p.stdout.read(n)
        if len(b)<n: break
        yield np.frombuffer(b,np.uint8).reshape(GH,GW,3).astype(np.float32)
    p.kill()
def plates():
    from collections import deque
    xf=int(XF*FPS)
    def blend(a,b,k):
        u=(k+1)/(xf+1); u=u*u*(3-2*u); return a*(1-u)+b*u
    head=None; tail=None
    for ci,(p,s,d) in enumerate(CLIPS):
        it=reader(p,s,d-s)
        if ci==0: head=[next(it) for _ in range(xf)]              # reserved for the closing fade
        else:
            for k in range(xf): yield blend(tail[k],next(it),k)
        dq=deque()
        for f in it:
            dq.append(f)
            if len(dq)>xf: yield dq.popleft()
        tail=list(dq)
    for k in range(xf): yield blend(tail[k],head[k],k)

GRADE=np.array([1.05,0.98,0.93],np.float32)
NZ=np.random.RandomState(9)
def frame(bg,t):
    bg=cv2.GaussianBlur(bg,(0,0),1.6)*GRADE*0.86
    img=glass(bg,t)
    br=np.clip(img-170,0,None); sm=cv2.resize(br,(W//6,H//6),interpolation=cv2.INTER_AREA)
    img+=cv2.resize(cv2.GaussianBlur(sm,(0,0),3),(W,H))*0.5
    X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
    img*=(1-0.22*(X**2*0.6+Y**2*0.8))[...,None]
    img+=NZ.randn(H//2,W//2).astype(np.float32).repeat(2,0).repeat(2,1)[...,None]*2.0
    return np.clip(img,0,255).astype(np.uint8)

if __name__=='__main__':
    out=O+sys.argv[1]; limit=float(sys.argv[2]) if len(sys.argv)>2 else 1e9
    print('cycle seconds',round(CYCLE,2),flush=True)
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    n=0
    for bg in plates():
        if n>=NF or n>=limit*FPS: break
        f=frame(bg,n/FPS)
        if n in (int(FPS*3),int(FPS*20)): cv2.imwrite(O+f'L_{n}.jpg',cv2.resize(f,(960,540)))
        enc.stdin.write(f.tobytes()); n+=1
        if n%300==0: print('frame',n,'/',NF,flush=True)
    enc.stdin.close(); enc.wait(); print('frames',n,'DONE',flush=True)
