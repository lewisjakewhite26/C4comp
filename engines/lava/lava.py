import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess
sys.path.insert(0,ZT+'jf')
import numpy as np, cv2
W,H,FPS=1920,1080,24
TAU=2*math.pi
# low-res simulation grid (field is smooth, so upsampling is invisible)
SW,SH=480,270
sx=np.linspace(0,W,SW,dtype=np.float32)[None,:]; sy=np.linspace(0,H,SH,dtype=np.float32)[:,None]
r=np.random.RandomState(12)
NB=13
B=[dict(x0=r.uniform(150,W-150),ax=r.uniform(40,140),fx=r.uniform(0.02,0.05),px=r.rand()*TAU,
        per=r.uniform(28,46),ph=r.rand(),rad=r.uniform(70,150),wob=r.rand()*TAU) for _ in range(NB)]
def blob_pos(b,t):
    # heat-rise / cool-sink cycle: slow at the ends, quicker in the middle
    u=((t/b['per'])+b['ph'])%1.0
    s=0.5-0.5*math.cos(TAU*u)                          # 0 bottom -> 1 top -> 0
    y=H+140-(H+320)*s
    x=b['x0']+b['ax']*math.sin(TAU*b['fx']*t+b['px'])
    v=math.sin(TAU*u)                                   # vertical speed sign/size
    return x,y,v
def field(t):
    F=np.zeros((SH,SW),np.float32)
    for b in B:
        x,y,v=blob_pos(b,t)
        rad=b['rad']*(1+0.08*math.sin(t*0.5+b['wob']))
        sx_=rad*(1-0.18*abs(v)); sy_=rad*(1+0.30*abs(v))   # stretch while moving
        d2=((sx-x)/sx_)**2+((sy-y)/sy_)**2
        F+=1.0/(d2+0.05)
    # wax pools at the bottom and gathers at the top
    F+=2.2*np.exp(-(H-sy)/70.0)*(1+0.25*np.sin(sx/160+t*0.3))
    F+=0.9*np.exp(-(sy)/55.0)*(1+0.2*np.sin(sx/210-t*0.25))
    return F
def shade(t,S=1):
    W_,H_=W//S,H//S
    F=cv2.resize(field(t),(W_,H_),interpolation=cv2.INTER_CUBIC)
    iso=1.15
    m=np.clip((F-iso*0.965)/(iso*0.07),0,1); m=m*m*(3-2*m)          # crisp but anti-aliased edge
    dome=1-np.exp(-np.clip(F-iso*0.965,0,None)/1.1)                   # rounded wax surface, no flat cap
    gy_,gx_=np.gradient(cv2.GaussianBlur(dome,(0,0),2.5/S))
    nx,ny=-gx_*55/S,-gy_*55/S; nz=np.ones_like(nx); nl=np.sqrt(nx*nx+ny*ny+nz*nz); nx/=nl; ny/=nl; nz/=nl
    L=np.array([-0.45,-0.55,0.70]); L/=np.linalg.norm(L)
    diff=np.clip(nx*L[0]+ny*L[1]+nz*L[2],0,1)
    Hh=np.array([L[0],L[1],L[2]+1]); Hh/=np.linalg.norm(Hh)
    spec=np.clip(nx*Hh[0]+ny*Hh[1]+nz*Hh[2],0,1)**40*m
    rim=np.clip(1-nz,0,1)**1.5
    yy=np.linspace(0,1,H_,dtype=np.float32)[:,None]
    core=np.clip((F-iso)/3.0,0,1)                                      # thick wax glows more
    # colours (BGR): hot pink at the bottom, violet higher up
    hot=np.array([180,60,255],np.float32); cool=np.array([255,70,170],np.float32)
    wax=(cool*(1-yy[...,None])+hot*yy[...,None])
    under=np.clip(-ny,0,1)                                              # lit from the lamp below
    wax=wax*(0.42+0.42*diff[...,None]+0.30*core[...,None]+0.25*under[...,None])+rim[...,None]*np.array([255,150,255],np.float32)*0.35
    wax+=spec[...,None]*np.array([255,240,255],np.float32)*0.9
    # liquid behind: deep indigo with a warm glow from the lamp base
    bg=np.array([40,6,18],np.float32)*(0.6+0.4*yy[...,None])+np.array([60,20,110],np.float32)*np.exp(-((1-yy)/0.25))[...,None]
    liq=cv2.resize(np.sin(sx/230+t*0.07)*np.sin(sy/170-t*0.05)+0*sx,(W_,H_)).astype(np.float32)
    bg=bg*(1+0.12*liq[...,None])
    glow=cv2.GaussianBlur(m,(0,0),30/S)[...,None]*np.array([160,40,170],np.float32)*0.30
    img=bg*(1-m[...,None])+wax*m[...,None]+glow*(1-m[...,None])
    # gentle bloom + vignette
    small=cv2.resize(img,(W//4,H//4)); img+=cv2.resize(cv2.GaussianBlur(small,(0,0),8),(W_,H_))*0.25
    X=np.linspace(-1,1,W_,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H_,dtype=np.float32)[:,None]
    img*=(1-0.35*(X**2*0.6+Y**2*0.9))[...,None]
    if S!=1: img=cv2.resize(img,(W,H),interpolation=cv2.INTER_LANCZOS4)
    return img

if __name__=='__main__':
    mode=sys.argv[1]
    timer = len(sys.argv)>2 and sys.argv[2]=='timer'
    if timer:
        import tim
    def fr(t):
        img=shade(t)
        if timer: img=tim.compose(img,150+(t-20.0),180)   # mid-countdown (0:30 left)
        return np.clip(img,0,255).astype(np.uint8)
    if mode=='still':
        cv2.imwrite(sys.argv[3] if len(sys.argv)>3 else 'still.png',fr(20.0))
    else:
        out=sys.argv[3]; dur=float(sys.argv[4])
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
        for f in range(int(dur*FPS)): p.stdin.write(fr(20.0+f/FPS).tobytes())
        p.stdin.close(); p.wait()
