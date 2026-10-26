# Maths-art engine: glowing line patterns inside a circle (bouncing light, spirograph, times-table string art).
# Each generator returns a list of polylines in unit-circle coordinates (radius 1), plus per-point "time" (0..1 draw order).
import numpy as np, cv2, math

def bounce(n_dots=3, gravity=0.0, steps=200000, seed=1, speed=1.0):
    """Dots bouncing elastically inside the unit circle. gravity>0 bends the paths into arcs."""
    rng=np.random.RandomState(seed); lines=[]
    for k in range(n_dots):
        p=np.array([rng.uniform(-0.3,0.3),rng.uniform(-0.3,0.3)]); a=rng.uniform(0,2*math.pi)
        v=np.array([math.cos(a),math.sin(a)])*speed; dt=0.0006/speed
        pts=[p.copy()]; bounces=[]
        for i in range(steps):
            v[1]+=gravity*dt; p=p+v*dt
            r=math.hypot(*p)
            if r>1:
                n=p/r; p=n*(2-r)                       # reflect position back inside
                v=v-2*np.dot(v,n)*n; bounces.append(i/steps)
            pts.append(p.copy())
        lines.append((np.array(pts),np.linspace(0,1,len(pts)),bounces))
    return lines

def spiro(R=1.0, r=0.37, d=0.62, loops=None):
    """Hypotrochoid scaled to fit the unit circle; closes after a whole number of turns."""
    from fractions import Fraction
    fr=Fraction(r).limit_denominator(200); turns=fr.numerator if loops is None else loops
    t=np.linspace(0,2*math.pi*turns,int(4000*turns))
    x=(R-r)*np.cos(t)+d*np.cos((R-r)/r*t); y=(R-r)*np.sin(t)-d*np.sin((R-r)/r*t)
    s=1/np.max(np.hypot(x,y)); pts=np.stack([x*s,y*s],1)
    return [(pts,np.linspace(0,1,len(pts)),[])]

def times_table(N=240, k=2.0):
    """String art: point i on the circle joined to point k*i (mod N)."""
    segs=[]
    for i in range(N):
        a=2*math.pi*i/N; b=2*math.pi*(k*i)/N
        segs.append(np.array([[math.cos(a),math.sin(a)],[math.cos(b),math.sin(b)]]))
    return segs

TEAL=np.array([235,215,40],np.float32)    # BGR
PINK=np.array([210,80,255],np.float32)
def render(W,H,cx,cy,R,polys,upto=1.0,alpha=0.08,width=1,ss=2,rim=True,dotpos=None):
    """Accumulate line density, then map density to a teal->pink glow on navy."""
    A0=np.zeros((H*ss,W*ss),np.float32); A1=np.zeros_like(A0)
    for pts,tt,_ in polys:
        m=tt<=upto
        if m.sum()<2: continue
        q=pts[m]; tq=tt[m]; q=np.stack([(cx+q[:,0]*R)*ss,(cy+q[:,1]*R)*ss],1).astype(np.int32)
        nch=48; idx=np.linspace(0,len(q)-1,nch+1).astype(int)
        for j in range(nch):
            a,b=idx[j],idx[j+1]+1
            if b-a<2: continue
            lay=np.zeros_like(A0); cv2.polylines(lay,[q[a:b]],False,1.0,max(1,width*ss),cv2.LINE_AA)
            w=float(tq[a]); A0+=lay*alpha*(1-w); A1+=lay*alpha*w
    A0=cv2.resize(A0,(W,H),interpolation=cv2.INTER_AREA); A1=cv2.resize(A1,(W,H),interpolation=cv2.INTER_AREA)
    return colourise(W,H,cx,cy,R,A0+A1,rim,dotpos,mix=A1/(A0+A1+1e-6))

def render_segs(W,H,cx,cy,R,segs,alpha=0.10,ss=2,rim=True):
    A0=np.zeros((H*ss,W*ss),np.float32); A1=np.zeros_like(A0); n=len(segs)
    for i,s in enumerate(segs):
        a=((cx+s[0,0]*R)*ss,(cy+s[0,1]*R)*ss); b=((cx+s[1,0]*R)*ss,(cy+s[1,1]*R)*ss)
        lay=np.zeros_like(A0); cv2.line(lay,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,ss,cv2.LINE_AA)
        w=i/max(1,n-1); A0+=lay*alpha*(1-w); A1+=lay*alpha*w
    A0=cv2.resize(A0,(W,H),interpolation=cv2.INTER_AREA); A1=cv2.resize(A1,(W,H),interpolation=cv2.INTER_AREA)
    return colourise(W,H,cx,cy,R,A0+A1,rim,None,mix=A1/(A0+A1+1e-6))

def colourise(W,H,cx,cy,R,acc,rim,dotpos,mix=None):
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    bg=np.zeros((H,W,3),np.float32)+np.array([34,18,12],np.float32)        # deep navy
    vig=1-0.55*np.clip(((xx-W/2)**2+(yy-H/2)**2)/(0.6*(W**2+H**2)/4),0,1)
    bg*=vig[...,None]
    dens=1-np.exp(-acc*4.5)                                                 # 0..1 brightness
    hue=mix if mix is not None else np.clip(np.log1p(acc*6)/math.log(1+6*1.2),0,1)
    col=TEAL[None,None,:]*(1-hue[...,None])+PINK[None,None,:]*hue[...,None]
    glow=cv2.GaussianBlur(dens,(0,0),6)*0.8+cv2.GaussianBlur(dens,(0,0),18)*0.5
    img=bg+col*(dens[...,None]*1.1)+col*glow[...,None]*0.55
    img+=np.array([255,250,250],np.float32)*np.clip(acc-0.9,0,1)[...,None]*0.4   # hot white where very dense
    if rim:
        rr=np.hypot(xx-cx,yy-cy); ring=np.exp(-((rr-R-6)/2.0)**2)
        img+=ring[...,None]*np.array([170,150,120],np.float32)*0.35
    if dotpos is not None:
        for (x,y) in dotpos:
            d=np.exp(-(((xx-cx-x*R)**2+(yy-cy-y*R)**2)/(2*5.0**2)))
            h=np.exp(-(((xx-cx-x*R)**2+(yy-cy-y*R)**2)/(2*22.0**2)))
            img+=d[...,None]*255+h[...,None]*np.array([220,200,140],np.float32)*0.6
    return np.clip(img,0,255).astype(np.uint8)

if __name__=='__main__':
    import sys
    W,H=540,960; cx,cy,R=270,500,220
    tiles=[]
    tiles.append(render(W,H,cx,cy,R,bounce(1,0.0,90000,seed=3),alpha=0.16))
    tiles.append(render(W,H,cx,cy,R,bounce(2,2.4,140000,seed=5,speed=2.3),alpha=0.10))
    tiles.append(render(W,H,cx,cy,R,spiro(r=0.37,d=0.62),alpha=0.16))
    tiles.append(render(W,H,cx,cy,R,spiro(r=0.29,d=0.95),alpha=0.16))
    tiles.append(render_segs(W,H,cx,cy,R,times_table(240,2),alpha=0.22))
    tiles.append(render_segs(W,H,cx,cy,R,times_table(240,3.4),alpha=0.22))
    s=np.hstack([np.pad(t,((0,0),(0,8),(0,0)),constant_values=255) for t in tiles])
    cv2.imwrite(sys.argv[1] if len(sys.argv)>1 else 'look.png',s)
