import numpy as np, cv2, math, sys, subprocess
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1920,1080,24
TAU=2*math.pi
rng=np.random.RandomState(7)
FONT='/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf'

# ---------- background ----------
yy=np.linspace(0,1,H,dtype=np.float32)[:,None]; xx=np.linspace(0,1,W,dtype=np.float32)[None,:]
top=np.array([34,18,6],np.float32); bot=np.array([7,3,1],np.float32)   # BGR, very dark navy
BG=(top*(1-yy[...,None])**1.6+bot*(1-(1-yy[...,None])**1.6))
BG=BG*np.ones((1,W,1),np.float32)
vig=1-0.45*((xx-0.5)**2*1.6+(yy-0.45)**2*2.2)
BG*=vig[...,None]
BG+=rng.randn(H,W,1).astype(np.float32)*0.6
def rays(t):
    """faint god-rays from the surface, slowly shifting"""
    r=np.zeros((H//4,W//4),np.float32); h,w=r.shape
    Y=np.arange(h,dtype=np.float32)[:,None]; X=np.arange(w,dtype=np.float32)[None,:]
    for k,(x0,sl,wd,ph) in enumerate([(120,0.35,22,0),(230,0.22,30,1.7),(330,0.30,18,3.1),(410,0.18,26,4.4)]):
        xc=x0+Y*sl+18*math.sin(0.05*t+ph)
        b=np.exp(-((X-xc)/wd)**2)*(0.55+0.45*math.sin(0.08*t+ph*1.3))
        r+=b
    r*=np.clip(1-Y/(h*0.85),0,1)**1.8
    r=cv2.resize(cv2.GaussianBlur(r,(0,0),6),(W,H))
    return r[...,None]*np.array([26,14,5],np.float32)

# ---------- particles ----------
NP=420
P=dict(x=rng.rand(NP)*W,y=rng.rand(NP)*H,z=rng.rand(NP)**1.5,ph=rng.rand(NP)*TAU,k=rng.uniform(0.1,0.4,NP),spark=rng.rand(NP)<0.12)
def particles(lay,t):
    for i in range(NP):
        z=P['z'][i]
        x=(P['x'][i]+12*math.sin(P['k'][i]*t+P['ph'][i])+4*t*(0.3+z))%W
        y=(P['y'][i]+(6+10*z)*t)%H
        if P['spark'][i]:
            b=max(0,math.sin(P['k'][i]*2.3*t+P['ph'][i]))**6
            if b<0.02: continue
            cv2.circle(lay,(int(x*4),int(y*4)),int(4*(1+1.2*z)),(255*b,190*b,90*b),-1,cv2.LINE_AA,2)
        else:
            a=0.10+0.25*z
            cv2.circle(lay,(int(x*4),int(y*4)),int(4*(0.6+1.4*z)),(120*a,130*a,140*a),-1,cv2.LINE_AA,2)

# ---------- jellyfish ----------
class Jelly:
    def __init__(s,x,y,R,per,ph,hue,z,ntent):
        s.x0,s.y0,s.R,s.per,s.ph,s.hue,s.z,s.nt=x,y,R,per,ph,hue,z,ntent
        s.tph=rng.rand(ntent)*TAU; s.tlen=rng.uniform(1.4,3.8,ntent); s.toff=rng.uniform(-0.04,0.04,ntent); s.tamp=rng.uniform(0.5,1.6,ntent); s.sway=rng.uniform(0.6,1.4)
        s.vy=-(4+7*z)*rng.uniform(0.8,1.25)
        s.dx=rng.uniform(30,90)
    def pulse(s,t):
        u=((t/s.per)+s.ph)%1.0
        # quick contraction, slow relaxation
        c=math.sin(math.pi*u/0.35)**2 if u<0.35 else 0.0
        return c,u
    def pos(s,t,span):
        y=(s.y0+s.vy*t)%span - 450
        return s.x0+s.dx*math.sin(0.009*t+s.ph*9)+18*math.sin(0.07*t+s.ph*5), y
    def draw(s,lay,t,span,SOFT):
        c,u=s.pulse(t)
        cx,cy=s.pos(t,span); cy+=0   # wrap above bottom
        R=s.R; bw=R*(1-0.16*c); bh=R*0.92*(1+0.14*c)
        tilt=0.10*math.sin(0.11*t+s.ph*3)
        col=np.array(s.hue,np.float32)
        Ssh=4  # supersample shift
        def P(px,py):   # rotate about centre by tilt
            dx,dy=px-cx,py-cy
            return (cx+dx*math.cos(tilt)-dy*math.sin(tilt), cy+dx*math.sin(tilt)+dy*math.cos(tilt))
        # bell outline (dome + slightly flared skirt)
        ang=np.linspace(math.pi,0,60)
        pts=[P(cx+bw*math.cos(a), cy-bh*math.sin(a)**0.9) for a in ang]
        skirt=[P(cx+bw*(1-0.06*c)*math.cos(a), cy+R*0.12*math.sin(a)*(1-c)) for a in np.linspace(0,math.pi,30)]
        poly=np.array([(int(px*Ssh),int(py*Ssh)) for px,py in pts+skirt],np.int32)
        bright=0.85+0.35*c
        pend=None
        # --- translucent bell: fresnel rim + soft body, built in a local patch ---
        allp=np.array(pts+skirt,np.float32)
        x0=int(allp[:,0].min())-8; y0=int(allp[:,1].min())-8; x1=int(allp[:,0].max())+9; y1=int(allp[:,1].max())+9
        pw,ph_=x1-x0,y1-y0
        if pw>4 and ph_>4:
            m=np.zeros((ph_,pw),np.uint8)
            cv2.fillPoly(m,[np.round((allp-[x0,y0])*4).astype(np.int32)],255,cv2.LINE_AA,2)
            d=cv2.distanceTransform((m>127).astype(np.uint8),cv2.DIST_L2,5)
            mf=m.astype(np.float32)/255
            dmax=max(1.0,d.max())
            rim=np.exp(-d/(R*0.07))*mf
            Y=np.arange(ph_,dtype=np.float32)[:,None]
            topw=np.clip(1-(Y-(cy-bh-y0))/(bh*1.3),0,1)
            body=(0.10+0.16*topw+0.10*(d/dmax))*mf
            inten=(rim*0.95+body)*bright
            cyan=np.array([255,235,170],np.float32)
            colp=inten[...,None]*col*0.9+(rim**2)[...,None]*cyan*0.35*bright
            xa,ya=max(0,x0),max(0,y0); xb,yb=min(W,x1),min(H,y1)
            if xb>xa and yb>ya:
                pend=((slice(ya,yb),slice(xa,xb)),colp[ya-y0:yb-y0,xa-x0:xb-x0])
        # radial canals (faint)
        for a in np.linspace(math.pi*0.12,math.pi*0.88,8):
            p0=P(cx,cy-bh*0.2); p1=P(cx+bw*0.93*math.cos(a),cy-bh*0.93*math.sin(a)**0.9)
            cv2.line(lay,(int(p0[0]*Ssh),int(p0[1]*Ssh)),(int(p1[0]*Ssh),int(p1[1]*Ssh)),tuple(float(v*0.045*bright) for v in col),max(1,int(R*0.015*Ssh)),cv2.LINE_AA,2)
        # marginal lights along the skirt edge
        for k in range(16):
            fx=-0.97+1.94*k/15
            mx,my=P(cx+bw*0.98*fx, cy+R*0.10*(1-fx*fx)*(1-c))
            tw=0.6+0.4*math.sin(t*2.0+k*1.3+s.ph*7)
            cv2.circle(lay,(int(mx*Ssh),int(my*Ssh)),max(2,int(R*0.022*Ssh)),tuple(float(min(255,v*1.1)*tw*bright) for v in col),-1,cv2.LINE_AA,2)
        # gonads: 4 soft loops near top
        for k in range(4):
            a=math.pi*(0.25+0.5*k); gx,gy=P(cx+bw*0.24*math.cos(a), cy-bh*0.55+bh*0.10*math.sin(a))
            cv2.ellipse(SOFT,(int(gx*Ssh),int(gy*Ssh)),(int(R*0.11*Ssh),int(R*0.06*Ssh)),math.degrees(a)*0.3,0,360,tuple(float(v*0.30*bright) for v in col),-1,cv2.LINE_AA,2)
        # tentacles from skirt edge
        swim=0.6+0.4*(1-c)
        for i in range(s.nt):
            fx=-1+2*(i+0.5)/s.nt+s.toff[i]
            bx,by=P(cx+bw*0.96*fx, cy+R*0.08*(1-fx*fx))
            L=R*s.tlen[i]; n=36; pts=[]
            for j in range(n):
                q=j/(n-1)
                delay=q*1.6
                sw=R*0.30*s.sway*s.tamp[i]*(q**1.2)*math.sin(TAU*(t-delay)/ (s.per*1.6) + s.tph[i]) \
                   + R*0.10*(q**1.5)*math.sin(TAU*(t-delay*0.7)/(s.per*0.9)+s.tph[i]*2)
                spread=fx*R*0.25*q*(1-0.5*c)
                pts.append((bx+sw+spread, by+L*q*swim*(1+0.08*math.sin(TAU*t/s.per+q*3))))
            for j in range(n-1):
                q=j/(n-1); a=(1-q)**1.3*0.38
                p0,p1=pts[j],pts[j+1]
                cv2.line(lay,(int(p0[0]*Ssh),int(p0[1]*Ssh)),(int(p1[0]*Ssh),int(p1[1]*Ssh)),tuple(float(v*a) for v in col),max(1,int(R*0.011*Ssh*(1-0.6*q))),cv2.LINE_AA,2)
        # oral arms: 4 frilly ribbons
        for k in range(4):
            fx=-0.3+0.2*k
            bx,by=P(cx+bw*fx*0.6,cy)
            n=26; L=R*1.5
            left=[];right=[]
            for j in range(n):
                q=j/(n-1); delay=q*1.2
                sw=R*0.16*q*math.sin(TAU*(t-delay)/(s.per*1.3)+k*1.7)
                x=bx+sw+fx*R*0.3*q; y=by+L*q*swim
                wdt=R*0.09*(1-q*0.7)*(1+0.5*math.sin(q*18+t*1.3+k))
                left.append((x-wdt,y)); right.append((x+wdt,y))
            poly=np.array([(int(px*Ssh),int(py*Ssh)) for px,py in left+right[::-1]],np.int32)
            cv2.fillPoly(SOFT,[poly],tuple(float(v*0.30) for v in col),cv2.LINE_AA,2)
            cv2.polylines(lay,[np.array([(int(px*Ssh),int(py*Ssh)) for px,py in left],np.int32)],False,tuple(float(v*0.12) for v in col),max(1,int(R*0.015*Ssh)),cv2.LINE_AA,2)
            cv2.polylines(lay,[np.array([(int(px*Ssh),int(py*Ssh)) for px,py in right],np.int32)],False,tuple(float(v*0.12) for v in col),max(1,int(R*0.015*Ssh)),cv2.LINE_AA,2)
        if pend is not None: lay[pend[0]]+=pend[1]

BLUES=[(255,190,90),(255,160,60),(255,210,130),(255,140,70),(240,200,120)]   # BGR blues/cyans
YOFF=[410, 1753, 1424, 599, 126, 1807, 1572, 987, 820, 1219, 789, 77, 598, 488]
def mk_jellies():
    global rng; rng=np.random.RandomState(7)
    J=[]
    spec=[ # x, y, R, period, z
        (330,980,175,4.6,1.0),(1600,700,150,4.0,0.95),(1000,1500,95,3.6,0.6),(560,450,70,3.2,0.45),
        (1380,380,62,3.7,0.4),(1820,1350,48,3.0,0.3),(120,560,42,3.4,0.25),(1150,920,36,2.9,0.2),(780,1250,30,3.1,0.12),(1700,140,28,3.3,0.1),
        (1250,600,130,4.3,0.9),(700,900,85,3.5,0.55),(1550,1200,40,3.2,0.22),(420,1500,26,2.9,0.1)]
    if YOFF is not None: spec=[(x,yo,R,p,z) for (x,_,R,p,z),yo in zip(spec,YOFF)]
    for i,(x,y,R,per,z) in enumerate(spec):
        J.append(Jelly(x,y,R,per,rng.rand(),BLUES[i%len(BLUES)],z,int(10+6*z)))
    return J
J=mk_jellies()
SPAN=H+900

# ---------- timer ----------
FNT=ImageFont.truetype(FONT,250)
_cache={}
def timer_layer(sec_left):
    if sec_left in _cache: return _cache[sec_left]
    m,s=divmod(sec_left,60); txt=f'{m}:{s:02d}'
    im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
    d.text((W/2,H/2),txt,font=FNT,fill=255,anchor='mm',features=['tnum'])
    a=np.asarray(im,np.float32)/255
    _cache.clear(); _cache[sec_left]=a; return a
def compose_timer(out,t,total):
    left=max(0,math.ceil(total-t-1e-6)) if t<total else 0
    a=timer_layer(left)
    # dark halo for readability
    halo=cv2.GaussianBlur(a,(0,0),18)
    out*=(1-0.65*np.clip(halo*2.2,0,1))[...,None]
    # progress ring
    ring=np.zeros((H,W),np.float32)
    frac=max(0.0,1-t/total)
    if frac>0:
        cv2.ellipse(ring,(W//2*4,H//2*4),(330*4,330*4),-90,0,360*frac,1.0,int(4*4),cv2.LINE_AA,2)
    track=np.zeros((H,W),np.float32); cv2.circle(track,(W//2*4,H//2*4),330*4,0.10,int(4*4),cv2.LINE_AA,2)
    rr=np.maximum(ring,track)
    end=0.0
    if t>=total:
        end=math.exp(-((t-total)-0.6)**2/0.8)
    gl=np.array([255,235,200],np.float32)
    txtcol=np.array([255,244,225],np.float32)
    out+=cv2.GaussianBlur(a,(0,0),10)[...,None]*np.array([255,170,60],np.float32)*(0.35+0.6*end)
    out=out*(1-a[...,None])+a[...,None]*txtcol*(1+0.0*end)
    out+=(rr[...,None]*np.array([255,200,110],np.float32)*0.6)+cv2.GaussianBlur(ring,(0,0),8)[...,None]*np.array([255,160,60],np.float32)*0.35
    return out,end

def frame(t,total=None):
    out=BG.copy()+rays(t)
    lay=np.zeros((H,W,3),np.float32)
    particles(lay,t)
    far=np.zeros((H,W,3),np.float32)
    soft=np.zeros((H,W,3),np.float32)
    for j in J:
        j.draw(far if j.z<0.5 else lay,t,SPAN,soft)
    lay+=cv2.GaussianBlur(soft,(0,0),5)
    lay=cv2.GaussianBlur(lay,(0,0),0.9)
    far=cv2.GaussianBlur(far,(0,0),2.2)*0.75
    L=lay+far
    end=0.0
    if total is not None and t>=total: end=math.exp(-((t-total)-0.6)**2/0.8)
    L*=1+0.8*end
    small=cv2.resize(L,(W//4,H//4),interpolation=cv2.INTER_AREA)
    bloom=cv2.resize(cv2.GaussianBlur(small,(0,0),3),(W,H))*0.9+cv2.resize(cv2.GaussianBlur(small,(0,0),12),(W,H))*0.6
    out=out+L+bloom
    if total is not None:
        out,_=compose_timer(out,t,total)
    return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='still':
        t=float(sys.argv[2]); tot=float(sys.argv[3]) if len(sys.argv)>4 else None
        cv2.imwrite(sys.argv[-1],frame(t,tot))
    elif cmd=='render':
        total=float(sys.argv[2]); tail=float(sys.argv[3]); out=sys.argv[4]; crf=sys.argv[5] if len(sys.argv)>5 else '20'
        st=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
        n=int((total+tail)*FPS)
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf',crf,'-preset','medium','-tune','animation','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
        for f in range(int(st*FPS),n): p.stdin.write(frame(f/FPS,total).tobytes())
        p.stdin.close(); p.wait()
