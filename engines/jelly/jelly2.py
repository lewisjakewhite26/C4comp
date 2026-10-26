import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess
sys.path.insert(0,ZT+'jelly')
import numpy as np, cv2
import jelly as base          # background, rays, particles, timer
W,H,FPS=base.W,base.H,base.FPS
TAU=2*math.pi
DT=1.0/FPS
SUB=3

def smooth_noise(seed,n=256,oct_=4):
    r=np.random.RandomState(seed); out=np.zeros((n,n),np.float32); amp=1.0
    for o in range(oct_):
        k=4*2**o
        g=r.rand(k,k).astype(np.float32)
        out+=cv2.resize(g,(n,n),interpolation=cv2.INTER_CUBIC)*amp; amp*=0.5
    out=(out-out.min())/(out.max()-out.min())
    return out

def ease(u): u=min(1.0,max(0.0,u)); return u*u*(3-2*u)

def chaikin(P):
    Q=0.75*P[:-1]+0.25*P[1:]; Rr=0.25*P[:-1]+0.75*P[1:]
    out=np.empty((2*len(Q),2),np.float32); out[0::2]=Q; out[1::2]=Rr
    return np.vstack([P[:1],out,P[-1:]])
class Chain:
    """verlet rope: nt strands x nn nodes"""
    def __init__(s,anchors,length,nn,stiff=0.0,sink=6.0,damp=0.94):
        s.nt=len(anchors); s.nn=nn
        s.seg=(np.asarray(length,np.float32)/(nn-1))[:,None]
        s.X=np.repeat(anchors[:,None,:],nn,1).astype(np.float32)
        for j in range(nn): s.X[:,j,1]+=s.seg[:,0]*j
        s.P=s.X.copy(); s.stiff=stiff; s.sink=sink; s.damp=damp
    def step(s,anchors,t,dt,flow):
        X,P=s.X,s.P
        V=(X-P)*s.damp
        P[:]=X
        acc=np.zeros_like(X); acc[...,1]+=s.sink
        acc+=flow(X,t)
        X+=V+acc*dt*dt
        X[:,0]=anchors
        for it in range(5):
            d=X[:,1:]-X[:,:-1]; L=np.linalg.norm(d,axis=2,keepdims=True)+1e-6
            corr=(L-s.seg[:,:,None])/L*d*0.5
            X[:,1:]-=corr; X[:,:-1]+=corr
            if s.stiff>0:
                d2=X[:,2:]-X[:,:-2]; L2=np.linalg.norm(d2,axis=2,keepdims=True)+1e-6
                c2=(L2-2*s.seg[:,:,None])/L2*d2*0.5*s.stiff
                X[:,2:]-=c2; X[:,:-2]+=c2
            X[:,0]=anchors
    def shift(s,dx,dy):
        s.X[...,0]+=dx; s.X[...,1]+=dy; s.P[...,0]+=dx; s.P[...,1]+=dy

def flow(X,t):
    """gentle turbulent water"""
    x=X[...,0]; y=X[...,1]
    fx=60*np.sin(0.0042*y+0.31*t)+35*np.sin(0.011*y-0.53*t+1.3)+25*np.sin(0.006*x+0.2*t)
    fy=30*np.sin(0.0050*x-0.27*t+0.7)+20*np.sin(0.013*x+0.44*t)
    return np.stack([fx,fy],-1).astype(np.float32)

SPECIES={
 #        h/R  ntent tlen(R)   tent_bright  narms armlen(R) armw  gonads canals
 'moon':  (0.50, 46, (0.35,0.9), 0.30,     4, (0.7,1.1), 0.10, True, 0.03),
 'crystal':(0.62, 60, (1.2,3.2), 0.20,     0, (0,0),     0.0,  False,0.07),
 'nettle':(0.88, 10, (2.6,4.6), 0.42,     4, (2.0,3.4), 0.08, False,0.025),
}
class Jelly:
    def __init__(s,sp,x,y,R,per,z,col,seed):
        r=np.random.RandomState(seed); s.r=r
        s.sp=sp; s.R=R; s.per=per; s.z=z; s.col=np.array(col,np.float32)
        hr,nt,tl,tb,na,al,aw,gon,can=SPECIES[sp]
        s.hr,s.tb,s.aw,s.gon,s.can=hr,tb,aw,gon,can
        s.prof={'moon':0.75,'crystal':0.9,'nettle':1.6}[sp]
        s.hx=x; s.p=np.array([x,y],np.float64); s.v=np.zeros(2); s.th=r.uniform(-0.3,0.3); s.ph=r.rand(); s.phase=r.rand()
        s.wob=r.rand(4)*TAU
        s.tf=np.sort(r.uniform(-1,1,nt)); s.tdepth=r.uniform(-1,1,nt)
        s.tlen=r.uniform(*tl,nt)*R
        s.tbr=r.uniform(0.5,1.0,nt)
        s.c=0.0; s.cprev=0.0
        s.tent=Chain(s.anchors(s.tf,s.tdepth),s.tlen,22,stiff=0.30,sink=140+60*z,damp=0.975)
        if na:
            s.af=np.array([-0.35,-0.12,0.12,0.35])*r.uniform(0.8,1.1)
            s.alen=r.uniform(*al,na)*R
            s.arms=Chain(s.anchors(s.af*0.6,np.zeros(na),inner=True),s.alen,26,stiff=0.35,sink=90,damp=0.975)
        else: s.arms=None
        s.noise=smooth_noise(seed+100)
        s.Lmax=max(s.tlen.max(), s.alen.max() if na else 0)+R
    # ---- geometry ----
    def axes(s):
        a=np.array([math.sin(s.th),-math.cos(s.th)])      # apex direction
        n=np.array([math.cos(s.th),math.sin(s.th)])        # across
        return a,n
    def to_world(s,u,w):
        a,n=s.axes()
        return s.p[0]+n[0]*u+a[0]*w, s.p[1]+n[1]*u+a[1]*w
    def margin_r(s):
        return s.R*(1-0.20*s.c)
    def anchors(s,f,dep,inner=False):
        Rm=s.margin_r()*(0.98 if not inner else 1.0)
        u=Rm*f; w=-s.R*0.06*dep*np.sqrt(np.clip(1-f*f,0,1))-s.R*0.02
        if inner: w=w+s.R*0.10
        X,Y=s.to_world(u,w)
        return np.stack([X,Y],-1).astype(np.float32)
    def outline(s,t,scale=1.0,hscale=1.0,wl=0.0):
        n=56; A=np.linspace(0,math.pi,n)
        R=s.R*scale; h=s.R*s.hr*(1+0.18*s.c)*hscale
        sa=np.sin(A)
        # contraction pulls the lower bell in more than the top
        rad=R*(1-0.20*s.c*(1-sa)**0.6 - 0.04*s.c)*(1+0.05*(1-s.c)*(1-sa)**4)
        # organic wobble, slightly asymmetric
        rad*=1+0.025*np.sin(2*A+0.7*t+s.wob[0])+0.015*np.sin(5*A-1.1*t+s.wob[1])+0.02*s.c*np.cos(A+s.wob[2])
        u=rad*np.cos(A)
        w=h*sa**s.prof*(1+0.03*np.sin(3*A+0.9*t+s.wob[3]))+wl
        return u,w
    # ---- simulation ----
    def step(s,t,dt):
        s.phase+=dt/(s.per*(1+0.08*math.sin(0.05*t+s.ph*9)))
        u=s.phase%1.0
        s.cprev=s.c
        s.c=ease(u/0.3) if u<0.3 else 1-ease((u-0.3)/0.7)**0.8
        dc=(s.c-s.cprev)/dt
        a,_=s.axes()
        thrust=max(0.0,dc)*s.R*0.55
        s.v+=a*thrust*dt
        s.v*=math.exp(-dt/1.1)
        s.p+=s.v*dt+np.array([5.0*math.sin(0.021*t+s.ph*4),-1.5])*(0.4+0.6*s.z)*dt
        # wandering heading, kept roughly upward, steer away from edges
        tgt=0.35*math.sin(0.031*t+s.ph*7)+0.2*math.sin(0.077*t+s.ph*3)
        hx=s.hx+0.16*W*math.sin(0.0045*t+s.ph*11)
        tgt+=max(-0.7,min(0.7,1.6*(hx-s.p[0])/W))
        mx=0.12*W
        if s.p[0]<mx: tgt+=0.6*(mx-s.p[0])/mx
        if s.p[0]>W-mx: tgt-=0.6*(s.p[0]-(W-mx))/mx
        s.th+=(tgt-s.th)*0.25*dt + 0.04*math.sin(TAU*u)*dt*(1 if s.ph>0.5 else -1)
        # wrap vertically once completely off screen
        if s.p[1] < -s.Lmax-40:
            dy=H+s.R+40-s.p[1]
            s.p[1]+=dy; s.tent.shift(0,dy)
            if s.arms: s.arms.shift(0,dy)
        for k in range(SUB):
            tt=t+k*dt/SUB
            s.tent.step(s.anchors(s.tf,s.tdepth),tt,dt/SUB,flow)
            if s.arms: s.arms.step(s.anchors(s.af*0.6,np.zeros(len(s.af)),inner=True),tt,dt/SUB,flow)
    # ---- drawing ----
    def draw(s,lay,soft,t):
        col=s.col; SS=4; R=s.R
        bright=0.8+0.3*s.c
        # tentacles (behind bell)
        X=s.tent.X
        for i in range(s.tent.nt):
            pts=chaikin(chaikin(X[i])); b=s.tb*s.tbr[i]; nn=len(pts)
            for j in range(nn-1):
                q=j/(nn-1); a=b*(1-q)**1.4
                if a<0.01: break
                cv2.line(lay,(int(pts[j,0]*SS),int(pts[j,1]*SS)),(int(pts[j+1,0]*SS),int(pts[j+1,1]*SS)),
                         tuple(float(v*a) for v in col),1 if R<120 or q>0.3 else 2,cv2.LINE_AA,2)
        # oral arms: frilly ribbons into soft layer
        if s.arms:
            for i in range(len(s.af)):
                pts=chaikin(s.arms.X[i]); n=len(pts)
                d=np.gradient(pts,axis=0); nrm=np.stack([-d[:,1],d[:,0]],1); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-6
                q=np.linspace(0,1,n)
                wdt=R*s.aw*(1-0.6*q)*(1+0.45*np.sin(q*22+t*1.1+i*1.9))
                Lp=pts+nrm*wdt[:,None]; Rp=pts-nrm*wdt[:,None]
                poly=np.concatenate([Lp,Rp[::-1]])
                cv2.fillPoly(soft,[np.round(poly*SS).astype(np.int32)],tuple(float(v*0.20) for v in col),cv2.LINE_AA,2)
                for side in (Lp,Rp):
                    cv2.polylines(lay,[np.round(side*SS).astype(np.int32)],False,tuple(float(v*0.10) for v in col),1,cv2.LINE_AA,2)
        # bell
        u,w=s.outline(t)
        BX,BY=s.to_world(u,w)
        poly=np.stack([BX,BY],1)
        # underside edge (slightly curved, closes the shape)
        uu=np.linspace(u[-1],u[0],20); ww=R*0.10*(1-(uu/(u[0]+1e-6))**2)*(0.6+0.4*s.c)
        UX,UY=s.to_world(uu,ww)
        full=np.concatenate([poly,np.stack([UX,UY],1)])
        # inner subumbrella
        ui,wi=s.outline(t,0.80,0.62,wl=-R*0.02)
        IX,IY=s.to_world(ui,wi); inner=np.stack([IX,IY],1)
        x0=int(full[:,0].min())-6; y0=int(full[:,1].min())-6; x1=int(full[:,0].max())+7; y1=int(full[:,1].max())+7
        pw,ph=x1-x0,y1-y0
        if pw<6 or ph<6: return
        m=np.zeros((ph,pw),np.uint8); mi=np.zeros((ph,pw),np.uint8)
        cv2.fillPoly(m,[np.round((full-[x0,y0])*SS).astype(np.int32)],255,cv2.LINE_AA,2)
        cv2.fillPoly(mi,[np.round((inner-[x0,y0])*SS).astype(np.int32)],255,cv2.LINE_AA,2)
        mf=m.astype(np.float32)/255; mif=mi.astype(np.float32)/255
        cur=np.full((ph,pw),255,np.uint8)
        cv2.polylines(cur,[np.round((poly-[x0,y0])*SS).astype(np.int32)],False,0,1,cv2.LINE_AA,2)
        d=cv2.distanceTransform((cur>128).astype(np.uint8),cv2.DIST_L2,5)
        di=cv2.distanceTransform((mi>127).astype(np.uint8),cv2.DIST_L2,5)
        rimw=R*(0.06+0.02*math.sin(t*0.6+s.ph*5))
        rim=np.exp(-d/rimw)*mf
        irim=cv2.GaussianBlur(np.exp(-di/(R*0.09))*mif,(0,0),max(1.0,R*0.03))
        tex=cv2.resize(s.noise[:max(8,int(128*ph/pw+8)),:128],(pw,ph))
        tex=0.65+0.7*tex
        body=(0.07+0.07*np.clip(1-d/(R*0.5),0,1))*mf*tex
        inten=(rim*0.75*(0.85+0.3*tex)+irim*0.16*tex+body)*bright
        colp=inten[...,None]*col
        colp+=(rim**3)[...,None]*np.array([255,240,215],np.float32)*0.22*bright   # pale specular edge
        # gonads (moon jelly) and radial canals
        sl=np.zeros((ph,pw,3),np.float32)
        if s.gon:
            for k in range(4):
                ang=math.pi*(0.25+0.5*k)
                gu=R*0.30*math.cos(ang)*(1-0.15*s.c); gw=R*s.hr*0.55+R*0.08*math.sin(ang)
                gx,gy=s.to_world(gu,gw)
                cv2.ellipse(sl,(int((gx-x0)*SS),int((gy-y0)*SS)),(int(R*0.11*SS),int(R*0.06*SS)),math.degrees(s.th)+math.degrees(ang)*0.2,0,300,
                            tuple(float(v*0.18) for v in col),max(1,int(R*0.018*SS)),cv2.LINE_AA,2)
        for k in range(16):
            A=math.pi*(k+0.5)/16
            p0=s.to_world(0,R*s.hr*0.85); p1=s.to_world(u[int(A/math.pi*55)]*0.97,w[int(A/math.pi*55)]*0.97)
            cv2.line(sl,(int((p0[0]-x0)*SS),int((p0[1]-y0)*SS)),(int((p1[0]-x0)*SS),int((p1[1]-y0)*SS)),tuple(float(v*s.can) for v in col),1,cv2.LINE_AA,2)
        colp+=cv2.GaussianBlur(sl,(0,0),1.2)*mf[...,None]
        # margin: faint fringe of light
        xa,ya=max(0,x0),max(0,y0); xb,yb=min(W,x1),min(H,y1)
        if xb>xa and yb>ya:
            lay[ya:yb,xa:xb]+=colp[ya-y0:yb-y0,xa-x0:xb-x0]

BL=[(255,175,80),(255,200,120),(250,150,60),(255,215,150),(240,165,95),(255,185,100)]
def make():
    spec=[ # species, x, y, R, period, depth
        ('nettle', 360, 520,150,4.8,1.00),('moon',1560,640,165,5.2,0.95),('crystal',1180,260,120,4.1,0.80),
        ('moon',   760,980, 90,4.4,0.60),('nettle',1700,180, 80,4.6,0.55),('crystal',220,900, 70,3.8,0.45),
        ('moon',  1350,1050,56,3.9,0.38),('crystal',560,200, 46,3.6,0.30),('moon',1840,880,40,3.5,0.25),
        ('nettle', 980,700, 36,3.9,0.20),('moon',  120,260, 30,3.3,0.14),('crystal',1450,420,26,3.2,0.10),
        ('moon', 560,1250,115,4.7,0.85),('crystal',1000,1500,100,4.3,0.75)]
    return [Jelly(sp,x,y,R,per,z,BL[i%len(BL)],11+i*7) for i,(sp,x,y,R,per,z) in enumerate(spec)]

class Scene:
    def __init__(s,warm=12.0):
        s.J=make(); s.t=-warm
        while s.t<0: s.step()
    def step(s):
        for j in s.J: j.step(s.t,DT)
        s.t+=DT
    def render(s,total=None):
        t=s.t
        out=base.BG.copy()+base.rays(t)
        near=np.zeros((H,W,3),np.float32); far=np.zeros((H,W,3),np.float32)
        snear=np.zeros((H,W,3),np.float32); sfar=np.zeros((H,W,3),np.float32)
        base.particles(near,t)
        for j in sorted(s.J,key=lambda j:j.z):
            if j.z<0.5: j.draw(far,sfar,t)
            else: j.draw(near,snear,t)
        far=cv2.GaussianBlur(far+cv2.GaussianBlur(sfar,(0,0),4),(0,0),2.0)*0.6
        near=cv2.GaussianBlur(near,(0,0),0.7)+cv2.GaussianBlur(snear,(0,0),4)
        L=near+far
        end=0.0
        if total is not None and t>=total: end=math.exp(-((t-total)-0.6)**2/0.8)
        L*=1+0.7*end
        small=cv2.resize(L,(W//4,H//4),interpolation=cv2.INTER_AREA)
        bloom=cv2.resize(cv2.GaussianBlur(small,(0,0),3),(W,H))*0.7+cv2.resize(cv2.GaussianBlur(small,(0,0),12),(W,H))*0.5
        out=out+L+bloom
        if total is not None: out,_=base.compose_timer(out,t,total)
        return np.clip(out,0,255).astype(np.uint8)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='stills':     # stills t1,t2,... total outprefix
        ts=[float(v) for v in sys.argv[2].split(',')]; tot=float(sys.argv[3]) if sys.argv[3]!='none' else None; pre=sys.argv[4]
        S=Scene()
        for t in sorted(ts):
            while S.t<t-1e-6: S.step()
            cv2.imwrite(f'{pre}_{t:g}.png',S.render(tot))
    elif cmd=='render':  # render total tail out crf [start]
        total=float(sys.argv[2]); tail=float(sys.argv[3]); out=sys.argv[4]; crf=sys.argv[5]
        st=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
        S=Scene()
        while S.t<st-1e-6: S.step()
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf',crf,'-preset','medium','-tune','animation','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
        n=int(round((total+tail-st)*FPS))
        for f in range(n):
            p.stdin.write(S.render(total).tobytes()); S.step()
        p.stdin.close(); p.wait()
