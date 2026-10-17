import sys, numpy as np, cv2, math, random, subprocess
import scene
from scipy.ndimage import gaussian_filter1d
W,H=1080,1920; F=1500.0; FPS=scene.FPS; S=scene.build(); POS=S['pos']; PL=S['plates']; N=scene.N; T0=scene.T0; BEAT=scene.BEAT; DUR=scene.DUR
NF=int(DUR*FPS)
def ease(x): x=min(max(x,0),1); return x*x*(3-2*x)
def uof(t): return min(max((t-T0)/BEAT,0.0),N-1.0)
# camera target per frame (smoothed, lookahead)
ts=np.arange(NF)/FPS
TG=np.array([POS(np.array(min(uof(t)+1.3,N-1+0.3))) for t in ts]); TG=gaussian_filter1d(TG,10,axis=0,mode='nearest')
def camera(f):
    t=f/FPS; tg=np.array([TG[f][0],TG[f][1],0.0])
    D=7.4+(1-ease(t/2.2))*6.0+ease((t-(T0+(N-1)*BEAT))/4.5)*9.0
    pitch=math.radians(52+5*math.sin(t*0.21)+ (1-ease(t/2.2))*14)
    yaw=math.radians(9*math.sin(t*0.27)+4*math.sin(t*0.61))
    off=np.array([math.sin(yaw)*math.cos(pitch)*-D*-1*-1*-1, -math.cos(yaw)*math.cos(pitch)*D, math.sin(pitch)*D])
    off[0]=math.sin(yaw)*math.cos(pitch)*D
    C=tg+off; fw=tg-C; fw/=np.linalg.norm(fw); r=np.cross(fw,[0,0,1.0]); r/=np.linalg.norm(r); u=np.cross(r,fw)
    return C,r,u,fw
class Cam:
    def __init__(s,f): s.C,s.r,s.u,s.f=camera(f)
    def proj(s,p):
        p=np.atleast_2d(p)-s.C; x=p@s.r; y=p@s.u; z=p@s.f
        return np.stack([W/2+F*x/np.maximum(z,1e-3),H/2-F*y/np.maximum(z,1e-3)],-1),z
def bgr(c,k=1.0): return (int(min(255,c[2]*k)),int(min(255,c[1]*k)),int(min(255,c[0]*k)))
# background
rs=np.random.RandomState(3)
BG=np.zeros((H,W,3),np.uint8); BG[:]=(18,10,8)
neb=np.zeros((H,W,3),np.float32)
for _ in range(9):
    cx,cy=rs.randint(0,W),rs.randint(0,H); col=rs.choice([0,1,2]); r=rs.randint(250,520)
    cv2.circle(neb,(cx,cy),r,[(60,25,30),(45,20,55),(50,30,20)][col],-1)
neb=cv2.GaussianBlur(neb,(0,0),140); BG=cv2.add(BG,neb.astype(np.uint8))
for _ in range(1300):
    x,y=rs.randint(0,W),rs.randint(0,H); b=int(rs.randint(90,255)); BG[y,x]=(b,b,min(255,b+15))
for _ in range(120):
    x,y=rs.randint(1,W-1),rs.randint(1,H-1); b=int(rs.randint(150,255)); cv2.circle(BG,(x,y),1,(b,b,b),-1,cv2.LINE_AA)
yy,xx=np.mgrid[0:H,0:W]; VIG=(1-0.38*(((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2*0.7)).clip(0.5,1)[...,None].astype(np.float32)
def box_faces(c,u2,v2,hu,hv,hz):
    u=np.array([u2[0],u2[1],0.0]);v=np.array([v2[0],v2[1],0.0]);z=np.array([0,0,1.0]); out=[]
    for ax,h in ((u,hu),(v,hv),(z,hz)):
        for sg in (1,-1):
            n=ax*sg; ctr=c+n*h
            o1,o2=[a for a in (u,v,z) if a is not ax]; h1,h2=[hh for a,hh in ((u,hu),(v,hv),(z,hz)) if a is not ax]
            q=[ctr+o1*h1*a+o2*h2*b for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))]
            out.append((n,ctr,np.array(q)))
    return out
LIGHT=np.array([-0.5,-0.45,0.75]); LIGHT/=np.linalg.norm(LIGHT)
def poly(img,cam,q,col):
    P,z=cam.proj(q)
    if (z<0.25).any(): return
    P=np.clip(P,-20000,20000).astype(np.int32); cv2.fillConvexPoly(img,P,col,cv2.LINE_AA)
cellset=set(S['cells'])
crnd=random.Random(9); CELLCOL={c:tuple(int(v+crnd.randint(-7,7)) for v in (146,128,156)) for c in S['cells']}
HITF={}  # cell idx -> plate
for p in PL: HITF[p['i']]=p
CIDX={c:i for i,c in enumerate(S['cells'])}
def glow_c(g,cam,p3,rad,col,k=1.0):
    P,z=cam.proj(p3)
    if z[0]<0.3: return
    x,y=P[0]; r=max(1,F*rad/z[0]/4)
    cv2.circle(g,(int(x/4),int(y/4)),int(r),bgr(col,k),-1,cv2.LINE_AA)
def sphere(img,cam,p3,rad,col,k=1.0):
    P,z=cam.proj(p3)
    if z[0]<0.3: return
    x,y=P[0]; r=F*rad/z[0]
    if r<1.2: return
    x,y,r=int(x),int(y),r
    cv2.circle(img,(x,y),int(r),bgr(col,0.62*k),-1,cv2.LINE_AA)
    cv2.circle(img,(int(x-r*0.12),int(y-r*0.16)),int(r*0.82),bgr(col,0.9*k),-1,cv2.LINE_AA)
    cv2.circle(img,(int(x-r*0.28),int(y-r*0.34)),max(1,int(r*0.42)),bgr(tuple(min(255,c+60) for c in col),1.0),-1,cv2.LINE_AA)
    cv2.circle(img,(int(x-r*0.34),int(y-r*0.42)),max(1,int(r*0.14)),(255,255,255),-1,cv2.LINE_AA)
def render(f):
    t=f/FPS; cam=Cam(f); img=BG.copy(); g=np.zeros((H//4,W//4,3),np.uint8); u=uof(t)
    # tint per cell from hit age
    def age(i):
        p=HITF.get(i); return (t-p['t']) if p else -1
    # floor cells far->near
    items=[]
    for ci,c in enumerate(S['cells']):
        ctr=np.array([c[0],c[1],-0.25]); d=np.linalg.norm(ctr-cam.C)
        if abs(c[1]-TG[f][1])>14 and d>26: continue
        items.append((d,0,c,ci))
    for (gx,gy,gz) in S['isl']: items.append((np.linalg.norm(np.array([gx,gy,gz])-cam.C),1,(gx,gy,gz),-1))
    items.sort(key=lambda a:-a[0])
    for d,kind,c,ci in items:
        if kind==0:
            ctr=np.array([c[0],c[1],-0.25]); hh=0.25; base=CELLCOL[c]; z0=0.0
        else:
            ctr=np.array([c[0],c[1],c[2]-0.25]); hh=0.25; base=(128,112,140); z0=c[2]
        for n,fc,q in box_faces(ctr,(1,0),(0,1),0.5,0.5,hh):
            if n[2]<0: continue
            if np.dot(n,cam.C-fc)<=0: continue
            if n[2]>0.5: continue
            if kind==0:
                nb=(c[0]+int(round(n[0])),c[1]+int(round(n[1])))
                if nb in cellset: continue
            sh=0.78+0.5*max(0,np.dot(n,LIGHT)); A=tuple(int(v*sh) for v in (52,42,70)); Bc=(14,11,22)
            top=q[2:4] if False else None
            q2=q.copy(); upper=q.copy(); lower=q.copy()
            # q vertices order: z-low?? build upper band explicitly
            zt=ctr[2]+hh; 
            
            band=2.5 if kind==0 else 0.8
            poly(img,cam,np.array([[p[0],p[1],(zt if abs(p[2]-(ctr[2]+hh))<1e-6 else zt-band)] for p in q]),bgr(A))
            dep=16 if kind==0 else 1.6
            lowq=np.array([[p[0],p[1],(zt-band if abs(p[2]-(ctr[2]+hh))<1e-6 else zt-dep)] for p in q])
            poly(img,cam,lowq,bgr(Bc))
        a=age(ci) if kind==0 else -1
        tcol=np.array(base,float)
        if a>=0:
            pc=np.array(HITF[ci]['col'],float); k=0.5*math.exp(-a/0.55); tcol=tcol*(1-k)+pc*k
        top=box_faces(ctr,(1,0),(0,1),0.5,0.5,hh)[4][2]  # +z face
        poly(img,cam,top,bgr(tuple(tcol)))
        # bricks
        z=ctr[2]+hh; lc=bgr(tuple(tcol*0.76))
        segs=[]
        for r in range(1,4): segs.append(((-0.5,-0.5+r*0.25),(0.5,-0.5+r*0.25)))
        for r in range(4):
            vs=(-0.17,0.17) if r%2==0 else (0.0,)
            for vv in vs: segs.append(((vv,-0.5+r*0.25),(vv,-0.5+(r+1)*0.25)))
        pts=np.array([[c[0]+a1[0],c[1]+a1[1],z] for a1,b1 in segs]+[[c[0]+b1[0],c[1]+b1[1],z] for a1,b1 in segs])
        P,zc=cam.proj(pts)
        if (zc>0.3).all():
            n_=len(segs)
            for k_ in range(n_):
                cv2.line(img,tuple(int(v) for v in P[k_]),tuple(int(v) for v in P[k_+n_]),lc,1,cv2.LINE_AA)
    # ripple rings + shadows (floor decals) into glow layer
    for p in PL:
        a=t-p['t']
        if 0<=a<1.2:
            cc=np.array([S['P'][p['i']][0],S['P'][p['i']][1],0.01]); rr=0.25+1.3*a; ang=np.linspace(0,2*np.pi,28)
            ring=np.stack([cc[0]+rr*np.cos(ang),cc[1]+rr*np.sin(ang),np.full(28,0.01)],-1); P,zc=cam.proj(ring)
            if (zc>0.3).all(): cv2.polylines(g,[(P/4).astype(np.int32)],True,bgr(p['col'],0.8*math.exp(-a/0.45)),2,cv2.LINE_AA)
    # objects
    obj=[]
    for p in PL:
        if abs(p['i']-u)>16: continue
        c3=np.array([p['c'][0],p['c'][1],0.4]); obj.append((np.linalg.norm(c3-cam.C),'plate',p))
    for i,(bp,ang) in enumerate(S['blocks']):
        if abs(i*5+2-u)>16: continue
        c3=np.array([bp[0],bp[1],0.2]); obj.append((np.linalg.norm(c3-cam.C),'block',(bp,ang)))
    for (ox,oy,oz,col,ph) in S['orbs']:
        if abs(oy-TG[f][1])>16: continue
        o3=np.array([ox,oy,oz+0.3*math.sin(t*0.8+ph)]); obj.append((np.linalg.norm(o3-cam.C),'orb',(o3,col)))
    # trail beads
    ds=0.34; kmax=int(u/ds*0+u/0.34)
    # bead absolute params along cell-index u: spacing 0.34 cell
    for k in range(max(0,int(u/ds)-16),int(u/ds)+1):
        uk=k*ds; beh=u-uk
        if beh<0.28 or beh>5.2: continue
        xy=POS(np.array(uk)); rad=0.19*(1-beh/5.6)**1.1; col=scene.PAL[k%6]
        b3=np.array([xy[0],xy[1],rad]); obj.append((np.linalg.norm(b3-cam.C),'bead',(b3,rad,col,1-beh/5.6)))
    bxy=POS(np.array(u)); bounce=0.07*abs(math.sin(math.pi*u)) if t>T0 else 0
    ball=np.array([bxy[0],bxy[1],0.33+bounce]); obj.append((np.linalg.norm(ball-cam.C),'ball',ball))
    obj.sort(key=lambda a:-a[0])
    shadows=[]
    for d,kd,dt in obj:
        if kd=='plate':
            p=dt; a=t-p['t']; fl=math.exp(-a/0.4) if a>=0 else 0
            col=np.array(p['col'],float)*(0.82+0.0)+255*0.0; col=tuple(col*(1-0.7*fl)+255*0.7*fl)
            c3=np.array([p['c'][0],p['c'][1],0.3]); uv=(math.cos(p['ang']),math.sin(p['ang'])); vv=(-uv[1],uv[0])
            for n,fc,q in box_faces(c3,uv,vv,0.2,0.04,0.3):
                if np.dot(n,cam.C-fc)<=0 or n[2]<-0.5: continue
                sh=0.62+0.55*max(0,np.dot(n,LIGHT)); poly(img,cam,q,bgr(col,min(1.25,sh)))
            if a>=0: glow_c(g,cam,c3,0.5+0.3*fl,p['col'],0.55*fl)
            else: glow_c(g,cam,c3,0.5,p['col'],0.12)
        elif kd=='block':
            bp,ang=dt; c3=np.array([bp[0],bp[1],0.13]); uv=(math.cos(ang),math.sin(ang)); vv=(-uv[1],uv[0])
            for n,fc,q in box_faces(c3,uv,vv,0.11,0.11,0.13):
                if np.dot(n,cam.C-fc)<=0 or n[2]<-0.5: continue
                sh=0.6+0.6*max(0,np.dot(n,LIGHT)); poly(img,cam,q,bgr((92,106,150),sh))
        elif kd=='orb':
            sphere(img,cam,dt[0],0.07,dt[1],1.0); glow_c(g,cam,dt[0],0.22,dt[1],0.55)
        elif kd=='bead':
            b3,rad,col,fa=dt; sphere(img,cam,b3,rad,col,0.7+0.3*fa); glow_c(g,cam,b3,rad*2.2,col,0.35*fa)
        elif kd=='ball':
            sphere(img,cam,dt,0.33,(150,205,255),1.0); glow_c(g,cam,dt,0.42,(130,200,255),0.22)
    # particles
    for p in PL:
        a=t-p['t']
        if not(0<=a<1.3): continue
        pr=random.Random(p['i']*7+1)
        for _ in range(9):
            vx,vy=pr.uniform(-0.8,0.8),pr.uniform(-0.8,0.8); vz=pr.uniform(1.0,2.6); col=scene.PAL[pr.randrange(6)]
            c0=np.array([p['c'][0],p['c'][1],0.4]); pos3=c0+np.array([vx*a,vy*a,vz*a-2.2*a*a])
            if pos3[2]<0.02: continue
            k=1-a/1.3; glow_c(g,cam,pos3,0.05+0.04*k,col,0.9*k)
            sphere(img,cam,pos3,0.035*k+0.01,col,1.0)
    # finale glow
    te=T0+(N-1)*BEAT
    if t>te:
        a=t-te; k=math.exp(-a/1.6)
        for ci,c in enumerate(S['cells'][::2]):
            glow_c(g,cam,np.array([c[0],c[1],0.05]),0.6,scene.PAL[ci%6],0.35*ease(a/1.5)*(0.7+0.3*math.sin(a*2+ci)))
    g=cv2.GaussianBlur(g,(0,0),7); gu=cv2.resize(g,(W,H),interpolation=cv2.INTER_LINEAR)
    img=cv2.add(img,(gu.astype(np.float32)*0.8).clip(0,255).astype(np.uint8))
    out=(img.astype(np.float32)*VIG)
    fade=min(1,t/0.5)*min(1,(DUR-t)/1.2)
    return (out*fade).clip(0,255).astype(np.uint8)
if __name__=='__main__':
    if sys.argv[1]=='test':
        fr=[render(int(t*FPS)) for t in (0.2,3,12,30,57.5,62)]
        cv2.imwrite('test.png',np.hstack([cv2.resize(x,(540,960)) for x in fr]))
    else:
        a,b,out=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
        pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
        for f in range(a,b): pr.stdin.write(render(f).tobytes())
        pr.stdin.close(); pr.wait()
