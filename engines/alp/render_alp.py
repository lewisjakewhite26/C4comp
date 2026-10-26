import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2,numpy as np,math,sys,subprocess,time
cv2.setNumThreads(2)
T=900; FPS=24; NF=T*FPS; W,H=1920,1080; TAU=2*math.pi
def om(k): return TAU*k/T
D=ZT+'alp/'
S=float(sys.argv[1]); cmd=sys.argv[2]; args=sys.argv[3:]
OW,OH=int(W*S),int(H*S)
im=cv2.imread(D+'clean_fg.png').astype(np.float32)
sky_a=cv2.imread(D+'sky_alpha.png',0).astype(np.float32)/255
man=cv2.imread(D+'man_mask_gc.png',0).astype(np.float32)/255
wf=cv2.imread(D+'m_wf.png',0).astype(np.float32)/255
rv=cv2.imread(D+'m_rv.png',0).astype(np.float32)/255
rng=np.random.RandomState(5)
# ---- sky_full: fill non-sky with plausible sky so the sky layer can drift under a fixed skyline
nosky=(sky_a<0.5).astype(np.uint8)*255
nosky=cv2.dilate(nosky,np.ones((9,9),np.uint8))
sm=cv2.resize(im.astype(np.uint8),(W//3,H//3),interpolation=cv2.INTER_AREA); nm=cv2.resize(nosky,(W//3,H//3),interpolation=cv2.INTER_NEAREST)
sky_full=cv2.resize(cv2.inpaint(sm,nm,12,cv2.INPAINT_TELEA),(W,H),interpolation=cv2.INTER_CUBIC).astype(np.float32)
sky_full=np.where((sky_a>0.5)[...,None]&True,im,sky_full)  # keep true sky pixels sharp
fg=im
# ---- displacement-field helpers (low-res fields, upsampled)
gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
manb=cv2.GaussianBlur(man,(0,0),22)
rampg=np.clip((gy-470)/610,0,1)**1.15
grass_w=rampg*(1-manb)*(1-cv2.GaussianBlur(rv,(0,0),8))*(1-cv2.GaussianBlur(wf,(0,0),8))
hsv=cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
tree_m=((hsv[...,0]>22)&(hsv[...,0]<80)&(hsv[...,1]>70)&(hsv[...,2]<170)).astype(np.float32)
tree_m[500:,:]*=0; tree_m*=(1-sky_a); tree_m=cv2.GaussianBlur(tree_m,(0,0),3)
tree_w=tree_m*np.clip(1-(gy-60)/480,0.15,1)
def G(cx,cy,sx,sy): return np.exp(-(((gx-cx)/sx)**2+((gy-cy)/sy)**2)/2).astype(np.float32)
CHEST=G(760,790,230,110)*manb; BEARD=G(790,700,130,55)*manb
# ---- tiles (periodic)
def tile(h,w,sx,sy,seed):
    r=np.random.RandomState(seed); n=r.randn(h,w).astype(np.float32)
    n=cv2.GaussianBlur(np.tile(n,(3,3)),(0,0),sigmaX=sx,sigmaY=sy)[h:2*h,w:2*w]; return (n-n.mean())/n.std()
WT1=tile(256,128,1.1,10,1); WT2=tile(256,128,0.8,5,2)
RT=tile(256,256,7,1.6,3); RT2=tile(256,256,2.2,1.0,4)
MT=tile(256,512,26,10,6)
def sample_tile(t,ox,oy,h=H,w=W):
    th,tw=t.shape
    mx=((gx[:h,:w]+ox)%tw).astype(np.float32); my=((gy[:h,:w]+oy)%th).astype(np.float32)
    return cv2.remap(np.tile(t,(2,2)),mx,my,cv2.INTER_LINEAR)
# ---- smoke
CH=(823,428); RATE=3.0; LIFE=17.0
def puff_profile(r):
    n=int(r*3)|1; y,x=np.mgrid[-(n//2):n//2+1,-(n//2):n//2+1].astype(np.float32)
    return np.exp(-(x*x+y*y)/(2*(r*0.55)**2))
PROF={r:puff_profile(r) for r in range(4,40)}
def smoke(t):
    a=np.zeros((H,W),np.float32)
    k1=int(math.floor(t*RATE)); k0=int(math.floor((t-LIFE)*RATE))
    for k in range(k0,k1+1):
        ts=k/RATE; age=t-ts
        if age<0 or age>LIFE: continue
        u=age/LIFE; ph=(k%(int(T*RATE)))*1.7
        wind=1.0+0.55*math.sin(om(7)*ts+0.8)+0.25*math.sin(om(19)*ts)
        x=CH[0]+ (11*age+0.5*age*age*0.35)*wind + 3.0*math.sin(age*0.9+ph)*(0.3+u)
        y=CH[1]- (9*age - 0.18*age*age) + 2.0*math.sin(age*0.7+ph*1.3)*(0.3+u)
        r=int(round(5+30*u**0.8)); 
        al=0.68*min(1,age/1.2)*(1-u)**1.4
        if al<0.01: continue
        p=PROF[r]; n=p.shape[0]; h=n//2; xi=int(x); yi=int(y)
        x0,y0=xi-h,yi-h; x1,y1=x0+n,y0+n
        sx0,sy0=max(0,-x0),max(0,-y0); ex,ey=min(n,W-x0),min(n,H-y0)
        if ex<=sx0 or ey<=sy0: continue
        a[y0+sy0:y0+ey,x0+sx0:x0+ex]=np.maximum(a[y0+sy0:y0+ey,x0+sx0:x0+ex],p[sy0:ey,sx0:ex]*al)
    return a

# ---- birds & fish (one-off events, periodic because times are inside the 900 s loop)
BIRDS=[(70.0,1,1.0,150),(192.0,1,0.8,95),(838.0,2,1.0,170),(884.0,1,0.9,120)]   # start time, count, scale, base y
FISH_T=160.0; FISH_P=(1770.0,712.0)
def draw_bird(out,x,y,sc,ph,alpha):
    # simple flapping gull: body + two wings (2-segment each)
    fl=math.sin(ph); ang=math.radians(18+34*fl)
    L1=15*sc; L2=17*sc
    def wing(sgn):
        ex=x+sgn*L1*math.cos(ang); ey=y-L1*math.sin(ang)
        ex2=ex+sgn*L2*math.cos(ang*0.35-0.15); ey2=ey+L2*math.sin(0.15-ang*0.35)*(-1)
        return [(x,y),(ex,ey),(ex2,ey2)]
    r=int(70*sc)+20; x0=int(max(0,x-r)); x1=int(min(W,x+r)); y0=int(max(0,y-r)); y1=int(min(H,y+r))
    if x1<=x0 or y1<=y0: return
    roi=out[y0:y1,x0:x1]; ov=roi.copy()
    for sgn in (-1,1):
        pts=np.array([[p[0]-x0,p[1]-y0] for p in wing(sgn)],np.float32)
        cv2.polylines(ov,[np.round(pts*8).astype(np.int32)],False,(52,40,48),2,cv2.LINE_AA,shift=3)
    cv2.circle(ov,(int(round((x-x0)*8)),int(round((y-y0)*8))),int(2.2*sc*8),(52,40,48),-1,cv2.LINE_AA,shift=3)
    out[y0:y1,x0:x1]=roi*(1-alpha)+ov*alpha
def birds(out,t):
    for t0,n,sc,by in BIRDS:
        dur=15.0
        if not (t0<=t<=t0+dur): continue
        u=(t-t0)/dur
        for i in range(n):
            uu=u-0.045*i
            if not 0<=uu<=1: continue
            x=1990-2090*uu; y=by+16*math.sin(uu*TAU*1.5+i)+uu*30+i*14
            s_=sc*(0.9+0.2*uu)*(1 if i==0 else 0.85)
            al=min(1,uu*8,(1-uu)*8)*0.9
            draw_bird(out,x,y,s_,TAU*3.2*(t-t0)+i*1.7,al)
def fish(out,t):
    u=(t-FISH_T)/0.95; px,py=FISH_P
    y0,y1,x0,x1=int(py-80),int(py+50),int(px-90),int(px+170)
    roi=out[y0:y1,x0:x1]; ov=roi.copy(); fx=np.zeros(roi.shape[:2],np.float32)
    # ripples: rings from take-off (u=0) and landing (u=1), flattened ellipses
    for tc,cx in ((FISH_T,px-6),(FISH_T+0.95,px+46)):
        age=t-tc
        if 0<age<3.2:
            for j in range(3):
                a2=age-j*0.45
                if a2<=0: continue
                r=6+26*a2; al=0.85*(1-a2/3.2)**1.3
                m=np.zeros(roi.shape[:2],np.uint8)
                cv2.ellipse(m,(int(cx-x0),int(py+2-y0)),(int(r),int(r*0.30)),0,0,360,255,1,cv2.LINE_AA)
                fx=np.maximum(fx,m.astype(np.float32)/255*al)
    # fish body along a parabola
    if 0<=u<=1:
        fxp=px-6+52*u; fyp=py-30*4*u*(1-u)
        vx=52/0.95; vy=-30*4*(1-2*u)/0.95
        a=math.degrees(math.atan2(vy,vx))
        c=(int(fxp-x0),int(fyp-y0))
        body=np.zeros(roi.shape[:2],np.uint8)
        cv2.ellipse(body,c,(16,5),a,0,360,255,-1,cv2.LINE_AA)
        tail=np.array([[c[0]-11*math.cos(math.radians(a)),c[1]-11*math.sin(math.radians(a))],
                       [c[0]-19*math.cos(math.radians(a))+5*math.sin(math.radians(a)),c[1]-19*math.sin(math.radians(a))-5*math.cos(math.radians(a))],
                       [c[0]-19*math.cos(math.radians(a))-5*math.sin(math.radians(a)),c[1]-19*math.sin(math.radians(a))+5*math.cos(math.radians(a))]],np.int32)
        cv2.fillConvexPoly(body,tail,255,cv2.LINE_AA)
        bm=cv2.GaussianBlur(body.astype(np.float32)/255,(0,0),0.6)[...,None]
        col=np.array([92,108,128],np.float32)  # silvery
        sheen=np.array([205,215,225],np.float32)
        ov=ov*(1-bm)+col*bm
        hl=np.zeros(roi.shape[:2],np.uint8); cv2.ellipse(hl,(c[0]+1,c[1]+2),(10,2),a,0,360,255,-1,cv2.LINE_AA)
        hm=cv2.GaussianBlur(hl.astype(np.float32)/255,(0,0),0.7)[...,None]*0.55
        ov=ov*(1-hm)+sheen*hm
        # droplets trailing
        for k in range(10):
            ang=0.8+k*0.5; sp=40+(k*37)%40
            for tc,cx,sgn in ((0.0,px-6,-1),(0.95,px+46,1)):
                ag=t-(FISH_T+tc)
                if 0<ag<0.9:
                    dx_=cx+sgn*sp*0.5*ag*math.cos(ang)*0.6; dy_=py-sp*ag*abs(math.sin(ang))+110*ag*ag
                    cv2.circle(ov,(int(dx_-x0),int(dy_-y0)),1,(235,240,245),-1,cv2.LINE_AA)
    else:
        for tc,cx,sgn in ((0.0,px-6,-1),(0.95,px+46,1)):
            ag=t-(FISH_T+tc)
            if 0<ag<0.9:
                for k in range(10):
                    ang=0.8+k*0.5; sp=40+(k*37)%40
                    dx_=cx+sgn*sp*0.5*ag*math.cos(ang)*0.6; dy_=py-sp*ag*abs(math.sin(ang))+110*ag*ag
                    cv2.circle(ov,(int(dx_-x0),int(dy_-y0)),1,(235,240,245),-1,cv2.LINE_AA)
    ov=ov*(1-fx[...,None])+np.array([232,236,240],np.float32)*fx[...,None]
    out[y0:y1,x0:x1]=ov

exec(open(D+'bf.py').read())
exec(open(D+'cycle.py').read())

def frame(f,tl=None,tc=None):
    t=f/FPS if tl is None else tl
    TC=t if tc is None else tc
    # displacement for fg (grass, trees, breathing, beard)
    gs=lambda x,y,k,ph: np.sin(0.011*x+0.006*y-om(k)*t*1.0+ph)
    dx=grass_w*(2.6*gs(gx,gy,31,0.0)+1.2*gs(gx*1.7,gy,53,1.3))+tree_w*(1.4*np.sin(om(19)*t+gx*0.012)+0.7*np.sin(om(41)*t+gy*0.02))
    dy=grass_w*0.9*gs(gx,gy,37,2.0)-CHEST*1.7*math.sin(om(225)*t)-BEARD*0.5*math.sin(om(150)*t+0.5)
    dx=dx+BEARD*1.1*math.sin(om(150)*t)
    mx=(gx-dx).astype(np.float32); my=(gy-dy).astype(np.float32)
    out=cv2.remap(fg,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    # sky drift
    sdx=np.full((H,W),6.0*math.sin(om(3)*t)+2.5*math.sin(om(13)*t+1.0),np.float32)+ (gy*0.004)*math.sin(om(5)*t)*4
    sdy=np.full((H,W),1.5*math.sin(om(4)*t),np.float32)
    sk=cv2.remap(sky_full,(gx-sdx).astype(np.float32),(gy-sdy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    drift=(sdx,sdy)
    # waterfall streaks
    s1=sample_tile(WT1,0,-t*85.0/1.0,h=H,w=W) if False else None
    ox=gx[:H,:W]
    # (sample tiles only over their regions for speed)
    y0,y1,x0,x1=225,495,1360,1460
    sub=lambda tl,sx_,sy_: cv2.remap(np.tile(tl,(2,2)),(((gx[y0:y1,x0:x1]+sx_)%tl.shape[1])).astype(np.float32),(((gy[y0:y1,x0:x1]+sy_)%tl.shape[0])).astype(np.float32),cv2.INTER_LINEAR)
    a1=sub(WT1,0,-t*85.333); a2=sub(WT2,37,-t*128.0)
    wfa=cv2.GaussianBlur(wf,(0,0),1.0)[y0:y1,x0:x1,None]
    streak=(a1*0.65+a2*0.45)[...,None]
    out[y0:y1,x0:x1]=np.clip(out[y0:y1,x0:x1]+wfa*(streak*22),0,255)
    # mist at base
    my0,my1,mx0,mx1=430,540,1260,1470
    mt=cv2.remap(np.tile(MT,(2,2)),(((gx[my0:my1,mx0:mx1]+t*512*5/900.0)%512)).astype(np.float32),(((gy[my0:my1,mx0:mx1]+t*(256/60.0))%256)).astype(np.float32),cv2.INTER_LINEAR)
    mm=G(1385,478,95,32)[my0:my1,mx0:mx1]*np.clip(0.55+0.35*mt,0,1)*(0.8+0.2*math.sin(om(6)*t))
    out[my0:my1,mx0:mx1]=out[my0:my1,mx0:mx1]*(1-mm[...,None]*0.28)+np.array([232,236,238],np.float32)*mm[...,None]*0.28
    # river shimmer
    ry0,ry1,rx0,rx1=480,760,1100,1920
    spd=256*60/900.0
    r1=cv2.remap(np.tile(RT,(2,2)),(((gx[ry0:ry1,rx0:rx1]-t*spd*1.0)%256)).astype(np.float32),(((gy[ry0:ry1,rx0:rx1]-t*spd*0.25)%256)).astype(np.float32),cv2.INTER_LINEAR)
    r2=cv2.remap(np.tile(RT2,(2,2)),(((gx[ry0:ry1,rx0:rx1]-t*spd*2.0)%256)).astype(np.float32),(((gy[ry0:ry1,rx0:rx1]-t*spd*0.5)%256)).astype(np.float32),cv2.INTER_LINEAR)
    ra=cv2.GaussianBlur(rv,(0,0),1.0)[ry0:ry1,rx0:rx1,None]
    spark=np.clip((r2-1.4)*1.5,0,1)[...,None]
    out[ry0:ry1,rx0:rx1]=np.clip(out[ry0:ry1,rx0:rx1]+ra*(r1[...,None]*9+spark*70),0,255)
    if FISH_T-0.1<=t<=FISH_T+4.2: fish(out,t)
    # smoke (tinted by golden light)
    sa=smoke(t)
    sa=cv2.GaussianBlur(sa,(0,0),1.5)
    col=np.array([226,232,240],np.float32)   # BGR warm white
    out=out*(1-sa[...,None])+col*sa[...,None]
    out=grade(out,TC)
    sk2=make_sky(TC,drift)
    a3=sky_a[:SKH,:,None]
    out[:SKH]=out[:SKH]*(1-a3)+sk2*a3
    nm=float(cyc(MOON,TC))
    if nm>0.01: out+=snow[...,None]*nm*np.array([70,52,40],np.float32)*0.55
    out=emissive(out,TC,tl=t)
    birds(out,t)
    out*=1.0+0.012*math.sin(om(7)*t)
    return np.clip(out,0,255).astype(np.uint8)
def post(img):
    return cv2.resize(img,(OW,OH),interpolation=cv2.INTER_AREA) if S!=1 else img
if cmd=='still':
    t=float(args[0]); cv2.imwrite(args[1],post(frame(int(round(t*FPS)))))
elif cmd=='time':
    f=int(args[0]); t0=time.time()
    for i in range(6): frame(f+i)
    print((time.time()-t0)/6,'s/frame')
elif cmd=='loopcheck':
    a=frame(0).astype(np.float32); b=frame(NF).astype(np.float32); print('0 vs NF',np.abs(a-b).mean(),np.abs(a-b).max())
elif cmd=='render':
    a,b=int(args[0]),int(args[1]); out=args[2]; crf=args[3]
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-','-c:v','libx264','-crf',crf,'-preset','medium','-tune','animation','-g','48','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for f in range(a,b): p.stdin.write(post(frame(f)).tobytes())
    p.stdin.close(); p.wait()
elif cmd=='fast':
    st=int(args[0]); out=args[1]
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','28','-preset','veryfast','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for f in range(0,NF,st): p.stdin.write(post(frame(f)).tobytes())
    p.stdin.close(); p.wait()
elif cmd=='tiktok':
    # whole day in DUR seconds; sky/light cycle sped up, local motion at normal speed; portrait crop
    DUR=float(args[0]); out=args[1]; X0=int(args[2]) if len(args)>2 else 520
    n=int(DUR*FPS); K=T/DUR; XF=int(1.5*FPS)
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','608x1080','-r',str(FPS),'-i','-',
        '-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for f in range(n):
        tl=f/FPS; tc=(f/FPS)*K
        img=frame(f,tl=tl,tc=tc).astype(np.float32)
        if f>=n-XF:
            a=(f-(n-XF)+0.5)/XF; a=a*a*(3-2*a)
            img=img*(1-a)+frame(f,tl=tl-DUR,tc=tc).astype(np.float32)*a
        p.stdin.write(np.clip(img[:,X0:X0+608],0,255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
