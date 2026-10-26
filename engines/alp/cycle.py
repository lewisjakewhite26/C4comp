# ---- day/night cycle: grade, procedural sky, stars, aurora, moon, windows, fireflies ----
def _ss(u): u=min(1,max(0,u)); return u*u*(3-2*u)
def cyc(keys,t):
    t=t%T
    for (t0,v0),(t1,v1) in zip(keys[:-1],keys[1:]):
        if t0<=t<=t1:
            u=_ss((t-t0)/(t1-t0)); return np.asarray(v0,np.float32)*(1-u)+np.asarray(v1,np.float32)*u
    return np.asarray(keys[0][1],np.float32)
A=lambda *a:np.array(a,np.float32)
GAIN=[(0,A(1,1,1)),(200,A(1,1,1)),(300,A(.96,.76,.72)),(390,A(.66,.58,.78)),(470,A(.34,.42,.74)),(540,A(.27,.38,.66)),(700,A(.27,.38,.66)),(770,A(.32,.40,.68)),(830,A(.86,.66,.70)),(875,A(1,.9,.82)),(900,A(1,1,1))]
LIFT=[(0,A(0,0,0)),(300,A(0,0,0)),(470,A(4,10,22)),(540,A(5,14,30)),(700,A(5,14,30)),(780,A(6,12,26)),(870,A(0,0,0)),(900,A(0,0,0))]
SATK=[(0,1.0),(300,1.05),(390,0.9),(470,0.8),(700,0.8),(830,0.95),(900,1.0)]
WPAL=[(0,0.0),(200,0.0),(300,1.0),(840,1.0),(900,0.0)]
ZEN=[(300,A(96,110,170)),(390,A(40,48,100)),(470,A(14,22,58)),(540,A(6,10,28)),(700,A(6,10,28)),(770,A(10,16,40)),(830,A(60,80,140)),(880,A(120,150,200)),(900,A(120,150,200))]
MID=[(300,A(225,130,120)),(390,A(120,80,130)),(470,A(30,44,96)),(540,A(12,20,50)),(700,A(12,20,50)),(770,A(30,44,90)),(830,A(180,120,140)),(880,A(240,170,150)),(900,A(240,170,150))]
HOR=[(300,A(255,176,110)),(390,A(225,120,110)),(470,A(70,70,120)),(540,A(24,36,74)),(700,A(24,36,74)),(770,A(64,70,118)),(830,A(255,160,120)),(880,A(255,200,140)),(900,A(255,200,140))]
STAR=[(0,0),(400,0),(520,1),(720,1),(800,0),(900,0)]
MOON=[(0,0),(450,0),(500,1),(740,1),(800,0),(900,0)]
AUR=[(0,0),(500,0),(570,1),(690,1),(750,0),(900,0)]
WIN=[(0,0),(320,0),(400,1),(800,1),(860,0),(900,0)]
FFL=[(0,0),(380,0),(470,1),(700,1),(790,0),(900,0)]
# static assets
lum0=cv2.cvtColor(sky_full.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
CL0=np.clip((lum0-cv2.GaussianBlur(lum0,(0,0),28))/38,-1,1)
SKH=420
yy_=np.arange(SKH,dtype=np.float32)[:,None]; xx_=np.arange(W,dtype=np.float32)[None,:]
vgrad=np.clip(yy_/330.0,0,1)
def _grad(zen,mid,hor):
    v=vgrad[...,None]
    a=_ss_arr(np.clip(v/0.55,0,1)); b=_ss_arr(np.clip((v-0.55)/0.45,0,1))
    return zen+(mid-zen)*a+(hor-mid)*b
def _ss_arr(u): return u*u*(3-2*u)
srng=np.random.RandomState(21)
NST=420
st_x=srng.randint(0,W,NST); st_y=(srng.rand(NST)**0.8*330).astype(int)+2; st_b=srng.rand(NST)**2.2*0.9+0.1; st_g=srng.randint(0,3,NST); st_ph=srng.rand(NST)*TAU
STARL=[np.zeros((SKH,W),np.float32) for _ in range(3)]
for i in range(NST):
    sz=st_b[i]; xi,yi=st_x[i],min(SKH-3,st_y[i])
    STARL[st_g[i]][yi,xi]=max(STARL[st_g[i]][yi,xi],sz)
    if sz>0.65:
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if 0<=xi+dx<W and 0<=yi+dy<SKH: STARL[st_g[i]][yi+dy,xi+dx]=max(STARL[st_g[i]][yi+dy,xi+dx],sz*0.35)
STARL=[cv2.GaussianBlur(l,(0,0),0.7)*2.2 for l in STARL]
# windows
WINS=[(722,506,15,17),(820,528,17,19),(902,528,17,19),(862,540,20,32)]
# fireflies
frng=np.random.RandomState(33); FF=[]
while len(FF)<46:
    bx,by=frng.uniform(120,1850),frng.uniform(620,1040)
    if man[int(by),int(bx)]>0.1 or (rv[int(min(by,H-1)),int(bx)]>0.1): continue
    FF.append((bx,by,frng.uniform(25,70),frng.uniform(12,32),frng.randint(5,16),frng.randint(6,17),frng.randint(90,230),frng.uniform(0,TAU),frng.uniform(0,TAU),frng.uniform(0,TAU)))
snow=((hsv[...,2]>205)&(hsv[...,1]<70)).astype(np.float32); snow[420:]=0; snow*=(1-sky_a); snow=cv2.GaussianBlur(snow,(0,0),1.5)
def grade(out,t):
    g=cyc(GAIN,t); l=cyc(LIFT,t); sat=float(cyc(SATK,t))
    gb=g[::-1]; lb=l[::-1]
    out=out*gb+lb
    if sat!=1.0:
        lu=out.mean(2,keepdims=True); out=lu+(out-lu)*sat
    return out
def make_sky(t,drift):
    """returns BGR float sky (rows 0:SKH) with clouds, stars, moon, aurora"""
    sx,sy=drift
    base=sky_full[:SKH]
    mapx=(gx[:SKH]-sx[:SKH]).astype(np.float32); mapy=(gy[:SKH]-sy[:SKH]).astype(np.float32)
    orig=cv2.remap(sky_full,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    w=float(cyc(WPAL,t))
    if w<=0.001: sky=orig.copy()
    else:
        zen,mid,hor=cyc(ZEN,t),cyc(MID,t),cyc(HOR,t)
        pal=_grad(zen[::-1],mid[::-1],hor[::-1])     # BGR
        # sun/after-glow behind the peaks
        glow=np.exp(-(((xx_-1080)/620)**2+((yy_-330)/150)**2))[...,None]
        gk=max(0,1-abs(t-340)/90)*0.5+max(0,1-abs(t-845)/70)*0.5
        pal=pal+glow*gk*A(60,120,200)[::-1]*0.5
        cl=cv2.remap(CL0,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[...,None]
        night=float(cyc(STAR,t))
        kc=0.42*(1-night)+0.16*night
        cloudwarm=np.clip(cl,0,1)*(A(40,70,120)[::-1]*max(0,1-abs(t-330)/110)+A(50,70,110)[::-1]*max(0,1-abs(t-850)/60))*0.6
        proc=pal*(1+kc*cl)+cloudwarm
        sky=orig*(1-w)+proc*w
    nstar=float(cyc(STAR,t))
    if nstar>0.01:
        tw=0
        for gi in range(3):
            tw=tw+STARL[gi]*(0.72+0.28*math.sin(om(31+gi*17)*t+gi*2.1))
        cl2=np.clip(cv2.remap(CL0,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT),0,1)
        sky=sky+(tw*(1-0.8*cl2))[...,None]*nstar*np.array([235,225,215],np.float32)*(1-0.55*np.clip(vgrad*1.0,0,1))[...,None]
    m=float(cyc(MOON,t))
    if m>0.01:
        u=(t-470)/230.0; mx=1130-150*u; my=125-30*u
        glow=np.exp(-(((xx_-mx)**2+(yy_-my)**2)/(2*90.0**2)))[...,None]*0.28
        sky=sky+glow*np.array([225,205,180],np.float32)*m
        d=np.hypot(xx_-mx,yy_-my); disc=np.clip((30-d)/1.6,0,1)
        # soft shading + maria
        mar=0.9-0.14*np.exp(-(((xx_-mx+8)**2+(yy_-my-6)**2)/60.0))-0.10*np.exp(-(((xx_-mx-9)**2+(yy_-my+8)**2)/40.0))-0.08*np.exp(-(((xx_-mx-4)**2+(yy_-my-14)**2)/25.0))-0.06*np.exp(-(((xx_-mx+12)**2+(yy_-my+7)**2)/20.0))
        lit=np.clip(((xx_-mx)*0.55+(yy_-my)*(-0.35)+30)/55,0.55,1.0)
        mcol=np.array([226,238,245],np.float32)*(mar*lit)[...,None]
        sky=sky*(1-(disc*m)[...,None])+mcol*(disc*m)[...,None]
    au=float(cyc(AUR,t))
    if au>0.01:
        def ribbon(ph,y_base,amp1,amp2,th0,sp):
            y0=y_base+amp1*np.sin(TAU*xx_/900+om(3)*t*sp+ph)+amp2*np.sin(TAU*xx_/410+om(7)*t*sp+1.1+ph)
            th=th0+22*np.sin(TAU*xx_/700+om(5)*t+0.4+ph)
            dyv=yy_-y0
            prof=np.where(dyv>0,np.exp(-(dyv/(th*0.55))**2),np.exp(-(-dyv/(th*2.3))**2))
            rays=0.5+0.5*np.sin(TAU*xx_/173+om(11)*t+ph)*np.sin(TAU*xx_/311-om(13)*t+2*ph)
            I=prof*(0.72+0.28*rays)*np.clip(0.8+0.2*np.sin(om(9)*t+xx_*0.004+ph),0,1)
            up=np.clip((y0-yy_)/(th*1.8),0,1)[...,None]
            col=np.array([120,235,140],np.float32)*(1-up)+np.array([215,120,160],np.float32)*up
            return I[...,None]*col
        aur=(ribbon(0.0,150,50,28,60,1.0)*0.60+ribbon(2.4,205,38,20,44,0.8)*0.30)
        aur=cv2.GaussianBlur(aur,(0,0),sigmaX=9,sigmaY=5)*0.75+cv2.GaussianBlur(aur,(0,0),28)*0.5
        sky=sky+aur*au*0.85
    return sky
def emissive(out,t,tl=None):
    tf=t if tl is None else tl
    wn=float(cyc(WIN,t))
    if wn>0.01:
        fl=1+0.04*math.sin(om(37)*t)+0.03*math.sin(om(91)*t+1.0)
        lay=np.zeros((120,300,3),np.float32); ox,oy=720,470
        for (cx,cy,w_,h_) in WINS:
            x0,y0=cx-w_//2-ox,cy-h_//2-oy
            lay[y0:y0+h_,x0:x0+w_]=(110,190,255) if w_<20 or h_<30 else (80,150,235)
        core=lay.copy()
        halo=cv2.GaussianBlur(core,(0,0),12)*0.9
        halo2=cv2.GaussianBlur(core,(0,0),4)*0.55
        add=(core*0.95+halo+halo2)*wn*fl
        y0_,y1_,x0_,x1_=oy,oy+120,ox,ox+300
        out[y0_:y1_,x0_:x1_]=np.clip(out[y0_:y1_,x0_:x1_]+add,0,255)
        # warm spill on the grass at the door
        sp=np.exp(-(((gx[560:640,780:960]-870)/55)**2+((gy[560:640,780:960]-572)/16)**2))[...,None]
        out[560:640,780:960]=np.clip(out[560:640,780:960]+sp*np.array([40,90,150],np.float32)*wn*fl*0.7,0,255)
    ff=float(cyc(FFL,t))
    if ff>0.01:
        lay=np.zeros((H,W,3),np.float32)
        for bx,by,ax,ay,k1,k2,k3,p1,p2,p3 in FF:
            x=bx+ax*math.sin(om(k1)*tf+p1)+0.5*ax*math.sin(om(k1*3)*tf+p2); y=by+ay*math.sin(om(k2)*tf+p2)
            b=max(0.0,math.sin(om(k3)*tf+p3))**2.5*ff
            if b<0.03: continue
            xi,yi=int(x),int(y)
            if not(8<=xi<W-8 and 8<=yi<H-8): continue
            patch=lay[yi-8:yi+9,xi-8:xi+9]
            gg=np.exp(-(((np.arange(17)-8)[None,:]**2+(np.arange(17)-8)[:,None]**2)/(2*3.0**2)))[...,None]
            cc=np.exp(-(((np.arange(17)-8)[None,:]**2+(np.arange(17)-8)[:,None]**2)/(2*1.0**2)))[...,None]
            patch+= (gg*np.array([60,210,170],np.float32)*0.9+cc*np.array([190,255,235],np.float32)*1.2)*b
        out+=lay
    return out
