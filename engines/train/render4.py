import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess, sys, math
U=ZT+'uploads/'
W,H=1920,1080; FPS=24; T=900; NF=T*FPS
ZOOM=1.03; BOTTOM=610; REFL_Y=812; SC=0.406
TAU=2*math.pi
def om(k): return TAU*k/T
# ---------- background (3% zoom so sky sway never shows an edge)
src=cv2.imread(U+'e045e710-image.jpg')
zw,zh=int(W*ZOOM),int(H*ZOOM)
big=cv2.resize(src,(zw,zh),interpolation=cv2.INTER_AREA)
ox,oy=(zw-W)//2,(zh-H)//2
bg0=big[oy:oy+H,ox:ox+W].astype(np.float32)
GX,GY=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
def sm(a,b,x):
    u=np.clip((x-a)/(b-a),0,1); return u*u*(3-2*u)
SKYW=(1-sm(390,476,GY)).astype(np.float32)
WATW=(sm(800,834,GY)*(1-sm(1000,1080,GY))).astype(np.float32)
GRAW=(sm(880,1040,GY)*(GX<1520)).astype(np.float32)
QW,QH=W//4,H//4
QX,QY=np.meshgrid(np.arange(QW,dtype=np.float32)*4,np.arange(QH,dtype=np.float32)*4)
QSKY=cv2.resize(SKYW,(QW,QH)); QWAT=cv2.resize(WATW,(QW,QH)); QGRA=cv2.resize(GRAW,(QW,QH))
def warp(t):
    s1=np.sin(om(1)*t)
    drift=(0.55+0.5*np.clip(1-QY/480,0,1))*60*s1
    bil=6.4*np.sin(QX/140+om(50)*t+QY/90)+4.0*np.sin(QX/74-om(72)*t+QY/160)
    bily=3.6*np.sin(QX/180+om(43)*t)+2.4*np.sin(QY/60+om(64)*t)
    dx=QSKY*(drift+bil); dy=QSKY*bily
    dx=dx+QWAT*(3.2*np.sin(QY/10+om(229)*t+QX/280)+1.6*np.sin(QY/4.6-om(301)*t))
    dx=dx+QGRA*(np.clip(QY-880,0,None)/200)**1.5*4.4*np.sin(QX/90+om(186)*t)
    dx=cv2.resize(dx.astype(np.float32),(W,H),interpolation=cv2.INTER_LINEAR); dy=cv2.resize(dy.astype(np.float32),(W,H),interpolation=cv2.INTER_LINEAR)
    return cv2.remap(bg0,GX-dx,GY-dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
# ---------- cyclic key lists over T
def cyc(keys,t):
    t=t%T
    for i in range(len(keys)-1):
        t0,v0=keys[i]; t1,v1=keys[i+1]
        if t0<=t<=t1:
            u=(t-t0)/(t1-t0); u=u*u*(3-2*u)
            return np.array(v0,np.float32)*(1-u)+np.array(v1,np.float32)*u
    return np.array(keys[0][1],np.float32)
GRADE=[(0,(1,1,1)),(180,(0.85,0.88,1.02)),(330,(0.70,0.58,0.82)),(480,(0.46,0.32,0.26)),(620,(0.40,0.28,0.23)),(760,(0.52,0.42,0.42)),(840,(0.86,0.83,0.96)),(900,(1,1,1))]
SKYADD=[(0,(0,0,0)),(180,(6,-2,16)),(330,(15,-3,18)),(480,(14,6,0)),(620,(14,6,0)),(760,(12,0,8)),(840,(10,-3,22)),(900,(0,0,0))]
NIGHTG=[(0,0),(180,0.12),(330,0.7),(480,1),(620,1),(760,0.85),(840,0.25),(900,0)]
STARA=[(0,0),(330,0.05),(450,0.9),(620,1),(740,0.85),(820,0.15),(860,0),(900,0)]
# ---------- sprites
def cut(fn):
    im=cv2.imread(U+fn).astype(np.float32); b,g,r=im[...,0],im[...,1],im[...,2]
    k=np.minimum(r-g,b-g)
    a=np.clip(1-(k-40)/80,0,1); a=cv2.GaussianBlur(a,(3,3),0)
    im[...,2]=np.minimum(r,g+(r-g)*a+20); im[...,0]=np.minimum(b,g+(b-g)*a+20)
    ys,xs=np.where(a>0.5); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    im=im[y0:y1+1,x0:x1+1]; a=a[y0:y1+1,x0:x1+1]
    s=(int(im.shape[1]*SC),int(im.shape[0]*SC))
    return cv2.resize(im,s,interpolation=cv2.INTER_AREA),cv2.resize(a,s,interpolation=cv2.INTER_AREA)
def prep(spr,al,mir):
    if mir: spr=cv2.flip(spr,1); al=cv2.flip(al,1)
    h,w=al.shape
    bright=np.clip((spr[...,2]-225)/30,0,1)*np.clip((spr[...,1]-175)/40,0,1)*np.clip((170-spr[...,0])/60,0,1)*al
    bl=cv2.GaussianBlur(bright,(0,0),12)
    fl=cv2.GaussianBlur(cv2.flip(spr,0),(0,0),3.0); fa=cv2.GaussianBlur(cv2.flip(al,0),(0,0),3.0)
    blf=cv2.flip(bl,0)
    cx=(w-1-0.064*w) if mir else 0.064*w
    return dict(spr=spr,al=al,h=h,w=w,bright=bright,bl=bl,fl=fl,fa=fa,blf=blf,cx=cx,cy=0.02*h)
P0=cut('640ec5d2-image.jpg'); F0=cut('007d2c99-image.jpg')
SPR={('P',-1):prep(*P0,False),('P',1):prep(*P0,True),('F',-1):prep(*F0,False),('F',1):prep(*F0,True)}
# passes: (kind, dir, t_start, speed px/s)
PASSES=[('P',1,35,225),('F',-1,170,190),('P',-1,290,225),('F',1,455,190),('P',1,520,225),('P',-1,700,225),('F',-1,810,190)]
TR=[dict(sp=SPR[(k,d)],dir=d,t0=t0,v=v) for k,d,t0,v in PASSES]
def xpos(tr,t):
    w=tr['sp']['w']; d=(t-tr['t0'])*tr['v']
    return (-w+d) if tr['dir']==1 else (W-d)
# ---------- misc layers
rng=np.random.RandomState(7)
ys=np.arange(H,dtype=np.float32)[:,None]; xs=np.arange(W,dtype=np.float32)[None,:]
vig=(1-0.35*(((xs-W/2)/(W/2))**2+((ys-H/2)/(H/2))**2)/2).astype(np.float32)
dith=(rng.rand(H,W).astype(np.float32)-0.5)*2.2
nz=rng.rand(40,200).astype(np.float32)
nz3=cv2.GaussianBlur(np.tile(nz,(1,3)),(0,0),3)[:,200:400]
nz=(nz3-nz3.min())/(nz3.max()-nz3.min())
MIST=cv2.resize(nz,(W,120),interpolation=cv2.INTER_CUBIC)
MIST=np.ascontiguousarray(MIST.astype(np.float32))
MY0=662
# cloud-ness for stars
cloudness=sm(190,230,cv2.GaussianBlur(bg0[...,2],(0,0),6))
STARS=[]
srng=np.random.RandomState(11)
while len(STARS)<260:
    x=srng.randint(4,W-4); y=srng.randint(10,470)
    if cloudness[y,x]<0.25 and SKYW[y,x]>0.9: STARS.append((x,y,srng.rand()*TAU,int(srng.randint(20,70)),0.5+0.5*srng.rand()))
FF=[]
frng=np.random.RandomState(5)
for i in range(34):
    FF.append((frng.randint(80,1750),frng.randint(860,1040),frng.randint(40,160),frng.randint(10,40),frng.randint(2,7),frng.randint(2,7),frng.rand()*TAU,frng.rand()*TAU,frng.randint(8,24)))
GK=cv2.getGaussianKernel(15,3.2).astype(np.float32); GK=GK@GK.T; GK/=GK.max()
def frame(fi):
    t=(fi/FPS)
    m=cyc(GRADE,t); g=float(cyc(NIGHTG,t)); sa=float(cyc(STARA,t)); sk=cyc(SKYADD,t)
    f=warp(t)*m
    f+=np.clip(1-ys/600,0,1)[...,None]*sk
    # stars
    if sa>0.01:
        for (x,y,ph,k,b) in STARS:
            a=sa*b*(0.55+0.45*math.sin(om(k)*t+ph))*(1-0.8*cloudness[y,x])
            f[y-1:y+2,x-1:x+2]+=a*np.array([150,170,175],np.float32)*np.array([[.25,.6,.25],[.6,1,.6],[.25,.6,.25]],np.float32)[...,None]
    # mist
    shift=t*8*W/T
    M=np.float32([[1,0,-shift],[0,1,0]])
    band=cv2.warpAffine(MIST,M,(W,120),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    ma=band*0.16*np.linspace(1,0.3,120,dtype=np.float32)[:,None]
    lum=float(np.clip(m.mean()*1.15,0.25,1))
    f[MY0:MY0+120]+=ma[...,None]*(np.array([230,235,240],np.float32)*lum*(1-0.3*g))
    smoke=np.zeros((H,W),np.float32)
    for tr in TR:
        sp=tr['sp']; w,h=sp['w'],sp['h']; x=xpos(tr,t)
        if t-tr['t0']>(W+w)/tr['v']+9 or t<tr['t0']-1: continue
        # smoke
        RATE=16; k1=int((t-tr['t0'])*RATE)
        for k in range(max(0,k1-120),k1+1):
            ts=tr['t0']+k/RATE; age=t-ts
            if age<0 or age>7: continue
            xs_=xpos(tr,ts)+sp['cx']; ys_=BOTTOM-h+sp['cy']
            if not (-200<xs_<W+300): 
                if not (-200<xs_+9*age<W+300): continue
            rg=np.random.RandomState(k*7+int(tr['t0']))
            ph=rg.rand()*6.28; ph2=rg.rand()*6.28; sz=0.8+0.5*rg.rand()
            chuff=1+0.7*max(0,math.sin(TAU*ts*2.4))
            rise=2*(26*(1-math.exp(-age/2.2))+3*age)
            wind=2*(9*age+2.5*age**1.3)
            tur=2*(4+1.8*age)*math.sin(age*1.3+ph)*min(age/1.2,1)
            px=xs_+wind+tur; py=ys_-rise+5*math.sin(age*1.9+ph2)*min(age,2)
            r=2*(3+4.2*age**0.85)*sz*(0.8+0.25*chuff)
            a_=min(age/0.35,1)*0.85*chuff**0.5*(1-age/7)**1.4
            for j in range(3):
                oxx=r*0.55*math.cos(ph+j*2.1); oyy=r*0.45*math.sin(ph2+j*2.1)
                X=px+oxx; Y=py+oyy
                if 0<=X<W and 0<=Y<H: cv2.circle(smoke,(int(X),int(Y)),int(r*(0.75 if j else 1.0)),a_*(0.55 if j else 0.8),-1,cv2.LINE_AA)
        if x>W or x+w<0: continue
        y=BOTTOM-h+int(round(1.0*math.sin(t*18+tr['t0'])))
        xi=int(round(x)); sx0=max(0,-xi); sx1=min(w,W-xi)
        if sx1<=sx0: continue
        al=sp['al'][:,sx0:sx1][...,None]; spr=sp['spr'][:,sx0:sx1]; br=sp['bright'][:,sx0:sx1,None]
        body=spr*(m*1.05)
        wind_=spr*(m*1.05+(1-m*1.05)*g)
        s=body*(1-br)+wind_*br
        reg=f[y:y+h,xi+sx0:xi+sx1]
        reg[:]=reg*(1-al)+s*al
        glow=sp['bl'][:,sx0:sx1]*g*1.7
        reg+=glow[...,None]*np.array([60,170,255],np.float32)+br*g*np.array([40,90,120],np.float32)
        # reflection
        fl=sp['fl']; fa=sp['fa']; blf=sp['blf']
        for r_ in range(0,h):
            yy=REFL_Y+r_
            if yy>=H: break
            dx=int(round(5*math.sin(r_*0.45+om(229)*t)))
            xa=xi+dx; x0_=max(0,xa); x1_=min(W,xa+w)
            if x1_<=x0_: continue
            seg=fl[r_,x0_-xa:x1_-xa]*m
            ga=(fa[r_,x0_-xa:x1_-xa]*(0.22+0.12*g)*(1-r_/h*0.6))[:,None]
            f[yy,x0_:x1_]=f[yy,x0_:x1_]*(1-ga)+seg*ga
            if g>0.02:
                f[yy,x0_:x1_]+=(blf[r_,x0_-xa:x1_-xa]*g*0.55)[:,None]*np.array([50,150,235],np.float32)
    # smoke composite (band only)
    y0,y1=120,760
    sb=np.clip(cv2.GaussianBlur(smoke[y0:y1],(0,0),6.0),0,1)
    tx=np.tile(MIST[:,:],(1,1))
    tex=cv2.warpAffine(cv2.resize(nz,(W,y1-y0),interpolation=cv2.INTER_LINEAR),np.float32([[1,0,-(t*28)],[0,1,0]]),(W,y1-y0),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    sb=np.clip(sb*(0.62+0.5*tex),0,0.92)[...,None]
    ms=0.45+0.55*float(m.mean())
    f[y0:y1]=f[y0:y1]*(1-sb)+sb*(np.array([105,108,125],np.float32)*ms)
    # fireflies
    fg=float(np.clip((g-0.45)*2,0,1))
    if fg>0.02:
        for (cx,cy,ax,ay,a,b,p1,p2,c) in FF:
            X=int(cx+ax*math.sin(om(a)*t+p1)); Y=int(cy+ay*math.sin(om(b)*t+p2))
            bl=max(0,math.sin(om(c*3)*t+p1))**2*fg
            if bl>0.02 and 8<X<W-8 and 8<Y<H-8:
                f[Y-7:Y+8,X-7:X+8]+=GK[...,None]*bl*np.array([70,230,255],np.float32)
    f*=vig[...,None]
    f+=dith[...,None]
    return np.clip(f,0,255).astype(np.uint8)
def enc(out):
    return subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','fast','-crf','19','-tune','animation','-g','48','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
if __name__=='__main__':
    c=sys.argv[1]
    if c=='still':
        cv2.imwrite(sys.argv[3],frame(int(float(sys.argv[2])*FPS)))
    elif c=='time':
        import time
        for fi in (int(float(sys.argv[2])*FPS),):
            t0=time.time(); frame(fi); print('s/frame',time.time()-t0)
    elif c=='loopcheck':
        a=frame(0).astype(int); b=frame(NF).astype(int); print('diff0vsNF',np.abs(a-b).mean(),np.abs(a-b).max())
        d=frame(NF-1).astype(int); print('diff NF-1 vs 0',np.abs(a-d).mean())
    else:
        a,b,out=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
        p=enc(out)
        for fi in range(a,b): p.stdin.write(frame(fi).tobytes())
        p.stdin.close(); p.wait()
