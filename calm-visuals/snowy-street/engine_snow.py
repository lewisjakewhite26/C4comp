import sys, os, math, random, subprocess
import numpy as np, cv2
SC=os.environ.get('SCENE','v'); D='/tmp/claude-0/-home-claude-c4comp/f7d1ccd4-c824-5d8b-9f33-8948edecece0/scratchpad/snow/'
FPS=24; T=float(os.environ.get('T','60'))
S=1080/768.0
VERT=(SC=='v'); W,H=(1080,1920) if VERT else (1920,1080)
src=cv2.imread(D+SC+'.jpg'); sh,sw=src.shape[:2]
bw,bh=int(round(sw*S)),int(round(sh*S)); XO=(bw-W)/2.0; YO=(bh-H)/2.0
def X(x): return x*S-XO
def Y(y): return y*S-YO
R=random.Random(7); rg=np.random.default_rng(7)
big=cv2.resize(src,(bw,bh),interpolation=cv2.INTER_LANCZOS4); x0=int(round(XO)); y0=int(round(YO))
BASE=big[y0:y0+H,x0:x0+W].astype(np.float32)
bl=cv2.GaussianBlur(BASE,(0,0),1.0); BASE=np.clip(BASE+0.35*(BASE-bl),0,255)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
lum=BASE.mean(2); Bc,Gc,Rc=BASE[...,0],BASE[...,1],BASE[...,2]
def gauss(cx,cy,sx,sy): return np.exp(-(((xx-X(cx))/(sx*S))**2+((yy-Y(cy))/(sy*S))**2)).astype(np.float32)
if VERT:
    LAMPS=[(268,650,70,1.0),(558,648,60,0.9),(362,800,28,0.6),(378,846,22,0.5),(444,838,22,0.5),(493,816,22,0.45)]
    NEONS=[(415,725,40,28,'pink'),(434,698,22,45,'pink'),(335,690,28,30,'cyan'),(536,742,28,22,'cyan'),(656,718,28,28,'pink'),(560,725,30,20,'orange'),(485,676,24,26,'white')]
    VEND=(268,965,45,60); HAZE=(500,610,260,110); SKY_Y=690; GROUND_Y=900; WIN=(540,1010)
else:
    LAMPS=[(408,192,60,1.0),(568,382,34,0.8),(810,423,22,0.5),(603,442,20,0.5),(722,468,20,0.45)]
    NEONS=[(796,322,24,40,'pink'),(1073,272,26,38,'cyan'),(480,350,24,24,'cyan'),(780,350,22,24,'cyan'),(484,330,16,22,'pink'),(897,350,22,22,'pink'),(505,350,18,18,'orange')]
    VEND=(1219,575,38,60); HAZE=(780,330,380,90); SKY_Y=335; GROUND_Y=480; WIN=(120,520)
COL={'pink':(190,90,255),'cyan':(255,230,80),'orange':(70,150,255),'white':(220,230,255)}   # BGR
# ---- glows ----
LG=[(gauss(a,b,r,r),p) for (a,b,r,p) in LAMPS]
NG=[(gauss(a,b,sx,sy),COL[c],i) for i,(a,b,sx,sy,c) in enumerate(NEONS)]
VG=gauss(VEND[0],VEND[1],VEND[2],VEND[3])
HG=gauss(HAZE[0],HAZE[1],HAZE[2],HAZE[3])
GLOWM=sum(g for g,_ in LG)*0.9
# ---- windows (warm lit components) ----
band=np.zeros((H,W),bool); band[int(Y(WIN[0])):int(Y(WIN[1])),:]=True
cand=(lum>150)&(Rc>Bc*1.25)&(Gc>Bc*1.0)&band
n,lab,st,_=cv2.connectedComponentsWithStats(cand.astype(np.uint8),connectivity=8)
ok=np.zeros(n,bool)
for i in range(1,n): ok[i]=(st[i,cv2.CC_STAT_AREA]>60) and st[i,cv2.CC_STAT_AREA]<40000
winm=(ok[lab]&cand)
PHW=rg.uniform(0,6.28,n); KW=rg.integers(1,6,n)
# ---- sky / clouds ----
SKY=np.clip((Y(SKY_Y)-yy)/(60*S),0,1)*np.clip((lum-48)/28,0,1); SKY=cv2.GaussianBlur(SKY.astype(np.float32),(0,0),2)
# ---- snow ----
def mk(cnt,rmin,rmax,vmin,vmax,al,blur,sway):
    L=[]
    for _ in range(cnt):
        r=R.uniform(rmin,rmax); Lc=H+4*r; v=R.uniform(vmin,vmax); k=max(1,round(v*T/Lc))
        L.append((R.uniform(-30,W+30),R.uniform(0,Lc),r,Lc,k,R.uniform(0,6.28),R.randint(1,4),R.uniform(0.6,1.0)*al))
    return L,blur,sway
n_s=W*H/ (1080*1920.)
SNOW=[mk(int(520*n_s),1.0,1.8,22,40,0.65,0,10),mk(int(230*n_s),2.0,3.4,45,80,0.8,0.8,16),mk(int(70*n_s),4.5,8.0,95,150,0.55,3.0,26)]
def snow(t):
    out=np.zeros((H,W),np.float32); g=math.sin(2*math.pi*3*t/T)*0.7+0.3*math.sin(2*math.pi*8*t/T+1)
    for L,blur,sway in SNOW:
        lay=np.zeros((H,W),np.float32)
        for (x,y,r,Lc,k,ph,m,al) in L:
            yv=(y+Lc*k*t/T)%Lc-2*r; xv=(x+sway*math.sin(2*math.pi*m*t/T+ph)+g*sway*1.4)
            cv2.circle(lay,(int(xv),int(yv)),max(1,int(round(r))),float(al),-1,cv2.LINE_AA)
        if blur>0: lay=cv2.GaussianBlur(lay,(0,0),blur)
        out+=lay
    return out
# ---- sparkles on the snow ----
SPK=[(R.uniform(0,W),R.uniform(Y(GROUND_Y),H),R.uniform(0,6.28),R.randint(4,14),R.uniform(1.0,2.2)) for _ in range(int(160*n_s))]
def sparkles(t):
    lay=np.zeros((H,W),np.float32)
    for (x,y,ph,k,r) in SPK:
        a=np.clip(math.sin(2*math.pi*k*t/T+ph)*3-2.0,0,1)
        if a>0.02: cv2.circle(lay,(int(x),int(y)),max(1,int(r)),float(a),-1,cv2.LINE_AA)
    return cv2.GaussianBlur(lay,(0,0),0.9)
MIST=cv2.GaussianBlur(rg.normal(0,1,(H//10,W//10)).astype(np.float32),(0,0),5); MIST=cv2.resize(MIST/MIST.std(),(W,H))
BAND=np.exp(-(((yy-Y(HAZE[1]+40))/(120*S))**2)).astype(np.float32)
GRAIN=[rg.normal(0,1.2,(H,W,1)).astype(np.float32) for _ in range(5)]
VIG=1-0.22*np.clip((np.hypot((xx-W/2)/(W/2),(yy-H/2)/(H/2))-0.55)/0.8,0,1)**1.6
def render(i):
    t=i/FPS; out=BASE.copy()
    # window flicker
    f=np.where(KW>0,1-0.07*(0.5+0.5*np.sin(2*np.pi*KW*t/T*2+PHW)),1).astype(np.float32); f[0]=1
    out*= (1-(1-f[lab])*winm*1.0)[...,None]
    # lamps
    for k,(g,p) in enumerate(LG):
        lb=1+0.06*math.sin(2*math.pi*(4+k)*t/T+k)+0.025*math.sin(2*math.pi*(27+5*k)*t/T)
        out+=g[...,None]*np.array([20,90,210],np.float32)*lb*0.45*p
    # neon: pulse + occasional flicker
    for (g,c,k) in NG:
        pl=0.72+0.28*math.sin(2*math.pi*(6+2*k)*t/T+k*1.7)
        if math.sin(2*math.pi*(11+3*k)*t/T+k)>0.992: pl*=0.45
        out+=g[...,None]*np.array(c,np.float32)*0.42*pl
    hz=0.7+0.3*math.sin(2*math.pi*3*t/T)
    out+=HG[...,None]*np.array([120,70,170],np.float32)*0.30*hz
    drift=np.roll(MIST,int(t*W/T*2),axis=1); out+=(BAND*np.clip(0.5+0.5*drift,0,1))[...,None]*np.array([200,170,230],np.float32)*0.05
    vb=1+0.05*math.sin(2*math.pi*30*t/T)+(-0.25 if math.sin(2*math.pi*9*t/T)>0.994 else 0)
    out+=VG[...,None]*np.array([255,225,120],np.float32)*0.32*vb
    # clouds drift + camera breathe
    mx=xx+SKY*(6.0*np.sin(2*np.pi*3*t/T+yy/260.0)+3.0*np.sin(2*np.pi*8*t/T))
    z=1+0.007*(1-math.cos(2*math.pi*t/T))/2; mx=(mx-W/2)/z+W/2+3*math.sin(2*math.pi*2*t/T); my=(yy-H/2)/z+H/2+2*math.sin(2*math.pi*3*t/T+1)
    out=cv2.remap(out,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    sn=snow(t)
    out+=sn[...,None]*np.array([255,245,235],np.float32)*0.80
    out+=(sn*GLOWM)[...,None]*np.array([40,140,255],np.float32)*0.9
    out+=sparkles(t)[...,None]*np.array([255,245,230],np.float32)*0.9
    out=out*VIG[...,None]+GRAIN[i%5]
    return np.clip(out,0,255).astype(np.uint8)
def worker(f0,f1,outf):
    pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','%dx%d'%(W,H),'-r',str(FPS),'-i','-','-c:v','libx264','-preset','veryfast','-crf','16','-pix_fmt','yuv420p','-g','48','-x264-params','threads=1',D+outf],stdin=subprocess.PIPE)
    for i in range(f0,f1):
        pr.stdin.write(render(i).tobytes())
        if (i-f0)%120==0: print('frame',i,flush=True)
    pr.stdin.close(); pr.wait(); print('done',outf,flush=True)
if __name__=='__main__':
    m=os.environ.get('MODE','test')
    if m=='test':
        import time
        for s in os.environ['TS'].split(','):
            t0=time.time(); cv2.imwrite(D+'t_%s_%s.png'%(SC,s),render(int(float(s)*FPS))); print(s,round(time.time()-t0,2),flush=True)
    else: worker(int(os.environ['F0']),int(os.environ['F1']),os.environ['OUT'])
