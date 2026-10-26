import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
D=ZT+''
O=D+'wall/'
W,H,FPS=1080,1920,24
T=float(os.environ.get('T','5'))
S0,S1=float(os.environ.get('S0','300')),float(os.environ.get('S1','15'))
SEC=23                                    # columns per building section

# ---------------- hashing
def hsh(i,j,k):
    x=(i.astype(np.int64)*73856093) ^ (j.astype(np.int64)*19349663) ^ (k*83492791)
    x=(x ^ (x>>13))*1274126177 & 0xffffffff
    x=x ^ (x>>16)
    return (x & 0xffffff)/float(0x1000000)

# ---------------- master cell template (BGR, 0..1), drawn at M px
M=512
def master():
    y=np.linspace(0,1,M,dtype=np.float32)[:,None]; x=np.linspace(0,1,M,dtype=np.float32)[None,:]
    one=np.ones((M,M),np.float32)
    wall=np.stack([one*0.235,one*0.175,one*0.135],-1)                     # blue-grey concrete
    wall*= (0.92+0.08*np.cos((x-0.5)*3))[...,None]
    slab=((y>0.80)).astype(np.float32)                                  # floor slab band
    wall=wall*(1-slab[...,None])+np.array([0.30,0.255,0.215],np.float32)*slab[...,None]
    wall=wall*(1-0.45*((y>0.795)&(y<0.81)).astype(np.float32))[...,None]   # slab shadow line
    gx=(x>0.10)&(x<0.90); gy=(y>0.20)&(y<0.72)
    G=(gx&gy).astype(np.float32)
    mull=(np.abs(x-0.5)<0.012)|(np.abs(y-0.36)<0.010)&gx                 # frame bars
    G[mull & gy]=0
    frame=((x>0.085)&(x<0.915)&(y>0.185)&(y<0.735)).astype(np.float32)-((gx&gy)).astype(np.float32)
    wall=wall*(1-frame[...,None])+np.array([0.20,0.17,0.15],np.float32)*frame[...,None]
    sill=((x>0.07)&(x<0.93)&(y>0.72)&(y<0.75)).astype(np.float32)
    wall=wall*(1-sill[...,None])+np.array([0.34,0.31,0.28],np.float32)*sill[...,None]
    # interior shading for lit windows: curtains at the sides, brighter middle, ceiling darker
    cur=np.clip(1-np.abs(x-0.5)/0.40,0,1)**0.6
    I=(0.55+0.45*cur)*(0.75+0.25*np.clip((y-0.2)/0.5,0,1))
    curtain_edge=((np.abs(x-0.22)<0.03)|(np.abs(x-0.78)<0.03)).astype(np.float32)
    I=I*(1-0.35*curtain_edge)
    # balcony variant: a railing across the lower window
    rail=((y>0.60)&(y<0.735)&(x>0.06)&(x<0.94)).astype(np.float32)
    bars=(((x*40)%1)<0.18).astype(np.float32)*rail
    railtop=((y>0.595)&(y<0.615)&(x>0.06)&(x<0.94)).astype(np.float32)
    R=np.clip(bars*0.85+railtop,0,1)
    blinds=(((y-0.2)*26)%1<0.55).astype(np.float32)*0.45+0.55
    return wall,G,I,R,blinds*np.ones_like(x)
WALL,GM,IM,RM,BM=master()
_tc={}
def tmpl(P):
    if P in _tc: return _tc[P]
    it=cv2.INTER_AREA if P<M else cv2.INTER_CUBIC
    r=lambda a: cv2.resize(a,(P,P),interpolation=it)
    _tc.clear(); _tc[P]=(r(WALL),np.clip(r(GM),0,1),r(IM),np.clip(r(RM),0,1),r(BM)); return _tc[P]

# ---------------- cat window sprite (cell-normalised), returns darkening mask at P
def cat_mask(P,t):
    SS=3 if P>300 else 4; N=P*SS; m=Image.new('L',(N,N),0); d=ImageDraw.Draw(m)
    b=1+0.03*math.sin(2*math.pi*t/3.2)                                   # breathing
    def E(cx,cy,rx,ry): d.ellipse([(cx-rx)*N,(cy-ry)*N,(cx+rx)*N,(cy+ry)*N],fill=255)
    o=0.07
    E(0.66+o,0.722-0.062*b,0.145,0.062*b)                              # body: a loaf on the sill
    d.rectangle([(0.56+o)*N,(0.722-0.03)*N,(0.76+o)*N,0.722*N],fill=255)
    E(0.505+o,0.672,0.050,0.044)                                        # head, resting low
    d.polygon([((0.463+o)*N,0.660*N),((0.470+o)*N,0.604*N),((0.500+o)*N,0.636*N)],fill=255)
    d.polygon([((0.508+o)*N,0.632*N),((0.536+o)*N,0.602*N),((0.546+o)*N,0.656*N)],fill=255)
    d.chord([(0.47+o)*N,0.700*N,(0.80+o)*N,0.744*N],180,360,fill=255) # tail curled along the front
    # round leafy plant in a pot on the left
    d.polygon([(0.17*N,0.722*N),(0.27*N,0.722*N),(0.262*N,0.662*N),(0.178*N,0.662*N)],fill=255)
    for cx,cy,r in ((0.22,0.62,0.045),(0.185,0.635,0.035),(0.255,0.632,0.036),(0.20,0.585,0.032),(0.245,0.59,0.030),(0.222,0.555,0.026)):
        d.ellipse([(cx-r)*N,(cy-r*0.85)*N,(cx+r)*N,(cy+r*0.85)*N],fill=255)
    return cv2.resize(np.asarray(m,np.float32)/255,(P,P),interpolation=cv2.INTER_AREA)
_lg={}
def cat_light(P):
    if P in _lg: return _lg[P]
    y=np.linspace(0,1,P,dtype=np.float32)[:,None]; x=np.linspace(0,1,P,dtype=np.float32)[None,:]
    lamp=0.70+0.75*np.exp(-((x-0.70)**2/0.05+(y-0.40)**2/0.06))
    folds=1-0.22*np.clip(1-np.abs(x-0.5)/0.4,0,1)**0*(np.abs(x-0.5)>0.30)*(0.5+0.5*np.sin(x*90))
    _lg.clear(); _lg[P]=(lamp*folds); return _lg[P]

# ---------------- per-cell state
LW=np.array([[0.45,0.72,1.00],[0.55,0.80,1.00],[0.62,0.86,1.00],[0.95,0.90,0.82],[0.40,0.62,1.00]],np.float32)
def cells(i,j,t):
    sec=np.floor_divide(i+1000,SEC)
    gap=((i+1000)%SEC)==0
    tint=0.80+0.40*hsh(sec,sec*0+7,3)[...,None]*np.array([1,0.95,0.9])
    tint=tint.astype(np.float32)
    balc=(hsh(sec,sec*0+5,4)>0.55)&(hsh(i,i*0+1,9)>0.15)
    r=hsh(i,j,1)
    p_on=0.05+0.09*hsh(sec,sec*0+2,5)                                   # how busy this block is
    # each window gets one toggle time inside the loop -> lights slowly change
    tog=hsh(i,j,11)*T*1.6
    flip=(hsh(i,j,12)<0.10)&(t>tog)
    lit=(r<p_on)^flip
    kind=(hsh(i,j,2)*len(LW)).astype(int)
    col=LW[kind]*(0.45+0.65*hsh(i,j,6)**0.7)[...,None]
    tv=hsh(i,j,8)<0.06
    fl=0.75+0.25*np.sin(t*7.3+hsh(i,j,13)*40)*np.sin(t*3.1+hsh(i,j,14)*20)
    blue=np.array([1.0,0.70,0.45],np.float32)
    col=np.where(tv[...,None],blue*fl[...,None],col)
    lm=lit.astype(np.float32)*(1-gap)
    dg=np.stack([0.20,0.13,0.085],-1).astype(np.float32)*(0.55+0.6*hsh(i,j,7))[...,None]   # glass reflecting the dusk sky
    cat=(i==0)&(j==0)
    lm=np.where(cat,1.0,lm); col=np.where(cat[...,None],np.array([0.42,0.70,1.0],np.float32),col)
    tint=tint*(0.93+0.14*hsh(i,j,21))[...,None].astype(np.float32)
    bl=(hsh(i,j,22)<0.35)&~cat
    return tint,balc&~cat,gap,lm,col,dg,bl

# ---------------- render
def scale_at(t):
    u=np.clip(t/T,0,1); u=u*u*(3-2*u)*0.25+u*0.75                      # gentle ease-in
    return S0*(S1/S0)**u
def facade(t):
    s=scale_at(t); P=max(int(math.ceil(s)),4)
    nx=int(math.ceil(W/(2*s)))+2; ny=int(math.ceil(H/(2*s)))+2
    ii=np.arange(-nx,nx+1); jj=np.arange(-ny,ny+1)
    I_,J_=np.meshgrid(ii,jj)
    tint,balc,gap,lm,col,dg,bl=cells(I_,J_,t)
    wall,G,Ii,R,Bm=tmpl(P)
    nyc,nxc=I_.shape
    rep=lambda a: np.repeat(np.repeat(a,P,0),P,1)
    tile=lambda a: np.tile(a,(nyc,nxc,1)) if a.ndim==3 else np.tile(a,(nyc,nxc))
    Wl=tile(wall)*rep(tint); Gl=tile(G); Il=tile(Ii); Bl=rep(bl.astype(np.float32)); Il=Il*(1-Bl+Bl*tile(Bm))
    Gl=Gl*(1-rep(gap.astype(np.float32)))
    LM=rep(lm)[...,None]
    glass=rep(dg)*(1-LM)+rep(col)*Il[...,None]*LM
    img=Wl*(1-Gl[...,None])+glass*Gl[...,None]
    Rl=tile(R)*rep(balc.astype(np.float32))
    img=img*(1-Rl[...,None])+np.array([0.16,0.14,0.13],np.float32)*Rl[...,None]
    gapcol=rep(gap.astype(np.float32))[...,None]
    img=img*(1-gapcol*0.55)
    # cat window detail
    if P>=12:
        y0,x0=ny*P,nx*P
        cm=cat_mask(P,t)
        sub=img[y0:y0+P,x0:x0+P]
        g=GM_P(P); sub*=(1+g*(cat_light(P)-1))[...,None]
        sub*= (1-0.92*cm*g)[...,None]
        img[y0:y0+P,x0:x0+P]=sub
    # map region -> screen
    k=s/P; cxr=nx*P+P/2; cyr=ny*P+P/2
    if k<0.999: img=cv2.GaussianBlur(img,(0,0),0.45/k)
    A=np.float32([[k,0,W/2-cxr*k],[0,k,H/2-cyr*k]])
    return cv2.warpAffine(img,A,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE),s
_gp={}
def GM_P(P):
    if P not in _gp: _gp.clear(); _gp[P]=tmpl(P)[1]
    return _gp[P]

# ---------------- rain (screen space, loops every 2 s)
def rain_tile(n,ln,seed,th,ang):
    r=np.random.RandomState(seed); tl=np.zeros((768,768),np.float32)
    for _ in range(n):
        x,y=r.randint(0,768,2); l=r.randint(ln//2,ln)
        cv2.line(tl,(x,y),(x+int(l*ang),y+l),r.uniform(0.4,1),th,cv2.LINE_AA)
    return np.tile(tl,(2,2))
RT=[(rain_tile(300,34,1,1,0.10),768/1.0,0.18),(rain_tile(120,60,2,1,0.10),768/0.6,0.26),(rain_tile(35,110,3,2,0.10),768/0.35,0.30)]
def rain(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in RT:
        oy=int(t*sp)%768; ox=int(t*sp*0.10)%768
        blk=np.roll(np.roll(tl,oy,0),ox,1)[:768,:768]
        acc+=np.tile(blk,(3,2))[:H,:W]*a
    return acc

FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',50)
def text_layer(s):
    im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
    lines=["somewhere in this city,","someone's cat is asleep"]
    for k,l in enumerate(lines): d.text((W/2,300+k*66),l,font=FB,fill=255,anchor='mm')
    a=np.asarray(im,np.float32)/255
    return a,cv2.GaussianBlur(a,(0,0),6)
TXT,TSH=text_layer(None)

def frame(f):
    t=f/FPS
    img,s=facade(t)
    img=img.astype(np.float32)*255
    # glow from lit windows
    br=np.clip(img-140,0,None)
    small=cv2.resize(br,(W//6,H//6),interpolation=cv2.INTER_AREA)
    img+=cv2.resize(cv2.GaussianBlur(small,(0,0),3),(W,H))*0.9
    # dusk haze + vignette
    yy=np.linspace(0,1,H,dtype=np.float32)[:,None,None]
    img=img*(0.92+0.08*yy)+np.array([22,10,4],np.float32)*(1-yy)
    X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
    img*=(1-0.30*(X**2*0.7+Y**2*0.5))[...,None]
    img+=rain(t)[...,None]*np.array([200,190,180],np.float32)*0.55
    img=img*(1-TSH[...,None]*0.45)
    img=img*(1-TXT[...,None])+TXT[...,None]*np.array([245,245,250],np.float32)
    return np.clip(img,0,255).astype(np.uint8)

if __name__=='__main__':
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','):
            cv2.imwrite(O+f's_{tt}.png',frame(int(float(tt)*FPS)))
    else:
        out=O+sys.argv[2]
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
            '-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',O+'v.mp4'],stdin=subprocess.PIPE)
        for f in range(int(T*FPS)): p.stdin.write(frame(f).tobytes())
        p.stdin.close(); p.wait()
