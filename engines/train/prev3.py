import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess, sys
U=ZT+'uploads/'
W,H=960,540; FPS=24; DUR=60; SC=0.197; BOTTOM=304
bg0=cv2.resize(cv2.imread(U+'e045e710-image.jpg'),(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
GX,GY=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
def sm(a,b,x): 
    u=np.clip((x-a)/(b-a),0,1); return u*u*(3-2*u)
SKYW=(1-sm(195,238,GY)).astype(np.float32)          # 1 in sky, 0 on hills/trees
WATW=(sm(398,415,GY)*(1-sm(500,540,GY))).astype(np.float32)  # river
GRAW=sm(440,520,GY)*(GX<760)*1.0                     # foreground grass sway (left/bottom)
QW,QH=W//4,H//4
QX,QY=np.meshgrid(np.arange(QW,dtype=np.float32)*4,np.arange(QH,dtype=np.float32)*4)
QSKY=cv2.resize(SKYW,(QW,QH)); QWAT=cv2.resize(WATW,(QW,QH)); QGRA=cv2.resize(GRAW.astype(np.float32),(QW,QH))
def warp(t):
    drift=(0.55+0.5*np.clip(1-QY/240,0,1))*t*0.9
    bil=3.2*np.sin(QX/70+t*0.35+QY/45)+2.0*np.sin(QX/37-t*0.5+QY/80)
    bily=1.8*np.sin(QX/90+t*0.3)+1.2*np.sin(QY/30+t*0.45)
    dx=QSKY*(drift+bil); dy=QSKY*bily
    dx=dx+QWAT*(1.6*np.sin(QY/5.0+t*1.6+QX/140)+0.8*np.sin(QY/2.3-t*2.1))
    dx=dx+QGRA*(np.clip(QY-440,0,None)/100)**1.5*2.2*np.sin(QX/45+t*1.3)
    dx=cv2.resize(dx.astype(np.float32),(W,H),interpolation=cv2.INTER_LINEAR); dy=cv2.resize(dy.astype(np.float32),(W,H),interpolation=cv2.INTER_LINEAR)
    return cv2.remap(bg0,(GX-dx).astype(np.float32),(GY-dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def cut(fn):
    im=cv2.imread(U+fn).astype(np.float32); b,g,r=im[...,0],im[...,1],im[...,2]
    k=np.minimum(r-g,b-g)  # magenta-ness
    a=np.clip(1-(k-40)/80,0,1)
    a=cv2.GaussianBlur(a,(3,3),0)
    # despill: pull magenta out of edge pixels
    spill=np.clip(np.minimum(r,b)-g,0,None)*(1-a>0.0)
    im[...,2]=np.minimum(r,g+ (r-g)*a+20); im[...,0]=np.minimum(b,g+(b-g)*a+20)
    ys,xs=np.where(a>0.5); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    im=im[y0:y1+1,x0:x1+1]; a=a[y0:y1+1,x0:x1+1]
    s=(int(im.shape[1]*SC),int(im.shape[0]*SC))
    return cv2.resize(im,s,interpolation=cv2.INTER_AREA),cv2.resize(a,s,interpolation=cv2.INTER_AREA)
P=cut('640ec5d2-image.jpg'); F=cut('007d2c99-image.jpg')
def mirror(t): return cv2.flip(t[0],1),cv2.flip(t[1],1)
P=mirror(P)  # passenger heads right
print(P[0].shape,F[0].shape,file=sys.stderr)
# chimney offset from sprite left (orig engine on left, x~213*? ) compute: chimney at ~165/1998 of image minus crop; approximate from sprite
def chim(spr,mir):
    h,w=spr[1].shape
    return ((w-1-0.06*w) if mir else 0.06*w, 0.02*h)
trains=[dict(spr=P,t0=3,dir=1,mir=True,cx=chim(P,True)),dict(spr=F,t0=32,dir=-1,mir=False,cx=chim(F,False))]
SPEED=110
def pos(tr,t):
    w=tr['spr'][0].shape[1]
    d=(t-tr['t0'])*SPEED
    return (-w+d) if tr['dir']==1 else (W-d)
def grade(t):
    u=np.clip(t/DUR,0,1); u=u*u*(3-2*u)
    gold=np.array([1,1,1],np.float32); dusk=np.array([0.70,0.58,0.82],np.float32) # BGR mult
    return gold*(1-u)+dusk*u, u
ys=np.arange(H)[:,None]
vig=1-0.35*(((np.arange(W)[None,:]-W/2)/(W/2))**2+((ys-H/2)/(H/2))**2)/2
rng=np.random.RandomState(1)
noise=cv2.GaussianBlur(rng.rand(40,200).astype(np.float32),(0,0),3); noise=cv2.resize((noise-noise.min())/(noise.max()-noise.min()),(W*2,60))
out=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','fast','-crf','31','-pix_fmt','yuv420p',sys.argv[1]],stdin=subprocess.PIPE)
for fi in range(DUR*FPS):
    t=fi/FPS; m,u=grade(t)
    f=warp(t)*m
    # extra sky warm->pink on top
    sky=np.clip(1-ys/300,0,1)[...,None]*u
    f=f+sky*np.array([25,-5,30],np.float32)*0.6
    # mist band drifting
    ox=int(t*8)%W; band=noise[:,ox:ox+W]
    ma=np.zeros((H,W),np.float32); ma[330:390]=band[:60]*0.16*np.linspace(1,0.3,60)[:,None]
    f=f+ma[...,None]*np.array([230,235,240],np.float32)*(1-0.4*u)
    smoke=np.zeros((H,W),np.float32); bloom=np.zeros((H,W,3),np.float32)
    for tr in trains:
        x=pos(tr,t); spr,al=tr['spr']; h,w=al.shape
        if x>W or x+w<0: continue
        y=BOTTOM-h+int(round(0.5*np.sin(t*18+tr['t0'])))
        xi=int(round(x))
        sx0=max(0,-xi); sx1=min(w,W-xi)
        if sx1<=sx0: continue
        reg=f[y:y+h,xi+sx0:xi+sx1]; a=al[:,sx0:sx1][...,None]
        s=spr[:,sx0:sx1]*(m*1.05)
        # bloom from lit windows
        bright=np.clip((spr[...,2]-225)/30,0,1)*np.clip((spr[...,1]-175)/40,0,1)*np.clip((170-spr[...,0])/60,0,1)*al
        bl=cv2.GaussianBlur(bright,(0,0),6)
        glow=bl[:,sx0:sx1]*u*1.2
        reg[:]=reg*(1-a)+s*a
        reg[:]+=glow[...,None]*np.array([60,170,255],np.float32)+bright[:,sx0:sx1,None]*u*np.array([40,90,120],np.float32)

        # water reflection
        fl=cv2.flip(spr,0); fa=cv2.flip(al,0)
        fl=cv2.GaussianBlur(fl,(0,0),2.0); fa=cv2.GaussianBlur(fa,(0,0),2.0)
        yr=402
        for r_ in range(h):
            dx=int(round(2.5*np.sin(r_*0.9+t*2.2)))
            xa=xi+dx
            x0_=max(0,xa); x1_=min(W,xa+w)
            if x1_<=x0_ or yr+r_>=H: continue
            seg=fl[r_,x0_-xa:x1_-xa]*m; ga=fa[r_,x0_-xa:x1_-xa][:,None]*0.22*(1-r_/h*0.6)
            f[yr+r_,x0_:x1_]=f[yr+r_,x0_:x1_]*(1-ga)+seg*ga
        # smoke: chuffing puffs, buoyant rise, wind drift, turbulence
        cxl,cyl=tr['cx']
        RATE=16
        for k in range(max(0,int((t-tr['t0'])*RATE)-110),int((t-tr['t0'])*RATE)+1):
            ts=tr['t0']+k/RATE; age=t-ts
            if age<0 or age>7: continue
            rg=np.random.RandomState(k*7+int(tr['t0']))
            ph=rg.rand()*6.28; ph2=rg.rand()*6.28; sz=0.8+0.5*rg.rand()
            chuff=1+0.7*max(0,np.sin(2*np.pi*(ts*2.4)))      # rhythmic chuffs
            xs_=pos(tr,ts)+cxl; ys_=BOTTOM-h+cyl
            rise=26*(1-np.exp(-age/2.2))*1.0+3*age
            wind=9*age+2.5*age**1.3
            tur=(4+1.8*age)*np.sin(age*1.3+ph)*min(age/1.2,1)
            px=xs_+wind+tur; py=ys_-rise+ 2.5*np.sin(age*1.9+ph2)*min(age,2)
            r=(3+4.2*age**0.85)*sz*(0.8+0.25*chuff)
            a_=min(age/0.35,1)*0.85*chuff**0.5*(1-age/7)**1.4
            for j in range(3):
                ox=r*0.55*np.cos(ph+j*2.1); oy=r*0.45*np.sin(ph2+j*2.1)
                if 0<=px+ox<W and 0<=py+oy<H: cv2.circle(smoke,(int(px+ox),int(py+oy)),int(r*(0.75 if j else 1.0)),a_*(0.55 if j else 0.8),-1,cv2.LINE_AA)
    smoke=np.clip(cv2.GaussianBlur(smoke,(0,0),3.0),0,1)
    # break up with drifting texture so it billows rather than a flat smear
    tx=cv2.resize(noise[:,:W//2],(W,H))
    sh=int(t*14)%max(1,W//2)
    tex=np.roll(tx,sh,axis=1)
    smoke=smoke*(0.62+0.5*tex)
    smoke=np.clip(smoke,0,0.92)
    f=f*(1-smoke[...,None]*0.9)+smoke[...,None]*0.9*(np.array([95,100,115],np.float32)*m)
    f=f*vig[...,None]
    # slow shimmer on water
    out.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
out.stdin.close(); out.wait()
