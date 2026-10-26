import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess, sys
U=ZT+'uploads/'
W,H=960,540; FPS=24; DUR=40; SC=0.1407; BOTTOM=304
bg=cv2.resize(cv2.imread(U+'e045e710-image.jpg'),(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
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
trains=[dict(spr=P,t0=3,dir=1,mir=True,cx=chim(P,True)),dict(spr=F,t0=22,dir=-1,mir=False,cx=chim(F,False))]
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
    f=bg*m
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
        # smoke
        cxl,cyl=tr['cx']; sxw=x+cxl; syw=y+cyl
        for k in range(int((t-tr['t0'])/0.09)+1):
            ts=tr['t0']+k*0.09; age=t-ts
            if age<0 or age>5: continue
            px=(x+(0 if True else 0))  # placeholder
            # spawn position follows train at spawn time
            xs_=pos(tr,ts)+cxl; ys_=BOTTOM-h+cyl
            px=xs_-tr['dir']*0.0+(-tr['dir']*(-1)*0)  # world-fixed after spawn
            px=px+ 6*age+ 2*np.sin(age*1.5+k); py=ys_-14*age
            r=5+age*4.5; a_=1.0*(1-age/5)**1.3
            if 0<=px<W and 0<=py<H: cv2.circle(smoke,(int(px),int(py)),int(r),a_,-1,cv2.LINE_AA)
    smoke=cv2.GaussianBlur(smoke,(0,0),2.2)
    f=f*(1-smoke[...,None]*0.9)+smoke[...,None]*0.9*(np.array([150,158,175],np.float32)*m)
    f=f*vig[...,None]
    # slow shimmer on water
    out.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
out.stdin.close(); out.wait()
