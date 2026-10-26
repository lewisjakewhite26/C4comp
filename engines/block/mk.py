import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, cv2, subprocess, math, sys
D=ZT+'block/'
base=cv2.imread(D+'base.png').astype(np.float32)          # 1080x1934 BGR
H0,W=base.shape[:2]; FPS=24; T=12.0; NF=int(T*FPS); TAU=2*math.pi
def om(k): return TAU*k/T
gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H0,dtype=np.float32))
rng=np.random.RandomState(9)
def soft(m,s): return cv2.GaussianBlur(m.astype(np.float32),(0,0),s)

# ---------- dark window panes that will light up
def pane(x0,y0,x1,y1):
    m=np.zeros((H0,W),np.float32); r=base[y0:y1,x0:x1].mean(2)
    sub=(r<28).astype(np.uint8)
    sub=cv2.morphologyEx(sub,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    sub=cv2.erode(sub,np.ones((3,3),np.uint8)); m[y0:y1,x0:x1]=sub; return soft(m,0.8)
PA=pane(604,1171,721,1286); PB=pane(150,852,276,966)
def interior(x0,y0,x1,y1,tint,flip=False,dim=0.55):
    # borrow the real lit flat (left two panes) so the new rooms are photographic
    src=base[816:905,470:620].copy()
    if flip: src=src[:,::-1]
    h,w=y1-y0,x1-x0
    room=cv2.resize(src,(w,h),interpolation=cv2.INTER_AREA)
    room=cv2.GaussianBlur(room,(0,0),1.6)                       # seen through glass + distance
    yy=np.linspace(0,1,h)[:,None]; xx=np.linspace(0,1,w)[None,:]
    nz=cv2.GaussianBlur(np.random.RandomState(5).rand(h,w).astype(np.float32),(0,0),0.7)
    net=0.93+0.07*(nz-0.5)*2
    folds=0.88+0.12*np.sin(xx*w/4.5+np.sin(yy*7))
    room=room*(net*folds)[...,None]*np.array(tint,np.float32)*dim
    big=np.zeros((H0,W,3),np.float32); big[y0:y1,x0:x1]=room; return big
IA=interior(595,1160,730,1295,(0.95,1.0,1.05))
IB=interior(140,840,285,975,(1.35,0.85,0.55),flip=True,dim=0.45)   # cool TV-lit room (BGR)
# shadow of someone walking past: the real figure's outline, softened by the net curtain
def person_mask(cx,y_top=1188):
    m=np.zeros((H0,W),np.float32)
    cv2.ellipse(m,(int(cx),y_top+13),(10,12),0,0,360,1,-1,cv2.LINE_AA)                 # head
    pts=np.array([[cx-9,y_top+24],[cx+9,y_top+24],[cx+27,y_top+40],[cx+30,y_top+110],[cx-30,y_top+110],[cx-27,y_top+40]],np.int32)
    cv2.fillPoly(m,[pts],1,cv2.LINE_AA)                                                  # shoulders + body
    return soft(m,4.0)*PA
# ---------- events within the 12 s loop
def env(t,on,off,ramp=0.12):
    if t<on or t>off: return 0.0
    return min(1.0,(t-on)/ramp,(off-t)/0.25)
TVB=[0.55+0.45*rng.rand() for _ in range(48)]
def tv_b(t):  # piecewise random brightness, loopable
    i=int(t/0.25)%48; return TVB[i]

# ---------- TV in the main flat
TVM=np.zeros((H0,W),np.float32); TVM[915:982,686:718]=1; TVM=soft(TVM,1.0)
TVGLOW=soft(np.where((gx>350)&(gx<750)&(gy>810)&(gy<1090),np.exp(-(((gx-700)/150)**2+((gy-950)/110)**2)),0),3)
# ---------- steam
SM=np.zeros((H0,W),np.float32); SM[860:975,505:575]=1; SM=soft(SM,10)
NZ=cv2.GaussianBlur(rng.randn(256,256).astype(np.float32),(0,0),6); NZ/=NZ.std()
# ---------- breathing (person torso)
BR=np.zeros((H0,W),np.float32); cv2.ellipse(BR,(480,1010),(55,70),0,0,360,1,-1); BR=soft(BR,12)
# ---------- wet road shimmer
RD=((gx>880)&(gy>1380)).astype(np.float32)*np.clip((base.mean(2)-40)/80,0,1); RD=soft(RD,3)
# ---------- rain layers (tiles scroll whole wraps per loop)
def rain_tile(n,len_,seed,th):
    r=np.random.RandomState(seed); t=np.zeros((512,512),np.float32)
    for _ in range(n):
        x,y=r.randint(0,512,2); l=r.randint(len_//2,len_)
        cv2.line(t,(x,y),(x+int(l*0.12),y+l),r.uniform(0.5,1),th,cv2.LINE_AA)
    t=np.tile(t,(2,2)); return t
RT=[rain_tile(260,40,1,1),rain_tile(110,70,2,1),rain_tile(40,110,3,2)]
RSPEED=[512*3/T*1, 512*5/T, 512*8/T]      # whole wraps per loop
lum=cv2.GaussianBlur(base.mean(2),(0,0),25)/255.0
RVIS=(0.04+2.2*lum**1.4)[...,None]            # rain catches the light near lamps and the window
def rain(t):
    acc=np.zeros((H0,W),np.float32)
    for k,(tl,sp,a) in enumerate(zip(RT,RSPEED,(0.22,0.30,0.38))):
        oy=int(t*sp)%512; ox=int(t*sp*0.12)%512
        big=np.tile(tl[oy:oy+512,ox:ox+512] if False else np.roll(np.roll(tl,oy,0),ox,1)[:512,:512],(4,3))[:H0,:W]
        acc+=big*a
    return acc[...,None]*RVIS*np.array([215,220,230],np.float32)*0.22

def frame(f):
    t=f/FPS
    img=base.copy()
    # breathing + steam via one displacement field
    dy=BR*1.3*math.sin(om(3)*t)
    ny=((gy*0.5+t*256/T*2)%256).astype(np.float32); nx=(gx*0.5%256).astype(np.float32)
    n1=cv2.remap(NZ,nx,ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    dx=SM*5.0*n1; dy=dy+SM*3.0*(1+0.5*np.sin(om(2)*t))
    img=cv2.remap(img,(gx-dx).astype(np.float32),(gy-dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    st=SM*(0.10+0.06*n1)
    img+=st[...,None]*np.array([200,215,225],np.float32)*0.5
    # TV flicker and its glow in the room
    b=tv_b(t)
    img=img*(1-TVM[...,None]*0.35)+TVM[...,None]*img*(0.65+0.7*b)
    img+=TVGLOW[...,None]*np.array([40,18,0],np.float32)*(b-0.75)
    # road shimmer
    sh=0.85+0.25*cv2.remap(NZ,((gx*0.7+t*256/T)%256).astype(np.float32),((gy*1.6)%256).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    img=img*(1-RD[...,None])+img*RD[...,None]*sh[...,None]
    # window A: light on 2.0-7.2 s, someone walks past 3.6-5.6 s
    ea=env(t,2.0,7.2,0.05)
    if t>2.0 and t<2.35: ea*=0.4+0.6*(int(t*40)%2)     # bulb catches on
    if ea>0:
        lit=IA*ea
        if 3.6<t<5.6:
            u=(t-3.6)/2.0; cx=600+u*130
            pm=person_mask(cx)
            lit=lit*(1-pm[...,None]*0.7)
        img=img*(1-PA[...,None]*ea)+lit*PA[...,None]
    # window B: TV comes on 8.2-11.2 s, flickering blue
    eb=env(t,8.2,11.2,0.2)
    if eb>0:
        fb=0.6+0.4*tv_b(t+3.1)
        img=img*(1-PB[...,None]*eb)+IB*PB[...,None]*eb*fb
    img+=rain(t)
    return np.clip(img[14:14+1920],0,255).astype(np.uint8)

if __name__=='__main__':
    if sys.argv[1]=='still':
        for t in map(float,sys.argv[2].split(',')): cv2.imwrite(D+f's_{t:g}.png',frame(int(t*FPS)))
    else:
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x1920','-r','24','-i','-','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',D+'video.mp4'],stdin=subprocess.PIPE)
        for f in range(NF): p.stdin.write(frame(f).tobytes())
        p.stdin.close(); p.wait()
