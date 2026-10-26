import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "every snowflake that escapes the globe adds two more" - a snow globe on a snowy windowsill.
import sys, os, math, subprocess
from collections import defaultdict
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
FW,FH=1080,1920                                  # simulation units = full-res pixels
W,H,FPS=int(FW*S),int(FH*S),30
DUR=float(os.environ.get('DUR','60')); CAP=int(os.environ.get('CAP','900'))
CX,CY,RG=600.0,1300.0,330.0                      # globe centre and inner radius
BASE_T,BASE_B,BASE_HW=1570,1700,200              # wooden base
SILL=1700.0                                      # sill surface line where the drift builds
R=9.0                                            # flake collision radius
GAPW=math.radians(float(os.environ.get('GAPDEG','16')))
SWIRL=float(os.environ.get('SWIRL','0.55')); SINK=float(os.environ.get('SINK','70'))
rng=np.random.RandomState(int(os.environ.get('SEED','3')))
def crack_ang(t):                                # the crack sweeps round the lower half of the globe
    return math.radians(90+75*math.sin(t*0.42+1.1))
def crack_open(a):                               # hidden (blocked) where the base covers the glass
    y=CY+RG*math.sin(a); return y<BASE_T-6
C=np.array([CX,CY])
# ---------------- simulation
SUB=2; dt=1/(FPS*SUB)
pos=np.zeros((0,2)); vel=np.zeros((0,2)); born=np.zeros(0); size=np.zeros(0)
def spawn(n,t):
    global pos,vel,born,size
    n=max(0,min(n,CAP-len(pos)))
    if n==0: return 0
    a=rng.uniform(0,2*math.pi,n); r=RG*0.55*np.sqrt(rng.uniform(0,1,n))
    p=np.stack([CX+r*np.cos(a),CY-RG*0.35+r*np.sin(a)*0.5],1)
    pos=np.vstack([pos,p]); vel=np.vstack([vel,rng.normal(0,40,(n,2))])
    born=np.concatenate([born,np.full(n,t)]); size=np.concatenate([size,rng.uniform(0.7,1.25,n)])
    return n
spawn(2,0.0)
OUT=[]                   # escaped: [x,y,vx,vy,t,landed_t]
EVENTS=[]; count=2
FR=[]; FRC=[]; FRA=[]; OUTPOS=[]
drift=np.zeros(FW,np.float32)
def collide():
    global pos,vel
    if len(pos)<2: return
    cell=2*R; gi=np.floor(pos/cell).astype(np.int64)
    b=defaultdict(list)
    for i,(x,y) in enumerate(gi): b[(x,y)].append(i)
    I=[];J=[]
    for (x,y),lst in b.items():
        for dx,dy in ((0,0),(1,-1),(1,0),(1,1),(0,1)):
            nb=b.get((x+dx,y+dy))
            if not nb: continue
            for a in lst:
                for c in nb:
                    if (dx,dy)==(0,0) and c<=a: continue
                    I.append(a); J.append(c)
    if not I: return
    I=np.array(I); J=np.array(J); d=pos[J]-pos[I]; dist=np.sqrt((d**2).sum(1))+1e-6
    h=dist<2*R
    if not h.any(): return
    I,J,d,dist=I[h],J[h],d[h],dist[h]; n=d/dist[:,None]; ov=(2*R-dist)[:,None]*0.5
    np.add.at(pos,I,-n*ov); np.add.at(pos,J,n*ov)
    rv=((vel[J]-vel[I])*n).sum(1); imp=np.where(rv<0,-rv*0.8,0)[:,None]*n*0.5
    np.add.at(vel,I,-imp); np.add.at(vel,J,imp)
for f in range(int(DUR*FPS)):
    for s in range(SUB):
        t=(f*SUB+s)*dt
        n=len(pos)
        if n:
            rel=pos-C
            flow=np.stack([-rel[:,1],rel[:,0]],1)*SWIRL+np.array([0,SINK])   # slow swirl in the water + sinking
            vel+=(flow-vel)*min(1,1.6*dt)+rng.normal(0,170*math.sqrt(dt),(n,2))
            pos+=vel*dt
            rel=pos-C; dist=np.sqrt((rel**2).sum(1))+1e-6
            hit=dist>RG-R
            if hit.any():
                a=crack_ang(t); ang=np.arctan2(rel[:,1],rel[:,0])
                da=np.abs((ang-a+math.pi)%(2*math.pi)-math.pi)
                esc=hit&(da<GAPW/2-R/RG)&crack_open(a)
                ins=hit&~esc
                nrm=rel[ins]/dist[ins,None]
                pos[ins]=C+nrm*(RG-R); vn=(vel[ins]*nrm).sum(1,keepdims=True); vel[ins]-=np.maximum(vn,0)*nrm*1.6
                # base top blocks the bottom of the glass too
                if esc.any():
                    idx=np.where(esc)[0]
                    for i in idx:
                        nrm_=rel[i]/dist[i]
                        OUT.append([pos[i,0]+nrm_[0]*12,pos[i,1]+nrm_[1]*12,nrm_[0]*90,max(20.0,nrm_[1]*90),t,None])
                    keep=~esc; pos,vel,born,size=pos[keep],vel[keep],born[keep],size[keep]
                    for _ in idx:
                        count+=spawn(2,t); EVENTS.append((t,count))
            if s==0 and len(pos)>1: collide()
        # escaped flakes fall to the sill and settle into the drift
        for o in OUT:
            if o[5] is not None: continue
            o[3]+=260*dt; o[2]*=0.995; o[0]+=o[2]*dt+18*math.sin(t*3+o[4]*7)*dt; o[1]+=o[3]*dt
            xi=int(min(FW-1,max(0,o[0])))
            if o[1]>=SILL-drift[xi] and not (CX-BASE_HW<o[0]<CX+BASE_HW and o[1]<BASE_B):
                o[5]=t; xs=np.arange(FW)
                drift[:]+=0.7*np.exp(-((xs-o[0])/26.0)**2)
    OUTPOS.append([(o[0],o[1]) for o in OUT if o[5] is None])
    FR.append((pos.copy(),born.copy(),size.copy())); FRC.append(count); FRA.append((crack_ang(f/FPS),drift.copy()))
print('final inside',len(pos),'total',count,'escapes',len(OUT),'first',[round(o[4],1) for o in OUT[:6]],'drift max',round(float(drift.max()),1),flush=True)
if os.environ.get('SIMONLY'): sys.exit()

# ---------------- background: Lewis's still, vertical crop, snow falling outside, lamp breathing
src=cv2.imread(O+'still.jpg').astype(np.float32)
x0=int(os.environ.get('X0','380')); cw=int(src.shape[0]*9/16)
bg0=cv2.resize(src[:,x0:x0+cw],(W,H),interpolation=cv2.INTER_CUBIC)
bg0=cv2.GaussianBlur(bg0,(0,0),1.6*S)
# window pane region (bright, low saturation) for snow outside
hsv=cv2.cvtColor(np.clip(bg0,0,255).astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
pane=((hsv[...,1]<70)&(hsv[...,2]>60)).astype(np.float32)
pane[int(H*0.80):]=0; pane=cv2.GaussianBlur(cv2.morphologyEx(pane,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8)),(0,0),6*S)
warm=np.clip((bg0[...,2]-bg0[...,0]-30)/90,0,1)*np.clip(bg0.mean(2)/160,0,1)
def snow_tile(n,rad,seed,blur):
    r=np.random.RandomState(seed); T=512; t=np.zeros((T,T),np.float32)
    for _ in range(n):
        x,y=r.randint(0,T,2)
        for ox in (-T,0,T):
            for oy in (-T,0,T): cv2.circle(t,(x+ox,y+oy),rad,r.uniform(0.4,1),-1,cv2.LINE_AA)
    return cv2.GaussianBlur(t,(0,0),blur)
ST=[(snow_tile(140,1,1,0.8),40,0.55),(snow_tile(60,2,2,1.5),75,0.6),(snow_tile(20,4,3,3.0),130,0.5)]
def bgframe(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in ST:
        oy=int(t*sp*S*2)%512; ox=int(18*math.sin(t*0.6+sp)*S)%512
        acc+=np.tile(np.roll(np.roll(tl,oy,0),ox,1),(H//512+1,W//512+1))[:H,:W]*a
    img=bg0*(1+warm*(0.035*math.sin(t*2.3)+0.02*math.sin(t*7.1)))[...,None]
    return img+(acc*pane)[...,None]*np.array([235,232,228],np.float32)*0.55
# ---------------- globe drawing (in output pixels)
def P(v): return v*S
cx,cy,rg=P(CX),P(CY),P(RG)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
dx=(xx-cx)/rg; dy=(yy-cy)/rg; rr=np.sqrt(dx*dx+dy*dy)
inside=np.clip((1-rr)*rg/1.5,0,1)
# lens: inverted, magnified view of what is behind the globe
k=np.clip(1-rr**2,0,1)
mapx=(cx-(xx-cx)*0.55*(0.6+0.4*k)).astype(np.float32); mapy=(cy-(yy-cy)*0.55*(0.6+0.4*k)-rg*0.25).astype(np.float32)
base_mask=np.zeros((H,W),np.float32)
pts=np.array([[P(CX-BASE_HW*0.78),P(BASE_T)],[P(CX+BASE_HW*0.78),P(BASE_T)],[P(CX+BASE_HW),P(BASE_B)],[P(CX-BASE_HW),P(BASE_B)]],np.int32)
cv2.fillPoly(base_mask,[pts],1.0,cv2.LINE_AA); base_mask=cv2.GaussianBlur(base_mask,(0,0),0.8)
bx=(xx-P(CX))/P(BASE_HW)
base_col=np.stack([14+8*(1-bx**2),24+14*(1-bx**2),42+26*(1-bx**2)],-1).astype(np.float32)     # dark wood, lit from the lamp side
base_col*=(1+0.35*np.clip(-bx,0,1))[...,None]
grain=(np.sin(yy*0.9+np.sin(xx*0.05)*3)*0.06+1)[...,None]; base_col*=grain
band=((yy>P(BASE_T+12))&(yy<P(BASE_T+22))).astype(np.float32)[...,None]
base_col=base_col*(1-band)+np.array([70,130,170],np.float32)*band*(0.8+0.4*np.clip(-bx,0,1))[...,None]  # brass band
rim=np.exp(-((rr-1)*rg/2.2)**2)
fres=np.clip((rr-0.75)/0.25,0,1)**2*inside
spec=np.exp(-(((dx+0.42)/0.16)**2+((dy+0.48)/0.11)**2))+0.6*np.exp(-(((dx-0.50)/0.05)**2+((dy+0.2)/0.35)**2))
warmrefl=np.exp(-(((dx+0.75)/0.18)**2+((dy-0.05)/0.25)**2))   # reflection of the lamp on the left side
FLK=np.array([248,246,244],np.float32)
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(96*S)))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(48*S)))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,P(190)),"every snowflake that escapes",font=FS,fill=255,anchor='mm')
d.text((W/2,P(252)),"the globe adds two more",font=FS,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255; HSH=cv2.GaussianBlur(HOOK,(0,0),8*S)
_cc={}
def count_mask(n):
    if n not in _cc:
        _cc.clear(); im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,P(375)),f"{n:,}",font=FB,fill=255,anchor='mm',features=['tnum'])
        _cc[n]=np.asarray(im,np.float32)/255
    return _cc[n]
def frame(f):
    t=f/FPS
    img=bgframe(t)
    a_cr,dr=FRA[f]
    # drift on the sill (drawn before the globe so the base sits on it)
    hpx=cv2.resize(dr[None,:],(W,1))[0]*S
    top=P(SILL)-hpx
    dm=np.clip((yy-top[None,:])/1.5,0,1)*np.clip((P(SILL)+P(14)-yy)/P(14),0,1)*(hpx[None,:]>0.3)
    dm=cv2.GaussianBlur(dm.astype(np.float32),(0,0),1.2)
    shade=np.clip(1-(yy-top[None,:])/np.maximum(hpx[None,:],1)*0.35,0.6,1)
    img=img*(1-dm[...,None])+dm[...,None]*np.array([235,228,222],np.float32)*shade[...,None]
    # glass interior: refracted room + flakes
    lens=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    lens=cv2.GaussianBlur(lens,(0,0),1.2*S*2)*np.array([1.02,0.97,0.92],np.float32)*0.82
    p,bn,sz=FR[f]
    lay=np.zeros((H,W),np.float32); glow=np.zeros((H//2,W//2),np.float32)
    for i in range(len(p)):
        a=min(1.0,(t-bn[i])/0.35)
        cv2.circle(lay,(int(P(p[i,0])*4),int(P(p[i,1])*4)),max(2,int(P(R)*0.95*sz[i]*4)),a,-1,cv2.LINE_AA,2)
    soft=cv2.GaussianBlur(lay,(0,0),0.9); soft=soft/(1+soft*0.25)
    g=cv2.resize(cv2.GaussianBlur(cv2.resize(soft,(W//4,H//4),interpolation=cv2.INTER_AREA),(0,0),3),(W,H))
    lm=lens.mean((0,1),keepdims=True); lens=lens*0.55+lm*0.45
    interior=lens+soft[...,None]*FLK*1.0+g[...,None]*np.array([110,120,130],np.float32)*1.4
    img=img*(1-inside[...,None])+interior*inside[...,None]
    # base
    img=img*(1-base_mask[...,None])+base_col*base_mask[...,None]
    # glass: fresnel edge, rim line, highlights, lamp reflection
    img+= (fres*0.22)[...,None]*np.array([235,235,240],np.float32)*inside[...,None]
    img+= (rim*0.55)[...,None]*np.array([220,225,230],np.float32)*(1-base_mask)[...,None]*(yy<P(BASE_T)+2)[...,None]
    img+= (spec*inside*0.85)[...,None]*np.array([255,252,248],np.float32)
    img+= (warmrefl*inside*0.30)[...,None]*np.array([70,150,255],np.float32)
    # the crack: a small dark jagged notch in the rim
    if crack_open(a_cr):
        ex,ey=P(CX+RG*math.cos(a_cr)),P(CY+RG*math.sin(a_cr))
        cr=np.zeros((H,W),np.float32)
        for j in (-1,0,1):
            aa=a_cr+j*GAPW*0.35
            cv2.line(cr,(int(P(CX+(RG-14)*math.cos(aa))),int(P(CY+(RG-14)*math.sin(aa)))),(int(P(CX+(RG+6)*math.cos(aa+0.03))),int(P(CY+(RG+6)*math.sin(aa+0.03)))),1.0,max(1,int(P(3))),cv2.LINE_AA)
        cr=cv2.GaussianBlur(cr,(0,0),0.6)
        img=img*(1-cr[...,None]*0.75)+cr[...,None]*np.array([255,250,240],np.float32)*0.25
    # falling escaped flakes, in front of everything
    fl=np.zeros((H,W),np.float32)
    for x_,y_ in OUTPOS[f]:
        cv2.circle(fl,(int(P(x_)*4),int(P(y_)*4)),max(2,int(P(R)*0.8*4)),1.0,-1,cv2.LINE_AA,2)
    fl=cv2.GaussianBlur(fl,(0,0),0.9)
    img=img*(1-fl[...,None])+fl[...,None]*FLK
    img=img*(1-HSH[...,None]*0.45)
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([240,242,248],np.float32)
    cm=count_mask(FRC[f]); img=img*(1-cm[...,None])+cm[...,None]*np.array([240,242,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)

# ---------------- sound: soft bell notes per escape over lo-fi
SR=48000
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*3.5)*np.minimum(1,tt*800)
    s_=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.1) for h,a in ((1,1),(2.76,0.35),(5.4,0.12)))   # bell-like partials
    return (s_*env*amp).astype(np.float32)
def audio(out):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'pondmusic/between_the_raindrops.wav','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    n=len(mus); root=329.63*2                      # track is in E
    sc=[root*2**(k/12) for k in (0,2,4,7,9,12,14,16)]
    m=np.zeros(n,np.float32); last=-1; k=0
    for t_,c in EVENTS:
        if t_-last<0.07: continue
        last=t_; k+=1
        s_=note(sc[(k*3)%len(sc)],1.4,0.08*(1-0.5*min(1,c/700))); a=int(t_*SR); q=min(len(s_),n-a); m[a:a+q]+=s_[:q]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    mix=mus*0.65+notes*(rms(mus)*0.65/rms(notes))
    f_=int(2*SR); mix[-f_:]*=np.linspace(1,0,f_)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','): cv2.imwrite(O+f'gs_{tt}.jpg',frame(min(len(FR)-1,int(float(tt)*FPS))))
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',O+'gv.mp4'],stdin=subprocess.PIPE)
    for f in range(len(FR)): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); audio(O+'ga.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'gv.mp4','-i',O+'ga.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
