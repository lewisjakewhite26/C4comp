import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "every firefly that escapes lights a window" - a jar of 170 fireflies on a windowsill, a tower block opposite in a power cut.
# Each firefly that slips out through the sliding gap in the lid flies across the road and switches on one flat's light.
import sys, os, math, subprocess
from collections import defaultdict
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
DUR=60.0
K=1080/1152                                    # source image -> full-res output
# ---------------- windows (source pixels), 10 columns x 17 floors
XC=[(80,158),(182,270),(288,355),(380,470),(475,565),(585,665),(678,780),(785,862),(888,985),(993,1073)]
BAL={1,3,6,8}
ROWS=[159,241,320,401,482,563,644,725,806,889,971,1055,1137,1222,1304,1385,1469]
WIN=[]
for r,y in enumerate(ROWS):
    yc=y+17+8*r/16
    for c,(x0,x1) in enumerate(XC):
        y0,y1=yc-24,yc+24+(14 if c in BAL else 0)
        WIN.append((x0*K,y0*K,x1*K,y1*K,r,c))
NW=len(WIN)
rng=np.random.RandomState(int(os.environ.get('SEED','7')))
# ---------------- jar physics (full-res units)
# mason jar sitting on the ledge (full-res px): lid 1478-1510, neck, shoulder 1528-1580, body to the base at 1846
JC=885.0; LIDT=1478; LIDY=1510; NECK1=1528; SHO1=1580; JY1=1826; BASEY=1846; HWB=130.0; HWN=84.0
JX0,JX1,JY0=JC-HWB,JC+HWB,LIDY
def hw(y):                                       # outer half-width of the glass at height y
    y=np.asarray(y,np.float64)
    u=np.clip((y-NECK1)/(SHO1-NECK1),0,1); w=HWN+(HWB-HWN)*(0.5-0.5*np.cos(np.pi*u))
    rb=26.0; dy=np.clip(y-(BASEY-rb),0,rb); w=w-(rb-np.sqrt(np.maximum(rb*rb-dy*dy,0)))
    return w
def ihw(y): return hw(y)-9
GW0=float(os.environ.get('GW0','12')); GW1=float(os.environ.get('GW1','90')); GP=float(os.environ.get('GP','3.2')); R=6.0
def gw(t): return GW0+(GW1-GW0)*min(1,t/52)**GP
LIFT=float(os.environ.get('LIFT','-260')); DRAG=0.6; WANDER=240.0; RESTI=0.7
def gap_x(t):
    span=max(0,(2*(HWN-9)-gw(t)-16)/2); mid=JC
    return mid+span*math.sin(t*0.6+0.4)
SUB=2; dt=1/(FPS*SUB)
pos=np.stack([rng.uniform(JC-90,JC+90,NW),rng.uniform(SHO1,JY1-R,NW)],1)
vel=rng.normal(0,60,(NW,2)); ph=rng.uniform(0,6.3,NW)
def collide():
    global pos,vel
    if len(pos)<2: return
    gi=np.floor(pos/(2*R)).astype(np.int64); b=defaultdict(list)
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
    rv=((vel[J]-vel[I])*n).sum(1); imp=np.where(rv<0,-rv*0.85,0)[:,None]*n*0.5
    np.add.at(vel,I,-imp); np.add.at(vel,J,imp)
def walls(top):
    global pos,vel
    m=pos[:,1]>JY1-R; pos[m,1]=JY1-R; vel[m,1]=-np.abs(vel[m,1])*RESTI
    if top is not None:
        m=pos[:,1]<top; pos[m,1]=top; vel[m,1]=np.abs(vel[m,1])*RESTI
    lim=ihw(np.maximum(pos[:,1],LIDY))-R
    dx=pos[:,0]-JC; m=np.abs(dx)>lim
    if m.any():
        sg=np.sign(dx[m]); pos[m,0]=JC+sg*lim[m]; vel[m,0]=-sg*np.abs(vel[m,0])*RESTI
        # sliding up under the shoulder: nudge inwards so they funnel to the neck
        vel[m,0]-=sg*40
# settle the crowd before t=0 (packed up under the lid)
for _ in range(240):
    vel+=np.array([0,LIFT])*dt+rng.normal(0,WANDER*math.sqrt(dt),(len(pos),2)); vel*=(1-DRAG*dt); pos+=vel*dt
    walls(LIDY+R)
    collide()
order=list(rng.permutation(NW))
LASTW=12*10+4                                   # make a mid-height, central flat the final one
order.remove(LASTW); order.append(LASTW)
ESC=[]                                          # [x0,y0,t,window,ctrlx,ctrly,dur,phase]
FR=[]
for f in range(int(DUR*FPS)):
    for s in range(SUB):
        t=(f*SUB+s)*dt; n=len(pos)
        if n==0: break
        vel+=np.array([0,LIFT])*dt+rng.normal(0,WANDER*math.sqrt(dt),(n,2)); vel*=(1-DRAG*dt)
        sp=np.sqrt((vel**2).sum(1)); vel*=np.minimum(1,380/(sp+1e-6))[:,None]
        pos+=vel*dt
        walls(None)
        gx=gap_x(t); atlid=pos[:,1]<LIDY+R; ingap=np.abs(pos[:,0]-gx)<gw(t)/2-R*0.5
        out=atlid&ingap&(pos[:,1]<LIDY-R); blk=atlid&~ingap
        pos[blk,1]=LIDY+R; vel[blk,1]=np.abs(vel[blk,1])*RESTI
        if out.any():
            for i in np.where(out)[0]:
                w=order[len(ESC)]; x0,y0,x1,y1=WIN[w][:4]; tx,ty=(x0+x1)/2,(y0+y1)/2
                dist=math.hypot(tx-pos[i,0],ty-pos[i,1])
                ESC.append([pos[i,0],pos[i,1],t,w,(pos[i,0]+tx)/2+rng.uniform(-220,220),min(pos[i,1],ty)-rng.uniform(60,260),(0.9+dist/900)*rng.uniform(0.85,1.15),ph[i]])
            keep=~out; pos,vel,ph=pos[keep],vel[keep],ph[keep]
        if s==0: collide()
    FR.append((pos.copy(),ph.copy()))
ESC=np.array(ESC); TON=np.full(NW,1e9)
for e in ESC: TON[int(e[3])]=e[2]+e[6]
tlast=TON[TON<1e8].max() if (TON<1e8).any() else DUR
print('escaped',len(ESC),'of',NW,'first',np.round(ESC[:5,2],2).tolist() if len(ESC) else None,'last lit',round(float(tlast),1),flush=True)
for q in (25,50,100,150,169):
    if len(ESC)>q: print('  window',q+1,'lit at',round(float(np.sort(TON)[q]),1))
if os.environ.get('SIMONLY'): sys.exit()
TSL=tlast+1.0                                   # streetlight comes back on
# ---------------- images
def ld(p): return cv2.resize(cv2.imread(O+p),(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
DARK=ld('dark.png'); LIT=ld('lit.png')
lum=cv2.cvtColor(DARK.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
# per-window light patches (additive light + soft spill on the concrete)
def kelvin(k):                                   # colour temperature -> BGR (Tanner Helland approximation)
    t=k/100.0
    r=255.0 if t<=66 else 329.7*((t-60)**-0.1332)
    g=99.47*math.log(t)-161.12 if t<=66 else 288.1*((t-60)**-0.0755)
    b=255.0 if t>=66 else (0.0 if t<=19 else 138.5*math.log(t-10)-305.0)
    return np.clip(np.array([b,g,r],np.float32),0,255)
def pick_light():
    u=rng.rand()
    if u<0.75: k=rng.uniform(2200,3000)              # most homes: warm bulbs and lamps
    elif u<0.975: k=rng.uniform(3000,4000)           # some neutral-white LEDs
    else: k=rng.uniform(4000,4800)                   # the odd cool one
    nm='fairy' if (k<2700 and rng.rand()<0.04) else 'bulb'
    return nm,k
BROKEN=int(os.environ.get('BROKEN','57'))        # the one flat with a dodgy flickering light
def P(v): return v*S
PATCH=[]
for w,(x0,y0,x1,y1,r,c) in enumerate(WIN):
    m=int(P(26)); X0=max(0,int(P(x0))-m); Y0=max(0,int(P(y0))-m); X1=min(W,int(P(x1))+m+1); Y1=min(H,int(P(y1))+m+1)
    hh,ww=Y1-Y0,X1-X0
    yy,xx=np.mgrid[0:hh,0:ww].astype(np.float32)
    rect=np.zeros((hh,ww),np.float32); cv2.rectangle(rect,(int(P(x0))-X0,int(P(y0))-Y0),(int(P(x1))-X0,int(P(y1))-Y0),1.0,-1)
    l=lum[Y0:Y1,X0:X1]; inside=l[rect>0.5]
    thr=np.percentile(inside,62)
    glass=rect*np.clip((thr-l)/12+0.5,0,1)                          # dark panes light up; frames/mullions stay
    glass=cv2.GaussianBlur(glass,(0,0),0.5)
    nm,kk=pick_light(); col=kelvin(kk)
    col=np.array(col,np.float32)*rng.uniform(0.92,1.06,3)
    br=rng.uniform(0.38,0.80)*(0.75 if kk<2500 else 1)
    lx=rng.uniform(P(x0),P(x1))-X0; ly=rng.uniform(P(y0),P(y1))-Y0          # where the light sits in the room
    grad=0.55+0.45*np.exp(-((xx-lx)**2+(yy-ly)**2)/(2*(P(x1-x0)*0.6)**2))
    tex=np.ones((hh,ww),np.float32)
    cur=rng.choice(['none','half','blinds','net','none'])
    if cur=='half':
        cut=rng.uniform(0.35,0.65)*(P(x1)-P(x0))+P(x0)-X0
        tex*=np.where((xx<cut)^(rng.rand()<0.5),0.35,1.0)
    elif cur=='blinds':
        tex*=0.65+0.35*(np.sin(yy/max(1.2,P(3.2))*math.pi)>0)
    elif cur=='net':
        tex*=0.8; glass=cv2.GaussianBlur(glass,(0,0),P(2))
    if nm=='fairy':
        dots=np.zeros((hh,ww),np.float32)
        for k_ in range(int(10*S*2)): cv2.circle(dots,(int(rng.uniform(P(x0),P(x1))-X0),int(P(y0)-Y0+P(4)+rng.uniform(0,P(6)))),max(1,int(P(2.5))),1.0,-1)
        tex=tex*0.6+dots*1.2
    L=glass*tex*grad*br
    spill=cv2.GaussianBlur(L,(0,0),P(10))*0.55+cv2.GaussianBlur(L,(0,0),P(3))*0.25
    if c in BAL:
        bal=np.zeros((hh,ww),np.float32); cv2.rectangle(bal,(int(P(x0))-X0,int(P(y1))-Y0),(int(P(x1))-X0,int(P(y1+22))-Y0),1.0,-1)
        spill+=cv2.GaussianBlur(bal,(0,0),P(4))*0.18*br
    mode='broken' if w==BROKEN else rng.choice(['instant','instant','stutter'])
    PATCH.append((Y0,Y1,X0,X1,L[...,None]*col,spill[...,None]*col,mode,rng.uniform(0,6.3)))
def inten(w,t):
    a=t-TON[w]
    if a<0: return 0.0
    Y0,Y1,X0,X1,L,Sp,mode,p=PATCH[w]
    if mode=='stutter':
        if a<0.6: return [1,0.1,0.8,0.05,1,1][int(a/0.1)]
    base=min(1.0,a/0.08)
    flash=1+0.35*math.exp(-a/0.25)
    if mode=='broken':                           # irregular dropouts, like a failing bulb
        q=int(t*12); h=((q*2654435761)^(q>>3))%100
        base*=0.15 if h<14 else (0.6 if h<22 else 1.0)
    return base*flash
# jar layers
# ---- glass jar look (precomputed in a region of interest)
RX0,RX1,RY0,RY1=max(0,int(P(JC-420))),min(W,int(P(JC+420))),int(P(LIDT-20)),min(H,int(P(BASEY+74)))
gy,gx_=np.mgrid[RY0:RY1,RX0:RX1].astype(np.float32)
fy=gy/S; fx=gx_/S
HWO=hw(np.clip(fy,LIDY,BASEY)).astype(np.float32)
U=(fx-JC)/HWO
inY=((fy>=LIDY)&(fy<=BASEY)).astype(np.float32)
GL=np.clip((1-np.abs(U))*HWO*S/1.2,0,1)*inY                      # glass coverage
GL=cv2.GaussianBlur(GL,(0,0),max(0.5,P(1.0)))
AU=np.clip(np.abs(U),0,1)
FRES=AU**9*GL
body=np.clip((fy-SHO1)/20,0,1)*np.clip((JY1-fy)/30,0,1)
HL=(np.exp(-((U+0.66)/0.07)**2)*0.9+np.exp(-((U-0.58)/0.03)**2)*0.4)*body*GL
shou=np.exp(-((fy-(NECK1+0.55*(SHO1-NECK1))-12*U**2)/6)**2)*np.exp(-((U+0.35)/0.35)**2)*GL*0.6
GLO=np.zeros_like(GL)
cv2.ellipse(GLO,(int(P(JC))-RX0,int(P(BASEY-12))-RY0),(int(P(HWB-6)),int(P(15))),0,0,180,1.0,max(1,int(P(3))),cv2.LINE_AA)
cv2.ellipse(GLO,(int(P(JC))-RX0,int(P(JY1-2))-RY0),(int(P(HWB-14)),int(P(11))),0,0,180,0.45,max(1,int(P(2))),cv2.LINE_AA)
cv2.ellipse(GLO,(int(P(JC))-RX0,int(P(NECK1-4))-RY0),(int(P(HWN)),int(P(7))),0,0,180,0.5,max(1,int(P(2))),cv2.LINE_AA)
GLO=cv2.GaussianBlur(GLO,(0,0),P(1.4))
baseg=np.clip((fy-(JY1-4))/6,0,1)*GL                              # thick glass base
EDGE=np.exp(-((np.abs(U)-1)*HWO/2.0)**2)*inY
# refraction: slightly magnified, shifted view of what's behind
MAPX=(JC+(fx-JC)*0.86)*S; MAPY=(fy-10+6*U**2)*S
MAPX=MAPX.astype(np.float32); MAPY=MAPY.astype(np.float32)
# lid: ridged metal band with an elliptical top
LIDM=np.zeros_like(GL); LHW=HWN+8
cv2.rectangle(LIDM,(int(P(JC-LHW))-RX0,int(P(LIDT))-RY0),(int(P(JC+LHW))-RX0,int(P(LIDY))-RY0),1.0,-1)
cv2.ellipse(LIDM,(int(P(JC))-RX0,int(P(LIDT))-RY0),(int(P(LHW)),int(P(9))),0,0,360,1.0,-1)
LU=np.clip((fx-JC)/LHW,-1,1)
LCOL=(np.array([70,95,115],np.float32)*(0.55+0.45*np.cos(LU*1.4))[...,None]*(0.85+0.15*np.sin(fy*1.6)[...,None]))
LCOL+=np.exp(-((LU+0.45)/0.12)**2)[...,None]*np.array([70,80,90],np.float32)
LTOP=np.zeros_like(GL); cv2.ellipse(LTOP,(int(P(JC))-RX0,int(P(LIDT))-RY0),(int(P(LHW)),int(P(9))),0,0,360,1.0,-1)
# shadow and light pool on the ledge
SH=np.exp(-(((fx-JC)/(HWB*1.05))**2+((fy-(BASEY-2))/7)**2)); POOL=np.exp(-(((fx-JC)/190)**2+((fy-(BASEY+6))/26)**2))*np.clip((fy-(BASEY-30))/24,0,1)
def jar_draw(img,t,nin,sl):
    roi=img[RY0:RY1,RX0:RX1]
    L=nin/NW
    roi*=(1-0.55*SH)[...,None]
    roi+=(POOL*(0.15+0.85*L))[...,None]*np.array([40,120,200],np.float32)*0.9
    refr=cv2.remap(img,MAPX,MAPY,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    refr=cv2.GaussianBlur(refr,(0,0),P(1.2))
    g=GL[...,None]
    roi[:]=roi*(1-g)+g*(refr*np.array([0.78,0.80,0.76],np.float32))
    warm=np.array([90,170,240],np.float32); cool=np.array([215,215,210],np.float32)
    roi+=(FRES*0.32)[...,None]*(cool*0.6+warm*0.4*L+np.array([60,130,255],np.float32)*0.5*sl)
    roi+=(HL*0.45)[...,None]*cool+(shou*0.45)[...,None]*cool
    roi+=(GLO*0.5)[...,None]*(cool*0.7+warm*0.5*L)
    roi+=(baseg*0.12)[...,None]*np.array([120,140,110],np.float32)
    roi*=(1-0.25*EDGE)[...,None]
    # lid with the sliding gap
    gx=gap_x(t); lm=LIDM.copy()
    cv2.rectangle(lm,(int(P(gx-gw(t)/2))-RX0,int(P(LIDT-14))-RY0),(int(P(gx+gw(t)/2))-RX0,int(P(LIDY+2))-RY0),0.0,-1)
    lm=cv2.GaussianBlur(lm,(0,0),P(0.9))
    lt=LTOP*lm
    col=LCOL*(1+0.25*lt)[...,None]+(lt*30)[...,None]
    roi[:]=roi*(1-lm[...,None])+lm[...,None]*col
    img[RY0:RY1,RX0:RX1]=roi
# text
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(P(92))))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(P(44))))
FL=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(P(34))))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,P(52)),"every firefly that escapes",font=FS,fill=255,anchor='mm')
d.text((W/2,P(106)),"lights a window",font=FS,fill=255,anchor='mm')
d.text((P(300),P(1668)),"windows lit",font=FL,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255; HSH=cv2.GaussianBlur(HOOK,(0,0),P(9))
topfade=np.clip(1-np.arange(H,dtype=np.float32)/P(190),0,1)[:,None]*np.ones((1,W),np.float32)
_cc={}
def count_mask(n):
    if n not in _cc:
        _cc.clear(); im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((P(300),P(1590)),f"{n} / {NW}",font=FB,fill=255,anchor='mm',features=['tnum'])
        _cc[n]=np.asarray(im,np.float32)/255
    return _cc[n]
FIRE=np.array([120,225,255],np.float32); HALO=np.array([60,150,230],np.float32)
def frame(f):
    t=f/FPS
    # streetlight comes back: flicker then hold
    if t<TSL: sl=0.0
    else:
        a=t-TSL; sl=min(1,a/1.2) if a>0.5 else [0,0.7,0,0.4,0.9][min(4,int(a/0.1))]
    img=DARK*(1-sl)+LIT*sl
    img=img*(1-0.45*topfade[...,None])
    for w in range(NW):
        k=inten(w,t)
        if k<=0: continue
        Y0,Y1,X0,X1,L,Sp,mode,p=PATCH[w]
        reg=img[Y0:Y1,X0:X1]
        img[Y0:Y1,X0:X1]=reg+(L*k)*(1-reg/300)+Sp*k*0.6
    # jar
    jar_draw(img,t,len(FR[f][0]),sl)
    # fireflies: in the jar, then flying across
    p,phs=FR[f]
    pts=[(p[i,0],p[i,1],0.62+0.38*(0.5+0.5*math.sin(t*3.6+phs[i]))**2,1.0,1.0) for i in range(len(p))]
    for e in ESC:
        a=(t-e[2])/e[6]
        if 0<=a<1:
            w=int(e[3]); x0,y0,x1,y1=WIN[w][:4]; tx,ty=(x0+x1)/2,(y0+y1)/2
            u=a*a*(3-2*a)
            x=(1-u)**2*e[0]+2*(1-u)*u*e[4]+u*u*tx+14*math.sin(a*9+e[7])*(1-u)
            y=(1-u)**2*e[1]+2*(1-u)*u*e[5]+u*u*ty+10*math.cos(a*7+e[7])*(1-u)
            pts.append((x,y,0.9,1-0.15*u,1-0.6*u))
    glow=np.zeros((H//2,W//2),np.float32); core=np.zeros((H,W),np.float32)
    for (x,y,b,al,sz) in pts:
        cv2.circle(glow,(int(P(x)/2),int(P(y)/2)),max(1,int(P(7)*sz)),b*al*0.7,-1,cv2.LINE_AA)
        cv2.circle(core,(int(P(x)),int(P(y))),max(1,int(P(5.5)*sz)),b*al,-1,cv2.LINE_AA)
    halo=cv2.resize(cv2.GaussianBlur(glow,(0,0),P(12)),(W,H))*5.0; halo=halo/(1+halo*0.9)
    soft=cv2.GaussianBlur(core,(0,0),P(1.8)); soft=soft/(1+soft*0.35)
    img+=(halo*0.55)[...,None]*HALO+np.clip(soft,0,1.4)[...,None]*FIRE
    # text
    img=img*(1-HSH[...,None]*0.5)
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([235,240,248],np.float32)
    nl=int((TON<=t).sum())
    cm=count_mask(nl); img=img*(1-cm[...,None])+cm[...,None]*np.array([235,240,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)
# ---------------- sound: a soft music-box note as each window lights, D minor pentatonic over the city-night bed
SR=48000
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*3.2)*np.minimum(1,tt*120)          # softer attack: no sharp 'ting'
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.8) for h,a in ((1,1),(2,0.15),(3,0.2)))
    return (s*env*amp).astype(np.float32)
def audio(out):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'citybed/city_night_bed.mp3','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    n=len(mus); root=293.66*2
    sc=[root*2**(k/12) for k in (0,3,5,7,10,12,15,17)]
    m=np.zeros(n,np.float32); last=-1; k=0
    for t_ in np.sort(TON[TON<1e8]):
        if t_-last<0.09: continue
        last=t_; k+=1
        s=note(sc[(k*3)%len(sc)],1.3,0.08); a=int(t_*SR); q=min(len(s),n-a); m[a:a+q]+=s[:q]
    ch=note(root/2,4,0.12)+note(root*2**(7/12)/2,4,0.08)+note(root*2**(3/12),4,0.06)
    a=int(TSL*SR); q=max(0,min(len(ch),n-a)); m[a:a+q]+=ch[:q]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    mix=mus*0.8+notes*(rms(mus)*float(os.environ.get('NOTEG','0.28'))/rms(notes))
    fo=int(2.5*SR); mix[-fo:]*=np.linspace(1,0,fo)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    if sys.argv[1]=='audio':
        audio(O+'ta.wav')
        subprocess.run(['ffmpeg','-y','-v','error','-i',O+sys.argv[2],'-i',O+'ta.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',O+sys.argv[3]])
        sys.exit()
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','): cv2.imwrite(O+f'ts_{tt}.jpg',frame(min(len(FR)-1,int(float(tt)*FPS))))
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',O+'tv.mp4'],stdin=subprocess.PIPE)
    for f in range(int(DUR*FPS)): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); audio(O+'ta.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'tv.mp4','-i',O+'ta.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
