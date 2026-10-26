import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# Firefly escape v2: tight spiral stack of rings, real gravity, gap-end collisions, soft pulsing firefly.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
T=float(os.environ.get('T','12')); TAIL=3.0
N=int(os.environ.get('NR','14'))
CX,CY=540,1080
R0,DR,THK=105,21,5.0
GAP=0.85; SPIR=0.20; OM=1.1
G=1500.0
rng=np.random.RandomState(int(os.environ.get('SEED','7')))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
C=np.array([CX,CY],float)

def brad(cleared): return 8.5+0.35*cleared
def simulate(seed,record=False):
    r_=np.random.RandomState(seed)
    rings=[dict(r=R0+k*DR,base0=k*SPIR,alive=True,tbreak=None) for k in range(N)]
    ph=r_.uniform(0,2*math.pi)
    a0=r_.uniform(0,2*math.pi); sp0=r_.uniform(700,1000)
    p=C+r_.uniform(-30,30,2); v=np.array([math.cos(a0),math.sin(a0)])*sp0
    POS=[]; GCS=[]; RAD=[]; BOUNCE=[]; BREAK=[]; TOUCH=[]
    free=False; nf=int((T+TAIL)*FPS) if record else int((T+1.0)*FPS)
    for f in range(nf):
        alive=[rg for rg in rings if rg['alive']]
        BRr=brad(N-len(alive))
        for s in range(SUB):
            t=(f*SUB+s)*dt
            if free:
                v=v*0.992+np.array([0,-30.0])*dt*20; p=p+v*dt*0.4; continue
            v=v+np.array([0,G])*dt; p=p+v*dt
            if not alive: free=True; continue
            rg=alive[0]; r=rg['r']-THK/2; gc=OM*t+rg['base0']+ph
            d=p-C; dist=math.hypot(d[0],d[1])
            if dist>rg['r']+BRr:                                   # it went out through the gap
                rg['alive']=False; rg['tbreak']=t; BREAK.append((t,rings.index(rg)))
                alive=[x for x in rings if x['alive']]; continue
            ang=math.atan2(d[1],d[0]); da=wrap(ang-gc)
            if abs(da)<GAP/2:                                       # inside the gap: only the two ends can be hit
                for sgn in (-1,1):
                    ea=gc+sgn*GAP/2; e=C+rg['r']*np.array([math.cos(ea),math.sin(ea)])
                    de=p-e; dl=math.hypot(de[0],de[1])
                    if 1e-6<dl<BRr+THK/2:
                        n=de/dl; vn=v@n
                        if vn<0:
                            v=v-2*vn*n; p=e+n*(BRr+THK/2+0.2)
                            BOUNCE.append((t,min(1,abs(vn)/800+0.2))); TOUCH.append((t,rings.index(rg),ea))
            elif dist+BRr>=r:
                n=d/dist; vn=v@n
                if vn>0:
                    v=v-2*vn*n; p=C+n*(r-BRr-0.2)                    # perfectly elastic
                    BOUNCE.append((t,min(1,abs(vn)/800+0.2))); TOUCH.append((t,rings.index(rg),ang))
        if record:
            POS.append(p.copy()); RAD.append(BRr)
            GCS.append([OM*(f/FPS)+rg['base0']+ph if rg['alive'] else None for rg in rings])
    return dict(rings=rings,POS=POS,GCS=GCS,RAD=RAD,BOUNCE=BOUNCE,BREAK=BREAK,TOUCH=TOUCH)

# search for a run whose last escape lands on the target time, with escapes spread through it
SUB=12; dt=1/(FPS*SUB)
def score(res):
    B=res['BREAK']
    if len(B)<N: return 1e9
    te=B[-1][0]; times=[x[0] for x in B]
    gaps=np.diff([0]+times)
    return abs(te-T)*10+max(0,max(gaps)-T*0.35)*2
best=None
import time; t0=time.time()
for sd in range(int(os.environ.get('TRIES','300'))):
    res=simulate(sd)
    sc=score(res)
    if best is None or sc<best[0]: best=(sc,sd)
    if sc<0.5: break
print('search',round(time.time()-t0,1),'s best',best)
res=simulate(best[1],record=True)
rings,POS,GCS,RAD,BOUNCE,BREAK,TOUCH=res['rings'],res['POS'],res['GCS'],res['RAD'],res['BOUNCE'],res['BREAK'],res['TOUCH']
TE=BREAK[-1][0] if len(BREAK)==N else T
print('TE',round(TE,2))
print('bounces',len(BOUNCE),'escapes',[(round(a,2),b) for a,b in BREAK])

# ---------------- background: soft distant rainy city
bg=np.zeros((H,W,3),np.float32)
yy=np.linspace(0,1,H,dtype=np.float32)[:,None,None]
bg+=np.array([30,14,8],np.float32)*(1-yy)+np.array([38,22,20],np.float32)*yy
br_=np.random.RandomState(2)
for i in range(240):
    x=br_.uniform(0,W); y=br_.uniform(0.25,1.0)*H; r=br_.uniform(3,11)*S*2
    col=np.array([90,170,255] if br_.rand()<0.7 else [255,200,150],np.float32)*br_.uniform(0.05,0.22)
    cv2.circle(bg,(int(x),int(y)),int(r),col.tolist(),-1,cv2.LINE_AA)
bg=cv2.GaussianBlur(bg,(0,0),12*S*2)
X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
bg*=(1-0.45*(X**2*0.6+Y**2*0.5))[...,None]
def rtile(n,ln,seed,th):
    r=np.random.RandomState(seed); tl=np.zeros((512,512),np.float32)
    for _ in range(n):
        x,y=r.randint(0,512,2); l=r.randint(ln//2,ln); cv2.line(tl,(x,y),(x+int(l*0.08),y+l),r.uniform(0.3,1),th,cv2.LINE_AA)
    return np.tile(tl,(2,2))
RT=[(rtile(160,24,1,1),512/0.9,0.08),(rtile(60,44,2,1),512/0.55,0.10)]
def rain(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in RT:
        oy=int(t*sp)%512; ox=int(t*sp*0.08)%512
        acc+=np.tile(np.roll(np.roll(tl,oy,0),ox,1)[:512,:512],(H//512+1,W//512+1))[:H,:W]*a
    return acc

FNT=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf',int(120*S))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',int(46*S))
def tmask(lines,font,cy,gap):
    im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
    for i,l in enumerate(lines): d.text((W/2,(cy+i*gap)*S),l,font=font,fill=255,anchor='mm',features=['tnum'])
    return np.asarray(im,np.float32)/255
HOOK=tmask(["can the little light find its way","home before the timer ends?"],FS,220,62)
def timer_mask(t):
    left=max(0.0,T*(TE-t)/TE)
    return tmask([f'{left:05.2f}'],FNT,1590,0)
def lerp(a,b,u): return a*(1-u)+b*u
RCOL=[lerp(np.array([120,205,255],np.float32),np.array([150,140,240],np.float32),k/(N-1)) for k in range(N)]   # amber -> rose
FIRE=np.array([140,235,255],np.float32)
sh_r=np.random.RandomState(11); SHIM=[]
def frame(f):
    t=f/FPS; sc=8*S
    img=bg.copy()
    lay=np.zeros((H,W,3),np.float32)
    for k,rg in enumerate(rings):
        gc=GCS[f][k]; a=0.85
        if gc is None:
            if rg['tbreak'] is None or t-rg['tbreak']>0.45: continue
            a=0.85*(1-(t-rg['tbreak'])/0.45); gc=GCS[max(0,int(rg['tbreak']*FPS)-1)][k] or 0.0
        gw=math.degrees(GAP)
        cv2.ellipse(lay,(int(CX*sc),int(CY*sc)),(int(rg['r']*sc),int(rg['r']*sc)),math.degrees(gc)+gw/2,0,360-gw,(RCOL[k]*a).tolist(),max(1,int(round(THK*S))),cv2.LINE_AA,3)
    # glows and motes go on their own layer and are ADDED, so they can never cut holes in a ring
    ringlay=lay; lay=np.zeros((H,W,3),np.float32)
    for tb,k,ang in TOUCH:
        dt_=t-tb
        if 0<=dt_<0.5:
            a=(1-dt_/0.5)**2; r=rings[k]['r']
            cv2.ellipse(lay,(int(CX*sc),int(CY*sc)),(int(r*sc),int(r*sc)),math.degrees(ang),-12,12,(np.array([200,245,255])*a).tolist(),max(1,int(round(THK*S*1.6))),cv2.LINE_AA,3)
    # shimmer motes left behind
    q=POS[f]
    for _ in range(2): SHIM.append([q[0],q[1],sh_r.uniform(-25,25),sh_r.uniform(-35,5),t])
    for m in SHIM:
        age=t-m[4]
        if 0<=age<0.9:
            a=(1-age/0.9)**2*0.5
            cv2.circle(lay,(int((m[0]+m[2]*age)*sc),int((m[1]+m[3]*age)*sc)),int(1.6*S*8),(FIRE*a).tolist(),-1,cv2.LINE_AA,3)
    lay=ringlay+lay*0.6
    small=cv2.resize(lay,(W//4,H//4),interpolation=cv2.INTER_AREA)
    img=img+cv2.resize(cv2.GaussianBlur(small,(0,0),3),(W,H))*0.9+lay
    # the firefly: soft core + wide halo, slow pulse
    endf=min(1.0,max(0.0,(t-TE)/TAIL))
    pulse=0.62+0.38*(0.5+0.5*math.sin(2*math.pi*t/1.7))**2+0.05*math.sin(t*17)
    fl=np.zeros((H//2,W//2),np.float32)
    cv2.circle(fl,(int(q[0]*S/2*4),int(q[1]*S/2*4)),int(RAD[f]*S/2*4),1.0,-1,cv2.LINE_AA,2)
    core=cv2.GaussianBlur(fl,(0,0),max(0.8,RAD[f]*S/2*0.35))
    halo=cv2.GaussianBlur(fl,(0,0),RAD[f]*S/2*3.2)*14
    glow=cv2.resize(core*1.6+halo,(W,H))*pulse*(1-endf)
    img+=glow[...,None]*FIRE
    if t>=TE:
        e=math.exp(-((t-TE)-0.9)**2/0.7)
        yy_,xx_=np.mgrid[0:H,0:W].astype(np.float32)
        img+=np.exp(-(((xx_-q[0]*S)**2+(yy_-q[1]*S)**2)/(2*(170*S)**2)))[...,None]*np.array([60,130,200],np.float32)*e
    img+=rain(t)[...,None]*np.array([200,195,190],np.float32)*0.35
    tm=timer_mask(t)
    u=min(1,max(0,(t-(TE-3))/3)); tc=lerp(np.array([235,240,250],np.float32),np.array([120,190,255],np.float32),u)
    img=img*(1-tm[...,None])+tm[...,None]*tc
    img=img*(1-HOOK[...,None]*0.9)+HOOK[...,None]*np.array([215,225,235],np.float32)*0.9
    return np.clip(img,0,255).astype(np.uint8)

# ---------------- sound: notes in the key of the music bed
SR=48000
D=ZT+''
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
def key_of(x):
    m=x.mean(1); n=len(m); hop=4096; chroma=np.zeros(12)
    for i in range(0,n-8192,hop):
        F=np.abs(np.fft.rfft(m[i:i+8192]*np.hanning(8192))); fq=np.fft.rfftfreq(8192,1/SR)
        sel=(fq>80)&(fq<2000); pc=np.round(12*np.log2(fq[sel]/261.63))%12
        np.add.at(chroma,pc.astype(int),F[sel]**2)
    maj=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    sc=[np.corrcoef(np.roll(maj,k),chroma)[0,1] for k in range(12)]
    return int(np.argmax(sc))
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*4.2)*np.minimum(1,tt*700)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.3) for h,a in ((1,1),(2,0.2),(3,0.45),(5,0.12)))
    return (s*env*amp).astype(np.float32)
def audio(out):
    dur=T+TAIL; n=int(dur*SR)
    mus=load(D+'rainmix/rain_house_1h_music.mp3',3314,dur)[:n]; rain_=load(D+'rainaudio/rain_house_1h_rain_only_single.mp3',600,dur)[:n]
    k=key_of(mus); root=261.63*2**(k/12)*2                 # an octave up: music-box register
    PENT=[root*2**(s/12) for s in (0,2,4,7,9,12,14,16)]
    print('music key pitch class',k)
    m=np.zeros(n,np.float32)
    for i,(tb,vel) in enumerate(BOUNCE):
        s=note(PENT[(i*3)%len(PENT)],1.3,0.09*vel); a=int(tb*SR); kk=min(len(s),n-a); m[a:a+kk]+=s[:kk]
    for tb,kk_ in BREAK:
        s=note(PENT[(kk_*2)%len(PENT)]*2,1.6,0.05); a=int(tb*SR); kk=min(len(s),n-a); m[a:a+kk]+=s[:kk]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    bed=mus*0.55+rain_*(rms(mus)*0.25/rms(rain_))
    mix=bed+notes*(rms(bed)*1.2/rms(notes))
    f=int(1.5*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
    mix/=max(1.0,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())

if __name__=='__main__':
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',O+'fv.mp4'],stdin=subprocess.PIPE)
    for f in range(int((T+TAIL)*FPS)): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait()
    audio(O+'fa.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'fv.mp4','-i',O+'fa.wav','-map','0:v','-map','1:a','-c:v','copy',
        '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out])
