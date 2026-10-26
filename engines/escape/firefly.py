import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# Direction A: a little warm light escaping lantern rings over a blurred rainy city. Quick half-res test.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
T=float(os.environ.get('T','10')); TAIL=3.0
N=int(os.environ.get('NR','6'))
CX,CY=540,1060                                     # sim works in full-res units; drawn scaled by S
R0,DR,THK=135,56,7
BR=14; SR=48000
rng=np.random.RandomState(int(os.environ.get('SEED','5')))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

# ---------------- simulate: gaps ease towards the ball's next contact point before each ring is due
SUB=10; dt=1/(FPS*SUB)
rings=[dict(r=R0+k*DR,om=(0.8+0.2*k)*(1 if k%2==0 else -1),ph=rng.uniform(0,2*math.pi),
            sched=T*(k+1)/N,alive=True,tbreak=None,off=0.0) for k in range(N)]
p=np.array([CX+20.0,CY-40.0]); v=np.array([520.0,-420.0]); G=320.0
POS=[]; GC=[]; BOUNCE=[]; BREAK=[]
GAP=0.42
def contact_angle(p,v,r):
    d=p-np.array([CX,CY]); b=np.dot(d,v); c=np.dot(d,d)-r*r; a=np.dot(v,v)
    s=(-b+math.sqrt(max(0,b*b-a*c)))/a
    q=d+v*s; return math.atan2(q[1],q[0])
done=False
for f in range(int((T+TAIL)*FPS)):
    for s in range(SUB):
        t=(f*SUB+s)*dt
        if not done:
            v[1]+=G*dt; p+=v*dt
            sp=np.linalg.norm(v)
            if sp>980: v*=980/sp
            elif sp<620: v*=620/sp
        alive=[rg for rg in rings if rg['alive']]
        if not alive:
            # free: drift up and away, gently
            v=v*0.995+np.array([0,-14.0]); p+=v*dt*0.35; continue
        rg=alive[0]
        base=rg['ph']+rg['om']*t
        w=min(1.0,max(0.0,(t-(rg['sched']-1.1))/0.8))           # steering weight ramps up before the due time
        if w>0:
            target=wrap(contact_angle(p,v,rg['r']-THK/2-BR)-base)
            rg['off']+=(target-rg['off'])*min(1.0,w*0.08)
        gc=base+rg['off']*w
        rg['gc']=gc
        d=p-np.array([CX,CY]); dist=np.linalg.norm(d)
        if dist+BR>=rg['r']-THK/2:
            ang=math.atan2(d[1],d[0]); da=wrap(ang-gc)
            half=GAP/2-BR/rg['r']+ (0.25*w if t>rg['sched']+0.4 else 0)
            if abs(da)<half and t>=rg['sched']-1.1:
                rg['alive']=False; rg['tbreak']=t; BREAK.append((t,rings.index(rg)))
            else:
                n=d/dist; v=v-2*np.dot(v,n)*n; v+=rng.uniform(-25,25,2)
                p=np.array([CX,CY])+n*(rg['r']-THK/2-BR-0.5)
                BOUNCE.append((t,min(1.0,abs(np.dot(v,n))/600+0.25)))
    POS.append(p.copy())
    GC.append([rg.get('gc',rg['ph']+rg['om']*f/FPS) if rg['alive'] else None for rg in rings])
print('bounces',len(BOUNCE),'escapes',[(round(a,2),b,round(rings[b]['sched'],2)) for a,b in BREAK])

# ---------------- background: blurred rainy city bokeh, drawn once
bg=np.zeros((H,W,3),np.float32)
yy=np.linspace(0,1,H,dtype=np.float32)[:,None,None]
bg+=np.array([34,16,10],np.float32)*(1-yy)+np.array([40,24,22],np.float32)*yy
br=np.random.RandomState(2)
for i in range(140):
    x=br.uniform(0,W); y=br.uniform(0.15,1.0)*H; r=br.uniform(6,30)*S*2
    warm=br.rand()<0.7
    col=np.array([90,170,255] if warm else [255,200,150],np.float32)*br.uniform(0.15,0.55)
    cv2.circle(bg,(int(x),int(y)),int(r),col.tolist(),-1,cv2.LINE_AA)
bg=cv2.GaussianBlur(bg,(0,0),9*S*2)
X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
bg*=(1-0.45*(X**2*0.6+Y**2*0.5))[...,None]
def rtile(n,ln,seed,th):
    r=np.random.RandomState(seed); tl=np.zeros((512,512),np.float32)
    for _ in range(n):
        x,y=r.randint(0,512,2); l=r.randint(ln//2,ln); cv2.line(tl,(x,y),(x+int(l*0.08),y+l),r.uniform(0.3,1),th,cv2.LINE_AA)
    return np.tile(tl,(2,2))
RT=[(rtile(160,24,1,1),512/0.9,0.10),(rtile(60,44,2,1),512/0.55,0.13)]
def rain(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in RT:
        oy=int(t*sp)%512; ox=int(t*sp*0.08)%512
        blk=np.roll(np.roll(tl,oy,0),ox,1)[:512,:512]
        acc+=np.tile(blk,(H//512+1,W//512+1))[:H,:W]*a
    return acc

FNT=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf',int(140*S))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',int(46*S))
def tmask(lines,font,cy,gap):
    im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
    for i,l in enumerate(lines): d.text((W/2,(cy+i*gap)*S),l,font=font,fill=255,anchor='mm',features=['tnum'])
    return np.asarray(im,np.float32)/255
HOOK=tmask(["can the little light find its way","home before the timer ends?"],FS,220,62)
_dc={}
def digits(sec):
    if sec not in _dc: _dc.clear(); _dc[sec]=tmask([f'{sec//60}:{sec%60:02d}'],FNT,470,0)
    return _dc[sec]
AMB=[np.array(c,np.float32) for c in ([120,200,255],[110,185,255],[100,170,250],[95,155,245],[90,140,235],[100,130,225])]
LIGHT=np.array([150,225,255],np.float32)
def frame(f):
    t=f/FPS; sc=8*S
    img=bg.copy()
    lay=np.zeros((H,W,3),np.float32)
    for k,rg in enumerate(rings):
        gc=GC[f][k]; a=1.0
        if gc is None:
            if rg['tbreak'] is None or t-rg['tbreak']>0.6: continue
            a=1-(t-rg['tbreak'])/0.6; gc=GC[int(rg['tbreak']*FPS)-1][k] or 0.0
        rr=rg['r']*(1+0.006*math.sin(t*1.1+k))
        gw=math.degrees(GAP)
        cv2.ellipse(lay,(int(CX*sc),int(CY*sc)),(int(rr*sc),int(rr*sc)),math.degrees(gc)+gw/2,0,360-gw,(AMB[k]*a).tolist(),max(1,int(THK*S)),cv2.LINE_AA,3)
    for i in range(16,0,-1):
        q=POS[max(0,f-i)]; a=(1-i/17)**2*0.30
        cv2.circle(lay,(int(q[0]*sc),int(q[1]*sc)),int(BR*S*(1-i/22)*8),(LIGHT*a).tolist(),-1,cv2.LINE_AA,3)
    q=POS[f]
    fl=1+0.08*math.sin(t*9)+0.05*math.sin(t*23)               # a living flicker
    endf=min(1.0,max(0.0,(t-T)/TAIL))
    cv2.circle(lay,(int(q[0]*sc),int(q[1]*sc)),int(BR*S*8),(LIGHT*1.4*fl*(1-endf)).tolist(),-1,cv2.LINE_AA,3)
    small=cv2.resize(lay,(W//4,H//4),interpolation=cv2.INTER_AREA)
    glow=cv2.resize(cv2.GaussianBlur(small,(0,0),4),(W,H))
    img=img+glow*1.2+lay*0.9
    # warm bloom at the finish
    if t>=T:
        e=math.exp(-((t-T)-0.8)**2/0.6)
        yy_,xx_=np.mgrid[0:H,0:W].astype(np.float32)
        img+=np.exp(-(((xx_-q[0]*S)**2+(yy_-q[1]*S)**2)/(2*(160*S)**2)))[...,None]*np.array([60,120,190],np.float32)*e
    img+=rain(t)[...,None]*np.array([200,195,190],np.float32)*0.35
    left=max(0,math.ceil(T-t-1e-6)); dm=digits(left)
    img=img*(1-dm[...,None])+dm[...,None]*np.array([235,240,250],np.float32)
    img=img*(1-HOOK[...,None]*0.9)+HOOK[...,None]*np.array([215,225,235],np.float32)*0.9
    return np.clip(img,0,255).astype(np.uint8)

# ---------------- sound: music-box notes over the lo-fi bed and rain
D=ZT+''
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*4.0)*np.minimum(1,tt*600)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.4) for h,a in ((1,1),(2,0.25),(3,0.5),(5,0.15)))   # music-box: strong odd partials
    return (s*env*amp).astype(np.float32)
PENT=[523.25,587.33,659.25,783.99,880.0,1046.5,1174.66]
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
def audio(out):
    dur=T+TAIL; n=int(dur*SR); m=np.zeros(n,np.float32)
    for i,(tb,vel) in enumerate(BOUNCE):
        s=note(PENT[(i*2)%len(PENT)],1.4,0.10*vel); a=int(tb*SR); k=min(len(s),n-a); m[a:a+k]+=s[:k]
    for tb,kk in BREAK:
        for fq in (659.25,783.99,1046.5):
            s=note(fq,2.2,0.06*(1.6 if kk==N-1 else 1)); a=int(tb*SR); k=min(len(s),n-a); m[a:a+k]+=s[:k]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    mus=load(D+'rainmix/rain_house_1h_music.mp3',3314,dur)[:n]; rain_=load(D+'rainaudio/rain_house_1h_rain_only_single.mp3',600,dur)[:n]
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    bed=mus*0.55+rain_*(rms(mus)*0.25/rms(rain_))
    mix=bed+notes*(rms(bed)*1.4/rms(notes))
    f=int(1.5*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
    mix/=max(1.0,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())

if __name__=='__main__':
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',O+'fv.mp4'],stdin=subprocess.PIPE)
    for f in range(int((T+TAIL)*FPS)):
        fr=frame(f); enc.stdin.write(fr.tobytes())
        if f in (60,240): cv2.imwrite(O+f'ff_{f}.jpg',fr)
    enc.stdin.close(); enc.wait()
    audio(O+'fa.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'fv.mp4','-i',O+'fa.wav','-map','0:v','-map','1:a','-c:v','copy',
        '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out])
