# "Can it escape before the timer ends?" - calm ball-escape countdown.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
W,H,FPS=1080,1920,30
T=float(os.environ.get('T','10')); TAIL=2.5
N=int(os.environ.get('NR','6'))
CX,CY=W/2,1080
R0,DR,THK=135,56,9
BR=17
SR=48000
rng=np.random.RandomState(int(os.environ.get('SEED','5')))

# ---------------- simulate (fixed steps), record events
SUB=10; dt=1/(FPS*SUB)
rings=[dict(r=R0+k*DR,om=(0.9+0.25*k)*(1 if k%2==0 else -1),ph=rng.uniform(0,2*math.pi),
            sched=T*(k+1)/N,alive=True,tbreak=None) for k in range(N)]
p=np.array([CX+20.0,CY-40.0]); v=np.array([520.0,-420.0]); G=320.0
POS=[]; BOUNCE=[]; BREAK=[]
def gapw(rg,t):
    base=0.30
    return base+1.1*max(0.0,(t-(rg['sched']-1.2))/1.2)          # widens towards its scheduled time
t=0.0
for f in range(int((T+TAIL)*FPS)):
    for s in range(SUB):
        t=(f*SUB+s)*dt
        v[1]+=G*dt; p+=v*dt
        sp=np.linalg.norm(v)
        if sp>980: v*=980/sp
        elif sp<620: v*=620/sp
        alive=[rg for rg in rings if rg['alive']]
        for rg in alive:
            if t>=rg['sched']+0.25 and rg['alive']:              # never late: it opens on time
                rg['alive']=False; rg['tbreak']=t; BREAK.append((t,rings.index(rg)))
        alive=[rg for rg in rings if rg['alive']]
        if not alive: continue
        rg=alive[0]
        d=p-np.array([CX,CY]); dist=np.linalg.norm(d)
        if dist+BR>=rg['r']-THK/2:
            ang=math.atan2(d[1],d[0]); gc=rg['ph']+rg['om']*t
            da=(ang-gc+math.pi)%(2*math.pi)-math.pi
            half=gapw(rg,t)/2-BR/rg['r']
            if abs(da)<half and t>=rg['sched']-1.2:              # slipped through the gap
                rg['alive']=False; rg['tbreak']=t; BREAK.append((t,rings.index(rg)))
            else:
                n=d/dist
                v=v-2*np.dot(v,n)*n
                v+=rng.uniform(-30,30,2)                          # tiny wobble keeps paths varied
                p=np.array([CX,CY])+n*(rg['r']-THK/2-BR-0.5)
                BOUNCE.append((t,min(1.0,abs(np.dot(v,n))/600+0.25)))
    POS.append(p.copy())
print('bounces',len(BOUNCE),'breaks',[(round(a,2),b) for a,b in BREAK])

# ---------------- visuals
FNT=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',150)
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',52)
def text_mask(s,font,cy):
    im=Image.new('L',(W,260),0); ImageDraw.Draw(im).text((W/2,130),s,font=font,fill=255,anchor='mm',features=['tnum'])
    return np.asarray(im,np.float32)/255,cy-130
HOOK,HY=text_mask("can it escape before time runs out?",FS,250)
_dc={}
def digits(sec):
    if sec not in _dc:
        _dc.clear(); _dc[sec]=text_mask(f'{sec//60}:{sec%60:02d}',FNT,470)
    return _dc[sec]
def hsv(h,s=0.55,v=1.0):
    c=cv2.cvtColor(np.uint8([[[h*179,s*255,v*255]]]),cv2.COLOR_HSV2BGR)[0,0].astype(np.float32); return c
RC=[hsv(0.50+0.06*k) for k in range(N)]                         # teal -> violet
yy=np.linspace(0,1,H,dtype=np.float32)[:,None,None]
BG=(np.array([38,16,8],np.float32)*(1-yy)+np.array([60,22,22],np.float32)*yy)
X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
BG=BG*(1-0.35*(X**2*0.6+Y**2*0.6))[...,None]
SPARKS=[]
for tb,k in BREAK:
    r=rings[k]['r']
    for i in range(46):
        a=rng.uniform(0,2*math.pi); s=rng.uniform(80,320)
        SPARKS.append((tb,CX+r*math.cos(a),CY+r*math.sin(a),math.cos(a)*s,math.sin(a)*s,k))
def frame(f):
    t=f/FPS
    img=BG.copy()
    lay=np.zeros((H,W,3),np.float32)
    for k,rg in enumerate(rings):
        a=1.0
        if rg['tbreak'] is not None and t>=rg['tbreak']:
            a=max(0.0,1-(t-rg['tbreak'])/0.35)
            if a<=0: continue
        gc=math.degrees(rg['ph']+rg['om']*t); gw=math.degrees(gapw(rg,t) if t<=rg['sched']+0.3 else 0.3)
        # breathe slightly
        rr=rg['r']*(1+0.006*math.sin(t*1.3+k))
        cv2.ellipse(lay,(int(CX*8),int(CY*8)),(int(rr*8),int(rr*8)),gc+gw/2,0,360-gw,(RC[k]*a).tolist(),THK,cv2.LINE_AA,3)
    # sparks from broken rings
    for tb,x,y,vx,vy,k in SPARKS:
        dt_=t-tb
        if 0<=dt_<1.4:
            a=(1-dt_/1.4)**1.5
            cv2.circle(lay,(int((x+vx*dt_)*8),int((y+vy*dt_+60*dt_*dt_)*8)),int(3.2*8),(RC[k]*a).tolist(),-1,cv2.LINE_AA,3)
    # ball + trail
    for i in range(14,0,-1):
        j=max(0,f-i)
        q=POS[j]; a=(1-i/15)**2*0.35
        cv2.circle(lay,(int(q[0]*8),int(q[1]*8)),int(BR*(1-i/20)*8),(np.array([250,235,255])*a).tolist(),-1,cv2.LINE_AA,3)
    q=POS[f]; cv2.circle(lay,(int(q[0]*8),int(q[1]*8)),int(BR*8),(255,248,255),-1,cv2.LINE_AA,3)
    small=cv2.resize(lay,(W//4,H//4),interpolation=cv2.INTER_AREA)
    glow=cv2.resize(cv2.GaussianBlur(small,(0,0),5),(W,H))
    img=img+glow*0.9+lay
    # countdown
    left=max(0,math.ceil(T-t-1e-6))
    dm,dy=digits(left)
    end=math.exp(-((t-T)-0.5)**2/0.5) if t>=T else 0.0
    col=np.array([250,245,255],np.float32)*(1-end)+np.array([200,255,210],np.float32)*end
    roi=img[dy:dy+260]; img[dy:dy+260]=roi*(1-dm[...,None])+dm[...,None]*col
    roi=img[HY:HY+260]; img[HY:HY+260]=roi*(1-HOOK[...,None]*0.92)+HOOK[...,None]*np.array([235,225,230],np.float32)*0.92
    return np.clip(img,0,255).astype(np.uint8)

# ---------------- sound: soft piano-ish notes, pentatonic, plus a chime per break
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*3.2)*np.minimum(1,tt*400)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*0.8) for h,a in ((1,1),(2,0.35),(3,0.12),(4,0.05)))
    return (s*env*amp).astype(np.float32)
PENT=[261.63,293.66,329.63,392.0,440.0,523.25,587.33,659.25,783.99]
def audio(out):
    n=int((T+TAIL+1)*SR); mix=np.zeros(n,np.float32)
    for i,(tb,vel) in enumerate(BOUNCE):
        fq=PENT[(i*3)%len(PENT)]; s=note(fq,1.6,0.16*vel); a=int(tb*SR); m=min(len(s),n-a); mix[a:a+m]+=s[:m]
    for tb,k in BREAK:
        for fq in (523.25,659.25,783.99):
            s=note(fq*(1+0.25*(k==N-1)),2.5,0.10); a=int(tb*SR); m=min(len(s),n-a); mix[a:a+m]+=s[:m]
    # simple room: two soft echoes
    out_=mix.copy()
    for d,g in ((0.11,0.28),(0.23,0.16),(0.41,0.08)): i=int(d*SR); out_[i:]+=mix[:-i]*g
    st=np.stack([out_,np.roll(out_,int(0.012*SR))],1)
    st/=max(1e-6,np.abs(st).max()/0.85)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=st.astype(np.float32).tobytes())

if __name__=='__main__':
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','): cv2.imwrite(O+f's_{tt}.jpg',cv2.resize(frame(int(float(tt)*FPS)),(540,960)))
    else:
        out=O+sys.argv[2]
        enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
            '-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',O+'v.mp4'],stdin=subprocess.PIPE)
        for f in range(int((T+TAIL)*FPS)): enc.stdin.write(frame(f).tobytes())
        enc.stdin.close(); enc.wait()
        audio(O+'a.wav')
        subprocess.run(['ffmpeg','-y','-v','error','-i',O+'v.mp4','-i',O+'a.wav','-map','0:v','-map','1:a','-c:v','copy',
            '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
