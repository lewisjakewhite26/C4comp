import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "every firefly that escapes brings back two friends" - fireflies float up inside a glass jar and escape through a sliding gap in the lid.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
DUR=float(os.environ.get('DUR','62')); CAP=int(os.environ.get('CAP','1000'))
JX0,JX1,JY0,JY1=230,850,720,1480                 # jar interior (full-res units)
LIDY=JY0
GW=float(os.environ.get("GW","150"))
R=9.0                                           # firefly collision radius
LIFT=float(os.environ.get("LIFT","-300"))
DRAG=0.6; WANDER=260.0; RESTI=0.75
rng=np.random.RandomState(int(os.environ.get('SEED','3')))
def gap_x(t):                                    # gap slides side to side along the lid
    span=(JX1-JX0-GW-40)/2; mid=(JX0+JX1)/2
    return mid+span*math.sin(t*0.55)

# ---------------- simulation (fixed step), with a grid for firefly-firefly collisions
SUB=2; dt=1/(FPS*SUB)
pos=np.zeros((0,2)); vel=np.zeros((0,2)); ph=np.zeros(0); born=np.zeros(0)
def spawn(n,t,near=None):
    global pos,vel,ph,born
    room=CAP-len(pos)
    n=max(0,min(n,room))
    if n==0: return 0
    if near is None:
        p=np.stack([rng.uniform(JX0+60,JX1-60,n),rng.uniform(JY1-260,JY1-60,n)],1)
    else:
        p=np.stack([rng.uniform(JX0+40,JX1-40,n),rng.uniform(JY1-120,JY1-30,n)],1)   # friends arrive from the bottom
    a=rng.uniform(0,2*math.pi,n); sp=rng.uniform(60,160,n)
    pos=np.vstack([pos,p]); vel=np.vstack([vel,np.stack([np.cos(a)*sp,np.sin(a)*sp],1)])
    ph=np.concatenate([ph,rng.uniform(0,2*math.pi,n)]); born=np.concatenate([born,np.full(n,t)])
    return n
spawn(2,0.0)
ESC=[]                         # escaped fireflies: [x,y,vx,vy,t_escape,phase]
EVENTS=[]                      # (t, count_after)
FR_IN=[]; FR_COUNT=[]; FR_GX=[]
count=2
cell=2*R
def collide():
    global pos,vel
    n=len(pos)
    if n<2: return
    gi=np.floor(pos/cell).astype(np.int64)
    key=gi[:,0]*100003+gi[:,1]
    order=np.argsort(key); ks=key[order]
    from collections import defaultdict
    buckets=defaultdict(list)
    for idx,k in zip(order,ks): buckets[k].append(idx)
    pairs_i=[];pairs_j=[]
    for (k,lst) in buckets.items():
        cx,cy=divmod(k,100003)
        for dx in (0,1):
            for dy in (-1,0,1):
                if dx==0 and dy<0: continue
                nb=buckets.get((cx+dx)*100003+(cy+dy))
                if not nb: continue
                for a in lst:
                    for b in nb:
                        if (dx==0 and dy==0) and b<=a: continue
                        pairs_i.append(a); pairs_j.append(b)
    if not pairs_i: return
    I=np.array(pairs_i); J=np.array(pairs_j)
    d=pos[J]-pos[I]; dist=np.sqrt((d**2).sum(1))+1e-6
    hit=dist<2*R
    if not hit.any(): return
    I,J,d,dist=I[hit],J[hit],d[hit],dist[hit]
    n_=d/dist[:,None]; over=(2*R-dist)[:,None]*0.5
    np.add.at(pos,I,-n_*over); np.add.at(pos,J,n_*over)
    rv=((vel[J]-vel[I])*n_).sum(1)
    imp=np.where(rv<0,-rv*0.9,0)[:,None]*n_*0.5
    np.add.at(vel,I,-imp); np.add.at(vel,J,imp)

t=0.0
for f in range(int(DUR*FPS)):
    for s in range(SUB):
        t=(f*SUB+s)*dt
        n=len(pos)
        if n:
            vel+=np.array([0,LIFT])*dt+rng.normal(0,WANDER*math.sqrt(dt),(n,2))
            vel*=(1-DRAG*dt)
            sp=np.sqrt((vel**2).sum(1)); lim=420.0; vel*=np.minimum(1,lim/(sp+1e-6))[:,None]
            pos+=vel*dt
            # walls
            for ax,lo,hi in ((0,JX0+R,JX1-R),(1,None,JY1-R)):
                if lo is not None:
                    m=pos[:,ax]<lo; pos[m,ax]=lo; vel[m,ax]=np.abs(vel[m,ax])*RESTI
                m=pos[:,ax]>hi; pos[m,ax]=hi; vel[m,ax]=-np.abs(vel[m,ax])*RESTI
            # lid with sliding gap
            gx=gap_x(t)
            atlid=pos[:,1]<LIDY+R
            ingap=np.abs(pos[:,0]-gx)<GW/2-R*0.6
            out=atlid&ingap&(pos[:,1]<LIDY-R)                 # fully through
            blk=atlid&~ingap
            pos[blk,1]=LIDY+R; vel[blk,1]=np.abs(vel[blk,1])*RESTI
            if out.any():
                idx=np.where(out)[0]
                for i in idx:
                    ESC.append([pos[i,0],pos[i,1],vel[i,0]*0.3,min(-60.0,vel[i,1]),t,ph[i]])
                keep=~out
                pos,vel,ph,born=pos[keep],vel[keep],ph[keep],born[keep]
                for _ in idx:
                    added=spawn(2,t,near=True)
                    count+=added
                    EVENTS.append((t,count))
            if s==0 and len(pos)>1: collide()
    FR_IN.append((pos.copy(),ph.copy(),born.copy())); FR_COUNT.append(count); FR_GX.append(gap_x(f/FPS))
print('final inside',len(pos),'total',count,'escapes',len(ESC),'first escapes',[round(e[4],1) for e in ESC[:6]],flush=True)
if os.environ.get('SIMONLY'): sys.exit()

# ---------------- background (Lewis's rainy-window clip, blurred, looped)
def load_bg():
    p=subprocess.Popen(['ffmpeg','-v','error','-i',D+'escape/bgclip.mp4','-vf',f'fps={FPS},scale={W}:{H}:flags=lanczos','-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    out=[]; n=W*H*3
    while True:
        b=p.stdout.read(n)
        if len(b)<n: break
        x=np.frombuffer(b,np.uint8).reshape(H,W,3).astype(np.float32)
        out.append(np.clip(cv2.GaussianBlur(x,(0,0),9*S)*0.62,0,255).astype(np.uint8))
    return out
BGF=load_bg(); BL=len(BGF); BXF=FPS
X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
VIG=(1-0.35*(X**2*0.6+Y**2*0.5))[...,None]
def bgframe(f):
    j=f%(BL-BXF); a=BGF[j].astype(np.float32)
    if j<BXF: u=j/BXF; a=a*u+BGF[j+BL-BXF].astype(np.float32)*(1-u)
    return a*VIG
def sc(v): return int(round(v*S))
# static jar layers
sill=np.zeros((H,W),np.float32); sill[sc(JY1+8):sc(JY1+60),:]=1; sill=cv2.GaussianBlur(sill,(0,0),3*S)
body=np.zeros((H,W),np.float32); cv2.rectangle(body,(sc(JX0),sc(JY0)),(sc(JX1),sc(JY1)),1.0,-1); body=cv2.GaussianBlur(body,(0,0),6*S)
edge=np.zeros((H,W),np.float32)
cv2.rectangle(edge,(sc(JX0),sc(JY0)),(sc(JX1),sc(JY1)),1.0,max(1,sc(5)),cv2.LINE_AA)
cv2.line(edge,(sc(JX0+40),sc(JY0+40)),(sc(JX0+40),sc(JY1-60)),0.55,max(1,sc(10)),cv2.LINE_AA)
cv2.line(edge,(sc(JX1-70),sc(JY0+90)),(sc(JX1-70),sc(JY0+330)),0.35,max(1,sc(6)),cv2.LINE_AA)
edge=cv2.GaussianBlur(edge,(0,0),2.0*S)
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',sc(96))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',sc(48))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,sc(170)),"every firefly that escapes",font=FS,fill=255,anchor='mm')
d.text((W/2,sc(232)),"brings back two friends",font=FS,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255
_cc={}
def count_mask(n):
    if n not in _cc:
        _cc.clear(); im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,sc(350)),f"{n:,}",font=FB,fill=255,anchor='mm',features=['tnum'])
        _cc[n]=np.asarray(im,np.float32)/255
    return _cc[n]
FIRE=np.array([120,225,255],np.float32); HALO=np.array([60,150,230],np.float32)
def frame(f):
    t=f/FPS
    img=bgframe(f)
    img=img*(1-0.55*sill[...,None])+sill[...,None]*np.array([30,45,70],np.float32)*0.55
    img=img*(1-0.10*body[...,None])+body[...,None]*np.array([40,38,30],np.float32)*0.10
    img+=edge[...,None]*np.array([215,225,235],np.float32)*0.45
    # lid
    gx=FR_GX[f]
    lid=np.zeros((H,W),np.float32)
    cv2.rectangle(lid,(sc(JX0-12),sc(LIDY-30)),(sc(JX1+12),sc(LIDY)),1.0,-1)
    cv2.rectangle(lid,(sc(gx-GW/2),sc(LIDY-40)),(sc(gx+GW/2),sc(LIDY+4)),0.0,-1)
    lid=cv2.GaussianBlur(lid,(0,0),1.2*S)
    img=img*(1-lid[...,None])+lid[...,None]*np.array([70,120,160],np.float32)
    # fireflies
    p,phs,bn=FR_IN[f]
    pts=[(p[i,0],p[i,1],0.62+0.38*(0.5+0.5*math.sin(t*3.6+phs[i]))**2,min(1.0,(t-bn[i])/0.4)) for i in range(len(p))]
    for e in ESC:
        a=t-e[4]
        if 0<=a<4.0:
            x=e[0]+e[2]*a+18*math.sin(a*2.1+e[5]); y=e[1]+e[3]*a-30*a*a
            if y<sc(420)/S: continue
            pts.append((x,y,0.8,max(0,1-a/4.0)*min(1,(y-420)/200)))
    many=len(pts)>200
    glow=np.zeros((H//2,W//2),np.float32); core=np.zeros((H,W),np.float32)
    for (x,y,b,a) in pts:
        cv2.circle(glow,(int(x*S/2),int(y*S/2)),max(1,int(9*S)),b*a*(0.6 if many else 1),-1,cv2.LINE_AA)
        cv2.circle(core,(int(x*S),int(y*S)),max(1,int(7*S)),b*a,-1,cv2.LINE_AA)
    halo=cv2.resize(cv2.GaussianBlur(glow,(0,0),14*S),(W,H))*5.0; halo=halo/(1+halo*0.9)
    soft=cv2.GaussianBlur(core,(0,0),2.2*S); soft=soft/(1+soft*0.35)
    img+=(halo*0.55)[...,None]*HALO+np.clip(soft,0,1.4)[...,None]*FIRE
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([235,240,248],np.float32)
    cm=count_mask(FR_COUNT[f]); img=img*(1-cm[...,None])+cm[...,None]*np.array([235,240,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)

# ---------------- sound: a music-box note per escape, over lo-fi + rain
SR=48000
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
def note(freq,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*4.5)*np.minimum(1,tt*700)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.3) for h,a in ((1,1),(2,0.2),(3,0.45),(5,0.12)))
    return (s*env*amp).astype(np.float32)
def audio(out):
    n=int(DUR*SR)
    mus=load(D+'rainmix/rain_house_1h_music.mp3',3314,DUR)[:n]; rain_=load(D+'rain2/rain_bed_11min.mp3',0,DUR)[:n]
    n=min(len(mus),len(rain_)); mus,rain_=mus[:n],rain_[:n]
    root=440.0*2   # music bed tested earlier as A
    PENT=[root*2**(s/12) for s in (0,2,4,7,9,12,14,16,19)]
    m=np.zeros(n,np.float32)
    # throttle: at most ~14 notes a second, quieter as it gets busy, so the cascade shimmers rather than clips
    last=-1; k=0
    for t_,c in EVENTS:
        if t_-last<0.07: continue
        last=t_; k+=1
        busy=min(1.0,len(EVENTS)/400)
        s=note(PENT[(k*3)%len(PENT)],1.2,0.09*(1-0.45*min(1,c/600))); a=int(t_*SR); kk=min(len(s),n-a); m[a:a+kk]+=s[:kk]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    bed=mus*0.6+rain_*(rms(mus)*0.25/rms(rain_))
    mix=bed+notes*(rms(bed)*1.0/rms(notes))
    f=int(2.0*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
    mix/=max(1.0,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())

if __name__=='__main__':
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','): cv2.imwrite(O+f'js_{tt}.jpg',frame(min(len(FR_IN)-1,int(float(tt)*FPS))))
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',O+'jv.mp4'],stdin=subprocess.PIPE)
    for f in range(int(DUR*FPS)): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait()
    audio(O+'ja.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'jv.mp4','-i',O+'ja.wav','-map','0:v','-map','1:a','-c:v','copy',
        '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
