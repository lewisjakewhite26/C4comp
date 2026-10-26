import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "every snowflake that touches it freezes to it" - a single ice crystal grows into a giant snowflake
# in front of a snowy night window. Growth shape: Reiter's hexagonal snowflake model; each frozen cell
# is delivered by a visible drifting flake that arrives exactly when that cell freezes.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from reiter import grow
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
FW,FH=1080,1920; W,H,FPS=int(FW*S),int(FH*S),30
DUR=60.0; T0,T1=0.6,55.0; PW=float(os.environ.get('PW','2.0'))
CX,CY=540.0,1000.0                    # crystal centre (full-res px)
A=4.1                                 # hex cell spacing (full-res px)
rng=np.random.RandomState(5)
# ---------------- growth
PAR=tuple(map(float,os.environ.get('PAR','1,0.5,0.001').split(',')))
ft,sv,_=grow(*PAR)
N=ft.shape[0]; c=N//2
ii,jj=np.where(ft>=0); it=ft[ii,jj]
order=np.lexsort((rng.rand(len(it)),it))          # freeze order, ties shuffled so symmetric copies arrive one by one
ii,jj,it=ii[order],jj[order],it[order]
qx=A*((ii-c)+(jj-c)/2); qy=A*(math.sqrt(3)/2)*(jj-c)    # crystal-space position (px, relative to centre)
val=sv[ii,jj]; val=np.clip(0.55+(val-1)*1.6,0.55,1.0)
n=len(ii); rank=np.arange(n)
ta=np.where(rank==0,0.0,T0+(T1-T0)*(rank/(n-1))**(1/PW))  # arrival times: slow start, snowballing finish
# incoming flake for each cell
rr=np.hypot(qx,qy); ang=np.arctan2(qy,qx)+np.where(rr<1,rng.uniform(0,6.3,n),0)
fdur=np.interp(ta,[0,20,55],[2.6,2.0,1.4])*rng.uniform(0.8,1.2,n)
ts=ta-fdur
sa=ang+rng.normal(0,0.45,n); sr=rr+rng.uniform(260,480,n)
wob=rng.uniform(-70,70,n); fsz=rng.uniform(0.8,1.25,n)
print('cells',n,flush=True)
def rot(t): return math.radians(0.35*t)
# ---------------- background: Lewis's frosty window, vertical crop, snow outside, lamp/house glow breathing
src=cv2.imread(O+'win.png').astype(np.float32)
sh,sw=src.shape[:2]; cw=int(sh*9/16); x0=int(os.environ.get('X0','330'))
bg0=cv2.resize(src[:,x0:x0+cw],(W,H),interpolation=cv2.INTER_CUBIC)
bg0=cv2.GaussianBlur(bg0,(0,0),3.0*S)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
k=W/cw
pane=np.zeros((H,W),np.float32)
cv2.rectangle(pane,(int((185-x0)*k),0),(int((1180-x0)*k),int(735*k)),1.0,-1)
pane=cv2.GaussianBlur(pane,(0,0),12*S)
warm=np.clip((bg0[...,2]-bg0[...,0]-25)/80,0,1)*np.clip(bg0.mean(2)/120,0,1)
# darken a soft disc behind the crystal so it reads
cd=np.exp(-(((xx-CX*S)**2+(yy-CY*S)**2)/(2*(440*S)**2)))
bg0*= (1-0.30*cd)[...,None]
def snow_tile(m,rad,seed,blur):
    r=np.random.RandomState(seed); T=512; t=np.zeros((T,T),np.float32)
    for _ in range(m):
        x,y=r.randint(0,T,2)
        for ox in (-T,0,T):
            for oy in (-T,0,T): cv2.circle(t,(x+ox,y+oy),rad,r.uniform(0.4,1),-1,cv2.LINE_AA)
    return cv2.GaussianBlur(t,(0,0),blur)
ST=[(snow_tile(110,1,1,0.8),35,0.45),(snow_tile(45,2,2,1.6),70,0.5),(snow_tile(14,3,3,3.0),120,0.4)]
def bgframe(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in ST:
        oy=int(t*sp*S*2)%512; ox=int(25*math.sin(t*0.5+sp)*S)%512
        acc+=np.tile(np.roll(np.roll(tl,oy,0),ox,1),(H//512+1,W//512+1))[:H,:W]*a
    img=bg0*(1+warm*(0.04*math.sin(t*2.1)+0.02*math.sin(t*6.7)))[...,None]
    return img+(acc*pane)[...,None]*np.array([235,228,222],np.float32)*0.5
# ---------------- crystal layer (crystal space, supersampled drawing), accumulated as cells arrive
CR=int(470*S); CL=2*CR+1
layer=np.zeros((CL,CL),np.float32)
def put(lay,i,amp):
    cv2.circle(lay,(int((CR+qx[i]*S)*4),int((CR+qy[i]*S)*4)),max(2,int(A*0.72*S*4)),float(amp),-1,cv2.LINE_AA,2)
ICE=np.array([255,246,236],np.float32); GLOW=np.array([255,205,160],np.float32)
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(96*S)))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(48*S)))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,P0:=190*S),"every snowflake that touches it",font=FS,fill=255,anchor='mm')
d.text((W/2,252*S),"freezes to it",font=FS,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255; HSH=cv2.GaussianBlur(HOOK,(0,0),8*S)
def count_mask(m):
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,375*S),f"{m:,}",font=FB,fill=255,anchor='mm',features=['tnum'])
    return np.asarray(im,np.float32)/255
nxt=0
def frame(f):
    global nxt
    t=f/FPS
    img=bgframe(t)
    while nxt<n and ta[nxt]<=t: put(layer,nxt,val[nxt]); nxt+=1
    # recent arrivals flash
    fl=np.zeros((CL,CL),np.float32)
    rec=np.where((ta<=t)&(ta>t-0.45))[0]
    for i in rec: put(fl,i,1.0-(t-ta[i])/0.45)
    a=rot(t); M=cv2.getRotationMatrix2D((CR,CR),-math.degrees(a),1.0); M[0,2]+=CX*S-CR; M[1,2]+=CY*S-CR
    core=cv2.warpAffine(cv2.GaussianBlur(layer,(0,0),0.6),M,(W,H))
    flash=cv2.warpAffine(fl,M,(W,H))
    bloom=cv2.resize(cv2.GaussianBlur(cv2.resize(core+flash*1.5,(W//4,H//4),interpolation=cv2.INTER_AREA),(0,0),5*S*2),(W,H))
    big=cv2.resize(cv2.GaussianBlur(cv2.resize(core,(W//8,H//8),interpolation=cv2.INTER_AREA),(0,0),6*S*2),(W,H))
    img+=big[...,None]*GLOW*0.30+bloom[...,None]*GLOW*0.40
    al=np.clip(core*0.82,0,0.88)
    img=img*(1-al[...,None])+al[...,None]*ICE*np.clip(core,0,1)[...,None]**0.3
    img+=flash[...,None]*np.array([255,250,245],np.float32)*0.8
    # incoming flakes (screen space)
    act=np.where((ts<=t)&(ta>t))[0]
    fk=np.zeros((H,W),np.float32)
    if len(act):
        u=(t-ts[act])/fdur[act]; e=1-(1-u)**2
        ca,sa_=math.cos(a),math.sin(a)
        ex=CX+qx[act]*ca-qy[act]*sa_; ey=CY+qx[act]*sa_+qy[act]*ca
        bx=CX+sr[act]*np.cos(sa[act]+a); by=CY+sr[act]*np.sin(sa[act]+a)
        dx_,dy_=ex-bx,ey-by; L=np.hypot(dx_,dy_)+1e-6
        px=bx+dx_*e-dy_/L*wob[act]*np.sin(np.pi*u); py=by+dy_*e+dx_/L*wob[act]*np.sin(np.pi*u)
        al_=np.clip(u/0.15,0,1)
        for x_,y_,a_,z in zip(px,py,al_,fsz[act]):
            cv2.circle(fk,(int(x_*S*4),int(y_*S*4)),max(2,int(4.2*z*S*4)),float(a_),-1,cv2.LINE_AA,2)
    fk=cv2.GaussianBlur(fk,(0,0),0.8)
    fg=cv2.resize(cv2.GaussianBlur(cv2.resize(fk,(W//4,H//4),interpolation=cv2.INTER_AREA),(0,0),2.5),(W,H))
    img+=fg[...,None]*GLOW*0.6
    img=img*(1-np.clip(fk,0,1)[...,None])+np.clip(fk,0,1)[...,None]*np.array([250,248,246],np.float32)
    # text
    img=img*(1-HSH[...,None]*0.45)
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([240,242,248],np.float32)
    cm=count_mask(nxt); img=img*(1-cm[...,None])+cm[...,None]*np.array([240,242,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)
# ---------------- sound: soft bells as flakes freeze on, over the lo-fi bed (E)
SR=48000
def note(freq,dur,amp):
    m=int(dur*SR); tt=np.arange(m)/SR
    env=np.exp(-tt*3.5)*np.minimum(1,tt*800)
    s_=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.1) for h,a in ((1,1),(2.76,0.35),(5.4,0.12)))
    return (s_*env*amp).astype(np.float32)
def audio(out):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'pondmusic/between_the_raindrops.wav','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    m=len(mus); root=329.63*2
    sc=[root*2**(q/12) for q in (0,2,4,7,9,12,14,16)]
    b=np.zeros(m,np.float32); last=-1; kk=0
    for t_ in ta[1:]:
        gap=np.interp(t_,[0,20,55],[0.0,0.12,0.16])
        if t_-last<gap: continue
        last=t_; kk+=1
        s_=note(sc[(kk*3)%len(sc)],1.4,0.08*np.interp(t_,[0,30,55],[1,0.6,0.4])); q=int(t_*SR); L=min(len(s_),m-q); b[q:q+L]+=s_[:L]
    s_=note(root/2,4.0,0.12)+note(root*1.5/2,4.0,0.08)+note(root,4.0,0.06); q=int(T1*SR); b[q:q+len(s_)]+=s_[:m-q]
    e=b.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=b[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    mix=mus*0.7+notes*(rms(mus)*0.6/rms(notes))
    f_=int(2.5*SR); mix[-f_:]*=np.linspace(1,0,f_)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    NF=int(DUR*FPS)
    if sys.argv[1]=='still':
        want=sorted(int(float(x)*FPS) for x in sys.argv[2].split(','))
        for f in range(NF):
            if f in want:
                cv2.imwrite(O+f'cs_{f//FPS}.jpg',frame(f))
                if f==want[-1]: break
            else:
                t=f/FPS
                while nxt<n and ta[nxt]<=t: put(layer,nxt,val[nxt]); nxt+=1
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',O+'cv.mp4'],stdin=subprocess.PIPE)
    for f in range(NF): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); audio(O+'ca.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'cv.mp4','-i',O+'ca.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
