import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "every snowflake that touches it freezes to it" v2 - real falling snow; only flakes that actually touch the crystal stick.
# Outside, macro-style: the crystal hangs in the snowfall with the blurred village behind.
# Shape: Reiter hex snowflake model gives the growth order; a flake touching a growth site freezes that site's
# whole symmetric orbit (so the crystal stays six-fold). A flake touching the crystal elsewhere sticks on and melts in.
import sys, os, math, subprocess
import numpy as np, cv2
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from reiter import grow
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
FW,FH=1080,1920; W,H,FPS=int(FW*S),int(FH*S),30
DUR=60.0; TEND=float(os.environ.get('TEND','58'))
CX,CY=540.0,1000.0; A=4.1
rng=np.random.RandomState(int(os.environ.get('SEED','5')))
# ---------------- growth order and symmetric orbits
ft,sv,_=grow(1,0.5,0.001)
N=ft.shape[0]; c=N//2
ii,jj=np.where(ft>=0); it=ft[ii,jj]
qx=A*((ii-c)+(jj-c)/2); qy=A*(math.sqrt(3)/2)*(jj-c); rr=np.hypot(qx,qy)
val=np.clip(0.55+(sv[ii,jj]-1)*1.6,0.55,1.0)
n=len(ii)
key=it.astype(np.int64)*100000+np.round(rr*10).astype(np.int64)
uk,orb=np.unique(key,return_inverse=True)             # orbit id per cell, orbits ordered by freeze iteration
NO=len(uk); members=[np.where(orb==o)[0] for o in range(NO)]
PRE=int(os.environ.get('PRE','20')); R0=float(os.environ.get('R0','28')); R1=float(os.environ.get('R1','18'))
frozen=orb<PRE; ofz=np.arange(NO)<PRE
K=int(os.environ.get('K','14')); LAG=int(os.environ.get('LAG','70'))
print('cells',n,'orbits',NO,flush=True)
def rot(t): return math.radians(0.35*t)
def wind(x,y,t):
    return 22+40*np.sin(0.0035*y-0.45*t)+28*np.sin(0.005*x+0.3*t+1.3*np.sin(0.002*y))
# ---------------- background: only the outside view, macro blur
src=cv2.imread(O+'win.png').astype(np.float32)
oy1=700; ow=int(oy1*9/16); ox0=int(os.environ.get('X0','360'))
bg0=cv2.resize(src[0:oy1,ox0:ox0+ow],(W,H),interpolation=cv2.INTER_CUBIC)
bg0=cv2.GaussianBlur(bg0,(0,0),11*S)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
warm=np.clip((bg0[...,2]-bg0[...,0]-20)/70,0,1)
bg0=bg0*1.05+warm[...,None]*np.array([20,60,120],np.float32)*0.5      # bokeh lights a touch richer
cd=np.exp(-(((xx-CX*S)**2+(yy-CY*S)**2)/(2*(420*S)**2))); bg0*=(1-0.25*cd)[...,None]
def snow_tile(m,rad,seed,blur):
    r=np.random.RandomState(seed); T=512; t=np.zeros((T,T),np.float32)
    for _ in range(m):
        x,y=r.randint(0,T,2)
        for ox in (-T,0,T):
            for oy in (-T,0,T): cv2.circle(t,(x+ox,y+oy),rad,r.uniform(0.4,1),-1,cv2.LINE_AA)
    return cv2.GaussianBlur(t,(0,0),blur)
ST=[(snow_tile(90,1,1,1.5),30,0.35),(snow_tile(30,3,2,3.5),55,0.35),(snow_tile(6,8,3,9.0),140,0.30)]
def bgframe(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in ST:
        oy=int(t*sp*S*2)%512; ox=int((-22*t*sp/60+25*math.sin(t*0.5+sp))*S)%512
        acc+=np.tile(np.roll(np.roll(tl,oy,0),ox,1),(H//512+1,W//512+1))[:H,:W]*a
    img=bg0*(1+warm*(0.05*math.sin(t*1.7)))[...,None]
    return img+acc[...,None]*np.array([235,230,225],np.float32)*0.55
# ---------------- simulation state
fx=np.zeros(0); fy=np.zeros(0); fv=np.zeros(0); fph=np.zeros(0); fw=np.zeros(0); fz=np.zeros(0); fb=np.zeros(0)
stuck=[]          # [crystal-space x,y,t_stuck]
EVT=[]            # (t,kind) kind 1 growth, 0 stuck
caught=0
CR=int(470*S); CL=2*CR+1
layer=np.zeros((CL,CL),np.float32)
def put(lay,x,y,amp,rad):
    cv2.circle(lay,(int((CR+x*S)*4),int((CR+y*S)*4)),max(2,int(rad*S*4)),float(amp),-1,cv2.LINE_AA,2)
for q in np.where(frozen)[0]: put(layer,qx[q],qy[q],val[q],A*0.72)
flashes=[]        # crystal-space x,y,t
def to_crys(x,y,a):
    dx,dy=x-CX,y-CY; ca,sa=math.cos(a),math.sin(a); return dx*ca+dy*sa,-dx*sa+dy*ca
def step(t,dt):
    global fx,fy,fv,fph,fw,fz,fb,caught
    p=frozen.mean(); tgt=min(1,(t/TEND)**2)
    rate=R0+R1*min(1,t/50)
    k=rng.poisson(rate*dt)
    if k:
        fx=np.concatenate([fx,rng.uniform(-250,1150,k)]); fy=np.concatenate([fy,rng.uniform(-60,-10,k)])
        fv=np.concatenate([fv,rng.uniform(75,125,k)]); fph=np.concatenate([fph,rng.uniform(0,6.3,k)])
        fw=np.concatenate([fw,rng.uniform(1.2,2.6,k)]); fz=np.concatenate([fz,rng.uniform(0.8,1.25,k)]); fb=np.concatenate([fb,np.full(k,t)])
    fx=fx+(wind(fx,fy,t)+18*np.sin(fw*t+fph))*dt; fy=fy+fv*dt
    keep=(fy<FH+30)&(fx<FW+300)
    a=rot(t); ca,sa=math.cos(a),math.sin(a)
    near=np.where(keep&(np.hypot(fx-CX,fy-CY)<rr[frozen].max()+40))[0]
    if len(near):
        fzi=np.where(frozen)[0]
        trF=cKDTree(np.stack([CX+qx[fzi]*ca-qy[fzi]*sa,CY+qx[fzi]*sa+qy[fzi]*ca],1))
        for f in near:
            d,j=trF.query([fx[f],fy[f]])
            if d>=A*0.8+4.2*fz[f]: continue
            keep[f]=False; caught+=1
            x_,y_=to_crys(fx[f],fy[f],a)
            uf=np.where(~ofz)[0]
            if len(uf):
                newest=np.where(ofz)[0].max()
                for o in uf[uf<newest-LAG]:          # shielded inner bits behind the growth front settle in quietly
                    m=members[o]; frozen[m]=True; ofz[o]=True
                    for q in m: put(layer,qx[q],qy[q],val[q],A*0.72)
                uf=np.where(~ofz)[0][:K]
            if len(uf):
                # the touching flake becomes the next growth: pick, among the next K orbits, the one with a copy nearest the touch
                best=None;bd=1e9
                for o in uf:
                    m=members[o]; dd=np.min((qx[m]-x_)**2+(qy[m]-y_)**2)
                    if dd<bd: bd=dd;best=o
                m=members[best]; m=m[~frozen[m]]; frozen[m]=True; ofz[best]=True
                for q in m: put(layer,qx[q],qy[q],val[q],A*0.72)
                qn=m[np.argmin((qx[m]-x_)**2+(qy[m]-y_)**2)]
                flashes.append((qx[qn],qy[qn],t)); EVT.append((t,1))
            else:
                stuck.append([x_,y_,t,fz[f]]); EVT.append((t,0))
    fx,fy,fv,fph,fw,fz,fb=fx[keep],fy[keep],fv[keep],fph[keep],fw[keep],fz[keep],fb[keep]
    if t>TEND+0.5 and not ofz.all():          # any shielded inner bits settle in quietly at the end
        for o in np.where(~ofz)[0][:8]:
            m=members[o]; frozen[m]=True; ofz[o]=True
            for q in m: put(layer,qx[q],qy[q],val[q],A*0.72)
ICE=np.array([255,246,236],np.float32); GLOW=np.array([255,205,160],np.float32)
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(96*S)))
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(48*S)))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,190*S),"every snowflake that touches it",font=FS,fill=255,anchor='mm')
d.text((W/2,252*S),"freezes to it",font=FS,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255; HSH=cv2.GaussianBlur(HOOK,(0,0),8*S)
def count_mask(m):
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,375*S),f"{m:,}",font=FB,fill=255,anchor='mm',features=['tnum'])
    return np.asarray(im,np.float32)/255
def frame(t):
    img=bgframe(t)
    a=rot(t)
    tmp=np.zeros((CL,CL),np.float32)
    # stuck flakes melt in over 1.2 s; then baked into the crystal
    for s_ in stuck:
        u=(t-s_[2])/1.2
        if u>=1 and len(s_)==4: put(layer,s_[0],s_[1],0.75,3.2*s_[3]); s_.append(1)
        elif u<1: put(tmp,s_[0],s_[1],1.0-0.25*u,4.2*s_[3]*(1-0.25*u))
    fl=np.zeros((CL,CL),np.float32)
    for x_,y_,t_ in flashes[-200:]:
        if t-t_<0.5: put(fl,x_,y_,1-(t-t_)/0.5,A*1.6)
    M=cv2.getRotationMatrix2D((CR,CR),-math.degrees(a),1.0); M[0,2]+=CX*S-CR; M[1,2]+=CY*S-CR
    core=cv2.warpAffine(cv2.GaussianBlur(layer+tmp,(0,0),0.6),M,(W,H))
    flash=cv2.warpAffine(cv2.GaussianBlur(fl,(0,0),1.5),M,(W,H))
    bloom=cv2.resize(cv2.GaussianBlur(cv2.resize(core+flash*1.5,(W//4,H//4),interpolation=cv2.INTER_AREA),(0,0),5*S*2),(W,H))
    big=cv2.resize(cv2.GaussianBlur(cv2.resize(core,(W//8,H//8),interpolation=cv2.INTER_AREA),(0,0),6*S*2),(W,H))
    img+=big[...,None]*GLOW*0.30+bloom[...,None]*GLOW*0.40
    al=np.clip(core*0.82,0,0.88)
    img=img*(1-al[...,None])+al[...,None]*ICE*np.clip(core,0,1)[...,None]**0.3
    img+=flash[...,None]*np.array([255,250,245],np.float32)*0.8
    fk=np.zeros((H,W),np.float32)
    for x_,y_,z in zip(fx,fy,fz):
        if -10<x_<FW+10: cv2.circle(fk,(int(x_*S*4),int(y_*S*4)),max(2,int(4.2*z*S*4)),1.0,-1,cv2.LINE_AA,2)
    fk=cv2.GaussianBlur(fk,(0,0),0.8)
    fg=cv2.resize(cv2.GaussianBlur(cv2.resize(fk,(W//4,H//4),interpolation=cv2.INTER_AREA),(0,0),2.5),(W,H))
    img+=fg[...,None]*GLOW*0.5
    img=img*(1-np.clip(fk,0,1)[...,None])+np.clip(fk,0,1)[...,None]*np.array([250,248,246],np.float32)
    img=img*(1-HSH[...,None]*0.45)
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([240,242,248],np.float32)
    cm=count_mask(caught); img=img*(1-cm[...,None])+cm[...,None]*np.array([240,242,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)
SR=48000
def note(freq,dur,amp):
    m=int(dur*SR); tt=np.arange(m)/SR
    env=np.exp(-tt*3.5)*np.minimum(1,tt*800)
    s_=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.1) for h,a in ((1,1),(2.76,0.35),(5.4,0.12)))
    return (s_*env*amp).astype(np.float32)
def audio(out,tdone):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'pondmusic/between_the_raindrops.wav','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    m=len(mus); root=329.63*2
    sc=[root*2**(q/12) for q in (0,2,4,7,9,12,14,16)]
    b=np.zeros(m,np.float32); last=-1; kk=0
    for t_,kd in EVT:
        if t_-last<0.11: continue
        last=t_; kk+=1
        s_=note(sc[(kk*3)%len(sc)]*(1 if kd else 0.5),1.4,0.08*np.interp(t_,[0,30,55],[1,0.6,0.45])); q=int(t_*SR); L=min(len(s_),m-q); b[q:q+L]+=s_[:L]
    s_=note(root/2,4.0,0.12)+note(root*1.5/2,4.0,0.08)+note(root,4.0,0.06); q=int(tdone*SR); L=min(len(s_),m-q); b[q:q+L]+=s_[:L]
    e=b.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=b[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    mix=mus*0.7+notes*(rms(mus)*0.6/rms(notes))
    f_=int(2.5*SR); mix[-f_:]*=np.linspace(1,0,f_)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    NF=int(DUR*FPS); SUB=3; mode=sys.argv[1]
    want=set(int(float(x)*FPS) for x in sys.argv[2].split(',')) if mode=='still' else set()
    # pre-roll so snow is already falling at frame 0
    for s in range(int(14*FPS*SUB)):
        tt=-14+s/(FPS*SUB); p0=caught
        fx=fx+(wind(fx,fy,tt)+18*np.sin(fw*tt+fph))/(FPS*SUB); fy=fy+fv/(FPS*SUB)
        k=rng.poisson(R0/(FPS*SUB))
        if k:
            fx=np.concatenate([fx,rng.uniform(-250,1150,k)]); fy=np.concatenate([fy,rng.uniform(-60,-10,k)])
            fv=np.concatenate([fv,rng.uniform(75,125,k)]); fph=np.concatenate([fph,rng.uniform(0,6.3,k)])
            fw=np.concatenate([fw,rng.uniform(1.2,2.6,k)]); fz=np.concatenate([fz,rng.uniform(0.8,1.25,k)]); fb=np.concatenate([fb,np.full(k,tt)])
        ok=(fy<FH+30)&(fx<FW+300)&(np.hypot(fx-CX,fy-CY)>40)
        fx,fy,fv,fph,fw,fz,fb=fx[ok],fy[ok],fv[ok],fph[ok],fw[ok],fz[ok],fb[ok]
    if mode!='still':
        enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',O+'cv2.mp4'],stdin=subprocess.PIPE)
    tdone=None
    for f in range(NF):
        for s in range(SUB): step(f/FPS+s/(FPS*SUB),1/(FPS*SUB))
        if tdone is None and frozen.all(): tdone=f/FPS
        if f%150==0: print(f//FPS,'s caught',caught,'frozen %.2f'%frozen.mean(),'flakes',len(fx),flush=True)
        if mode=='still':
            if f in want:
                cv2.imwrite(O+f'c2_{f//FPS}.jpg',frame(f/FPS))
                if f==max(want): break
        else: enc.stdin.write(frame(f/FPS).tobytes())
    print('done growing at',tdone,'caught',caught,flush=True)
    if mode=='still': sys.exit()
    enc.stdin.close(); enc.wait(); audio(O+'ca2.wav',tdone or 55)
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'cv2.mp4','-i',O+'ca2.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',O+mode])
    print('done',flush=True)
