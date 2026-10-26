import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# "can maths draw a capybara?" - a chain of spinning circles (Fourier epicycles) traces the capybara outline.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from maths import colourise
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
DUR=60.0
NC=128                                                     # most circles used (final stage)
SS=2; ALPHA=0.9
# ---------------- outline from the silhouette
sil=cv2.imread(O+os.environ.get('SIL','capy_sil.jpg'),cv2.IMREAD_GRAYSCALE)
m=(sil<128).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
c=max(cs,key=cv2.contourArea)[:,0,:].astype(np.float64)
# start at the nose (leftmost point) and go round
i0=int(np.argmin(c[:,0])); c=np.roll(c,-i0,0)
seg=np.hypot(*np.diff(np.vstack([c,c[:1]]),axis=0).T); L=np.concatenate([[0],np.cumsum(seg)])
M=2048; s=np.linspace(0,L[-1],M,endpoint=False)
x=np.interp(s,L,np.append(c[:,0],c[0,0])); y=np.interp(s,L,np.append(c[:,1],c[0,1]))
# fit into the frame: centred, about 860 px wide (full-res units)
bw,bh=x.max()-x.min(),y.max()-y.min(); sc=min(860/bw,760/bh)
CX,CY=540,1010
z=((x-(x.min()+x.max())/2)*sc+CX)+1j*((y-(y.min()+y.max())/2)*sc+CY)
Z=np.fft.fft(z)/M
freqs=np.fft.fftfreq(M,1/M).astype(int)
order=np.argsort(-np.abs(Z)); order=[k for k in order if freqs[k]!=0][:NC]
C0=Z[freqs==0][0]
coef=[(freqs[k],Z[k]) for k in order]                     # biggest circles first
STAGES=[1,2,4,8,16,32,64,128]
SD=[3.6,3.6,4.0,4.4,4.8,5.2,5.6,7.0]                        # seconds per loop for each stage
ST=[]; t_=0.6
for d_ in SD: ST.append((t_,t_+d_)); t_+=d_+0.5
TD1=ST[-1][1]
def stage_at(t):
    for i,(a,b) in enumerate(ST):
        if t<b+0.5: return i, float(np.clip((t-a)/(b-a),0,1))
    return len(ST)-1,1.0
def chain(u,n=NC):
    """positions of each circle centre and the pen at phase u (0..1 = once round)."""
    p=[C0]; acc=C0
    for f,a in coef[:n]:
        acc=acc+a*np.exp(2j*math.pi*f*u); p.append(acc)
    return np.array(p)
FR_=np.array([f for f,a in coef]); AM_=np.array([a for f,a in coef])
def pen_at(u,n=NC):
    return C0+np.sum(AM_[:n]*np.exp(2j*math.pi*FR_[:n]*u))
def uphase(t):
    v=np.clip((t-TD0)/(TD1-TD0),0,1)
    return v*v*(3-2*v)*0.25+v*0.75                         # gentle ease at the start and end
# text
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',max(8,int(64*S)))
FL=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',max(8,int(40*S)))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,170*S),os.environ.get('HOOK1','can maths draw'),font=FB,fill=255,anchor='mm')
d.text((W/2,250*S),os.environ.get('HOOK2','a capybara?'),font=FB,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255
FC=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(58*S)))
_lab={}
def label(n):
    if n not in _lab:
        im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,1640*S),f"{n} spinning circle"+("" if n==1 else "s"),font=FC,fill=255,anchor='mm',features=['tnum'])
        _lab[n]=np.asarray(im,np.float32)/255
    return _lab[n]
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
# silhouette fill for the ending
FILL=np.zeros((H,W),np.float32)
cv2.fillPoly(FILL,[np.stack([z.real*S,z.imag*S],1).astype(np.int32)],1.0,cv2.LINE_AA)
FILL=cv2.GaussianBlur(FILL,(0,0),2*S)
# eye position estimated from the silhouette image (relative to the head)
EX,EY=float(os.environ.get('EYEX','828')),float(os.environ.get('EYEY','537'))   # eye in silhouette-image pixels
EYE=complex((EX-(x.min()+x.max())/2)*sc+CX,(EY-(y.min()+y.max())/2)*sc+CY)
A0=np.zeros((H*SS,W*SS),np.float32); A1=np.zeros_like(A0); GH=np.zeros_like(A0); st=[0,0.0]   # current stage, last phase drawn
def extend(i,u0,u1):
    n=STAGES[i]; us=np.linspace(u0,u1,max(2,int((u1-u0)*3000)))
    pts=np.array([pen_at(v,n) for v in us])
    q=np.stack([pts.real*S*SS,pts.imag*S*SS],1).astype(np.int32)
    lay=np.zeros_like(A0); cv2.polylines(lay,[q],False,1.0,max(1,int(3*S*SS)),cv2.LINE_AA)
    w=i/(len(STAGES)-1); A0[:]+=lay*ALPHA*(1-w); A1[:]+=lay*ALPHA*w
def advance(t):
    i,u=stage_at(t)
    while st[0]<i:                                       # finish the old loop, fade it into the ghost layer
        if st[1]<1: extend(st[0],st[1],1.0)
        GH[:]=GH*0.45+(A0+A1); A0[:]=0; A1[:]=0; st[0]+=1; st[1]=0.0
    if u>st[1]: extend(i,st[1],u); st[1]=u
    return i,u
def frame(f):
    t=f/FPS
    i,u=advance(t)
    n=STAGES[i]
    a0=cv2.resize(A0,(W,H),interpolation=cv2.INTER_AREA); a1=cv2.resize(A1,(W,H),interpolation=cv2.INTER_AREA)
    gh=cv2.resize(GH,(W,H),interpolation=cv2.INTER_AREA)*0.28
    acc=a0+a1+gh
    w=i/(len(STAGES)-1)
    mix=(a1+gh*w)/(acc+1e-6)
    img=colourise(W,H,0,0,0,acc,False,None,mix=mix).astype(np.float32)
    fade=float(np.clip((t-TD1-0.3)/1.5,0,1))
    if fade>0:
        img+=FILL[...,None]*np.array([90,55,70],np.float32)*0.55*fade
        ex,ey=EYE.real*S,EYE.imag*S; ef=float(np.clip((t-TD1-1.5)/0.6,0,1))
        blink=0.15 if (t>TD1+4.2 and (t-TD1-4.2)%3.2<0.15) else 1.0
        if ef>0.5: cv2.ellipse(img,(int(ex),int(ey)),(max(1,int(9*S)),max(1,int(9*S*blink))),0,0,360,(255,250,245),-1,cv2.LINE_AA)
    if fade<1:
        p=chain(u,n); arms=np.zeros((H,W),np.float32); circ=np.zeros((H,W),np.float32)
        for k in range(len(p)-1):
            r=abs(p[k+1]-p[k])
            if r*S>1.0: cv2.circle(circ,(int(p[k].real*S),int(p[k].imag*S)),int(r*S),1.0,max(1,int(1.5*S)),cv2.LINE_AA)
            cv2.line(arms,(int(p[k].real*S),int(p[k].imag*S)),(int(p[k+1].real*S),int(p[k+1].imag*S)),1.0,max(1,int(2*S)),cv2.LINE_AA)
        img+=(circ*0.38+arms*0.85)[...,None]*np.array([235,225,215],np.float32)*(1-fade)
        px,py=p[-1].real*S,p[-1].imag*S
        g=np.exp(-(((xx-px)**2+(yy-py)**2)/(2*(14*S)**2)))
        img+=g[...,None]*np.array([255,240,200],np.float32)*(1-fade)
        cv2.circle(img,(int(px),int(py)),max(2,int(5*S)),(255,255,255),-1,cv2.LINE_AA)
    sh=cv2.GaussianBlur(HOOK,(0,0),10*S); img=img*(1-np.clip(sh*1.5,0,0.8))[...,None]
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([245,242,240],np.float32)
    lb=label(n); img=img*(1-lb[...,None]*0.9)+lb[...,None]*np.array([225,220,235],np.float32)*0.9
    return np.clip(img,0,255).astype(np.uint8)
SR=48000
def note(freq,dur,amp,att=120):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*3.0)*np.minimum(1,tt*att)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.8) for h,a in ((1,1),(2,0.15),(3,0.2)))
    return (s*env*amp).astype(np.float32)
def audio(out):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'citybed/city_night_bed.mp3','-ss','200','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    n=len(mus); root=293.66*2
    m=np.zeros(n,np.float32)
    ch=note(root/2,4,0.12,40)+note(root*2**(7/12)/2,4,0.08,40)+note(root*2**(3/12),4,0.06,40)
    a=int((TD1+0.3)*SR); q=max(0,min(len(ch),n-a)); m[a:a+q]+=ch[:q]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    mix=mus*0.85+notes*(rms(mus)*0.35/rms(notes))
    fo=int(2.5*SR); mix[-fo:]*=np.linspace(1,0,fo)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    NF=int(DUR*FPS)
    if sys.argv[1]=='still':
        want=sorted(int(float(v)*FPS) for v in sys.argv[2].split(','))
        for f in range(want[-1]+1):
            if f in want: cv2.imwrite(O+f'cf_{f//FPS}.jpg',frame(f))
            else: advance(f/FPS)
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',O+'cfv.mp4'],stdin=subprocess.PIPE)
    for f in range(NF): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); audio(O+'cfa.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'cfv.mp4','-i',O+'cfa.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
