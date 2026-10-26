import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
# Maths-art Short: "join each dot to its double" - times-table string art drawn line by line, then morphed.
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from maths import colourise
O=os.path.dirname(os.path.abspath(__file__))+'/'
D=ZT+''
S=float(os.environ.get('S','0.5'))
W,H,FPS=int(1080*S),int(1920*S),30
DUR=60.0; N=int(os.environ.get('N','240'))
TA=float(os.environ.get('TA','36'))          # end of the drawing phase
TB0,TB1=38.0,56.0                            # morph phase
K1=float(os.environ.get('K1','5'))           # where the morph ends
cx,cy,R=W/2,1000*S,430*S
ALPHA=float(os.environ.get('ALPHA','0.20'))
SS=2
ang=2*math.pi*np.arange(N)/N+math.pi/2       # point 0 at the bottom, so the heart sits upright
PX=cx+R*np.cos(ang); PY=cy+R*np.sin(ang)
# chord i starts drawing at T[i]: slow at first, then a rush
u=np.arange(N)/(N-1)
T=0.4+(TA-1.0)*u**1.9
DRAW=np.interp(T,[0,10,TA],[0.9,0.45,0.18])  # seconds to draw each chord
def kval(t):
    if t<TB0: return 2.0
    v=min(1,(t-TB0)/(TB1-TB0)); v=v*v*(3-2*v)
    return 2.0+(K1-2.0)*v
def endp(i,k):
    a=2*math.pi*(k*i)/N+math.pi/2
    return cx+R*math.cos(a), cy+R*math.sin(a)
def draw_chords(A0,A1,items):
    Hh,Ww=A0.shape
    for (x0,y0,x1,y1,w) in items:                    # draw each line in its own small patch (fast at full res)
        X0,Y0,X1,Y1=int(x0*SS),int(y0*SS),int(x1*SS),int(y1*SS)
        bx0,by0=max(0,min(X0,X1)-3),max(0,min(Y0,Y1)-3); bx1,by1=min(Ww,max(X0,X1)+4),min(Hh,max(Y0,Y1)+4)
        if bx1<=bx0 or by1<=by0: continue
        p=np.zeros((by1-by0,bx1-bx0),np.float32); cv2.line(p,(X0-bx0,Y0-by0),(X1-bx0,Y1-by0),1.0,SS,cv2.LINE_AA)
        A0[by0:by1,bx0:bx1]+=p*ALPHA*(1-w); A1[by0:by1,bx0:bx1]+=p*ALPHA*w
# text
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',max(8,int(64*S)))
FK=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf',max(8,int(84*S)))
im=Image.new('L',(W,H),0); d=ImageDraw.Draw(im)
d.text((W/2,170*S),os.environ.get('HOOK1','join each dot to its double'),font=FB,fill=255,anchor='mm')
d.text((W/2,250*S),os.environ.get('HOOK2','…then wait for it'),font=FB,fill=255,anchor='mm')
HOOK=np.asarray(im,np.float32)/255
_kc={}
def kmask(txt):
    if txt not in _kc:
        _kc.clear(); im=Image.new('L',(W,H),0); ImageDraw.Draw(im).text((W/2,1600*S),txt,font=FK,fill=255,anchor='mm',features=['tnum'])
        _kc[txt]=np.asarray(im,np.float32)/255
    return _kc[txt]
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
DOTS=np.zeros((H,W),np.float32)
for x,y in zip(PX,PY): cv2.circle(DOTS,(int(x),int(y)),max(1,int(2.2*S)),1.0,-1,cv2.LINE_AA)
DOTS=cv2.GaussianBlur(DOTS,(0,0),0.6)
# state for the drawing phase: finished chords accumulate once
FA0=np.zeros((H*SS,W*SS),np.float32); FA1=np.zeros_like(FA0); done=0
def frame(f):
    global done
    t=f/FPS
    heads=[]
    if t<TB0:
        while done<N and T[done]+DRAW[done]<=t:
            x1,y1=endp(done,2.0); draw_chords(FA0,FA1,[(PX[done],PY[done],x1,y1,done/(N-1))]); done+=1
        A0=FA0.copy(); A1=FA1.copy(); items=[]
        for i in range(done,N):
            if T[i]>t: break
            a=(t-T[i])/DRAW[i]; a=1-(1-a)**2
            x1,y1=endp(i,2.0); xh=PX[i]+(x1-PX[i])*a; yh=PY[i]+(y1-PY[i])*a
            items.append((PX[i],PY[i],xh,yh,i/(N-1))); heads.append((xh,yh))
        draw_chords(A0,A1,items)
    else:
        k=kval(t); A0=np.zeros_like(FA0); A1=np.zeros_like(FA0)
        draw_chords(A0,A1,[(PX[i],PY[i],*endp(i,k),i/(N-1)) for i in range(N)])
    A0=cv2.resize(A0,(W,H),interpolation=cv2.INTER_AREA); A1=cv2.resize(A1,(W,H),interpolation=cv2.INTER_AREA)
    acc=A0+A1
    img=colourise(W,H,cx,cy,R,acc,True,None,mix=A1/(acc+1e-6)).astype(np.float32)
    img+=DOTS[...,None]*np.array([200,190,170],np.float32)*0.55
    for (x,y) in heads:                                   # bright pen tips
        cv2.circle(img,(int(x),int(y)),max(2,int(5*S)),(255,255,255),-1,cv2.LINE_AA)
    if heads:
        hl=np.zeros((H,W),np.float32)
        for (x,y) in heads: cv2.circle(hl,(int(x),int(y)),max(2,int(10*S)),1.0,-1)
        img+=cv2.GaussianBlur(hl,(0,0),12*S)[...,None]*np.array([255,230,160],np.float32)*1.2
    # shimmer when the heart completes and at the very end
    for ts in (TA+0.3,TB1+0.4):
        if ts<=t<ts+1.6:
            p=(t-ts)/1.6; band=np.exp(-(((xx+yy*0.4)/(W+H*0.4)-p)/0.06)**2)
            img*= (1+0.5*band*(acc>0.02))[...,None]
    # text
    sh=cv2.GaussianBlur(HOOK,(0,0),10*S); img=img*(1-np.clip(sh*1.5,0,0.8))[...,None]
    img=img*(1-HOOK[...,None])+HOOK[...,None]*np.array([245,242,240],np.float32)
    k=kval(t); km=kmask(f"× {k:.2f}" if t>=TB0 else "× 2")
    img=img*(1-km[...,None])+km[...,None]*np.array([235,230,250],np.float32)*0.95
    return np.clip(img,0,255).astype(np.uint8)
# ---------------- audio: a soft pluck as each line lands, chord at the heart and at the end
SR=48000
def note(freq,dur,amp,att=120):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.exp(-tt*3.0)*np.minimum(1,tt*att)
    s=sum(a*np.sin(2*np.pi*freq*h*tt)*np.exp(-tt*h*1.8) for h,a in ((1,1),(2,0.15),(3,0.2)))
    return (s*env*amp).astype(np.float32)
def audio(out):
    mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',D+'citybed/city_night_bed.mp3','-ss','200','-t',str(DUR),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
    n=len(mus); root=293.66*2
    sc=[root*2**(q/12) for q in (0,3,5,7,10,12,15,17)]
    m=np.zeros(n,np.float32); last=-1; kk=0
    for i in range(N):
        t_=T[i]+DRAW[i]
        if t_-last<0.2: continue
        last=t_; kk+=1
        s=note(sc[(kk*3)%len(sc)],1.3,0.08); a=int(t_*SR); q=min(len(s),n-a); m[a:a+q]+=s[:q]
    for ts,amp in ((TA+0.3,1.0),(TB1+0.4,1.2)):
        ch=note(root/2,4,0.12*amp,40)+note(root*2**(7/12)/2,4,0.08*amp,40)+note(root*2**(3/12),4,0.06*amp,40)
        a=int(ts*SR); q=max(0,min(len(ch),n-a)); m[a:a+q]+=ch[:q]
    e=m.copy()
    for d_,g in ((0.12,0.3),(0.27,0.18),(0.45,0.1)): i=int(d_*SR); e[i:]+=m[:-i]*g
    notes=np.stack([e,np.roll(e,int(0.01*SR))],1)
    rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
    g=float(os.environ.get('NOTEG','0'))                 # Lewis: no chimes on this one, music only
    mix=mus*0.8+(notes*(rms(mus)*g/rms(notes)) if g>0 else 0)
    fo=int(2.5*SR); mix[-fo:]*=np.linspace(1,0,fo)[:,None]; mix/=max(1,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
if __name__=='__main__':
    NF=int(DUR*FPS)
    if sys.argv[1]=='still':
        want=sorted(int(float(x)*FPS) for x in sys.argv[2].split(','))
        for f in range(want[-1]+1):
            if f in want: cv2.imwrite(O+f'ms_{f//FPS}.jpg',frame(f))
            elif f/FPS<TB0:                      # keep the incremental state moving cheaply
                t=f/FPS
                while done<N and T[done]+DRAW[done]<=t:
                    x1,y1=endp(done,2.0); draw_chords(FA0,FA1,[(PX[done],PY[done],x1,y1,done/(N-1))]); done+=1
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',O+'mv.mp4'],stdin=subprocess.PIPE)
    for f in range(NF): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); audio(O+'ma.wav')
    subprocess.run(['ffmpeg','-y','-v','error','-i',O+'mv.mp4','-i',O+'ma.wav','-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',out])
    print('done',flush=True)
