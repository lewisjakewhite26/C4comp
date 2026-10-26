import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess, os
D=ZT+''
sys.path.insert(0,D+'jf'); sys.path.insert(0,D+'lava'); sys.path.insert(0,D+'jelly')
import numpy as np, cv2
import lava, tim
from jaudio import chime
W,H,FPS=1920,1080,24; CX,CY=W//2,H//2; R=330; TAIL=6.0
# ---------- ring works on a square region around the clock only
P=R+110; X0,Y0=CX-P,CY-P; S=2*P
Y,X=np.mgrid[Y0:Y0+S,X0:X0+S].astype(np.float32)
DD=np.sqrt((X-CX)**2+(Y-CY)**2); ANG=(np.arctan2(X-CX,-(Y-CY))%(2*np.pi))
YN=(Y/H)[...,None]
COL=np.array([255,70,170],np.float32)*(1-YN)+np.array([180,60,255],np.float32)*YN
L=np.array([-0.45,-0.55,0.70]); L/=np.linalg.norm(L); HH=np.array([L[0],L[1],L[2]+1]); HH/=np.linalg.norm(HH)
TRACK=np.clip(1-np.abs(DD-R)/22,0,1)*0.08
SPL=[(0.6,0.0),(1.9,1.7),(3.1,3.3),(4.4,0.9),(5.5,2.5)]
TXT=np.array([250,240,255],np.float32)
SK={k:(np.sin(k*ANG).astype(np.float32),np.cos(k*ANG).astype(np.float32)) for k in (3,5,7,11,13,23)}
def sk(k,ph): s_,c_=SK[k]; return s_*math.cos(ph)+c_*math.sin(ph)     # sin(kA+ph)
SPG=[np.exp(-(np.angle(np.exp(1j*(ANG-ak)))/0.11)**2).astype(np.float32) for ak,_ in SPL]
def ring(roi,t,total):
    frac=max(0.0,1-t/total); A=ANG
    wv=0.30*sk(3,0.40*t)+0.22*sk(7,-0.65*t+1.3)+0.14*sk(13,0.9*t+2.1)+0.08*sk(23,-1.3*t)
    w=40*(1+0.5*np.clip(wv,-0.55,1.2))
    for (ak,ph),g in zip(SPL,SPG):
        w=w+(22*(0.75+0.25*math.sin(0.8*t+ph)))*g
    off=5*sk(5,0.3*t+0.7)+3*sk(11,-0.5*t)
    d=DD-R-off
    across=np.clip(1-np.abs(d)/w,0,1)
    m=np.clip(across*w/2.0,0,1)
    end=2*np.pi*frac
    ang_ok=np.clip((end-A)*R/3.0,0,1)*np.clip((A-0.10)*R/3.0,0,1)
    m=m*ang_ok; acr=across*ang_ok
    if frac>0.002:
        ex,ey=CX+R*math.sin(end),CY-R*math.cos(end)
        eb=np.sqrt((X-ex)**2+(Y-ey)**2); rb=50+6*math.sin(t*1.7)
        cm=np.clip((rb-eb)/1.5,0,1); m=np.maximum(m,cm); acr=np.maximum(acr,np.clip(1-eb/rb,0,1)*cm)
    if frac>0.02:
        sx0,sy0=CX+R*math.sin(0.10),CY-R*math.cos(0.10)
        sb=np.sqrt((X-sx0)**2+(Y-sy0)**2); sm=np.clip((44-sb)/1.5,0,1)
        m=np.maximum(m,sm); acr=np.maximum(acr,np.clip(1-sb/44,0,1)*sm)
    dome=np.sqrt(acr)
    gy_,gx_=np.gradient(cv2.GaussianBlur(dome,(0,0),2))
    nx,ny=-gx_*30,-gy_*30; nl=np.sqrt(nx*nx+ny*ny+1); nx/=nl; ny/=nl; nz=1/nl
    diff=np.clip(nx*L[0]+ny*L[1]+nz*L[2],0,1); spec=np.clip(nx*HH[0]+ny*HH[1]+nz*HH[2],0,1)**35
    wax=COL*(0.45+0.5*diff[...,None]+0.25*dome[...,None])+spec[...,None]*np.array([255,240,255],np.float32)*0.9
    roi=roi+TRACK[...,None]*np.array([255,210,255],np.float32)
    roi=roi*(1-m[...,None])+wax*m[...,None]
    gl=cv2.resize(cv2.GaussianBlur(cv2.resize(m,(S//4,S//4),interpolation=cv2.INTER_AREA),(0,0),4),(S,S))
    roi=roi+gl[...,None]*np.array([200,60,230],np.float32)*0.35*(1-m[...,None])
    return roi
def frame(t,total):
    img=lava.shade(t,2)
    if t>=total:
        e=math.exp(-((t-total)-0.7)**2/0.9)
        small=cv2.resize(img,(W//4,H//4)); img=img*(1+0.5*e)+cv2.resize(cv2.GaussianBlur(small,(0,0),10),(W,H))*0.7*e
    img[Y0:Y0+S,X0:X0+S]=ring(img[Y0:Y0+S,X0:X0+S],t,total)
    # digits
    left=max(0,math.ceil(total-t-1e-6)) if t<total else 0
    a,mul,_=tim.digits(left)
    roi=img[tim.RY0:tim.RY0+tim.RH,tim.RX0:tim.RX0+tim.RW]
    glow=cv2.GaussianBlur(a[...,0],(0,0),12)[...,None]*np.array([200,60,230],np.float32)
    end=math.exp(-((t-total)-0.6)**2/0.8) if t>=total else 0.0
    roi=roi*mul+glow*(0.45+0.6*end); roi=roi*(1-a)+a*TXT
    img[tim.RY0:tim.RY0+tim.RH,tim.RX0:tim.RX0+tim.RW]=roi
    return np.clip(img,0,255).astype(np.uint8)
def audio(total,out):
    SR=48000; n=int((total+TAIL)*SR)
    raw=subprocess.run(['ffmpeg','-v','error','-i',D+'nordic_audio/nordic_music_30min.wav','-t',str(total),'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
    mus=np.zeros((n,2),np.float32); x=np.frombuffer(raw,np.float32).reshape(-1,2); mus[:len(x)]=x
    a=int((total-5)*SR); b=int(total*SR)
    mus[a:b]*=np.linspace(1,0,b-a,dtype=np.float32)[:,None]**1.5; mus[b:]=0
    mix=mus+chime(n,total).astype(np.float32)*0.16
    mix/=max(1.0,np.abs(mix).max()/0.89)
    subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.tobytes())
if __name__=='__main__':
    mins=int(sys.argv[1]); total=mins*60.0; crf=sys.argv[2] if len(sys.argv)>2 else '23'
    start=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; dur=float(sys.argv[4]) if len(sys.argv)>4 else None
    tag=f'{mins}min' if dur is None else f'{mins}min_test'
    v=D+f'lava/v_{tag}.mp4'; a=D+f'lava/a_{tag}.wav'; out=D+f'lava/lava_lamp_timer_{tag}.mp4'
    n0=int(start*FPS); n1=int((total+TAIL)*FPS) if dur is None else n0+int(dur*FPS)
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx265','-preset','medium','-crf',crf,'-x265-params','log-level=error','-tag:v','hvc1','-pix_fmt','yuv420p',v],stdin=subprocess.PIPE)
    for f in range(n0,n1): p.stdin.write(frame(f/FPS,total).tobytes())
    p.stdin.close(); p.wait()
    audio(total,a)
    subprocess.run(['ffmpeg','-y','-v','error','-i',v,'-ss',str(start),'-i',a,'-map','0:v','-map','1:a','-c:v','copy','-af','loudnorm=I=-16:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',out])
    os.remove(v); os.remove(a)
    print('done',mins,os.path.getsize(out))
