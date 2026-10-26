import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess
sys.path.insert(0,ZT+'jelly')
import numpy as np, cv2
import jelly as base
D=ZT+'jf/'
W,H,FPS=1920,1080,24
GRADE="scale=1920:1080:flags=lanczos,colorchannelmixer=rr=0.80:rb=0.05:gg=0.97:bb=1.0,eq=gamma=0.96:saturation=1.04"
TAIL=6.0
RX0,RY0,RW,RH=W//2-420,H//2-420,840,840
_c={}
def digits(sec):
    if sec in _c: return _c[sec]
    a=base.timer_layer(sec)[RY0:RY0+RH,RX0:RX0+RW].copy()
    halo=cv2.GaussianBlur(a,(0,0),18)
    mul=(1-0.65*np.clip(halo*2.2,0,1))[...,None]
    glow=cv2.GaussianBlur(a,(0,0),10)[...,None]*np.array([255,170,60],np.float32)
    _c.clear(); _c[sec]=(a[...,None],mul,glow); return _c[sec]
TXT=np.array([255,244,225],np.float32)
def compose(img,t,total):
    left=max(0,math.ceil(total-t-1e-6)) if t<total else 0
    a,mul,glow=digits(left)
    end=math.exp(-((t-total)-0.6)**2/0.8) if t>=total else 0.0
    roi=img[RY0:RY0+RH,RX0:RX0+RW]
    roi=roi*mul+glow*(0.35+0.6*end)
    roi=roi*(1-a)+a*TXT
    ring=np.zeros((RH,RW),np.float32); track=np.zeros((RH,RW),np.float32)
    c=(RW//2*4,RH//2*4); frac=max(0.0,1-t/total)
    if frac>0: cv2.ellipse(ring,c,(330*4,330*4),-90,0,360*frac,1.0,16,cv2.LINE_AA,2)
    cv2.circle(track,c,330*4,0.10,16,cv2.LINE_AA,2)
    rr=np.maximum(ring,track)
    rs=cv2.resize(ring,(RW//4,RH//4),interpolation=cv2.INTER_AREA)
    rg=cv2.resize(cv2.GaussianBlur(rs,(0,0),2),(RW,RH))
    roi=roi+rr[...,None]*np.array([255,200,110],np.float32)*0.6+rg[...,None]*np.array([255,160,60],np.float32)*0.35
    img[RY0:RY0+RH,RX0:RX0+RW]=roi
    return img
def render(total,out,crf='23',start=0.0,dur=None):
    n_total=int(round((total+TAIL)*FPS))
    f0=int(round(start*FPS)); f1=n_total if dur is None else min(n_total,f0+int(dur*FPS))
    src=subprocess.Popen(['ffmpeg','-v','error','-stream_loop','-1','-i',D+'loopM.mp4','-ss',str(start),'-vf',GRADE,'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx265','-preset','medium','-crf',crf,'-x265-params','log-level=error','-tag:v','hvc1','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    fs=W*H*3
    for f in range(f0,f1):
        buf=src.stdout.read(fs)
        if len(buf)<fs: break
        img=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)
        t=f/FPS
        if t>=total:
            e=math.exp(-((t-total)-0.6)**2/0.8)
            small=cv2.resize(img,(W//4,H//4),interpolation=cv2.INTER_AREA)
            img=img*(1+0.45*e)+cv2.resize(cv2.GaussianBlur(small,(0,0),10),(W,H))*0.6*e
        img=compose(img,t,total)
        enc.stdin.write(np.clip(img,0,255).astype(np.uint8).tobytes())
    src.kill(); enc.stdin.close(); enc.wait()

if __name__=='__main__':
    total=float(sys.argv[1]); out=sys.argv[2]
    start=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
    dur=float(sys.argv[4]) if len(sys.argv)>4 else None
    import os
    render(total,out,crf=os.environ.get('CRF','23'),start=start,dur=dur)
