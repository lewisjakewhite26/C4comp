import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess
from scipy.signal import butter, sosfilt
D=ZT+'hut/'
F=D+'src.mp4'; N=24; SR=48000
# ---- video loop (motion-matched blend of last second into first)
c=cv2.VideoCapture(F); S=[]
while True:
    ok,f=c.read()
    if not ok: break
    S.append(f)
n=len(S); h,w=S[0].shape[:2]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx,gy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
def warp(img,fl,s): return cv2.remap(img,gx-s*fl[...,0],gy-s*fl[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
loop=S[N:n-N]
for k in range(N):
    E=S[n-N+k].astype(np.float32); B=S[k].astype(np.float32)
    u=(k+0.5)/N; a=u*u*(3-2*u)
    ge=cv2.cvtColor(S[n-N+k],cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(S[k],cv2.COLOR_BGR2GRAY)
    feb=cv2.GaussianBlur(dis.calc(ge,gb,None),(0,0),4); fbe=cv2.GaussianBlur(dis.calc(gb,ge,None),(0,0),4)
    loop.append(np.clip(warp(E,feb,a)*(1-a)+warp(B,fbe,1-a)*a,0,255).astype(np.uint8))
L=len(loop); dur=L/24.0
# ---- blizzard wind: shaped noise, synthesised longer then looped with a crossfade
r=np.random.RandomState(4); T=int((dur+2)*SR); t=np.arange(T)/SR
def bp(x,lo,hi,o=2): return sosfilt(butter(o,[lo,hi],'band',fs=SR,output='sos'),x,axis=0)
def lp(x,f,o=2): return sosfilt(butter(o,f,'low',fs=SR,output='sos'),x,axis=0)
gust=0.55+0.30*np.sin(2*np.pi*t/4.3+0.4)+0.15*np.sin(2*np.pi*t/1.7+1.1)+0.10*lp(r.randn(T),0.8,1)*30
gust=np.clip(gust,0.15,1.2)
wn=r.randn(T,2)
body=lp(bp(wn,120,900),700)*gust[:,None]                          # roar
# whistle: narrow resonance whose pitch follows the gusts
wh=np.zeros((T,2))
for lo,hi,g in ((520,640,0.5),(780,900,0.35)):
    wh+=bp(r.randn(T,2),lo,hi,3)*g
wh*=(np.clip(gust-0.45,0,None)**1.5)[:,None]
hiss=bp(r.randn(T,2),2500,7000)*0.08*gust[:,None]                # snow hiss
wind=body+wh*1.4+hiss
def loopify(x,start):
    a0=int(start*SR); m=int(dur*SR); xf=int(N/24*SR)
    seg=x[a0:a0+m].copy(); pre=x[a0-xf:a0]
    wv=np.linspace(0,1,xf,dtype=np.float32)[:,None]
    seg[-xf:]=seg[-xf:]*np.cos(wv*np.pi/2)+pre*np.sin(wv*np.pi/2)
    return seg
wind=loopify(wind,1.0)
mus=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss','40','-t','12','-i',D+'../nordic_audio/02f108b2-beneath_the_arctic_veil.mp3','-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
mus=loopify(mus,1.0)
def rms(x): return float(np.sqrt((x**2).mean()))
mix=mus+wind*(rms(mus)*0.8/rms(wind))
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',D+'audio.wav'],input=mix.astype(np.float32).tobytes())
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','-','-i',D+'audio.wav',
    '-map','0:v','-map','1:a','-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',D+'tiktok_blizzard_hut.mp4'],stdin=subprocess.PIPE)
for f in loop: enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait()
g=[cv2.cvtColor(cv2.resize(f,(180,320)),cv2.COLOR_BGR2GRAY).astype(np.float32) for f in loop]
st=[np.abs(g[i]-g[i+1]).mean() for i in range(len(g)-1)]
print('loop',L,'frames',dur,'s; wrap',round(float(np.abs(g[-1]-g[0]).mean()),2),'median step',round(float(np.median(st)),2),'max',round(max(st),2))
