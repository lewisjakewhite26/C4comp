import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess
from scipy.signal import butter, sosfilt
D=ZT+''
B=D+'block/'; SR=48000; N=29
c=cv2.VideoCapture(B+'v1.mp4'); F=[]
while True:
    ok,f=c.read()
    if not ok: break
    F.append(f)
n=len(F)
seq=[f.astype(np.float32) for f in F[N:]]
for k in range(N):
    u=(k+0.5)/N; a=u*u*(3-2*u)
    seq[len(seq)-N+k]=seq[len(seq)-N+k]*(1-a)+F[k].astype(np.float32)*a
L=len(seq); dur=L/24.0
# ---------------- audio
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
xf=int(N/24*SR); m=int(dur*SR)
def loopify(x):
    seg=x[xf:xf+m].copy(); pre=x[:xf]
    w=np.linspace(0,1,xf,dtype=np.float32)[:,None]
    seg[-xf:]=seg[-xf:]*np.cos(w*np.pi/2)+pre*np.sin(w*np.pi/2); return seg
mus=loopify(load(D+'rainmix/rain_house_1h_music.mp3',3314,dur+2))
rain=loopify(load(D+'rainaudio/rain_house_1h_rain_only_single.mp3',600,dur+2))
r=np.random.RandomState(2); t=np.arange(m)/SR
def lp(x,f,o=2): return sosfilt(butter(o,f,'low',fs=SR,output='sos'),x,axis=0)
def hp(x,f,o=2): return sosfilt(butter(o,f,'high',fs=SR,output='sos'),x,axis=0)
# distant traffic: low rumble, periodic so it loops
traffic=lp(hp(r.randn(m,2),35),220)*(0.8+0.2*np.sin(2*np.pi*t/dur))[:,None]
# distant UK two-tone siren passing between 1.8 s and dur-1.2 s
s0,s1=2.5,6.3
ph=np.zeros(m); f=np.zeros(m)
cyc=(t-s0)/0.75                                          # one tone change every 0.75 s
hi=(np.floor(cyc)%2==0)
prog=np.clip((t-s0)/(s1-s0),0,1)
dop=1.03-0.06*prog                                       # slight doppler fall as it passes
f=np.where(hi,935.0,740.0)*dop
f=lp(f,25,1)                                             # soften the switch between tones
ph=2*np.pi*np.cumsum(f)/SR
tone=np.sin(ph)+0.35*np.sin(2*ph)+0.15*np.sin(3*ph)
envs=np.sin(np.pi*prog)**2*(t>s0)*(t<s1)
sir=lp(tone*envs,2200,2)
# small-room-free "city" reverb: a few diffuse echoes off buildings
rev=np.zeros_like(sir)
for d,g in ((0.09,0.25),(0.19,0.12)):
    i=int(d*SR); rev[i:]+=sir[:-i]*g
sir=sir+lp(rev,1200)
pan=0.25+0.5*prog
siren=np.stack([sir*np.cos(pan*np.pi/2),sir*np.sin(pan*np.pi/2)],1)
def rms(x): return float(np.sqrt((x**2).mean()))
mix=mus+rain*(rms(mus)*0.14/rms(rain))
pk_s=np.abs(siren).max(); mix+=siren/pk_s*rms(mus)*0.55
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',B+'a1.wav'],input=mix.astype(np.float32).tobytes())
h,w=F[0].shape[:2]
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','-','-i',B+'a1.wav',
    '-map','0:v','-map','1:a','-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',B+'tiktok_block_close.mp4'],stdin=subprocess.PIPE)
for f in seq: enc.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait(); print('frames',L,'dur',round(dur,2))
