import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess
D=ZT+'harb2/'
F=D+'m3.mp4'
N=24; SR=48000
# ---- video: morph the last second into the first so the 9 s loop is seamless
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
# ---- audio: Veo storm sound + lo-fi, both looped the same way
def load(args):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error']+args+['-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
storm=load(['-ss','600','-t','12','-i',D+'../rainaudio/rain_house_1h_rain_only_single.mp3']); mus=load(['-ss','3420','-t','12','-i',D+'../rainmix/rain_house_1h_music.mp3'])
def lp_loop(x,start):
    a0=int(start*SR); xf=int(N/24*SR); m=int(dur*SR)
    seg=x[a0:a0+m].copy(); tail=x[a0+m:a0+m+xf] if False else None
    # crossfade the final second into the material just before the loop start
    pre=x[a0-xf:a0]
    w=np.linspace(0,1,xf,dtype=np.float32)[:,None]
    seg[-xf:]=seg[-xf:]*np.cos(w*np.pi/2)+pre*np.sin(w*np.pi/2)
    return seg
st=lp_loop(storm,N/24); mu=lp_loop(mus,1.0)
def rms(x): return float(np.sqrt((x**2).mean()))
mix=mu+st*(rms(mu)*0.18/rms(st))
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',D+'audio.wav'],input=mix.astype(np.float32).tobytes())
# ---- encode, upscaled to 1080x1920
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','-','-i',D+'audio.wav',
    '-map','0:v','-map','1:a','-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',D+'short_keeper.mp4'],stdin=subprocess.PIPE)
for f in loop: enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait()
g=[cv2.cvtColor(cv2.resize(f,(180,320)),cv2.COLOR_BGR2GRAY).astype(np.float32) for f in loop]
st_=[np.abs(g[i]-g[i+1]).mean() for i in range(len(g)-1)]
print('loop frames',L,'dur',dur,'wrap',round(float(np.abs(g[-1]-g[0]).mean()),2),'median step',round(float(np.median(st_)),2),'max',round(max(st_),2))
