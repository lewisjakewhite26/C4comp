import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys; sys.path.insert(0,ZT+'harb2')
import cv2, numpy as np, subprocess
from boat import portrait, D
N=36; END=156; SR=48000
c=cv2.VideoCapture(D+'harb2/m1.mp4'); F=[]
while len(F)<END:
    ok,f=c.read()
    if not ok: break
    F.append(f)
h,w=F[0].shape[:2]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx,gy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
def warp(img,fl,s): return cv2.remap(img,gx-s*fl[...,0],gy-s*fl[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
n=len(F); loop=F[N:n-N]
for k in range(N):
    E=F[n-N+k]; B=F[k]; u=(k+0.5)/N; a=u*u*(3-2*u)
    ge=cv2.cvtColor(E,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(B,cv2.COLOR_BGR2GRAY)
    feb=cv2.GaussianBlur(dis.calc(ge,gb,None),(0,0),4); fbe=cv2.GaussianBlur(dis.calc(gb,ge,None),(0,0),4)
    loop.append(np.clip(warp(E.astype(np.float32),feb,a)*(1-a)+warp(B.astype(np.float32),fbe,1-a)*a,0,255).astype(np.uint8))
L=len(loop); reps=2; dur=L*reps/24
g=[cv2.resize(f,(160,90)).astype(np.float32) for f in loop]
st=[np.abs(g[i]-g[i+1]).mean() for i in range(L-1)]
print('loop',L,'frames; wrap',round(float(np.abs(g[-1]-g[0]).mean()),2),'median',round(float(np.median(st)),2),'max',round(max(st),2))
# music: "Paper Boats in Twilight", looped to the clip with a crossfade
raw=subprocess.run(['ffmpeg','-v','error','-ss','2025','-t',str(dur+2),'-i',D+'rainmix/rain_house_1h_music.mp3','-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
x=np.frombuffer(raw,np.float32).reshape(-1,2).copy(); m=int(dur*SR); xf=SR
seg=x[xf:xf+m].copy(); wv=np.linspace(0,1,xf,dtype=np.float32)[:,None]
seg[-xf:]=seg[-xf:]*np.cos(wv*np.pi/2)+x[:xf]*np.sin(wv*np.pi/2)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',D+'harb2/ba.wav'],input=seg.tobytes())
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x1920','-r','24','-i','-','-i',D+'harb2/ba.wav',
    '-map','0:v','-map','1:a','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',D+'harb2/short_sunset_boat.mp4'],stdin=subprocess.PIPE)
P=[np.clip(portrait(f),0,255).astype(np.uint8) for f in loop]
for r in range(reps):
    for f in P: enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait()
