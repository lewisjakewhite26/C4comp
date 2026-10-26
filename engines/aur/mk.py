import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess
D=ZT+''
SR=48000; X=24
c=cv2.VideoCapture(D+'aur/a2.mp4'); F=[]
while True:
    ok,f=c.read()
    if not ok: break
    F.append(f)
def ease(u): return u*u*(3-2*u)
A=F[:240]; B=F[240:]
seq=A[:-X]+[(A[-X+k].astype(np.float32)*(1-ease((k+.5)/X))+B[k].astype(np.float32)*ease((k+.5)/X)).astype(np.uint8) for k in range(X)]+B[X:]
# loop: blend the last second into the first second
seq2=seq[X:]
for k in range(X):
    u=ease((k+.5)/X); seq2[len(seq2)-X+k]=(seq2[len(seq2)-X+k].astype(np.float32)*(1-u)+seq[k].astype(np.float32)*u).astype(np.uint8)
L=len(seq2); dur=L/24
raw=subprocess.run(['ffmpeg','-v','error','-ss','700','-t',str(dur+2),'-i',D+'nordic_audio/nordic_music_30min.wav','-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
x=np.frombuffer(raw,np.float32).reshape(-1,2).copy(); m=int(dur*SR); xf=SR
seg=x[xf:xf+m].copy(); w=np.linspace(0,1,xf,dtype=np.float32)[:,None]
seg[-xf:]=seg[-xf:]*np.cos(w*np.pi/2)+x[:xf]*np.sin(w*np.pi/2)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',D+'aur/a.wav'],input=seg.tobytes())
h,wd=F[0].shape[:2]
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{wd}x{h}','-r','24','-i','-','-i',D+'aur/a.wav',
    '-map','0:v','-map','1:a','-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',D+'aur/short_aurora_river.mp4'],stdin=subprocess.PIPE)
for f in seq2: enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait(); print('frames',L,'dur',round(dur,2))
