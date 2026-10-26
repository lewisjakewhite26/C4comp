import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, subprocess, sys
SR=48000; dur=float(sys.argv[1]); out=sys.argv[2]
D=ZT+''
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
n=int(dur*SR)
mus=load(D+'rainmix/rain_house_1h_music.mp3',0,dur)[:n]; rain=load(D+'rainaudio/rain_house_1h_rain_only_single.mp3',0,dur)[:n]
m=min(len(mus),len(rain)); mus,rain=mus[:m],rain[:m]
rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
mix=mus+rain*(rms(mus)*0.22/rms(rain))
f=int(3*SR); mix[:f]*=np.linspace(0,1,f)[:,None]; mix[-f:]*=np.linspace(1,0,f)[:,None]
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
