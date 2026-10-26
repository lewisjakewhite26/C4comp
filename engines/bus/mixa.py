import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, subprocess, sys
SR=48000; dur=float(sys.argv[1]); plate,pss=sys.argv[2],float(sys.argv[3]); out=sys.argv[4]
D=ZT+''
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
n=int(dur*SR)
def fit(x): y=np.zeros((n,2),np.float32); y[:min(n,len(x))]=x[:n]; return y
mus=fit(load(D+'rainmix/rain_house_1h_music.mp3',3314,dur)); rain=fit(load(D+'rainaudio/rain_house_1h_rain_only_single.mp3',600,dur)); hum=fit(load(plate,pss,dur))
rms=lambda x: float(np.sqrt((x**2).mean()))+1e-9
from scipy.signal import butter,sosfilt
hum=sosfilt(butter(2,900,'low',fs=SR,output='sos'),hum,axis=0).astype(np.float32)   # muffled, as if heard from inside
mix=mus+rain*(rms(mus)*0.22/rms(rain))+hum*(rms(mus)*0.18/rms(hum))
f=int(0.3*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
