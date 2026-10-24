# Mix music (main layer) with a rain bed under it.
# usage: python3 audio.py <seconds> <music file> <music start s> <rain file> <rain start s> <out.wav>
# Music must be cleared (Lyria mixes). Keep rain recordings out of this public repo.
import numpy as np, subprocess, sys
SR=48000
dur=float(sys.argv[1]); mus_p,mus_ss=sys.argv[2],float(sys.argv[3]); rain_p,rain_ss=sys.argv[4],float(sys.argv[5]); out=sys.argv[6]
def load(p,ss,t):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',str(ss),'-t',str(t),'-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
n=int(dur*SR)
mus=load(mus_p,mus_ss,dur)[:n]; rain=load(rain_p,rain_ss,dur)[:n]
rms=lambda x: float(np.sqrt((x**2).mean()))
mix=mus+rain*(rms(mus)*0.22/rms(rain))          # rain sits well under the music
f=int(0.25*SR); mix[-f:]*=np.linspace(1,0,f)[:,None]
mix/=max(1.0,np.abs(mix).max()/0.89)
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=mix.astype(np.float32).tobytes())
