# Build a long, non-repeating-sounding rain bed from a short recording:
# random 7-13 s slices chained with 2.5 s equal-power crossfades, with slight per-slice level and stereo variation.
import numpy as np, subprocess, sys
SR=48000
src,dur,out=sys.argv[1],float(sys.argv[2]),sys.argv[3]
x=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',src,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32).reshape(-1,2).copy()
x=x[int(0.5*SR):-int(0.5*SR)]                                # trim any fade at the ends
x/=np.abs(x).max()/0.7                                       # the file clips slightly at full scale
r=np.random.RandomState(7); n=int(dur*SR); XF=int(2.5*SR)
y=np.zeros((n+SR*15,2),np.float32); pos=0; first=True
w=np.linspace(0,1,XF,dtype=np.float32)[:,None]
while pos<n:
    L=int(r.uniform(7,13)*SR); s=r.randint(0,len(x)-L)
    seg=x[s:s+L]*r.uniform(0.9,1.08)
    if r.rand()<0.5: seg=seg[:,::-1]                          # swap left/right now and then
    if first: y[:L]=seg; pos=L; first=False; continue
    st=pos-XF
    y[st:st+XF]=y[st:st+XF]*np.cos(w*np.pi/2)+seg[:XF]*np.sin(w*np.pi/2)
    y[st+XF:st+L]=seg[XF:]
    pos=st+L
y=y[:n]
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-c:a','libmp3lame','-q:a','2',out],input=y.astype(np.float32).tobytes())
print('built',out,dur)
