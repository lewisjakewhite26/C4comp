import numpy as np, scipy.signal as sg
sr=44100; D=312; n=sr*D; t=np.arange(n,dtype=np.float32)/sr; rng=np.random.default_rng(11)
def lp(x,f,o=2): return sg.sosfilt(sg.butter(o,f,'low',fs=sr,output='sos'),x).astype(np.float32)
L=np.zeros(n,np.float32); R=np.zeros(n,np.float32)
ch=[[110,164.81,220,261.63,329.63],[87.31,130.81,174.61,220,261.63],[130.81,196,261.63,329.63,392],[98,146.83,196,246.94,293.66]]
seg=26.0; ncs=int(D/seg)+2
for k in range(ncs):
    c=ch[k%4]; cen=k*seg+seg/2
    w=np.exp(-0.5*((t-cen)/(seg*0.36))**2).astype(np.float32)
    if w.max()<1e-3: continue
    for f in c:
        for det,X,ph in ((0.0,L,0),(0.28,R,1.1)):
            X+=w*np.sin(2*np.pi*(f+det)*t+ph).astype(np.float32)/(1+f/160)*0.05
# slow breathing noise
nzl=lp(rng.standard_normal(n).astype(np.float32),700,3); nzr=lp(rng.standard_normal(n).astype(np.float32),700,3)
br=(0.5+0.5*np.sin(2*np.pi*t/40)).astype(np.float32)*0.03
L+=nzl*br; R+=nzr*br
def bell(f,at,amp,dec=1.4):
    i=int(at*sr); m=min(n-i,sr*9); tt=np.arange(m)/sr
    s=(np.sin(2*np.pi*f*tt)+0.25*np.sin(2*np.pi*f*2.0*tt)*np.exp(-tt*3))*np.exp(-tt*dec)*np.minimum(tt/0.006,1)*amp
    return i,s.astype(np.float32)
def put(f,at,amp,dec=1.4,pan=0.5):
    i,s=bell(f,at,amp,dec); L[i:i+len(s)]+=s*(1-pan); R[i:i+len(s)]+=s*pan
sc=[440,493.88,587.33,659.25,783.99,880]
at=6.0
while at<296:
    put(sc[rng.integers(0,len(sc))],at,0.035,1.4,rng.uniform(.25,.75)); at+=rng.uniform(4.5,9.0)
for m in (63,123,183,243):                       # minute chimes
    put(523.25,m,0.10,0.9); put(659.25,m+0.18,0.07,0.9); 
for m in (253,263,273,283,293):                  # last-minute warm chimes
    put(392,m,0.10,0.8,0.4); put(587.33,m+0.12,0.05,0.8,0.6)
# final singing bowl at 0:00 (video t=303)
for f,a,d in ((196,0.20,0.45),(392.7,0.12,0.5),(588.4,0.07,0.6),(784,0.04,0.7)):
    put(f,303.0,a,d,0.5)
ir=(rng.standard_normal(sr*2)*np.exp(-np.arange(sr*2)/sr/0.55)).astype(np.float32); ir=lp(ir,4000)
rev=lambda x: sg.fftconvolve(x,ir*0.01,mode='full')[:n].astype(np.float32)
L=L+rev(L)*0.9; R=R+rev(R)*0.9
fade=np.clip(np.minimum(t/2.0,(D-t)/3.0),0,1).astype(np.float32); L*=fade; R*=fade
st=np.stack([L,R],1); st*=0.15/np.sqrt((st**2).mean()); st=np.tanh(st)
open('a5.raw','wb').write((st*32767*0.9).astype('<i2').tobytes()); print('peak',float(np.abs(st).max()))
