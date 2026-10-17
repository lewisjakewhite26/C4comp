import numpy as np, scene, subprocess
from scipy.signal import fftconvolve
SR=44100; S=scene.build(); DUR=scene.DUR; n=int(DUR*SR)+SR*2
L=np.zeros(n); R=np.zeros(n); wl=np.zeros(n); wr=np.zeros(n)
rng=np.random.RandomState(2)
def hz(m): return 440*2**((m-69)/12)
def add(buf,t0,x):
    i=int(t0*SR); e=min(n,i+len(x)); buf[i:e]+=x[:e-i]
def marimba(f,d=2.4,v=1.0):
    t=np.arange(int(d*SR))/SR
    x=np.sin(2*np.pi*f*t)*np.exp(-t/0.55)+0.45*np.sin(2*np.pi*f*3.93*t)*np.exp(-t/0.07)+0.2*np.sin(2*np.pi*f*9.2*t)*np.exp(-t/0.025)+0.18*np.sin(2*np.pi*f*2.0*t)*np.exp(-t/0.3)
    x*=np.minimum(1,t/0.003)*np.minimum(1,(d-t)/0.2); return x*v
def pad(f,d,v):
    t=np.arange(int(d*SR))/SR; env=np.minimum(1,t/0.5)*np.minimum(1,(d-t)/0.7)
    return (np.sin(2*np.pi*f*t)+np.sin(2*np.pi*f*1.004*t)*0.8+0.3*np.sin(2*np.pi*f*2*t+1))*env*v
def bass(f,d=1.2,v=1.0):
    t=np.arange(int(d*SR))/SR; return np.sin(2*np.pi*f*t)*np.exp(-t/0.5)*np.minimum(1,t/0.01)*v
# melody on plate hits
for p in S['plates']:
    if p['i']==scene.N-1: continue
    v=0.85+0.15*np.random.RandomState(p['i']).rand(); x=marimba(hz(p['midi']),2.4,0.30*v); pan=0.5+0.22*p['side']
    add(L,p['t'],x*(1-pan)*1.2); add(R,p['t'],x*pan*1.2)
    add(wl,p['t'],x*0.5*(1-pan)); add(wr,p['t'],x*0.5*pan)
# bars: chords + bass + soft echo notes on off-beats (octave up quiet)
prog=[(48,[60,64,67]),(45,[57,60,64]),(41,[53,57,60]),(43,[55,59,62])]
nb=(scene.N-1)//4+1
for b in range(nb):
    t0=scene.T0+(1+4*b)*scene.BEAT-0.0; root,ch=prog[b%4]; d=4*scene.BEAT+0.8
    for m in ch:
        x=pad(hz(m),d,0.045); add(L,t0,x); add(R,t0,x); add(wl,t0,x*0.8); add(wr,t0,x*0.8)
    for k,vv in ((0,0.30),(2,0.2)):
        x=bass(hz(root-12 if root>43 else root),1.3,vv); add(L,t0+k*scene.BEAT,x); add(R,t0+k*scene.BEAT,x)
    # sparkle on the offbeat of beat 3
    for k,m in ((1.5,ch[2]+24),(3.5,ch[1]+24)):
        x=marimba(hz(m),1.6,0.06); add(wl,t0+k*scene.BEAT,x); add(wr,t0+k*scene.BEAT,x*0.8)
# finale
te=scene.T0+(scene.N-1)*scene.BEAT
for k,m in enumerate([72,76,79,83,86,91]):
    x=marimba(hz(m),3.5,0.34); add(L,te+k*0.11,x); add(R,te+k*0.11,x); add(wl,te+k*0.11,x*0.9); add(wr,te+k*0.11,x*0.9)
for m in (48,60,64,67,71,74):
    x=pad(hz(m),6.0,0.07); add(L,te,x); add(R,te,x); add(wl,te,x); add(wr,te,x)
# reverb
ir_n=int(2.0*SR); ir=rng.randn(2,ir_n)*np.exp(-np.arange(ir_n)/SR/0.55)*0.04
wl_=fftconvolve(wl,ir[0])[:n]; wr_=fftconvolve(wr,ir[1])[:n]
Lm=L*0.8+wl_*1.5; Rm=R*0.8+wr_*1.5
m=np.stack([Lm,Rm],1)[:int(DUR*SR)]
m*=np.minimum(1,np.arange(len(m))/(SR*0.4))[:,None]*np.minimum(1,(len(m)-np.arange(len(m)))/(SR*1.8))[:,None]
m*=0.85/np.abs(m).max()
open('a.raw','wb').write((m*32767).astype('<i2').tobytes())
subprocess.run(['ffmpeg','-y','-loglevel','error','-f','s16le','-ar',str(SR),'-ac','2','-i','a.raw','-c:a','aac','-b:a','192k','a.m4a'],check=True)
print('ok',len(m)/SR)
