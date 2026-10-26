import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, subprocess, sys
from scipy.signal import butter, sosfilt
SR=48000
MUSIC=ZT+'uploads/891c6635-beneath_the_blue_margin.mp3'
TAU=2*np.pi

def load(path):
    raw=subprocess.run(['ffmpeg','-v','error','-i',path,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).reshape(-1,2).copy()

def level(x,target_db=-20.0,win=3.0):
    """slow automatic gain: evens out the track's gradual build"""
    n=int(SR*win); m=x.mean(1)
    pad=np.pad(m**2,(n//2,n//2),mode='edge')
    c=np.cumsum(pad); rms=np.sqrt((c[n:]-c[:-n])/n)[:len(m)]+1e-6
    g=10**(target_db/20)/rms
    g=np.clip(g,0.25,4.0)
    # smooth the gain heavily (~6 s)
    sos=butter(1,0.15,'low',fs=SR,output='sos'); g=sosfilt(sos,sosfilt(sos,g))
    return x*g[:,None]

def music_bed(T):
    m=load(MUSIC)
    # skip the near-silent first 6 s and the fade-out tail, then even the level
    m=m[int(6*SR):int((len(m)/SR-6)*SR)]
    m=level(m)
    xf=int(8*SR)
    out=m.copy()
    while len(out)<T*SR:
        a=out[-xf:]; b=m[:xf]; w=np.linspace(0,1,xf)[:,None]
        out=np.concatenate([out[:-xf],a*np.sqrt(1-w)+b*np.sqrt(w),m[xf:]])
    out=out[:int(T*SR)]
    fi=int(3*SR); out[:fi]*=np.linspace(0,1,fi)[:,None]
    return out

def hum(n):
    t=np.arange(n)/SR
    L=np.zeros(n); R=np.zeros(n)
    comps=[(55.0,1.0),(110.0,0.55),(165.0,0.30),(220.0,0.20),(330.0,0.07),(82.4,0.25)]
    for k,(f,a) in enumerate(comps):
        L+=a*np.sin(TAU*f*t+k)
        R+=a*np.sin(TAU*(f*1.002)*t+k*1.7)     # tiny detune: slow natural beating, wide image
    breathe=0.72+0.28*np.sin(TAU*t/9.0)        # 9 s breathing swell
    breathe*=0.9+0.1*np.sin(TAU*t/23.0+1)
    return np.stack([L,R],1)*breathe[:,None]

def brown(n,seed=3):
    r=np.random.RandomState(seed); w=r.randn(n,2)
    sos=butter(1,20,'low',fs=SR,output='sos'); b=sosfilt(sos,w,axis=0)
    sos2=butter(2,[30,500],'band',fs=SR,output='sos'); b=sosfilt(sos2,b,axis=0)
    return b/np.sqrt(np.mean(b**2))

def chime(n,at):
    out=np.zeros((n,2));
    for k,(f0,dt,pan) in enumerate([(659.25,0.0,0.45),(987.77,0.32,0.55)]):
        s=int((at+dt)*SR); m=min(n-s,int(7*SR))
        if m<=0: continue
        t=np.arange(m)/SR
        v=np.zeros(m)
        for ratio,amp,dec in [(1,1.0,3.6),(2.0,0.35,2.2),(2.76,0.22,1.4),(5.4,0.08,0.6)]:
            v+=amp*np.sin(TAU*f0*ratio*t)*np.exp(-t/dec)
        v*=np.clip(t/0.006,0,1)
        out[s:s+m,0]+=v*(1-pan); out[s:s+m,1]+=v*pan
    return out

def build(T,tail=6.0):
    n=int((T+tail)*SR)
    mus=np.zeros((n,2)); mb=music_bed(T); mus[:len(mb)]=mb
    # music eases out over the last 5 s before zero
    a=int((T-5)*SR); b=int(T*SR); mus[a:b]*=np.linspace(1,0,b-a)[:,None]**1.5; mus[b:]=0
    h=hum(n); bn=brown(n)
    env=np.ones(n); fi=int(4*SR); env[:fi]=np.linspace(0,1,fi)
    e0=int(T*SR); env[e0:]=np.linspace(1,0,n-e0)**1.3
    mix=mus*1.0 + h*0.045*env[:,None] + bn*0.012*env[:,None] + chime(n,T)*0.16
    mix/=max(1.0,np.abs(mix).max()/0.89)
    return mix.astype(np.float32)

if __name__=='__main__':
    T=float(sys.argv[1]); out=sys.argv[2]
    x=build(T)
    p=subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-',out],input=x.tobytes())
