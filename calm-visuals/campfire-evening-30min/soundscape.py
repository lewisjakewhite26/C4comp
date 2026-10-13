import numpy as np, wave, sys, time
from scipy.signal import butter, sosfilt
SR=44100; DUR=int(sys.argv[1]) if len(sys.argv)>1 else 1800     # seconds (1800 for the real thing)
N=SR*DUR; TAU=2*np.pi
rng=np.random.default_rng(2026)
L=np.zeros(N,np.float32); R=np.zeros(N,np.float32)
layers={}
def pan_g(p): a=(p+1)*np.pi/4; return np.cos(a),np.sin(a)      # p -1..1
def put_st(sig,t,p=0.0,gain=1.0,layer=None):
    gl,gr=pan_g(p); i=int(t*SR)%N; n=len(sig); s=sig.astype(np.float32)*gain; o=0
    while n>0:
        k=min(n,N-i); L[i:i+k]+=s[o:o+k]*gl; R[i:i+k]+=s[o:o+k]*gr; n-=k; o+=k; i=0
def put_lr(sl,sr,t,gain=1.0):
    i=int(t*SR)%N; n=len(sl); o=0
    while n>0:
        k=min(n,N-i); L[i:i+k]+=sl[o:o+k]*gain; R[i:i+k]+=sr[o:o+k]*gain; n-=k; o+=k; i=0
def circ(x,lo=None,hi=None,order=2,slope=None):
    X=np.fft.rfft(x); fr=np.fft.rfftfreq(len(x),1/SR); fr[0]=1e-3
    g=np.ones_like(fr)
    if lo: r=(fr/lo)**(2*order); g*=r/(1+r)
    if hi: g/=np.sqrt(1+(fr/hi)**(2*order))
    if slope: g*=(fr/100)**(-slope/2)
    g[0]=0; o=np.fft.irfft(X*g,len(x)).astype(np.float32); return o/o.std()
def pnoise(seed,rate,kmax,pw=0.8):
    n=int(rate*DUR); r=np.random.default_rng(seed); K=min(kmax,n//2)
    spec=np.zeros(n//2+1,complex); k=np.arange(1,K+1)
    spec[1:K+1]=(r.standard_normal(K)+1j*r.standard_normal(K))/k**pw
    o=np.fft.irfft(spec,n); return (o/o.std()).astype(np.float32)
def add_tex(tile,ctrl,rate,gain,p=0.0,extra=None):
    T=len(tile); B=SR*30; gl,gr=pan_g(p)
    for s in range(0,N,B):
        e=min(B,N-s); idx=(s+np.arange(e))%T
        pos=(s+np.arange(e))/(SR/rate)
        env=np.interp(pos,np.arange(len(ctrl)),ctrl,period=len(ctrl)).astype(np.float32)
        x=tile[idx]*env*gain
        L[s:s+e]+=x*gl; R[s:s+e]+=x*gr
t0=time.time()
def lg(m): print(f'{time.time()-t0:6.0f}s {m}',flush=True)
# ---------------- reverb IR
NFFT=1<<19
def make_ir(seed):
    r=np.random.default_rng(seed); n=int(SR*4.2); t=np.arange(n)/SR
    ir=r.standard_normal(n)*np.exp(-t*1.7); ir[:int(0.02*SR)]*=0.1
    ir=sosfilt(butter(2,3300,fs=SR,output='sos'),ir); ir=ir/np.sqrt((ir**2).sum()); return ir.astype(np.float32)
IRL=np.fft.rfft(make_ir(1),NFFT); IRR=np.fft.rfft(make_ir(2),NFFT)
def place(sig,t,p=0.0,wet=0.5,gain=1.0):
    sig=sig.astype(np.float32); sig=sig[:NFFT-int(SR*4.3)]
    S=np.fft.rfft(sig,NFFT); wl=np.fft.irfft(S*IRL,NFFT)[:len(sig)+int(SR*4.2)]; wr=np.fft.irfft(S*IRR,NFFT)[:len(sig)+int(SR*4.2)]
    gl,gr=pan_g(p); n=len(wl); d=np.zeros(n,np.float32); d[:len(sig)]=sig
    put_lr(d*gl*(1-wet)+wl*wet,d*gr*(1-wet)+wr*wet,t,gain)
def sched(count,seed,margin=0.15):
    r=np.random.default_rng(seed); seg=DUR/count
    return [(i+margin+(1-2*margin)*r.uniform())*seg for i in range(count)]
def smooth(a,b,x): t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
# ================= FIRE =================
lg('fire')
roar=circ(np.random.default_rng(10).standard_normal(SR*30),hi=260,order=2)
roar+=0.5*circ(np.random.default_rng(11).standard_normal(SR*30),lo=400,hi=1400)
hiss=circ(np.random.default_rng(12).standard_normal(SR*30),lo=4000,hi=9000)
env_slow=pnoise(21,10,300); env_fast=pnoise(22,40,3000,0.5)
add_tex(roar,(0.75+0.12*env_slow+0.1*env_fast)/1.0,10,0.016,-0.05) if False else None
ctrl=(0.8+0.12*np.repeat(env_slow,4)[:len(env_fast)]+0.10*env_fast).astype(np.float32); ctrl=np.clip(ctrl,0.2,None)
add_tex(roar,ctrl,40,0.020,-0.05)
add_tex(hiss,np.clip(0.4+0.2*np.repeat(env_slow,4)[:len(env_fast)]+0.25*env_fast,0.05,None).astype(np.float32),40,0.0035,-0.05)
# pop pool
pool=[]
for i in range(240):
    typ=rng.choice(3,p=[0.5,0.38,0.12])
    if typ==0: n=int(rng.uniform(0.003,0.011)*SR); tau=rng.uniform(0.0008,0.003); b=(1500,7000)
    elif typ==1: n=int(rng.uniform(0.03,0.12)*SR); tau=rng.uniform(0.008,0.028); b=(600,4500)
    else: n=int(rng.uniform(0.05,0.14)*SR); tau=rng.uniform(0.02,0.045); b=(120,1100)
    t=np.arange(n)/SR; x=rng.standard_normal(n)*np.exp(-t/tau)
    x=sosfilt(butter(2,b,btype='band',fs=SR,output='sos'),x)
    if typ==2: x+=np.sin(2*np.pi*rng.uniform(110,260)*t)*np.exp(-t/0.03)*x.std()*4
    pool.append((x/np.abs(x).max()).astype(np.float32))
rate_ctrl=pnoise(31,1,200,0.9)
for s in range(DUR):
    lam=8+4.5*np.tanh(rate_ctrl[s]); 
    for _ in range(rng.poisson(lam)):
        x=pool[rng.integers(len(pool))]; g=float(np.clip(0.055*rng.lognormal(0,0.7),0.004,0.22))
        put_st(x,s+rng.uniform(),rng.uniform(-0.18,0.1),g)
for t in sched(DUR//45,41):     # occasional bigger snap / settle
    x=pool[rng.integers(len(pool))]; put_st(x,t,rng.uniform(-0.2,0.1),0.2)
    for j in range(rng.integers(1,4)): put_st(pool[rng.integers(len(pool))],t+0.04*(j+1)+rng.uniform(0,0.05),rng.uniform(-0.2,0.1),0.1)
# ================= WIND / RIVER =================
lg('wind river')
wind=circ(np.random.default_rng(50).standard_normal(SR*40),lo=120,hi=900,order=2)
rust=circ(np.random.default_rng(51).standard_normal(SR*24),lo=1500,hi=5500)
gust=pnoise(52,10,130,0.7); g=np.clip(gust-0.2,0,None)**1.6
add_tex(wind,(0.22+0.9*g).astype(np.float32),10,0.017,0.0)
add_tex(rust,(0.05+1.1*g**1.2).astype(np.float32),10,0.0045,0.3)
riv=circ(np.random.default_rng(60).standard_normal(SR*25),lo=500,hi=2800)
amr=1+0.45*circ(np.random.default_rng(61).standard_normal(SR*25),hi=18)
add_tex((riv*np.clip(amr,0.2,None)).astype(np.float32),(0.8+0.2*pnoise(62,5,120)).astype(np.float32),5,0.0045,0.45)
# ================= PAD =================
lg('pad')
mf=lambda m:440*2**((m-69)/12)
chordset=[[38,45,50,53,57],[46,53,57,60,62],[43,50,53,58,62],[45,52,55,60,64]]
step=40.0; plen=70.0
for i in range(int(DUR/step)):
    ch=chordset[i%4]; n=int(plen*SR); t=np.arange(n,dtype=np.float32)/SR
    env=(np.sin(np.pi*np.clip(t/plen,0,1))**2).astype(np.float32)
    sl=np.zeros(n,np.float32); sr=np.zeros(n,np.float32)
    for v,m in enumerate(ch):
        vr=np.random.default_rng(1000*i+v); f=mf(m)*(1+vr.uniform(-0.001,0.001)); pp=vr.uniform(-0.8,0.8); gl,gr=pan_g(pp)
        x=np.zeros(n,np.float32)
        for h,wt in ((1,1.0),(2,0.5),(3,0.28),(4,0.14),(6,0.06)):
            for d in (-0.0012,0.0012):
                x+=(wt*np.sin(TAU*f*h*(1+d)*t+vr.uniform(0,6.28))).astype(np.float32)
        x*=env*(0.55 if m<44 else 0.4)
        sl+=x*gl; sr+=x*gr
    put_lr(sl,sr,i*step,0.0075)
# ================= GUITAR / SLIDE / LOW PIANO =================
lg('guitar')
def pluck(m,dur=4.0,vel=0.5):
    f=mf(m); t=np.arange(int(SR*dur))/SR; s=np.zeros_like(t)
    for h in range(1,9):
        s+=np.sin(TAU*f*h*np.sqrt(1+0.0003*h*h)*t+rng.uniform(0,6.28))*np.exp(-t*(1.0+0.85*h*h**0.2+f/600))/h**0.95
    s+=rng.standard_normal(len(t))*np.exp(-t*220)*0.08
    s*=np.minimum(1,t/0.003); return (s*vel*0.45).astype(np.float32)
pent=[50,53,55,57,60,62,65,67,69]
for t in sched(26,71):
    r=np.random.default_rng(int(t*1000)); k=int(r.integers(3,7)); idx=int(r.integers(1,6)); tt=t
    for j in range(k):
        idx=int(np.clip(idx+r.integers(-2,3),0,len(pent)-1)); place(pluck(pent[idx],vel=r.uniform(0.35,0.6)),tt,r.uniform(-0.3,0.3),0.45,0.55)
        tt+=r.uniform(1.0,2.6)
def piano(m,dur=6.0,vel=0.5):
    f=mf(m); t=np.arange(int(SR*dur))/SR; s=np.zeros_like(t)
    for h in range(1,10):
        s+=np.sin(TAU*f*h*np.sqrt(1+0.0004*h*h)*t+rng.uniform(0,6.28))*np.exp(-t*(0.7+0.5*h))/h**1.2
    return (s*np.minimum(1,t/0.006)*vel*0.4).astype(np.float32)
for i,t in enumerate(sched(14,72)):
    place(piano([38,45,43,41,50][i%5]),t,rng.uniform(-0.2,0.2),0.6,0.5)
def slide(m0,m1,dur=5.0):
    t=np.arange(int(SR*dur))/SR; f=mf(m0)*(2**((m1-m0)/12*smooth(0,1.2,t)))*(1+0.006*np.sin(TAU*5*t)*smooth(1.0,2.5,t))
    ph=TAU*np.cumsum(f)/SR; s=np.sin(ph)+0.35*np.sin(2*ph)+0.12*np.sin(3*ph)
    env=smooth(0,0.8,t)*(1-smooth(dur-2.5,dur,t)); return (s*env*0.22).astype(np.float32)
for i,t in enumerate(sched(9,73)):
    a,b=[(57,62),(58,62),(60,65),(62,65),(55,57)][i%5]; place(slide(a-1.5,b),t,rng.uniform(-0.4,0.4),0.6,0.30)
# ================= WOLVES / OWL / BIRDS =================
lg('wolves')
def wolf(dur=6.5,base=370,seed=0):
    r=np.random.default_rng(seed); t=np.arange(int(SR*dur))/SR
    f=base*(1+0.42*smooth(0,1.4,t)-0.08*smooth(3.0,4.2,t)-0.17*smooth(4.4,dur,t))
    f*=1+0.012*np.sin(TAU*5.3*t+r.uniform(0,6))*smooth(1.5,2.8,t)
    ph=TAU*np.cumsum(f)/SR; s=np.zeros_like(t)
    for h in range(1,13):
        w=np.exp(-((h*f-950)/800)**2)/h**0.4; s+=w*np.sin(h*ph)
    env=smooth(0,0.8,t)*(1-smooth(dur-2.2,dur,t))*(1-0.12*smooth(2,3,t))
    br=sosfilt(butter(2,[700,2200],btype='band',fs=SR,output='sos'),r.standard_normal(len(t)))*0.06*env
    s=s*env/ (np.abs(s).max()+1e-6)+br; return sosfilt(butter(2,3200,fs=SR,output='sos'),s).astype(np.float32)
wt=sched(6,81,0.25)
for i,t in enumerate(wt):
    p=[-0.8,0.7,-0.6,0.8,-0.75,0.65][i]; place(wolf(base=rng.uniform(330,400),seed=i),t,p,0.7,0.11)
    if i%2==1: place(wolf(dur=5.5,base=rng.uniform(300,350),seed=100+i),t+rng.uniform(1.6,3.0),p*0.9,0.75,0.08)
lg('owl birds')
def hoot(dur=0.45,f0=400):
    t=np.arange(int(SR*dur))/SR; f=f0*(1-0.06*t/dur); ph=TAU*np.cumsum(f)/SR
    s=np.sin(ph)+0.25*np.sin(2*ph)+0.06*np.sin(3*ph); env=smooth(0,0.08,t)*(1-smooth(dur*0.55,dur,t)); return (s*env*0.5).astype(np.float32)
for t in sched(6,91,0.2):
    out=np.zeros(int(SR*5.0),np.float32)
    for off,d in ((0,0.45),(1.0,0.28),(1.45,0.28),(2.25,0.7)):
        h=hoot(d); i0=int(off*SR); out[i0:i0+len(h)]+=h
    place(out,t,rng.choice([-0.7,0.7]),0.6,0.09)
def note(dur,fa,fb,am=0):
    t=np.arange(int(SR*dur))/SR; f=fa+(fb-fa)*smooth(0,1,t/dur); ph=TAU*np.cumsum(f)/SR
    s=np.sin(ph)+0.12*np.sin(2*ph); env=smooth(0,0.15,t/dur)*(1-smooth(0.5,1,t/dur))
    if am: s*=1+0.5*np.sin(TAU*am*t)
    return (s*env).astype(np.float32)
for i,t in enumerate(sched(20,95,0.1)):
    r=np.random.default_rng(300+i); typ=i%3; parts=[]
    if typ==0:
        for _ in range(int(r.integers(3,6))):
            f=r.uniform(2400,3600); parts.append(note(r.uniform(0.09,0.2),f,f*r.uniform(0.8,1.25))); parts.append(np.zeros(int(SR*r.uniform(0.07,0.22)),np.float32))
    elif typ==1:
        parts=[note(0.34,3300,2800),np.zeros(int(SR*0.12),np.float32),note(0.38,2500,3000)]
    else:
        parts=[note(0.8,3700,3300,am=r.uniform(25,38))]
    ph=np.concatenate(parts); place(ph,t,r.uniform(-0.8,0.8),0.45,0.035)
# ================= MASTER =================
lg('master')
dec=slice(0,N,97); rms=float(np.sqrt(((L[dec]**2+R[dec]**2)/2).mean())); print('mix rms',rms)
sc=0.075/rms
w=wave.open(f'campfire_ambient_{DUR}s.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
pk=0
for s in range(0,N,SR*30):
    a=np.tanh(L[s:s+SR*30]*sc*1.0)*0.97; b=np.tanh(R[s:s+SR*30]*sc*1.0)*0.97; pk=max(pk,float(np.abs(a).max()),float(np.abs(b).max()))
    w.writeframes((np.stack([a,b],1)*32767).astype(np.int16).tobytes())
w.close()
print('peak',pk,'seam L',float(L[0]*sc),float(L[-1]*sc),'typ step',float(np.abs(np.diff(L[:SR*5])).mean()*sc))
lg('done')
