import numpy as np, wave
SR=44100; D=600; N=SR*D; rng=np.random.default_rng(2026)
t=(np.arange(N,dtype=np.float64)/SR).astype(np.float32)
def cyc(n,ph=0): return np.sin(2*np.pi*n*t/D+ph).astype(np.float32)
def circ_bp(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    m=(1/(1+(lo/np.maximum(f,1e-3))**4))*(1/(1+(f/hi)**4)); return np.fft.irfft(X*m,len(x)).astype(np.float32)
def noise(lo,hi):
    x=circ_bp(rng.standard_normal(N).astype(np.float32),lo,hi); return x/np.abs(x).max()
def put(buf,x,t0):
    i=int(t0*SR)%N; n=len(x)
    if i+n<=N: buf[i:i+n]+=x
    else:
        k=N-i; buf[i:]+=x[:k]; buf[:n-k]+=x[k:]
def sbp(n,lo,hi):
    X=np.fft.rfft(n); f=np.fft.rfftfreq(len(n),1/SR); m=(1/(1+(lo/np.maximum(f,1e-3))**4))*(1/(1+(f/hi)**4)); return np.fft.irfft(X*m,len(n))
L=np.zeros(N,np.float32); R=np.zeros(N,np.float32)
# --- water bed
for B,ph in ((L,0.0),(R,1.7)):
    w=noise(120,1300)
    am=0.55+0.25*cyc(75,ph)+0.2*cyc(55,ph*1.3+1)   # waves ~8s & ~11s
    B+=0.20*w*am
# --- gentle wind
for B,ph in ((L,0.5),(R,2.2)):
    w=noise(250,900); B+=0.05*w*(0.5+0.5*cyc(7,ph))*(0.6+0.4*cyc(11,ph+1))
# --- deep pad (sleepy)
def q(f): return round(f*D)/D
chord=[(55,.5),(82.5,.4),(110,.5),(164.8,.35),(220,.25),(329.6,.12)]
for k,(f,g) in enumerate(chord):
    for B,d in ((L,0.0),(R,0.17)):
        ff=q(f+d+0.05*k)
        env=0.6+0.4*cyc(3+k,k)
        B+=(0.055*g*np.sin(2*np.pi*ff*t+0.8*np.sin(2*np.pi*q(0.1)*t))*env).astype(np.float32)
# --- event buses (go through reverb)
EL=np.zeros(N,np.float32); ER=np.zeros(N,np.float32)
def tn(d): return np.arange(int(d*SR))/SR
# lapping plops
for _ in range(int(D*0.32)):
    d=rng.uniform(0.25,0.6); tt=tn(d); f0=rng.uniform(180,320); f=f0+f0*1.6*(1-np.exp(-tt*14))
    s=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-tt*rng.uniform(9,16))
    nz=sbp(rng.standard_normal(len(tt)),300,1500)*np.exp(-tt*20)*0.5
    x=((s+nz)*np.minimum(1,tt/0.02)*rng.uniform(0.02,0.07)).astype(np.float32); p=rng.uniform(0.2,0.8)
    t0=rng.uniform(0,D); put(EL,x*(1-p),t0); put(ER,x*p,t0)
# fish splashes
for _ in range(int(D/80)):
    d=0.7; tt=tn(d); nz=sbp(rng.standard_normal(len(tt)),150,1400)*np.exp(-tt*7)*np.minimum(1,tt/0.01)
    x=nz.astype(np.float32)*0.07; p=rng.uniform(0.2,0.8); t0=rng.uniform(0,D); put(EL,x*(1-p),t0); put(ER,x*p,t0)
    for j in range(3):
        dd=tn(0.25); pl=np.sin(2*np.pi*(500+j*120+200*(1-np.exp(-dd*20)))*dd)*np.exp(-dd*18)*0.025
        put(EL,pl.astype(np.float32)*(1-p),t0+0.25+0.18*j); put(ER,pl.astype(np.float32)*p,t0+0.25+0.18*j)
# frogs
for _ in range(int(D/18)):
    t0=rng.uniform(0,D); p=rng.uniform(0.1,0.9); f0=rng.uniform(150,230)
    for j in range(int(rng.integers(2,4))):
        tt=tn(0.22); s=np.sin(2*np.pi*(f0*(1+0.3*tt)*tt))*(0.5+0.5*np.sin(2*np.pi*28*tt))*np.sin(np.pi*tt/0.22)
        s=sbp(s,100,700)*0.05; put(EL,s.astype(np.float32)*(1-p),t0+j*0.26); put(ER,s.astype(np.float32)*p,t0+j*0.26)
# distant owl
for _ in range(int(D/110)):
    t0=rng.uniform(5,D-5); p=rng.uniform(0.2,0.8)
    for j,(dl,dur) in enumerate(((0,0.35),(0.5,0.35),(1.05,0.9))):
        tt=tn(dur); h=np.sin(2*np.pi*(340-35*tt/dur)*tt+2*np.sin(2*np.pi*5*tt))*np.sin(np.pi*tt/dur)**1.6
        h=(sbp(h,150,900)*0.05).astype(np.float32); put(EL,h*(1-p),t0+dl); put(ER,h*p,t0+dl)
# reverb: exponential noise IR (circular)
def reverb(x,rt=2.6,mix=1.0):
    n=int(rt*SR); ir=rng.standard_normal(n)*np.exp(-np.arange(n)/(SR*rt/6.9)); ir=sbp(ir,100,3500)
    X=np.fft.rfft(x); I=np.fft.rfft(np.pad(ir/np.abs(ir).sum()*8,(0,N-n)))
    return np.fft.irfft(X*I,N).astype(np.float32)
for B,E in ((L,EL),(R,ER)):
    B+=E*1.0+reverb(E)*1.2
del EL,ER
# --- crickets (dry, spread)
CL=np.zeros(N,np.float32); CR=np.zeros(N,np.float32)
for c,(f0,per,p,n_lfo) in enumerate([(4350,1.15,0.25,3),(4700,0.95,0.75,5),(3950,1.4,0.5,4),(5000,1.25,0.4,2)]):
    nper=int(D/per); per2=D/nper
    lfo=cyc(n_lfo,c*1.3)
    for k in range(nper):
        base=k*per2+rng.uniform(0,0.15)
        lv=0.5+0.5*lfo[int(base*SR)%N]
        if lv<0.35 or rng.random()<0.3: continue
        for j in range(4):
            tt=tn(0.04); ch=np.sin(2*np.pi*f0*tt)*np.sin(np.pi*tt/0.04)**2*(0.6+0.4*np.sin(2*np.pi*90*tt))
            x=(ch*0.011*lv).astype(np.float32); put(CL,x*(1-p),base+j*0.065); put(CR,x*p,base+j*0.065)
L+=CL; R+=CR; del CL,CR
# slow breathing across the loop (periodic)
br=0.9+0.1*cyc(10,0); L*=br; R*=br
pk=max(np.abs(L).max(),np.abs(R).max()); L*=0.55/pk; R*=0.55/pk
w=wave.open('lake_night_loop10.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
st=np.empty((N,2),'<i2'); st[:,0]=(L*32767).astype('<i2'); st[:,1]=(R*32767).astype('<i2'); w.writeframes(st.tobytes()); w.close()
print('ok',20*np.log10(np.sqrt((L**2).mean())))
