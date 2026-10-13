import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR=44100; COUNT=300; TOT=306; n=TOT*SR; rng=np.random.default_rng(21); t=np.arange(n)/SR
mtof=lambda m:440*2**((m-69)/12)
L=np.zeros(n); R=np.zeros(n)
def add(sig,pan,start):
    start=int(start*SR); e=min(n,start+len(sig)); s=sig[:e-start]
    L[start:e]+=s*np.cos(pan*np.pi/2); R[start:e]+=s*np.sin(pan*np.pi/2)
def env(dur,a,r):
    m=int(dur*SR); x=np.ones(m); na=int(a*SR); nr=int(r*SR); x[:na]=np.linspace(0,1,na)**1.5; x[-nr:]=np.minimum(x[-nr:],np.linspace(1,0,nr)**1.5); return x
def box(m,dur=2.2,g=0.1):   # music-box pluck
    x=np.arange(int(dur*SR))/SR; f=mtof(m)
    return (np.sin(2*np.pi*f*x)+0.35*np.sin(2*np.pi*f*2.0*x)*np.exp(-x*5)+0.2*np.sin(2*np.pi*f*4.1*x)*np.exp(-x*12))*np.exp(-x*2.6)*np.minimum(1,x*600)*g
def pad(ms,dur,g):
    x=np.arange(int(dur*SR))/SR; y=0
    for m in ms:
        for d in (-0.0015,0.0015): y=y+np.sin(2*np.pi*mtof(m)*(1+d)*x+rng.uniform(0,6.28))
    return y/len(ms)/2*env(dur,3,3)*g
chords=[([48,55,60,64],[60,64,67,72]),([45,52,57,60],[57,60,64,69]),([41,48,53,57],[60,65,69,72]),([43,50,55,59],[59,62,67,71])]
bar=4.0
for bi in range(int(TOT/bar)+1):
    ch,arp=chords[bi%4]; st=bi*bar
    add(pad(ch,bar+3,0.07),0.5,st)
    # music-box arpeggio, sparser at start and end
    for k in range(8):
        tt=st+k*0.5
        if tt>COUNT+1: break
        if rng.random()<0.72:
            m=arp[[0,1,2,3,2,1,3,2][k]]+(12 if rng.random()<0.15 else 0)
            add(box(m,g=0.07),0.25+0.5*rng.random(),tt+rng.uniform(0,0.03))
# soft rolling rumble following the track slope
b,a=sg.butter(2,[90/(SR/2),500/(SR/2)],'band')
ru=sg.lfilter(b,a,rng.normal(size=n))*(0.012+0.010*np.sin(2*np.pi*t/9)**2)
L+=ru; R+=np.roll(ru,300)
# minute gates
for k in range(5):
    for i,m in enumerate([81,88]):
        x=np.arange(int(4*SR))/SR
        s=(np.sin(2*np.pi*mtof(m)*x)+0.3*np.sin(2*np.pi*mtof(m)*2.76*x)*np.exp(-x*3))*np.exp(-x*1.1)*0.07
        add(s,0.4+0.2*i,60*k+0.05*i)
# finish chime + sparkle
for i,m in enumerate([76,83,88,91,95]):
    x=np.arange(int(6*SR))/SR
    s=(np.sin(2*np.pi*mtof(m)*x)+0.3*np.sin(2*np.pi*mtof(m)*2.76*x)*np.exp(-x*3))*np.exp(-x*0.8)*0.10
    add(s,0.25+0.12*i,COUNT+0.12*i)
irn=int(3.0*SR); ir=rng.normal(size=irn)*np.exp(-np.arange(irn)/SR*1.8); ir[0]=0
b3,a3=sg.butter(1,5000/(SR/2),'low'); ir=sg.lfilter(b3,a3,ir); ir/=np.sqrt((ir**2).sum())
wl=sg.fftconvolve(L,ir)[:n]; wr=sg.fftconvolve(R,ir[::-1])[:n]
out=np.stack([L*0.55+wl*0.8,R*0.55+wr*0.8],1)
fi=np.minimum(1,t/2.5)**2; fo=np.minimum(1,(TOT-t)/2.5); out*=(fi*fo)[:,None]
rms=np.sqrt((out[:COUNT*SR]**2).mean()); out*=10**(-20/20)/rms; out=np.tanh(out*1.1)
print('peak',abs(out).max())
wf.write('marble_audio.wav',SR,(out*32767*0.95).astype(np.int16))
