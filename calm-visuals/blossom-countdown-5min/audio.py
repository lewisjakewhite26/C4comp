import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR=44100; COUNT=300; TOT=306; n=TOT*SR; rng=np.random.default_rng(9); t=np.arange(n)/SR
mtof=lambda m:440*2**((m-69)/12)
L=np.zeros(n); R=np.zeros(n)
def add(sig,pan,start):
    e=min(n,start+len(sig)); s=sig[:e-start]
    L[start:e]+=s*np.cos(pan*np.pi/2); R[start:e]+=s*np.sin(pan*np.pi/2)
def env(dur,a,r):
    m=int(dur*SR); x=np.ones(m); na=int(a*SR); nr=int(r*SR); x[:na]=np.linspace(0,1,na)**1.5; x[-nr:]=np.minimum(x[-nr:],np.linspace(1,0,nr)**1.5); return x
def tone(f,dur,harm=(1,0.35,0.15,0.08),det=0.0):
    tt=np.arange(int(dur*SR))/SR; y=0
    for k,a in enumerate(harm,1): y=y+a*np.sin(2*np.pi*f*k*(1+det)*tt+rng.uniform(0,6.28))
    return y
chords=[[50,57,61,64,69],[47,54,59,62,66,69],[43,50,55,59,62,69],[45,52,57,59,64,69]]
seg=12
for ci in range(26):
    ch=chords[ci%4]; st=ci*seg; dur=seg+5
    for k,m in enumerate(ch):
        for det in (-0.0012,0.0012):
            s=tone(mtof(m),dur,det=det)*env(dur,4.5,5)*(0.05 if m<55 else 0.035)
            add(s,0.5+0.35*(det/0.0012)*(0.6 if k%2 else -0.6),int(st*SR))
    add(tone(mtof(ch[0]-12),dur,harm=(1,0.2))*env(dur,4,5)*0.08,0.5,int(st*SR))
# wind shaped like the simulation's base gusts
base=95+110*np.sin(2*np.pi*t/17+0.5)+60*np.sin(2*np.pi*t/6.3+1.7)
amp=np.clip((base+80)/280,0.08,1.0)
b,a=sg.butter(2,[160/(SR/2),1100/(SR/2)],'band')
w1=sg.lfilter(b,a,rng.normal(size=n))*(0.01+0.045*amp**1.5); w2=sg.lfilter(b,a,rng.normal(size=n))*(0.01+0.045*np.roll(amp,int(0.35*SR))**1.5)
hb,ha=sg.butter(2,[1800/(SR/2),4200/(SR/2)],'band')
sh=sg.lfilter(hb,ha,rng.normal(size=n))*0.006*amp**2
L+=w1+sh; R+=w2+sh
# landing plinks: pentatonic, register rises as the screen fills
lands=np.load('lands.npy')
pent=[0,2,4,7,9]
idx=np.arange(len(lands)); last=-9
for i in idx[::1]:
    tl,col,hue=lands[i]
    if tl-last<0.22 or rng.random()>0.045 or tl>COUNT+0.5: continue
    last=tl
    prog=min(1,tl/COUNT); deg=pent[int(hue)%5]; octv=int(prog*2.2)
    m=62+deg+12*(octv%3)+ (0 if prog<0.7 else 0)
    dur=3.5; x=np.arange(int(dur*SR))/SR
    s=(np.sin(2*np.pi*mtof(m)*x)+0.3*np.sin(2*np.pi*mtof(m)*2.76*x)*np.exp(-x*4))*np.exp(-x*1.6)*0.04*np.minimum(1,x*400)
    pan=0.15+0.7*col/320.0; add(s,pan,int(tl*SR))
# end chime
for i,m in enumerate([74,81,86,90]):
    dur=6; x=np.arange(int(dur*SR))/SR
    s=(np.sin(2*np.pi*mtof(m)*x)+0.3*np.sin(2*np.pi*mtof(m)*2.76*x)*np.exp(-x*3))*np.exp(-x*0.7)*0.09
    add(s,0.3+0.13*i,int((COUNT+0.05*i)*SR))
irn=int(3.2*SR); ir=rng.normal(size=irn)*np.exp(-np.arange(irn)/SR*1.7); ir[0]=0
b3,a3=sg.butter(1,5000/(SR/2),'low'); ir=sg.lfilter(b3,a3,ir); ir/=np.sqrt((ir**2).sum())
wl=sg.fftconvolve(L,ir)[:n]; wr=sg.fftconvolve(R,ir[::-1])[:n]
out=np.stack([L*0.45+wl*0.9,R*0.45+wr*0.9],1)
fi=np.minimum(1,t/2.5)**2; fo=np.minimum(1,(TOT-t)/2.5); out*=(fi*fo)[:,None]
rms=np.sqrt((out[:COUNT*SR]**2).mean()); out*=10**(-20/20)/rms; out=np.tanh(out*1.1)
print('peak',abs(out).max())
wf.write('particles_audio.wav',SR,(out*32767*0.95).astype(np.int16))
