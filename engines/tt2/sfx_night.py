import numpy as np, scipy.signal as sg, subprocess, sys
SR=44100; DUR=21; N=SR*DUR
rng=np.random.RandomState(3)
def bp(x,lo,hi,o=2): 
    b=sg.butter(o,[lo,hi],'band',fs=SR,output='sos'); return sg.sosfilt(b,x)
def lp(x,f,o=2): return sg.sosfilt(sg.butter(o,f,'low',fs=SR,output='sos'),x)
L=np.zeros(N); R=np.zeros(N)
def add(sig,t,pan,gain):
    i=int(t*SR)
    if i<0 or i>=N: return
    s=sig[:N-i]; th=(pan+1)*np.pi/4
    L[i:i+len(s)]+=s*gain*np.cos(th); R[i:i+len(s)]+=s*gain*np.sin(th)
# train geometry, seconds in clip; passenger train crossing left->right, 225 px/s, 972 px long
W_=1920; TL=972; V=225; T0=4.5
def xc(t): return -TL/2+(t-T0)*V        # centre
def xf(t): return -TL+(t-T0)*V+TL       # engine (front)
def gdist(x): return np.exp(-((x-960)/1250)**2)*0.9+0.04
def pan(x): return float(np.clip((x-960)/1100,-1,1))
# chuff templates (strong / weak)
def chuff(strong):
    n=int(0.22*SR); t=np.arange(n)/SR
    nz=rng.randn(n); body=bp(nz,180,1400)*np.exp(-t/0.055)*(1-np.exp(-t/0.004))
    thump=np.sin(2*np.pi*(78-30*t)*t)*np.exp(-t/0.07)*0.7
    hiss=bp(rng.randn(n),2500,6000)*np.exp(-t/0.03)*0.18
    s=body*1.6+thump+hiss
    return s*(1.0 if strong else 0.55)
CH=[chuff(True),chuff(False),chuff(False),chuff(False)]
# chuff events: speed ramps up a little
t=T0-1.0; k=0
while t<T0+14:
    x=xf(t); add(CH[k%4],t,pan(x),0.5*gdist(x)*(0.9+0.2*rng.rand()))
    t+=1/ (3.4+ 0.9*np.clip((t-T0)/6,0,1)) * (1+0.03*rng.randn()); k+=1
# wheel clacks: clickety pattern along the train body
def clack():
    n=int(0.12*SR); t=np.arange(n)/SR
    c=bp(rng.randn(n),900,4200)*np.exp(-t/0.012)
    r=np.sin(2*np.pi*1350*t)*np.exp(-t/0.02)*0.35+np.sin(2*np.pi*620*t)*np.exp(-t/0.03)*0.3
    return (c*1.2+r)
CK=[clack() for _ in range(6)]
t=T0+0.5; j=0
while t<T0+13:
    # each pass-through of rail joint: bogie pair double-click
    for d,a in ((0,1),(0.085,0.8)):
        pos=xc(t)+ (rng.rand()-0.5)*TL*0.9     # random bogie along train
        add(CK[j%6],t+d,pan(pos),0.17*a*gdist(pos)*(0.8+0.4*rng.rand()))
    j+=1; t+=0.37*(1+0.08*rng.randn())/(1+0.0)
# rumble bed
n=N; ru=lp(np.cumsum(rng.randn(n))*0.002,160); ru=ru-lp(ru,18)
tt=np.arange(n)/SR; xx=xc(tt); env=np.exp(-((xx-960)/1000)**2)*(tt>T0-1)*(tt<T0+14)
env=lp(env,3,1)
th=pan(960)
rumble=ru/np.abs(ru).max()*0.25*env
# stereo pan on rumble following train
pn=np.clip((xx-960)/1100,-1,1); thp=(pn+1)*np.pi/4
L+=rumble*np.cos(thp); R+=rumble*np.sin(thp)
# whistle: two-tone, soft, distant, at engine entering
def whistle(dur=1.6):
    n=int(dur*SR); t=np.arange(n)/SR
    vib=1+0.004*np.sin(2*np.pi*5.2*t)
    s=sum(a*np.sin(2*np.pi*f*vib*t) for f,a in ((392,1),(493.9,0.8),(784,0.3),(987,0.2)))
    ed=np.clip(t/0.12,0,1)*np.where(t<dur-0.5,1,np.exp(-(t-(dur-0.5))/0.18))
    br=bp(rng.randn(n),1500,3500)*0.05
    return lp((s*0.22+br)*ed,3600)
wh=whistle()
for tw,gw in ((T0+2.6,0.55),):
    x=xf(tw); add(wh,tw,pan(x),gw*gdist(x))
# echo on whistle via simple delays
ech=np.zeros((2,N))
for d,g in ((0.23,0.28),(0.47,0.15)):
    i=int(d*SR); ech[0,i:]+=L[:-i]*g*0; # keep dry; reverb only via tails below
# subtle stereo "air" reverb: lowpassed delayed copy
def tail(x):
    y=np.zeros_like(x)
    for d,g in ((0.031,0.25),(0.057,0.2),(0.089,0.15)):
        i=int(d*SR); y[i:]+=x[:-i]*g
    return lp(y,2500)
L=L+tail(R)*0.5; R=R+tail(L)*0.5
tr=np.stack([L,R],1)
# music
raw=subprocess.run(['ffmpeg','-v','error','-ss','1530','-t',str(DUR),'-i','../trainaudio/train_bridge_1h.mp3','-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
mus=np.frombuffer(raw,np.float32).reshape(-1,2)[:N].copy()
def rms(x): return np.sqrt((x**2).mean())
trs=tr[int(T0*SR):int((T0+12)*SR)]

tr*= (rms(mus)*0.55)/rms(trs)       # train sits ~5 dB under the music while passing
mix=mus*0.9+tr
fade=np.ones(N)
pk=np.abs(mix).max(); print('peak',pk,file=sys.stderr)
mix*=0.85/pk if pk>0.85 else 1
sol=(tr*fade[:,None]); 
open('night.f32','wb').write(mix.astype(np.float32).tobytes())
