import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, scipy.signal as sg, subprocess, sys, math, gc
SR=44100; T=900; NC=4; W_=1920; TL=972
U=ZT+'uploads/'
PASSES=[('P',1,35,225),('F',-1,170,190),('P',-1,290,225),('F',1,455,190),('P',1,520,225),('P',-1,700,225),('F',-1,810,190)]
def bp(x,lo,hi,o=2): return sg.sosfilt(sg.butter(o,[lo,hi],'band',fs=SR,output='sos'),x)
def lp(x,f,o=2): return sg.sosfilt(sg.butter(o,f,'low',fs=SR,output='sos'),x)
def cyc(keys,t):
    t=t%T
    for (t0,v0),(t1,v1) in zip(keys[:-1],keys[1:]):
        if t0<=t<=t1:
            u=(t-t0)/(t1-t0); u=u*u*(3-2*u); return v0*(1-u)+v1*u
    return keys[0][1]
BIRD=[(0,1.0),(180,0.8),(330,0.25),(420,0.0),(640,0.0),(760,0.12),(840,0.8),(900,1.0)]
CRIK=[(0,0),(330,0),(430,0.7),(480,1),(620,1),(740,0.5),(800,0),(900,0)]
def rms(x): return float(np.sqrt((x.astype(np.float64)**2).mean()))
def gdist(x): return np.exp(-((x-960)/1250)**2)*0.9+0.04
def pan(x): return float(np.clip((x-960)/1100,-1,1))
# ---- pass renderer
def render_pass(kind,d,t0,V,rng):
    pre=3.0; dur=(W_+TL)/V+pre+3.0; n=int(dur*SR); buf=np.zeros((n,2),np.float32); tb=t0-pre
    def xc(t): return (-TL/2+(t-t0)*V) if d==1 else (W_+TL/2-(t-t0)*V)
    def xf(t): return ((t-t0)*V) if d==1 else (W_-(t-t0)*V)
    def add(sig,t,p,g):
        i=int((t-tb)*SR)
        if i<0 or i>=n: return
        s=sig[:n-i]; th=(p+1)*np.pi/4
        buf[i:i+len(s),0]+=(s*g*np.cos(th)).astype(np.float32); buf[i:i+len(s),1]+=(s*g*np.sin(th)).astype(np.float32)
    heavy=(kind=='F')
    def chuff(strong):
        m=int(0.24*SR); t=np.arange(m)/SR
        body=bp(rng.randn(m),170 if heavy else 180,1300,2)*np.exp(-t/(0.065 if heavy else 0.055))*(1-np.exp(-t/0.004))
        thump=np.sin(2*np.pi*((70 if heavy else 78)-30*t)*t)*np.exp(-t/0.08)*(0.9 if heavy else 0.7)
        hiss=bp(rng.randn(m),2500,6000)*np.exp(-t/0.03)*0.18
        return (body*1.6+thump+hiss)*(1.0 if strong else 0.55)
    CH=[chuff(True),chuff(False),chuff(False),chuff(False)]
    t=t0-pre; k=0; tend=t0+(W_+TL)/V+2
    while t<tend:
        x=xf(t); add(CH[k%4],t,pan(x),0.5*gdist(x)*(0.9+0.2*rng.rand()))
        rate=(2.9+0.7*np.clip((t-t0)/6,0,1)) if heavy else (3.4+0.9*np.clip((t-t0)/6,0,1))
        t+=1/rate*(1+0.03*rng.randn()); k+=1
    def clack():
        m=int(0.12*SR); t=np.arange(m)/SR
        return bp(rng.randn(m),900,4200)*np.exp(-t/0.012)*1.2+(np.sin(2*np.pi*1350*t)*np.exp(-t/0.02)*0.35+np.sin(2*np.pi*620*t)*np.exp(-t/0.03)*0.3)
    CK=[clack() for _ in range(6)]
    t=t0+0.3; j=0; gap=0.31 if heavy else 0.37
    while t<t0+(W_+TL)/V-0.5+ (0 if d==1 else 0):
        for dd,a in ((0,1),(0.085,0.8)):
            pos=xc(t)+(rng.rand()-0.5)*TL*0.9
            add(CK[j%6],t+dd,pan(pos),(0.20 if heavy else 0.17)*a*gdist(pos)*(0.8+0.4*rng.rand()))
        j+=1; t+=gap*(1+0.08*rng.randn())
    # rumble
    tt=tb+np.arange(n)/SR; xx=xc(tt)
    env=lp(np.exp(-((xx-960)/1000)**2)*((tt>t0-pre)&(tt<tend)),3,1)
    ru=lp(np.cumsum(rng.randn(n))*0.002,160); ru=ru-lp(ru,18); ru=ru/np.abs(ru).max()*(0.33 if heavy else 0.25)*env
    thp=(np.clip((xx-960)/1100,-1,1)+1)*np.pi/4
    buf[:,0]+=(ru*np.cos(thp)).astype(np.float32); buf[:,1]+=(ru*np.sin(thp)).astype(np.float32)
    # whistle
    f1,f2=(330,415) if heavy else (392,493.9)
    dur_w=2.0 if heavy else 1.6; m=int(dur_w*SR); tw_=np.arange(m)/SR
    vib=1+0.004*np.sin(2*np.pi*5.2*tw_)
    s=sum(a*np.sin(2*np.pi*f*vib*tw_) for f,a in ((f1,1),(f2,0.8),(f1*2,0.3),(f2*2,0.2)))
    ed=np.clip(tw_/0.12,0,1)*np.where(tw_<dur_w-0.5,1,np.exp(-(tw_-(dur_w-0.5))/0.18))
    wh=lp((s*0.22+bp(rng.randn(m),1500,3500)*0.05)*ed,3600)
    tw=t0+2.6; x=xf(tw); add(wh,tw,pan(x),0.55*gdist(x))
    # reverb air
    L=buf[:,0].astype(np.float64); R=buf[:,1].astype(np.float64)
    def tail(xx_):
        y=np.zeros_like(xx_)
        for dl,g in ((0.031,0.25),(0.057,0.2),(0.089,0.15)):
            i=int(dl*SR); y[i:]+=xx_[:-i]*g
        return lp(y,2500)
    buf[:,0]=(L+tail(R)*0.5).astype(np.float32); buf[:,1]=(R+tail(L)*0.5).astype(np.float32)
    return tb,buf
# ---- beds
def load_birds():
    raw=subprocess.run(['ffmpeg','-v','error','-i',U+'3a312f9a-wr-sound-library-deep-forest-birds-singing-no-copyright-infinite-568105.mp3','-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
    b=np.frombuffer(raw,np.float32).reshape(-1,2).copy(); x=8*SR
    f=np.linspace(0,1,x,dtype=np.float32)[:,None]
    loop=b[x:].copy(); loop[-x:]=b[-x:]*np.cos(f*np.pi/2)*0+0  # placeholder replaced below
    core=b[x:-x]; head=b[:x]; tail=b[-x:]
    seam=tail*np.cos(f*np.pi/2)+head*np.sin(f*np.pi/2)
    return np.concatenate([seam,core]).astype(np.float32)   # circular, seamless
BIRDS=load_birds(); BG=0.0335/rms(BIRDS)
def crickets(c0,rng):
    out=np.zeros((T*SR,2),np.float32)
    voices=[(4200+rng.randint(-250,250),rng.uniform(1.9,2.6),rng.uniform(28,36),rng.uniform(-0.8,0.8),rng.rand()*7) for _ in range(7)]
    B=30*SR
    for b0 in range(0,T*SR,B):
        tm=(b0+np.arange(B))/SR+c0*T; lvl=np.array([cyc(CRIK,tm[0]),cyc(CRIK,tm[-1])]).mean()
        if lvl<0.02: continue
        lv=np.interp(tm,[tm[0],tm[-1]],[cyc(CRIK,tm[0]),cyc(CRIK,tm[-1])])
        for f,cr,pr,pn,ph in voices:
            drift=0.03*np.sin(tm*0.07+ph)
            gate=((tm*(cr+drift)+ph)%1<0.38).astype(np.float32)
            pulse=(((tm*pr)%1)<0.55).astype(np.float32)
            env=lp(gate*pulse,900,1).astype(np.float32)
            sig=(np.sin(2*np.pi*f*tm)*env*lv*0.012).astype(np.float32)
            th=(pn+1)*np.pi/4
            out[b0:b0+B,0]+=sig*np.cos(th); out[b0:b0+B,1]+=sig*np.sin(th)
    return out
def river(rng):
    out=np.zeros((T*SR,2),np.float32)
    sos=sg.butter(2,[250,2200],'band',fs=SR,output='sos'); zi=np.zeros((sos.shape[0],2,2))
    B=30*SR
    for b0 in range(0,T*SR,B):
        y,zi=sg.sosfilt(sos,rng.randn(2,B),axis=1,zi=zi)
        m=(0.7+0.3*np.sin(2*np.pi*(b0+np.arange(B))/SR*0.13))
        out[b0:b0+B]=(y.T*m[:,None]*0.0045).astype(np.float32)
    return out
# ---- music decoder
def music(c):
    raw=subprocess.run(['ffmpeg','-v','error','-ss',str(c*T),'-t',str(T),'-i','train_bridge_1h.mp3','-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
    m=np.frombuffer(raw,np.float32).reshape(-1,2).copy()
    if len(m)<T*SR: m=np.concatenate([m,np.zeros((T*SR-len(m),2),np.float32)])
    return m[:T*SR]
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled','-c:a','libmp3lame','-b:a','192k','train_bridge_1h_sfx.mp3'],stdin=subprocess.PIPE)
MUS_RMS=0.0956; TRSCALE=1.02
for c in range(NC):
    rng=np.random.RandomState(100+c)
    mix=music(c)*0.9
    # birds (periodic bed following day cycle)
    i0=(c*T*SR)%len(BIRDS)
    for b0 in range(0,T*SR,10*SR):
        idx=(i0+b0+np.arange(10*SR))%len(BIRDS)
        g=np.interp(np.arange(10*SR),[0,10*SR-1],[cyc(BIRD,b0/SR),cyc(BIRD,(b0+10*SR)/SR)])
        mix[b0:b0+10*SR]+=(BIRDS[idx]*BG*g[:,None]).astype(np.float32)
    mix+=crickets(c,rng); mix+=river(rng)
    for kind,d,t0,V in PASSES:
        tb,buf=render_pass(kind,d,t0,V,rng); i=int(tb*SR)
        a=max(0,-i); n=min(len(buf)-a,T*SR-(i+a))
        mix[i+a:i+a+n]+=buf[a:a+n]*(TRSCALE*MUS_RMS*0.55/0.0516)
        print('cycle',c,kind,t0,file=sys.stderr)
    if c==0:
        n=3*SR; mix[:n]*=np.linspace(0,1,n,dtype=np.float32)[:,None]
    if c==NC-1:
        n=8*SR; mix[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]
    enc.stdin.write(mix.tobytes()); del mix; gc.collect()
enc.stdin.close(); enc.wait()
