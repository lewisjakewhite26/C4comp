import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, scipy.signal as sg, subprocess, sys, math, gc
SR=44100; T=900; TAU=2*math.pi
MUSIC=ZT+'rainmix/rain_house_1h_music.mp3'
def om(k): return TAU*k/T
def inten(t): return 0.72+0.2*np.sin(om(5)*t+0.6)+0.08*np.sin(om(17)*t)
def bp(x,lo,hi,o=2,ax=-1): return sg.sosfilt(sg.butter(o,[lo,hi],'band',fs=SR,output='sos'),x,axis=ax)
def lp(x,f,o=2,ax=-1): return sg.sosfilt(sg.butter(o,f,'low',fs=SR,output='sos'),x,axis=ax)
def hp(x,f,o=2,ax=-1): return sg.sosfilt(sg.butter(o,f,'high',fs=SR,output='sos'),x,axis=ax)
def pan(x,p):  # mono -> stereo
    th=(p+1)*np.pi/4; return np.stack([x*np.cos(th),x*np.sin(th)],1).astype(np.float32)
# ---- events from the video's own schedule
THUNDER=[(212.0,3.6,1.0),(541.0,4.6,0.8),(777.0,3.0,1.1)]   # flash time, delay, strength
DR=[(1093,165,3.1,0.4),(1360,157,3.9,1.7),(860,186,2.8,2.2),(650,292,4.4,0.9),(463,236,3.6,2.9)]
def th_of(t): return 2.4*np.sin(om(214)*t+0.4)*(0.55+0.45*np.sin(om(6)*t+1.0))+0.8*np.sin(om(97)*t)
def chime_strikes():
    t=np.arange(0,T,0.02); a=np.abs(th_of(t))
    pk=[i for i in range(1,len(t)-1) if a[i]>a[i-1] and a[i]>=a[i+1] and a[i]>2.7]
    out=[]; last=-9
    for i in pk:
        if t[i]-last<3.5: continue
        out.append((t[i],min(1,(a[i]-2.5)/1.4))); last=t[i]
    return out
CHIME=chime_strikes()
def pr_events():
    pr=np.random.RandomState(8); RIP=[]
    for i in range(int(900*5.5)): RIP.append((pr.uniform(0,T),pr.uniform(1250,1580),pr.uniform(715,770),pr.uniform(0.6,1.0)))
    RIP.sort(); return RIP
RIP=pr_events()
def place(buf,sig,t,p,g,t0):
    i=int((t-t0)*SR)
    if i>=len(buf) or i+len(sig)<=0: return
    a=max(0,-i); s=sig[a:]; j=i+a; n=min(len(s),len(buf)-j)
    if n<=0: return
    if s.ndim==2: buf[j:j+n]+=s[:n]*g; return
    th=(p+1)*np.pi/4; buf[j:j+n,0]+=s[:n]*g*math.cos(th); buf[j:j+n,1]+=s[:n]*g*math.sin(th)
def bell(rng):
    m=int(3.2*SR); t=np.arange(m)/SR
    s=sum(a*np.sin(TAU*f*t+rng.rand()*6)*np.exp(-t/d) for f,a,d in ((1865,1,1.6),(1865*1.006,0.7,1.4),(2790,0.45,0.9),(4340,0.25,0.5),(5900,0.1,0.25)))
    return (s*(1-np.exp(-t/0.002))).astype(np.float32)
def plop(rng,f0):
    m=int(0.09*SR); t=np.arange(m)/SR
    return (np.sin(TAU*(f0*(1+0.9*t/0.09)*t))*np.exp(-t/0.022)*(1-np.exp(-t/0.0015))).astype(np.float32)
def thunder(rng,strength):
    d=9.5; m=int(d*SR); t=np.arange(m)/SR
    n=rng.randn(m)
    low=lp(n,90,2)+0.6*bp(n,90,260,2)+0.15*bp(n,260,700,2)
    # lobes: a few rolling swells
    env=np.zeros(m)
    for k,(st,w,a) in enumerate(((0.0,1.4,1.0),(1.6,1.8,0.7),(3.4,2.2,0.55),(5.8,2.4,0.3))):
        env+=a*np.exp(-0.5*((t-st-w)/ (w*0.55))**2)*(1+0.25*np.sin(TAU*(0.9+0.3*k)*t))
    env*=np.exp(-t/6.5)*(1-np.exp(-t/0.35)); env=lp(env,6,1)
    crack=bp(rng.randn(m),350,1200,2)*np.exp(-t/0.12)*(1-np.exp(-t/0.01))*0.25
    s=(low*env*1.0+crack)
    s=s/np.abs(s).max()
    L=s; R=np.roll(s,int(0.012*SR))*0.9+lp(rng.randn(m),400)*env*0.05
    return (np.stack([L,R],1)*strength).astype(np.float32)
def bed(c0,n,rng,tstart):
    """rain bed n samples starting at absolute time tstart (s); intensity follows tstart%900."""
    out=np.zeros((n,2),np.float32); B=20*SR
    zis={}
    for b0 in range(0,n,B):
        nb=min(B,n-b0); tt=(tstart+(b0+np.arange(nb))/SR)
        I=inten(tt)[:,None].astype(np.float32)
        x=rng.randn(nb,2)
        wash=lp(x,4200,3,0)*0.28+lp(x,1100,2,0)*0.55+lp(x,240,2,0)*0.5
        # drop patter: sparse impulses, density follows intensity
        dens=300*I[:,0]/SR
        imp=(rng.rand(nb,2)<dens[:,None])*rng.rayleigh(1.0,(nb,2))
        pat=bp(imp,900,3600,2,0)*6.0
        # roof tiles (left-weighted), larger drops
        d2=26*I[:,0]/SR; imp2=(rng.rand(nb)<d2)*rng.rayleigh(1.0,nb)
        roof=lp(bp(imp2,500,2600,2),2800)*7.0
        roofs=np.stack([roof*0.8,roof*0.35],1)
        seg=(wash*0.9+pat)*(I**1.3)+roofs*I
        out[b0:b0+nb]=seg.astype(np.float32)
    return out
RG=0.12
MG=0.5
def build(t0,t1,music_slice_from=None):
    n=int((t1-t0)*SR); rng=np.random.RandomState(int(t0)%997+7)
    R=bed(0,n,rng,t0); 
    cur=np.zeros((n,2),np.float32)
    for k in range(int(t0//T)-1,int(t1//T)+1):
        base=k*T
        for x,y,per,ph in DR:
            nn=int(round(T/per)); pp=T/nn
            for j in range(nn):
                tt=base+ph+j*pp+0.52
                if t0-1<tt<t1: place(cur,plop(rng,rng.uniform(1300,2100)),tt,(x-960)/1100,0.10*(0.7+0.6*rng.rand()),t0)
        for tb,x,y,s in RIP:
            tt=base+tb
            if t0-1<tt<t1: place(cur,plop(rng,rng.uniform(600,1000)),tt,(x-960)/1100,0.012*s*(inten(tb)),t0)
        for tc,a in CHIME:
            tt=base+tc
            if t0-4<tt<t1: place(cur,bell(rng),tt,0.15,0.055*a,t0)
        for tf,dl,s in THUNDER:
            tt=base+tf+dl
            if t0-10<tt<t1: place(cur,thunder(rng,0.75*s),tt,0,1.0,t0)
    return R,cur
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='test':
        t0,t1=float(sys.argv[2]),float(sys.argv[3])
        R,cur=build(t0,t1)
        raw=subprocess.run(['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i',MUSIC,'-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
        m=np.frombuffer(raw,np.float32).reshape(-1,2)
        mix=np.zeros_like(R); mix[:len(m)]=m[:len(R)]*MG
        r=lambda x: float(np.sqrt((x.astype(np.float64)**2).mean()))
        print('rms music %.4f bed %.4f fx %.4f'%(r(mix),r(R),r(cur)),file=sys.stderr)
        mix=mix+R*RG+cur
        out=sys.argv[4]
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled','-c:a','libmp3lame','-b:a','192k',out],stdin=subprocess.PIPE); p.stdin.write(mix.tobytes()); p.stdin.close(); p.wait()
    if mode=='full':
        TOT=3550.0; BL=300
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled','-c:a','libmp3lame','-b:a','192k','rain_house_1h_with_rain.mp3'],stdin=subprocess.PIPE)
        for t0 in range(0,int(TOT),BL):
            t1=min(t0+BL,TOT); R,cur=build(t0,t1)
            raw=subprocess.run(['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i',MUSIC,'-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
            m=np.frombuffer(raw,np.float32).reshape(-1,2); mix=np.zeros_like(R); k=min(len(m),len(R)); mix[:k]=m[:k]*MG
            mix=mix+R*RG+cur
            if t0==0:
                n=4*SR; mix[:n]*=np.linspace(0,1,n,dtype=np.float32)[:,None]
            if t1>=TOT:
                n=8*SR; mix[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]
            p.stdin.write(mix.tobytes()); del R,cur,mix; gc.collect(); print('block',t0,file=sys.stderr,flush=True)
        p.stdin.close(); p.wait()
    if mode=='stems':
        TOT=3550.0; BL=300
        def enc(name): return subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled','-c:a','libmp3lame','-b:a','128k',name],stdin=subprocess.PIPE)
        pa=enc('rain_house_1h_rain_only.mp3'); pb=enc('rain_house_1h_accents_only.mp3')
        for t0 in range(0,int(TOT),BL):
            t1=min(t0+BL,TOT); R,cur=build(t0,t1)
            a=R*RG*2.2+cur*1.0; b=cur*1.0
            for x in (a,b):
                if t0==0:
                    n=4*SR; x[:n]*=np.linspace(0,1,n,dtype=np.float32)[:,None]
                if t1>=TOT:
                    n=8*SR; x[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]
            pa.stdin.write(a.tobytes()); pb.stdin.write(b.tobytes()); del R,cur,a,b; gc.collect(); print('block',t0,file=sys.stderr,flush=True)
        pa.stdin.close(); pb.stdin.close(); pa.wait(); pb.wait()
