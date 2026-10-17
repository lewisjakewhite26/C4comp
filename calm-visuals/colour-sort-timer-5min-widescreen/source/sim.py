import numpy as np, sys, time
N=520; FPS=30; DUR=312; TS=3.0; SUB=2; AR=16/9
R0=0.0100; RING_R=0.30; RING_W=0.12; CEN=np.array([AR/2,0.44],np.float32); NSEC=6
rng=np.random.RandomState(4)
hue=((np.arange(N)+rng.uniform(-0.3,0.3,N))/N)%1.0; rng.shuffle(hue); hue=hue.astype(np.float32)
x0=np.stack([rng.uniform(0.08,AR-0.08,N),rng.uniform(0.06,0.84,N)],1).astype(np.float32)
order=np.argsort(hue); rank=np.empty(N,int); rank[order]=np.arange(N)
th=2*np.pi*(rank/N)-np.pi/2; rho=RING_R+(rng.rand(N)-0.5)*RING_W
TG=(CEN+np.stack([np.cos(th)*rho,np.sin(th)*rho],1)).astype(np.float32)
sec=np.minimum((hue*NSEC).astype(int),NSEC-1); sang=2*np.pi*(sec+0.5)/NSEC-np.pi/2
rr=0.085*np.sqrt(rng.rand(N)); ra=rng.rand(N)*2*np.pi
CL=(CEN+np.stack([np.cos(sang)*RING_R+np.cos(ra)*rr,np.sin(sang)*RING_R+np.sin(ra)*rr],1)).astype(np.float32)
ps=0.05+0.33*rng.rand(N); pr=0.58+0.22*rng.rand(N); W1=0.17; W2=0.18
ph=rng.rand(N,2)*6.28; fq=rng.uniform(0.05,0.12,(N,2))
dh=np.abs(hue[:,None]-hue[None,:]); dh=np.minimum(dh,1-dh); F=np.cos(2*np.pi*dh).astype(np.float32); np.fill_diagonal(F,0)
def smooth(a): a=np.clip(a,0,1); return a*a*(3-2*a)
def home(p,t):
    e1=smooth((p-ps)/W1)[:,None]; e2=smooth((p-pr)/W2)[:,None]
    H=x0+e1*(CL-x0)+e2*(TG-CL)
    amp=0.014*(1-0.85*e2)
    H=H+amp*np.stack([np.sin(t*fq[:,0]*6.28+ph[:,0]),np.cos(t*fq[:,1]*6.28+ph[:,1])],1)
    return H
def relax(x):
    for _ in range(2):
        d=x[None,:,:]-x[:,None,:]; r=np.sqrt((d**2).sum(2))+1e-4
        ov=np.clip(2*R0*1.04-r,0,None); np.fill_diagonal(ov,0)
        x-=0.5*((ov/r)[...,None]*d).sum(1)*0.6
    return x
def run(frames):
    x=x0.copy(); pos=np.zeros((frames,N,2),np.float16); dt=1/FPS/SUB; lam=0.06; A=0.004
    for f in range(frames):
        t=f/FPS; p=min(max((t-TS)/300.0,0),1)
        for s in range(SUB):
            tt=t+s*dt; H=home(p,tt)
            d=x[None,:,:]-x[:,None,:]; r=np.sqrt((d**2).sum(2))+1e-4
            g=np.exp(-r/lam)*(r>2*R0*1.2)
            v=((F*g*A/lam)[...,None]*(d/r[...,None])).sum(1)*0.05*lam
            v+=1.3*(H-x)
            for ax,lo,hi in ((0,0.05,AR-0.05),(1,0.04,0.84)):
                v[:,ax]+=np.clip(lo-x[:,ax],0,None)*20-np.clip(x[:,ax]-hi,0,None)*20
            sp=np.linalg.norm(v,axis=1,keepdims=True); v=v*np.minimum(1,0.30/np.maximum(sp,1e-6))
            x=x+v*dt; x=relax(x)
        pos[f]=x
    return pos
if __name__=='__main__':
    n=int(sys.argv[1]); t0=time.time(); P=run(n); print('time',time.time()-t0)
    np.save(sys.argv[2],P); np.save('hue.npy',hue)
