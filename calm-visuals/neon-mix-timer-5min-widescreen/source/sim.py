import numpy as np, sys, time
W,H=16/9.0,0.86
N=320; NC=4; FPS=30; SUBS=2; DT=1.0/(FPS*SUBS)
PRE=45.0; VID=312.0
rng=np.random.default_rng(7)
col=np.repeat(np.arange(NC),N//NC)
pos=np.stack([rng.uniform(0.06,W-0.06,N),rng.uniform(0.06,H-0.06,N)],1); vel=np.zeros((N,2))
S0=0.040; RO=0.26; RB=0.34; KB=1.8; KCORE=30.0; KREP=1.1; PULL=0.50; SIGM=0.50
SAME=(col[:,None]==col[None,:]); EYE=np.eye(N,dtype=bool)
def sstep(x): x=min(1,max(0,x)); return x*x*(3-2*x)
tau=np.zeros(N)
for c in range(NC):
    idx=np.where(col==c)[0]; perm=rng.permutation(len(idx))
    tau[idx]=13.0+270.0*((perm+rng.uniform(0,1,len(idx)))/len(idx))**1.15
def sv(t): return np.clip((t-tau)/4.0,0,1)**2*(3-2*np.clip((t-tau)/4.0,0,1))
def step(tt,pos,vel):
    t=tt-PRE
    s=sv(t)[:,None] if t>0 else np.zeros((N,1))
    dx=pos[None,:,0]-pos[:,None,0]; dy=pos[None,:,1]-pos[:,None,1]
    r=np.sqrt(dx*dx+dy*dy)+1e-6; ux=dx/r; uy=dy/r
    core=np.where((r<S0)&~EYE,-KCORE*(S0-r)/S0,0.0)
    shortc=np.clip(1-(r-S0)/(1.3*S0),0,1)*((r>=S0)&(r<2.3*S0))
    cohA=np.where(SAME,0.9*shortc,0.0)*(1-s)
    cohB=np.where(~SAME,0.45*shortc,0.0)*s
    repA=np.where(~SAME&(r<RO),-KREP*(1-r/RO),0.0)*(1-s)
    repB=np.where(SAME&(r<RB),-KB*(1-r/RB),0.0)*s
    attB=np.where(~SAME&(r>=2.3*S0)&(r<0.5),0.30*(1-r/0.5),0.0)*s
    f=core+cohA+cohB+repA+repB+attB
    Fx=(f*ux).sum(1); Fy=(f*uy).sum(1)
    wl=0.20; sc=s[:,0]
    Fx+=10.0*(np.maximum(0,wl-pos[:,0])-np.maximum(0,pos[:,0]-(W-wl)))
    Fy+=10.0*(np.maximum(0,wl-pos[:,1])-np.maximum(0,pos[:,1]-(H-wl)))
    cm=pos.mean(0); Fx+=-0.3*(pos[:,0]-W/2)-6.0*(cm[0]-W/2); Fy+=-0.3*(pos[:,1]-H/2)-6.0*(cm[1]-H/2)
    w=np.where(SAME&~EYE,np.exp(-(r/SIGM)**2),0.0); ws=w.sum(1)+1e-6
    mx=(w*dx).sum(1)/ws; my=(w*dy).sum(1)/ws; mm=np.sqrt(mx*mx+my*my)+1e-6
    sat=np.minimum(mm/0.12,1.0)
    pf=sstep((tt-1.2)/4.5)*(1-sc)
    k=pf*PULL*sat*np.minimum(1,ws/3.0)*22.0
    Fx+=k*mx/mm; Fy+=k*my/mm
    v=np.stack([Fx,Fy],1)*0.012
    nb=(r<2.2*S0)&~EYE; cnt=nb.sum(1)
    avg=np.where(cnt[:,None]>0,np.stack([nb@vel[:,0],nb@vel[:,1]],1)/np.maximum(cnt,1)[:,None],vel)
    vel=0.55*vel+0.45*v; vel=vel+0.35*(avg-vel)
    sp=np.linalg.norm(vel,axis=1,keepdims=True); vel=vel*np.minimum(1,0.12/np.maximum(sp,1e-6))
    amp=np.where(tt<PRE,0.03*np.exp(-tt/9.0),0.004)+sc*0.03*np.exp(-0.0*t)
    vel=vel+rng.normal(0,1,(N,2))*amp[:,None]*0.5
    vel=vel-vel.mean(0)
    pos=pos+vel*DT*SUBS
    mg=0.03
    for kk,(lo,hi) in enumerate(((mg,W-mg),(mg,H-mg))):
        low=pos[:,kk]<lo; high=pos[:,kk]>hi
        pos[low,kk]=lo; pos[high,kk]=hi; vel[low|high,kk]*=-0.2
    return pos,vel
if __name__=='__main__':
    t0=time.time(); nf=int(VID*FPS); tt=0.0; out=np.zeros((nf,N,2),np.float32)
    for _ in range(int(PRE*FPS)*SUBS): pos,vel=step(tt,pos,vel); tt+=DT
    for f in range(nf):
        for k in range(SUBS): pos,vel=step(tt,pos,vel); tt+=DT
        out[f]=pos
        if f%600==0: print(f,round(time.time()-t0),flush=True)
    np.save('pos.npy',out); print('done',time.time()-t0,flush=True)
