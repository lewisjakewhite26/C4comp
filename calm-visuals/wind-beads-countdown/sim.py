import numpy as np, math, json, sys
W,H=1920,1080; CELL=6; GW,GH=W//CELL,H//CELL
FPS=30; COUNT=60.0; TOT=66.0; NF=int(TOT*FPS); N=GW*GH; LAG=2.6
rng=np.random.default_rng(42)
def wind(t,x,y):
    base=95+110*np.sin(2*np.pi*t/17+0.5)+60*np.sin(2*np.pi*t/6.3+1.7)
    gust=1+0.55*np.sin((x-260*t)/520.0)
    vx=base*gust+55*np.sin(y/190.0+1.3*t+x/420.0)
    vy=50*np.sin(x/170.0-1.1*t)+35*np.sin(y/260.0+0.7*t)
    return vx,vy
GH_=np.zeros((NF,GH,GW),np.uint8); GJ_=np.zeros((NF,GH,GW),np.uint8)
hue=np.zeros((GH,GW),np.uint8); jit=np.zeros((GH,GW),np.uint8)
# particles: real (settling) and dust
P={k:np.zeros(0,np.float32) for k in ('x','y','vx','vy','fall','hue','jit')}
def padd(n,x,y,vx,vy,fall,h,j):
    for k,v in zip(('x','y','vx','vy','fall','hue','jit'),(x,y,vx,vy,fall,h,j)): P[k]=np.concatenate([P[k],np.asarray(v,np.float32)])
ND=900
D=dict(x=rng.uniform(0,W,ND).astype(np.float32),y=rng.uniform(0,H,ND).astype(np.float32),vx=np.zeros(ND,np.float32),vy=np.zeros(ND,np.float32),
       fall=rng.uniform(15,60,ND).astype(np.float32),hue=rng.integers(1,251,ND).astype(np.float32),r=rng.uniform(0.8,2.2,ND).astype(np.float32))
frames_p=[]; frames_d=[]; lands=[]   # lands: (t, col, hue)
settled=0
def step_ca(f):
    global hue,jit
    occ=hue>0
    for _ in range(2):
        mv=occ[:-1]&~occ[1:]
        idx=np.argwhere(mv)
        if len(idx):
            r,c=idx[:,0],idx[:,1]
            hue[r+1,c]=hue[r,c]; jit[r+1,c]=jit[r,c]; hue[r,c]=0; jit[r,c]=0
            occ=hue>0
    # diagonal slides (parity columns to avoid conflicts)
    for side in ((1,-1) if f%2 else (-1,1)):
        occ=hue>0
        blocked=np.zeros_like(occ); blocked[:-1]=occ[1:]; blocked[-1]=True
        can=occ&blocked
        tgt_ok=np.zeros_like(occ)
        if side==-1:
            ok=np.zeros_like(occ); ok[:-1,1:]=~occ[1:,:-1]&~occ[:-1,:-1] ; # down-left empty and left empty
            ok2=np.zeros_like(occ); ok2[:-1,1:]=(~occ[1:,:-1])&(~occ[:-1,:-1]); tgt_ok=ok2
        else:
            ok2=np.zeros_like(occ); ok2[:-1,:-1]=(~occ[1:,1:])&(~occ[:-1,1:]); tgt_ok=ok2
        par=(np.arange(GW)%2==(f//2)%2)[None,:]
        mv=can&tgt_ok&par
        idx=np.argwhere(mv)
        if len(idx):
            r,c=idx[:,0],idx[:,1]; c2=c+side
            hue[r+1,c2]=hue[r,c]; jit[r+1,c2]=jit[r,c]; hue[r,c]=0; jit[r,c]=0
for f in range(NF):
    t=f/FPS; dt=1.0/FPS
    prog=min(1.0,(t+LAG)/COUNT)
    # spawn
    airborne=len(P['x'])
    need=int(round(N*prog-settled-airborne)) if t<COUNT-0.9 else 0
    need=max(0,min(need,90))
    if need>0:
        top=np.where((hue>0).any(0),(hue>0).argmax(0),GH)
        cc_=np.arange(GW); hT=GH*prog+(1-prog**3)*(9*np.sin(cc_/31.0+t*0.45)+5*np.sin(cc_/13.0-t*0.8)+3*np.sin(cc_/7.0+t*1.3))
        cur=GH-top
        defi=np.maximum(0,hT-cur)+0.05
        p=np.sqrt(defi); p=p/p.sum(); cols=rng.choice(GW,need,p=p)
        lx=(cols+rng.random(need))*CELL
        fall=rng.uniform(200,340,need)
        sy=-rng.uniform(5,60,need)
        ly=(GH-cur[cols])*CELL
        tf=np.maximum(0.3,(ly-sy)/fall)
        sx=lx-95*tf*0.8+rng.normal(0,60,need)
        hh=t/COUNT*1.15+rng.normal(0,0.05,need); hh=np.where(rng.random(need)<0.10,rng.random(need),hh); h=1+(hh%1.0)*249
        padd(need,sx,sy,np.zeros(need),np.zeros(need)+fall,fall,np.floor(h),rng.integers(60,256,need))
    # final top-up
    if abs(t-(COUNT-0.9))<dt/2+1e-6:
        e=np.argwhere(hue==0)
        if len(e)>0:
            n=len(e); r,c=e[:,0],e[:,1]
            padd(n,(c+0.5)*CELL,(r*CELL-30-rng.random(n)*90),np.zeros(n),np.full(n,380.0),np.full(n,380.0),
                 np.floor(1+((t/COUNT*1.15+rng.normal(0,0.05,n))%1.0)*249),rng.integers(0,256,n))
    # integrate (3 substeps)
    x,y,vx,vy,fall=P['x'],P['y'],P['vx'],P['vy'],P['fall']
    keep=np.ones(len(x),bool)
    top=np.where((hue>0).any(0),(hue>0).argmax(0),GH)
    sub=3; h_=dt/sub
    for s in range(sub):
        if len(x)==0: break
        wx,wy=wind(t+s*h_,x,y)
        vx+=(wx-vx)*2.4*h_; vy+=((fall+wy)-vy)*2.4*h_
        x[:]=x+vx*h_; y+=vy*h_
        c=np.clip((x//CELL).astype(int),0,GW-1); r=(y//CELL).astype(int)
        inside=(x>=0)&(x<W); hit=keep&inside&(r>=top[c])
        keep&=~((x<-900)|(x>W+150))
        if hit.any():
            for i in np.where(hit)[0]:
                cc=int(c[i]); lo=max(0,cc-30); hi=min(GW,cc+31)
                seg=top[lo:hi].astype(np.float32); cc=lo+int(np.argmax(seg-0.10*np.abs(np.arange(lo,hi)-cc))) if len(seg) else cc
                rr=top[cc]-1
                if rr>=0:
                    hue[rr,cc]=int(P['hue'][i]); jit[rr,cc]=int(P['jit'][i]); top[cc]=rr; settled+=1; lands.append((t,int(cc),int(P['hue'][i])))
                keep[i]=False
    if not keep.all():
        for k in P: P[k]=P[k][keep]
    # dust
    wx,wy=wind(t,D['x'],D['y'])
    D['vx']+=(wx*0.8-D['vx'])*2.0*dt; D['vy']+=((D['fall']+wy*0.5)-D['vy'])*2.0*dt
    D['x']=(D['x']+D['vx']*dt)%W; D['y']=(D['y']+D['vy']*dt)
    wrap=D['y']>H+10; D['y'][wrap]=-10
    step_ca(f); step_ca(f+1)
    GH_[f]=hue; GJ_[f]=jit
    frames_p.append(np.stack([P['x'],P['y'],P['vx'],P['vy'],P['hue']],1).astype(np.float32))
    frames_d.append(np.stack([D['x'],D['y'],D['vx'],D['vy'],D['hue'],D['r']],1).astype(np.float32))
    if f%150==0: print(f,round(t,1),'settled',settled,'air',len(P['x']),'filled',round((hue>0).mean(),3),flush=True)
np.save('GH.npy',GH_); np.save('GJ.npy',GJ_)
cnt=np.array([len(a) for a in frames_p]); np.save('pcnt.npy',cnt); np.save('pall.npy',np.concatenate(frames_p))
np.save('dall.npy',np.stack(frames_d)); np.save('lands.npy',np.array(lands,np.float32))
print('final filled',(GH_[int(COUNT*FPS)]>0).mean(),(GH_[-1]>0).mean(), 'frames',NF)
