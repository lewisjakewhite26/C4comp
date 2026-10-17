import numpy as np, random
from scipy.ndimage import gaussian_filter1d
FPS=30; BPM=95; BEAT=60/BPM; T0=1.3; N=90
DUR=T0+(N-1)*BEAT+6.4
PAL=[(255,170,205),(170,240,205),(255,240,170),(160,215,255),(205,185,255),(255,205,170)]
def build():
    rnd=random.Random(11); W=7; x=3; y=0; cells=[(x,y)]; lat=rnd.choice([-1,1])
    while len(cells)<N:
        for _ in range(rnd.randint(2,3)):
            y+=1; cells.append((x,y))
            if len(cells)>=N: break
        if len(cells)>=N: break
        d=lat
        if not(0<=x+2*d<=W-1): d=-d
        lat=-d
        for _ in range(rnd.randint(2,4)):
            if 0<=x+d<=W-1: x+=d; cells.append((x,y))
            if len(cells)>=N: break
    P=np.array(cells,float)
    # smoothed path as function of cell-index parameter u
    pad=2.0; d0=P[0]-P[1]; d1=P[-1]-P[-2]
    pts=np.vstack([P[0]+d0*pad,P,P[-1]+d1*pad]); idx=np.arange(-1,N+1,1.0); idx[0]=-pad; idx[-1]=N-1+pad
    ug=np.arange(-pad,N-1+pad,0.02)
    X=np.interp(ug,idx,pts[:,0]); Y=np.interp(ug,idx,pts[:,1])
    X=gaussian_filter1d(X,15,mode='nearest'); Y=gaussian_filter1d(Y,15,mode='nearest')
    def pos(u):
        u=np.clip(u,-pad+0.01,N-1+pad-0.01); return np.stack([np.interp(u,ug,X),np.interp(u,ug,Y)],-1)
    # plates
    plates=[]
    for i in range(1,N):
        a=P[min(i+1,N-1)]-P[i-1]; a=a/np.linalg.norm(a); n=np.array([-a[1],a[0]])
        side=(1 if i%2 else -1)*(1 if rnd.random()>0.25 else -1)
        c=P[i]+n*side*0.41; ang=np.arctan2(a[1],a[0])+rnd.uniform(-0.45,0.45)
        plates.append(dict(i=i,c=c,side=side,ang=ang,col=PAL[rnd.randrange(6)],t=T0+i*BEAT))
    # islands and orbs
    cs=set(cells); isl=[]
    for _ in range(60):
        k=rnd.randrange(3,N-3); gx=cells[k][0]+rnd.choice([-1,1])*rnd.randint(3,6); gy=cells[k][1]+rnd.randint(-2,2)
        if all(abs(gx-a)+abs(gy-b)>2 for a,b in cells) and all(abs(gx-a)+abs(gy-b)>1 for a,b,_ in isl):
            isl.append((gx,gy,-rnd.uniform(1.0,4.0)))
        if len(isl)>=16: break
    orbs=[]
    for _ in range(26):
        k=rnd.uniform(0,N-1); b=pos(np.array(k)); orbs.append((b[0]+rnd.uniform(-7,7),b[1]+rnd.uniform(-4,6),rnd.uniform(0.8,5),PAL[rnd.randrange(6)],rnd.uniform(0,6.28)))
    blocks=[]
    for i in range(2,N-1,5):
        a=P[min(i+1,N-1)]-P[i-1]; a=a/np.linalg.norm(a); n=np.array([-a[1],a[0]]); s=rnd.choice([-1,1])
        blocks.append((P[i]+n*s*0.40+a*0.5, np.arctan2(a[1],a[0])))
    # melody
    S=[60,62,64,67,69,72,74,76,79,81,84]; mel=[]; mrnd=random.Random(5); cur=4
    def motif():
        m=[]; c=cur
        for k in range(8):
            c=int(np.clip(c+mrnd.choice([-2,-1,-1,0,1,1,2]),1,9)); m.append(c)
        return m
    seq=[]; M=None
    for blk in range(0,N//8+2):
        if blk%3==0: M=motif()
        A=list(M); 
        if blk%2==1:
            for q in (5,6,7): A[q]=int(np.clip(A[q]+mrnd.choice([-2,-1,1,2]),1,9))
        seq+=A
    for p in plates: p['midi']=S[seq[p['i']-1]]
    return dict(P=P,pos=pos,plates=plates,isl=isl,orbs=orbs,blocks=blocks,cells=cells)
