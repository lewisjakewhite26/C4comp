import numpy as np, random
FPS=30; BPM=95; BEAT=60/BPM; T0=1.3; N=90   # N = number of straight segments (= beats of travel)
DUR=T0+N*BEAT+6.4
PAL=[(255,170,205),(170,240,205),(255,240,170),(160,215,255),(205,185,255),(255,205,170)]
def build():
    rnd=random.Random(11); W=7; x=3; y=0; cells=[(x,y)]; corners=[(x,y)]; lat=rnd.choice([-1,1])
    for k in range(N):
        L=rnd.choice([2,3])
        if k%2==0:
            for _ in range(L): y+=1; cells.append((x,y))
        else:
            d=lat
            if not(0<=x+L*d<=W-1): d=-d
            lat=-d
            for _ in range(L): x+=d; cells.append((x,y))
        corners.append((x,y))
    K=np.array(corners,float); xp=np.arange(N+1.0)
    def pos(u):
        u=np.clip(u,0,N); return np.stack([np.interp(u,xp,K[:,0]),np.interp(u,xp,K[:,1])],-1)
    plates=[]
    for i in range(1,N+1):
        din=K[i]-K[i-1]; din=din/np.linalg.norm(din)
        c=K[i]+din*0.40; ang=np.arctan2(din[1],din[0])+np.pi/2     # plate face is square-on to the ball
        plates.append(dict(i=i,c=c,cp=K[i],din=din,side=(1 if i%2 else -1),ang=ang,col=PAL[rnd.randrange(6)],t=T0+i*BEAT,cell=corners[i]))
    isl=[]
    for _ in range(80):
        k=rnd.randrange(3,len(cells)-3); gx=cells[k][0]+rnd.choice([-1,1])*rnd.randint(3,6); gy=cells[k][1]+rnd.randint(-2,2)
        if all(abs(gx-a)+abs(gy-b)>2 for a,b in cells) and all(abs(gx-a)+abs(gy-b)>1 for a,b,_ in isl):
            isl.append((gx,gy,-rnd.uniform(1.0,4.0)))
        if len(isl)>=18: break
    orbs=[]
    for _ in range(30):
        b=pos(np.array(rnd.uniform(0,N))); orbs.append((b[0]+rnd.uniform(-7,7),b[1]+rnd.uniform(-4,6),rnd.uniform(0.8,5),PAL[rnd.randrange(6)],rnd.uniform(0,6.28)))
    blocks=[]
    for i in range(2,len(cells)-1,4):
        a=np.array(cells[i+1],float)-np.array(cells[i-1],float); a=a/np.linalg.norm(a); n=np.array([-a[1],a[0]]); s=rnd.choice([-1,1])
        blocks.append((np.array(cells[i],float)+n*s*0.40, np.arctan2(a[1],a[0])))
    S=[60,62,64,67,69,72,74,76,79,81,84]; mrnd=random.Random(5); cur=4
    def motif():
        m=[]; c=cur
        for k in range(8):
            c=int(np.clip(c+mrnd.choice([-2,-1,-1,0,1,1,2]),1,9)); m.append(c)
        return m
    seq=[]; M=None
    for blk in range(0,N//8+2):
        if blk%3==0: M=motif()
        A=list(M)
        if blk%2==1:
            for q in (5,6,7): A[q]=int(np.clip(A[q]+mrnd.choice([-2,-1,1,2]),1,9))
        seq+=A
    for p in plates: p['midi']=S[seq[p['i']-1]]
    return dict(P=K,pos=pos,plates=plates,isl=isl,orbs=orbs,blocks=blocks,cells=cells)
