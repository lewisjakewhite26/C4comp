import sys, os, math, random, subprocess, json
import numpy as np, cv2
D='/tmp/claude-0/-home-claude-c4comp/f7d1ccd4-c824-5d8b-9f33-8948edecece0/scratchpad/city/'
FPS=24; T=180.0; W,H=1920,1080; S=1920/1376.0
R=random.Random(21); rg=np.random.default_rng(21)
src=cv2.imread(D+'c.jpg'); BASE=cv2.resize(src,(W,H),interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
bl=cv2.GaussianBlur(BASE,(0,0),1.0); BASE=np.clip(BASE+0.3*(BASE-bl),0,255)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
def X(x): return x*S
B,G,Rc=BASE[...,0],BASE[...,1],BASE[...,2]; lum=BASE.mean(2)
# region of glass panes (source coords x 155-1205, y 0-650)
glass=np.zeros((H,W),np.float32); glass[:int(585*S),int(150*S):int(1205*S)]=1
for a,b in ((440,475),(910,945)): glass[:,int(a*S):int(b*S)]=0
glass=cv2.GaussianBlur(glass,(0,0),3); glass[int(540*S):]*=np.clip(1-(yy[int(540*S):]-540*S)/(45*S),0,1)
# --- window lights: small warm components in the city band, away from lamp ---
cand=((lum>110)&(Rc>B*0.95)&(yy>X(150))&(yy<X(520))&(xx>X(150))&(xx<X(1205))).astype(np.uint8)
cand[int(X(400)):,int(X(1090)):]=0
n,lab,st,cen=cv2.connectedComponentsWithStats(cand,connectivity=8)
small=[i for i in range(1,n) if 6<=st[i,cv2.CC_STAT_AREA]<=170 and st[i,cv2.CC_STAT_WIDTH]<=26 and st[i,cv2.CC_STAT_HEIGHT]<=16]
print('windows found',len(small),flush=True)
R.shuffle(small)
nt=14   # only ~6% ever toggle
small=[i for i in small if cen[i][1]<X(425)]; tog=small[:nt]
TG=[]
for i in tog:
    m=(lab==i).astype(np.float32)
    x,y,w,h=st[i,0],st[i,1],st[i,2],st[i,3]
    pad=4; x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad)
    mm=cv2.GaussianBlur(m[y0:y1,x0:x1],(0,0),0.9); 
    TG.append((x0,y0,x1,y1,mm,R.uniform(0,T),R.uniform(18,50),R.choice([1,2])))
# bokeh street lights (bigger warm blobs low in the frame): tiny twinkle
bk=((lum>120)&(Rc>B*1.2)&(yy>X(440))&(yy<X(600))&(xx>X(150))&(xx<X(1205))).astype(np.uint8)
bk[int(X(380)):,int(X(1090)):]=0
n2,lab2,st2,_=cv2.connectedComponentsWithStats(bk,connectivity=8)
BKM=np.zeros((H,W),np.float32); BKPH=rg.uniform(0,6.28,n2); BKK=rg.integers(2,7,n2)
okb=np.array([0]+[1 if 20<=st2[i,cv2.CC_STAT_AREA]<=900 else 0 for i in range(1,n2)])
BKA=okb[lab2].astype(np.float32); BKI=lab2
# lamps
LAMP=np.exp(-(((xx-X(1255))/X(70))**2+((yy-X(470))/X(60))**2)).astype(np.float32)
LAMP2=np.exp(-(((xx-X(1130))/X(55))**2+((yy-X(455))/X(45))**2)).astype(np.float32)
# sky mask for lightning
SKY=np.clip((X(430)-yy)/X(300),0,1)*glass
SKYC=np.exp(-(((xx-X(700))/X(260))**2+((yy-X(210))/X(90))**2)).astype(np.float32)
# lightning events (seconds within the 180s loop): list of (t, strength, double)
LT=[(11.0,1.0),(37.5,0.7),(38.1,0.9),(63.0,0.8),(92.0,1.0),(92.45,0.6),(121.0,0.7),(149.5,1.0),(165.0,0.6)]
json.dump([t for t,_ in LT],open(D+'lightning.json','w'))
def flash(t):
    f=0.0
    for (tc,s) in LT:
        d=(t-tc)%T
        if d<0.9: f+=s*(math.exp(-d*9)*(0.55+0.45*math.sin(d*50)**2))
        elif d>T-0.05: pass
    return min(f,1.4)
# rain on glass: sliding drops with trails (loop-safe, integer cycles)
DR=[]
panes=[(165,435),(480,905),(950,1200)]
for _ in range(210):
    a,b=R.choice(panes); x=X(R.uniform(a,b)); ln=R.uniform(20,80); k=R.randint(1,4)
    DR.append((x,R.uniform(0,H+ln),ln,k,R.uniform(0.25,0.7),R.uniform(1,2.2),R.uniform(0,6.28)))
STR=[]
for _ in range(120):
    a,b=R.choice(panes); x=X(R.uniform(a,b)); ln=R.uniform(40,160); k=R.randint(1,3)
    STR.append((x,R.uniform(0,H+ln),ln,k,R.uniform(0.12,0.28)))
def rain(t):
    lay=np.zeros((H,W),np.float32)
    for (x,y0,ln,k,al,r,p) in DR:
        Lc=H+ln; y=(y0+Lc*k*t/T)%Lc-ln
        xo=x+3*math.sin(2*math.pi*(3*k)*t/T+p)
        cv2.line(lay,(int(xo),int(y-ln)),(int(xo),int(y)),al*0.28,1,cv2.LINE_AA)
        cv2.circle(lay,(int(xo),int(y)),max(1,int(r)),al,-1,cv2.LINE_AA)
    for (x,y0,ln,k,al) in STR:
        Lc=H+ln; y=(y0+Lc*k*t/T)%Lc-ln
        cv2.line(lay,(int(x),int(y-ln)),(int(x+2*math.sin(y/37.0)),int(y)),al*0.8,1,cv2.LINE_AA)
    return lay*glass
BED=np.clip((yy-X(560))/X(60),0,1)
CLOUD=np.clip((X(330)-yy)/X(250),0,1)*glass
VIG=1-0.35*np.clip((np.hypot((xx-W/2)/(W/2),(yy-H/2)/(H/2))-0.55)/0.8,0,1)**1.6
GRAIN=[rg.normal(0,1.2,(H,W,1)).astype(np.float32) for _ in range(5)]
def render(i):
    t=i/FPS; ph=2*math.pi*t/T
    out=BASE.copy()
    # subtle window toggles
    for (x0,y0,x1,y1,mm,tc,dur,_k) in TG:
        d=((t-tc+T/2)%T)-T/2
        if abs(d)<dur/2+3:
            u=np.clip((dur/2-abs(d))/3.0,0,1); u=u*u*(3-2*u)   # smooth off
            if u>0: out[y0:y1,x0:x1]*=(1-0.85*u*mm)[...,None]
    # bokeh twinkle (very small)
    tw=1+0.06*np.sin(ph*BKK+BKPH)
    out*=(1+(tw[BKI]-1)*BKA)[...,None]
    # lamp breathe
    lb=1+0.03*math.sin(ph*30)+0.015*math.sin(ph*71+1)
    out+=LAMP[...,None]*np.array([20,70,160],np.float32)*0.14*lb+LAMP2[...,None]*np.array([20,60,130],np.float32)*0.08*lb
    # lightning + faint cloud glow
    f=flash(t)
    amb=0.04*(0.5+0.5*math.sin(ph*6))
    out+=(SKY*(f*0.55+amb))[...,None]*np.array([255,215,150],np.float32)*0.55
    out+=(SKYC*glass*(f*0.6))[...,None]*np.array([255,230,190],np.float32)*0.5
    out*=(1+0.10*f*glass)[...,None]
    # clouds drift + duvet breathe + tiny camera breathe
    mx=xx+CLOUD*(5.0*np.sin(ph*3+yy/220.0)+2.5*np.sin(ph*7))
    my=yy-BED*2.2*np.sin(ph*30)
    z=1+0.005*(1-math.cos(ph))/2; mx=(mx-W/2)/z+W/2; my=(my-H/2)/z+H/2
    out=cv2.remap(out,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    rl=rain(t)
    out+=rl[...,None]*np.array([225,205,170],np.float32)*(0.55+0.5*f)
    out=out*VIG[...,None]+GRAIN[i%5]
    return np.clip(out,0,255).astype(np.uint8)
def worker(f0,f1,outf):
    pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(FPS),'-i','-','-c:v','libx264','-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-g','48','-x264-params','threads=1',D+outf],stdin=subprocess.PIPE)
    for i in range(f0,f1): pr.stdin.write(render(i).tobytes())
    pr.stdin.close(); pr.wait(); print('done',outf,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='test':
        import time
        for s in sys.argv[2:]:
            t0=time.time(); cv2.imwrite(D+'t_%s.png'%s,render(int(float(s)*FPS))); print(s,round(time.time()-t0,2),flush=True)
    else: worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
