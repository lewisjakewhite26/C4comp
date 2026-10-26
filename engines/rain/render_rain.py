import cv2, numpy as np, subprocess, sys, math
T=900; FPS=24; TAU=2*math.pi
S=float(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].replace('.','').isdigit() else 1.0
W,H=int(1920*S),int(1080*S)
def om(k): return TAU*k/T
def P(x,y): return (int(round(x*S)),int(round(y*S)))
base_full=cv2.imread('base.png')
# ---------- chime layers (full-res masks)
lum=cv2.cvtColor(base_full,cv2.COLOR_BGR2GRAY)
cm=np.zeros(lum.shape,np.uint8)
box=(slice(148,386),slice(1104,1164))
dark=(lum[box]<150).astype(np.uint8)*255
cm[box]=dark
cv2.fillPoly(cm,[np.array([(1122,313),(1141,313),(1128,381),(1110,381)],np.int32)],255)
cm=cv2.morphologyEx(cm,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
cm[:148]=0
# the bell must stay a solid blob: fill bell polygon
cv2.fillPoly(cm,[np.array([(1111,236),(1131,230),(1155,238),(1158,290),(1110,292)],np.int32)],255)
cmd=cv2.dilate(cm,np.ones((5,5),np.uint8))
clean_full=cv2.inpaint(base_full,cmd,5,cv2.INPAINT_TELEA)
bell_m=cm.copy(); bell_m[313:]=0              # string+bell
strip_m=cm.copy(); strip_m[:313]=0
def sprite(m):
    ys,xs=np.where(m>0); y0,y1,x0,x1=ys.min()-3,ys.max()+4,xs.min()-3,xs.max()+4
    a=cv2.GaussianBlur(m[y0:y1,x0:x1].astype(np.float32)/255,(3,3),0)
    return (x0,y0,base_full[y0:y1,x0:x1].astype(np.float32),a)
BELL=sprite(bell_m); STRIP=sprite(strip_m)
base=cv2.resize(clean_full,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
# ---------- masks
def poly_mask(pts,feather=3):
    m=np.zeros((H,W),np.float32); cv2.fillPoly(m,[np.array([P(*p) for p in pts],np.int32)],1.0)
    return cv2.GaussianBlur(m,(0,0),max(0.5,feather*S))
RAINM=poly_mask([(430,150),(1590,120),(1590,835),(1200,822),(900,806),(700,786),(430,786)])
pill=np.zeros((H,W),np.float32); cv2.rectangle(pill,P(515,130),P(568,800),1.0,-1)
RAINM=np.clip(RAINM-cv2.GaussianBlur(pill,(0,0),2*S),0,1)
OUTD=poly_mask([(430,150),(1590,120),(1590,835),(1200,822),(900,806),(700,786),(430,786)],12)  # light weighting
GARDEN=poly_mask([(720,700),(1590,690),(1590,830),(1200,822),(900,806),(700,786)])
POND=poly_mask([(1240,712),(1400,715),(1590,705),(1590,775),(1500,768),(1380,760),(1290,740)],2)
# foliage sway mask
hsv=cv2.cvtColor(clean_full,cv2.COLOR_BGR2HSV)
fm=((hsv[...,0]>30)&(hsv[...,0]<90)&(hsv[...,1]>55)&(hsv[...,2]>55)).astype(np.float32)
reg=np.zeros(fm.shape,np.float32); reg[110:670,700:1600]=1; cv2.rectangle(reg,(930,490),(1060,720),0,-1)
fm=cv2.GaussianBlur(fm*reg,(0,0),4)
FM=cv2.resize(fm,(W,H))
yy=np.arange(H,dtype=np.float32)[:,None]/S
FW=FM*np.clip((670-yy)/420,0.15,1)
GX,GY=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
QW,QH=max(2,W//4),max(2,H//4)
QX,QY=np.meshgrid(np.arange(QW,dtype=np.float32)*4/S,np.arange(QH,dtype=np.float32)*4/S)
QF=cv2.resize(FW,(QW,QH))
def foliage(t):
    dx=QF*(3.4*np.sin(QX/130+om(40)*t+QY/90)+2.0*np.sin(QX/55-om(61)*t)+1.2*np.sin(om(7)*t))*S
    dy=QF*(1.5*np.sin(QX/110+om(53)*t)+0.8*np.sin(QY/40+om(37)*t))*S
    dx=cv2.resize(dx.astype(np.float32),(W,H)); dy=cv2.resize(dy.astype(np.float32),(W,H))
    return cv2.remap(base,GX-dx,GY-dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
# ---------- rain layers
rg=np.random.RandomState(4)
def mk(n,v,ln,w,al):
    return dict(x=rg.uniform(380,1640,n),y0=rg.uniform(0,1080,n),h=rg.rand(n),v=v,ln=ln,w=w,al=al)
LAY=[mk(900,600,16,1,0.30),mk(520,900,30,1,0.42),mk(90,1200,58,2,0.50)]
for L in LAY: L['v']=round(L['v']*T/1080)*1080/T      # whole number of wraps per loop
SLANT=0.10
def inten(t): return 0.72+0.2*math.sin(om(5)*t+0.6)+0.08*math.sin(om(17)*t)   # 0.44..1.0
def rain_layer(t):
    lay=np.zeros((H,W),np.float32); I=inten(t)
    for L in LAY:
        y=(L['y0']+L['v']*t)%1080
        act=L['h']<I
        for x,yv,hh in zip(L['x'][act],y[act],L['h'][act]):
            a=L['al']*min(1,(I-hh)*6)
            x0=x-SLANT*yv*0  # constant slant offset per streak
            cv2.line(lay,P(x0,yv),P(x0-SLANT*L['ln'],yv-L['ln']),a,max(1,int(round(L['w']*S))) if S>=1 else 1,cv2.LINE_AA)
    return lay
# ---------- ripples / splashes
pr=np.random.RandomState(8)
RIP=[]
nr=int(900*5.5)
for i in range(nr):
    RIP.append((pr.uniform(0,T),pr.uniform(1250,1580),pr.uniform(715,770),pr.uniform(0.6,1.0)))
RIP.sort()
RT=np.array([r[0] for r in RIP])
SPL=[(pr.uniform(0,T),pr.uniform(730,1560),pr.uniform(715,815)) for _ in range(int(900*28))]
SPL.sort(); ST=np.array([s[0] for s in SPL])
def pond_layer(t):
    lay=np.zeros((H,W),np.float32)
    life=1.9
    lo=t-life
    idx=[]
    for a,b in ((lo,t),(lo+T,t+T),(lo-T,t-T)):
        i0=np.searchsorted(RT,a); i1=np.searchsorted(RT,b)
        for i in range(i0,i1): idx.append((i,(t-(RT[i]-(a-lo)) ) ))
    for i,_ in idx:
        tb,x,y,s=RIP[i]
        age=(t-tb)%T
        if age>life: continue
        u=age/life; r=3+26*u*s
        cv2.ellipse(lay,P(x,y),(max(1,int(r*S)),max(1,int(r*0.30*S))),0,0,360,0.55*(1-u)**1.3,1,cv2.LINE_AA)
    # splashes on stones/garden floor
    sl=0.28
    for a,b in ((t-sl,t),(t-sl+T,t+T),(t-sl-T,t-T)):
        i0=np.searchsorted(ST,a); i1=np.searchsorted(ST,b)
        for i in range(i0,i1):
            tb,x,y=SPL[i]; age=(t-tb)%T
            if age>sl: continue
            u=age/sl; r=1.5+4*u
            cv2.ellipse(lay,P(x,y),(max(1,int(r*S)),max(1,int(r*0.35*S))),0,0,360,0.5*(1-u),1,cv2.LINE_AA)
    return lay
# ---------- eave drips
DR=[(1093,165,3.1,0.4),(1360,157,3.9,1.7),(860,186,2.8,2.2),(650,292,4.4,0.9),(463,236,3.6,2.9)]
def drips(t,lay):
    for x,y,per,ph in DR:
        n=int(round(T/per)); per=T/n
        tau=((t-ph)%per)
        if tau<0.55:
            d=0.5*1800*tau*tau; a=0.8*(1-tau/0.7)
            cv2.line(lay,P(x,y+d),P(x,y+d+9),a,1 if S<1 else 2,cv2.LINE_AA)
# ---------- steam
def steam(t,lay):
    RATE=6
    for sx,sy,sc in ((537,883,1.0),(620,896,0.8)):
        k0=int((t-7)*RATE)
        for k in range(k0,int(t*RATE)+1):
            kk=k%(T*RATE); rg2=np.random.RandomState(kk*2+int(sx)); ts=k/RATE; age=t-ts
            if age<0 or age>7: continue
            ph=rg2.rand()*6.28
            x=sx+ (6*age+3*math.sin(age*1.1+ph)*min(age,2)); y=sy-(34*(1-math.exp(-age/2.4))+3*age)*sc
            r=(2+3.2*age**0.8)*sc; a=0.30*min(age/0.5,1)*(1-age/7)**1.4
            cv2.circle(lay,P(x,y),max(1,int(r*S)),a,-1,cv2.LINE_AA)
# ---------- thunder
FL=[(212,0.55),(212.22,0.35),(541,0.45),(777,0.6),(777.3,0.3)]
def flash(t):
    f=0
    for te,a in FL:
        d=t-te
        if d>=0: f+=a*math.exp(-d/0.13)
    return f
# ---------- noise / mist
rn=np.random.RandomState(2)
nz3=cv2.GaussianBlur(np.tile(rn.rand(30,160).astype(np.float32),(1,3)),(0,0),3)[:,160:320]
nz=(nz3-nz3.min())/(nz3.max()-nz3.min())
MIST=cv2.resize(nz,(W,int(260*S)),interpolation=cv2.INTER_CUBIC).astype(np.float32)
vig=(1-0.30*(((np.arange(W)[None,:]-W/2)/(W/2))**2+((np.arange(H)[:,None]-H/2)/(H/2))**2)/2).astype(np.float32)
dith=(rn.rand(H,W).astype(np.float32)-0.5)*2.0
RAINC=np.array([228,232,226],np.float32)
def frame(fi):
    t=fi/FPS
    f=foliage(t)
    # chime
    th=2.4*math.sin(om(214)*t+0.4)*(0.55+0.45*math.sin(om(6)*t+1.0))+0.8*math.sin(om(97)*t)
    ths=th+5.5*math.sin(om(214)*t-0.9)*(0.6+0.4*math.sin(om(6)*t+1.0))
    for spr,piv,ang in ((BELL,(1132,150),th),(STRIP,(1132,313),ths)):
        x0,y0,im,a=spr
        M=cv2.getRotationMatrix2D((float(piv[0]-x0),float(piv[1]-y0)),float(ang),1.0)
        M[0,2]+=0; 
        h,w=a.shape
        # place in scaled canvas
        Ms=np.vstack([M,[0,0,1]]); T1=np.array([[S,0,x0*S],[0,S,y0*S],[0,0,1]],np.float64)
        Mf=T1@Ms
        wi=cv2.warpAffine(im,Mf[:2],(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        wa=cv2.warpAffine(a,Mf[:2],(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[...,None]
        f=f*(1-wa)+wi*wa
    # cat breathing
    k=1+0.012*math.sin(om(225)*t)
    cx0,cy0,cx1,cy1=[int(v*S) for v in (1165,812,1360,912)]
    roi=f[cy0:cy1,cx0:cx1]
    pivy=(905*S-cy0)
    Mc=np.float32([[1,0,0],[0,k,pivy*(1-k)]])
    wr=cv2.warpAffine(roi,Mc,(cx1-cx0,cy1-cy0),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    fe=np.ones((cy1-cy0,cx1-cx0),np.float32); b=max(2,int(6*S)); fe[:b]=np.linspace(0,1,b)[:,None]; fe[:,:b]*=np.linspace(0,1,b)[None,:]; fe[:,-b:]*=np.linspace(1,0,b)[None,:]
    f[cy0:cy1,cx0:cx1]=roi*(1-fe[...,None])+wr*fe[...,None]
    # light variation (outdoor weighted) + thunder
    br=0.05*math.sin(om(6)*t+0.3)+0.025*math.sin(om(14)*t+1.2)
    f*=(1+br*(0.35+0.65*OUTD))[...,None]
    fl=flash(t)
    if fl>0.005: f+=(fl*(0.25+0.75*OUTD))[...,None]*np.array([70,60,55],np.float32)
    # mist
    ox=(t*8*W/T)  # 8 widths per loop
    M2=np.float32([[1,0,-ox],[0,1,0]])
    band=cv2.warpAffine(MIST,M2,(W,MIST.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    y0=int(520*S); ma=np.zeros((H,W),np.float32); ma[y0:y0+band.shape[0]]=band*0.10*np.linspace(0.2,1,band.shape[0])[:,None]
    f+=(ma*RAINM)[...,None]*np.array([225,230,225],np.float32)
    # rain, ripples, drips, steam
    lay=rain_layer(t); drips(t,lay)
    f+= (lay*RAINM)[...,None]*RAINC*0.62
    pl=pond_layer(t)
    f+= (pl*np.maximum(POND,GARDEN*0.7))[...,None]*np.array([235,235,230],np.float32)*0.8
    sl=np.zeros((H,W),np.float32); steam(t,sl)
    sl=cv2.GaussianBlur(sl,(0,0),max(1.0,3*S))
    f=f*(1-sl[...,None]*0.55)+sl[...,None]*0.55*np.array([225,228,230],np.float32)
    f*=vig[...,None]; f+=dith[...,None]
    return np.clip(f,0,255).astype(np.uint8)
def enc(out,crf):
    return subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','fast','-crf',str(crf),'-tune','animation','-g','48','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
if __name__=='__main__':
    c=sys.argv[2]
    if c=='still': cv2.imwrite(sys.argv[4],frame(int(float(sys.argv[3])*FPS)))
    elif c=='time':
        import time; t0=time.time(); frame(int(float(sys.argv[3])*FPS)); print('s/frame',time.time()-t0)
    elif c=='loopcheck':
        a=frame(0).astype(int); b=frame(T*FPS).astype(int); print('0 vs NF',np.abs(a-b).mean(),np.abs(a-b).max())
    else:
        a,b,out,crf=int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],int(sys.argv[6])
        p=enc(out,crf)
        for fi in range(a,b): p.stdin.write(frame(fi).tobytes())
        p.stdin.close(); p.wait()
