import numpy as np, cv2, sys, subprocess, time
cv2.setNumThreads(2)
FPS=24; P=36; N=FPS*P; W,H=1920,1080
MODE=sys.argv[1]
base=cv2.imread('base.png').astype(np.float32)/255
rng=np.random.default_rng(5)
X=np.tile(np.arange(W,dtype=np.float32)[None,:],(H,1)); Y=np.tile(np.arange(H,dtype=np.float32)[:,None],(1,W))
xs=np.arange(W,dtype=np.float32); ys=np.arange(H,dtype=np.float32)
TAU=2*np.pi; K=0.56    # scale of effects relative to the vertical version
def sstep(a,b,v): t=np.clip((v-a)/(b-a),0,1); return t*t*(3-2*t)
FX,FY=983.,900.
ty=sstep(250,330,ys)*(1-sstep(470,540,ys))
ry=np.exp(-((ys-690)/100)**2); rx=np.exp(-((xs-980)/190)**2)
def tnoise(seed,h=135,w=240,s=3.2):
    r=np.random.default_rng(seed).standard_normal((h,w)).astype(np.float32)
    F=np.fft.fft2(r); fy=np.fft.fftfreq(h)[:,None]*h; fx=np.fft.fftfreq(w)[None,:]*w
    F*=np.exp(-((fx/(s*1.8))**2+(fy/s)**2)); o=np.fft.ifft2(F).real.astype(np.float32); return o/o.std()
NA,NB=tnoise(1),tnoise(2); NC,ND=tnoise(3,s=6),tnoise(4,s=6)
mist_my=(np.exp(-((ys-640)/190)**2)*0.9+np.exp(-((ys-520)/110)**2)*0.5)[:,None].astype(np.float32)
mist_mx=(0.3+0.7*np.exp(-((xs-1000)/650)**2))[None,:].astype(np.float32)
mist_col=np.array([205,232,255],np.float32)/255
SM=[dict(o=i/14+rng.uniform(-.01,.01),ph=rng.uniform(0,6.28),dx=rng.uniform(-30,60)*K,r0=rng.uniform(16,26)*K) for i in range(14)]
EM=[dict(o=rng.uniform(0,1),k=int(rng.choice([6,9,9,12])),ph=rng.uniform(0,6.28),w=rng.uniform(18,45)*K,sz=rng.uniform(1.8,3.2)*K*1.3) for i in range(16)]
MO=[dict(x=rng.uniform(0,W),y=rng.uniform(60,800),a=int(rng.choice([1,-1,0,0])),ph=rng.uniform(0,6.28),k=int(rng.choice([3,4,5,6])),amp=rng.uniform(15,60)*0.8,sz=rng.uniform(1.5,3.5),tw=int(rng.choice([9,12,18,24])),ax=rng.uniform(20,70)) for i in range(60)]
TONG=tuple((ox*K,k,ph,hh*K,ww*K) for (ox,k,ph,hh,ww) in ((0,107,0.0,46,15),(-14,169,1.7,34,11),(13,145,3.1,38,12),(3,79,4.4,60,13),(-6,209,5.2,28,9)))
def blob(ov,cx,cy,r,a,col=None):
    R=int(r*2.6)+1; x0,x1=max(0,int(cx)-R),min(W,int(cx)+R+1); y0,y1=max(0,int(cy)-R),min(H,int(cy)+R+1)
    if x0>=x1 or y0>=y1: return
    g=np.exp(-((xs[x0:x1][None,:]-cx)**2+(ys[y0:y1][:,None]-cy)**2)/(r*r))*a
    ov[y0:y1,x0:x1]+=g[...,None] if col is not None else g
def sceneframe(f):
    u=f/N
    S=lambda k,ph=0:np.sin(TAU*(k*u)+ph)
    mapx=X.copy(); mapy=Y.copy()
    mapx[250:540]+=(0.7*ty[250:540,None]*np.sin(TAU*2*u+xs/260)[None,:]).astype(np.float32)
    aa=TAU*18*u; s_=np.sin(aa+ys/6)[:,None]*np.cos(xs/38)[None,:]+np.cos(aa+ys/6)[:,None]*np.sin(xs/38)[None,:]
    mapy[560:840]+=(1.0*ry[560:840,None]*rx[None,:]*s_[560:840]).astype(np.float32)
    img=cv2.remap(base,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    fl=1+0.11*S(107,0.5)+0.08*S(169,2.2)+0.07*S(55,1.0)+0.05*S(245,3.0)
    y0,y1,x0,x1=350,1080,500,1500
    yy=ys[y0:y1][:,None]; xx=xs[x0:x1][None,:]
    d2=(xx-FX)**2+(yy-FY)**2
    glow=(np.exp(-d2/(170**2))*0.20+np.exp(-d2/(55**2))*0.26)*fl
    col=np.array([60,150,255],np.float32)/255
    roi=img[y0:y1,x0:x1]; roi[:]=1-(1-roi)*(1-glow[...,None]*col)
    sg=np.exp(-(((X[:500:2,::2]-300)**2+(Y[:500:2,::2]-180)**2)/(600**2)))*0.05*(1+0.4*S(1))
    sg=cv2.resize(sg,(W,500))
    img[:500]=1-(1-img[:500])*(1-sg[...,None]*np.array([140,205,255],np.float32)/255)
    th=TAU*u
    m=NA*np.cos(th)+NB*np.sin(th); m2=NC*np.cos(2*th+1)+ND*np.sin(2*th+1)
    mm=cv2.resize(0.8*m+0.45*m2,(W,H),interpolation=cv2.INTER_CUBIC)
    al=np.clip(0.42+0.32*mm,0,1)*mist_my*mist_mx*0.24
    img+=(mist_col[None,None,:]-img)*al[...,None]
    fadd=np.zeros((H,W,3),np.float32)
    for (ox,k,ph,hh,ww) in TONG:
        wob=TAU*k*u+ph
        h=hh*(0.75+0.25*np.sin(wob)+0.12*np.sin(wob*2+1))
        cx=FX+ox+6*K*np.sin(TAU*(k//2)*u+ph*2); cy=FY-12*K-0.55*h
        R=int(h*1.6)+20; x0=int(cx)-R; x1=int(cx)+R; y0=int(cy)-R; y1=int(cy)+R
        gxs=xs[x0:x1][None,:]; gys=ys[y0:y1][:,None]
        sx=ww*(1-0.5*np.clip((FY-12*K-gys)/(1.1*h+1),0,1))
        g=np.exp(-((gxs-cx)/sx)**2-((gys-cy)/(0.62*h+3))**2)*(gys<FY-3)
        fadd[y0:y1,x0:x1,0]+=g*0.10; fadd[y0:y1,x0:x1,1]+=g*0.40; fadd[y0:y1,x0:x1,2]+=g*0.85
    img=1-(1-img)*(1-np.clip(fadd,0,1)*0.55)
    ov=np.zeros((H,W),np.float32)
    for s in SM:
        v=(4*u+s['o'])%1.0
        cx=FX+s['dx']*v+30*K*np.sin(TAU*(v*1.4)+s['ph']); cy=FY-60*K-430*K*v
        blob(ov,cx,cy,s['r0']+85*K*v,0.20*np.sin(np.pi*v)**1.3)
    img+=((np.array([228,228,225],np.float32)/255)[None,None,:]-img)*np.clip(ov,0,0.5)[...,None]*0.55
    add=np.zeros((H,W,3),np.float32); ec=np.array([60,170,255],np.float32)/255
    for e in EM:
        v=(e['k']*u+e['o'])%1.0
        cx=FX+15*K*np.sin(TAU*v*2+e['ph'])+e['w']*v*np.sin(e['ph']); cy=FY-30*K-340*K*v
        blob(add,cx,cy,e['sz']*1.5,(1-v)**1.3*np.sin(np.pi*min(1,v*8)),col=1)
    add*=ec
    for mo in MO:
        x=(mo['x']+mo['a']*W*u+mo['ax']*np.sin(TAU*mo['k']*u+mo['ph']*3))%W; y=mo['y']+mo['amp']*np.sin(TAU*mo['k']*u+mo['ph'])
        blob(add,x,y,mo['sz'],0.55*(0.45+0.45*np.sin(TAU*mo['tw']*u+mo['ph']*2)),col=1)
    img=1-(1-img)*(1-np.clip(add,0,1))
    vg=(1-0.26*((X[::4,::4]-960)/960)**2-0.18*((Y[::4,::4]-540)/540)**2)
    img*=cv2.resize(vg,(W,H))[...,None]
    return np.clip(img,0,1)
if MODE=='test':
    fr=[0,200,400]
    cv2.imwrite('test.jpg',np.vstack([cv2.resize((sceneframe(f)*255).astype(np.uint8),(960,540)) for f in fr[:2]]))
    cr=[(sceneframe(f)*255).astype(np.uint8)[780:960,860:1110] for f in (0,3,6,9)]
    cv2.imwrite('flame.jpg',cv2.resize(np.hstack(cr),None,fx=2,fy=2))
    a=sceneframe(0);b=sceneframe(N);print('seam exact',np.abs(a-b).max()*255)
elif MODE=='final':
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','24','-i','-','-an','-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p','-g','48','master36.mp4'],stdin=subprocess.PIPE)
    t0=time.time()
    for f in range(N):
        p.stdin.write((sceneframe(f)*255+0.5).astype(np.uint8).tobytes())
        if f%72==0: print(f,time.time()-t0,flush=True)
    p.stdin.close(); p.wait(); print('done',time.time()-t0)
