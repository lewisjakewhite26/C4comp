# Light the street, whimsical version: the firefly drifts and loops about, notices a lantern, floats over,
# settles on the glass, and the lantern's own glass panes warm up and glow. Then it lingers and wanders off again.
import cv2, numpy as np, json, math, subprocess, sys, os
O=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/'
CLIP=os.environ.get('CLIP','slow1.mp4'); TRK=os.environ.get('TRK','tracks_slow.json')
VP=np.array([float(v) for v in os.environ.get('VP','360,570').split(',')])
cap=cv2.VideoCapture(O+CLIP); FR=[]
while True:
    ok,f=cap.read()
    if not ok: break
    FR.append(f)
N=len(FR); H,W=FR[0].shape[:2]; FPS=24; DUR=N/FPS
tr=json.load(open(O+TRK))
WTRK=os.environ.get('WTRK','tracks_win.json')
if os.path.exists(O+WTRK): tr.update(json.load(open(O+WTRK)))
MERGE={'Rb9':'Rb','Lb9':'Lb'}
lamps={}
for k,v in tr.items():
    kk=MERGE.get(k,k); d={int(a):b for a,b in v.items()}
    lamps.setdefault(kk,{}).update({f:p for f,p in d.items() if f not in lamps.get(kk,{})})
for k in lamps:
    fs=sorted(lamps[k]); P=np.array([lamps[k][f] for f in fs],np.float64)
    if len(P)>9:
        ker=np.exp(-np.arange(-6,7)**2/(2*2.5**2)); ker/=ker.sum()
        Pp=np.pad(P,((6,6),(0,0)),mode='edge'); P=np.stack([np.convolve(Pp[:,c],ker,'valid') for c in (0,1)],1)
    lamps[k]={f:tuple(P[i]) for i,f in enumerate(fs)}
# ---------------- time warp: the opening runs at half this speed, easing up to normal, so the firefly has time near the camera
RAMP=float(os.environ.get('RAMP','0.5'))
if RAMP<1:
    rate=lambda t: RAMP if t<7 else (RAMP+(1-RAMP)*min(1,(t-7)/4))
    src=[0.0]
    while src[-1]<N-1: src.append(src[-1]+rate(len(src)/FPS))
    src=np.array(src[:-1]); NF=[]; NL={k:{} for k in lamps}
    for g,u in enumerate(src):
        i0=int(u); a_=u-i0; i1=min(i0+1,N-1)
        NF.append(cv2.addWeighted(FR[i0],1-a_,FR[i1],a_,0) if a_>0.01 else FR[i0])
        for k,d in lamps.items():
            if i0 in d and i1 in d: NL[k][g]=tuple(np.array(d[i0])*(1-a_)+np.array(d[i1])*a_)
    FR=NF; lamps=NL; N=len(FR)
    print('warped length',round(N/FPS,1),'s')
WINK=['W0','W1','W2','W3']
if all(k in lamps for k in WINK):
    fs=set.intersection(*[set(lamps[k]) for k in WINK])
    lamps['WIN']={f:tuple(np.mean([lamps[k][f] for k in WINK],0)+np.array([4,40])) for f in fs}
def lpf(k,f):
    """lantern position at frame f; after it leaves the frame, keep it moving outwards the way it was going."""
    d=lamps[k]; fs=sorted(d)
    if f in d: return np.array(d[f])
    if f<fs[0]: return np.array(d[fs[0]])
    fl=fs[-1]; fb=fs[max(0,len(fs)-11)]
    pl,pb=np.array(d[fl]),np.array(d[fb])
    g=(np.linalg.norm(pl-VP)/max(1e-3,np.linalg.norm(pb-VP)))**(1/max(1,fl-fb))
    return VP+(pl-VP)*g**(f-fl)
def lp(k,t): return lpf(k,int(round(t*FPS)))
def scl(p): return max(0.08,np.linalg.norm(np.asarray(p)-VP)/400)
# ---------------- the firefly: a little physics creature with momentum, steering towards whatever it is interested in
PLAN=json.loads(os.environ.get('PLAN','[["La",0.0,0.4,0.5,300],["Ra",0.0,0.3,0.3,300],["WIN",12.2,1.6,1.0,170],["Rb",20.6],["Lb",27.8]]'))   # [lantern, moment it decides to go]
NEAR=1.15; NOTICE=float(os.environ.get('NOTICE','0.6')); HOVER=1.2; DBG=os.environ.get('DBG')
def home(t):                                          # a lazy spot just in front of us that drifts about
    return np.array([360+130*math.sin(0.21*t+1.0)+50*math.sin(0.53*t+2.0),
                     815+70*math.sin(0.17*t)+35*math.sin(0.41*t+0.7)])
rng=np.random.RandomState(int(os.environ.get('SEED','4'))); PH=rng.uniform(0,6.3,8)
def breeze(t):                                        # gentle air currents
    return 55*np.array([math.sin(0.9*t+PH[0])+0.6*math.sin(1.7*t+PH[1])+0.3*math.sin(3.1*t+PH[4]),
                        math.cos(0.8*t+PH[2])+0.6*math.sin(1.3*t+PH[3])+0.3*math.cos(2.7*t+PH[5])])
def glass_pt(k,t):
    p=lp(k,t); return p+np.array([0,14])*scl(p)
POS=np.zeros((N,2)); SZ=np.zeros(N); NOT=np.zeros(N); TON={}
p=np.array([360.0,800.0]); v=np.array([-40.0,-120.0]); z=NEAR; hov=HOVER; ntc=NOTICE; vgo=230
state='home'; plan=list(PLAN); cur=None; tst=0.0; tv=np.zeros(2); SUB=6; dt=1/(FPS*SUB)
for f in range(N):
    for sub in range(SUB):
        t=(f*SUB+sub)*dt
        if state in('home','back') and plan and t>=plan[0][1]-NOTICE:
            it_=plan.pop(0); k_=it_[0]; hov=it_[2] if len(it_)>2 else HOVER; ntc=it_[3] if len(it_)>3 else NOTICE; vgo=it_[4] if len(it_)>4 else 230
            if np.linalg.norm(lp(k_,t+2.5)-VP)<700 and 10<lp(k_,t+1.5)[1]: state,cur,tst='notice',k_,t   # still in view when we'd get there
        if state=='go' and lp(cur,t)[1]<5: state='back'; tst=t                                   # it slipped out of frame: give up
        if state=='notice' and t>=tst+ntc: state='go'; tst=t
        if DBG and (state!=getattr(sys,'_ls',None)): print(round(t,2),state,cur,np.round(p),flush=True); sys._ls=state
        tv=np.zeros(2)
        if state in('home','notice'): tgt=home(t); vmax=90 if state=='home' else 22; arr=140; zt=NEAR
        else:
            tgt=glass_pt(cur,t); tv=(glass_pt(cur,t+0.1)-tgt)/0.1; tgt=tgt+tv*0.35; zt=scl(tgt); arr=55*zt+10
            vmax={'go':vgo,'hover':40,'back':120}[state]
            if state=='back': tgt=home(t); arr=140; zt=NEAR
        if state=='hover':
            tgt=tgt+np.array([0,2.0*math.sin(t*8)])*zt
            if t>=tst+hov: state='back'; tst=t
        if state=='back' and np.linalg.norm(home(t)-p)<60: state='home'; cur=None
        d=tgt-p; dist=np.linalg.norm(d)
        if state=='go' and dist<10+6*zt: state='hover'; tst=t; TON[cur]=t+0.35
        desired=d/(dist+1e-6)*vmax*min(1,dist/arr)+(tv if state in('go','hover') else 0)
        steer=desired-v; mag=np.linalg.norm(steer); fmax=260 if state!='hover' else 400
        if mag>fmax: steer*=fmax/mag
        wind=breeze(t)*(0.25 if state in('hover','go') else 1.0)
        v=v+(steer+wind)*dt; v*=(1-0.5*dt); p=p+v*dt
        z+=(zt-z)*min(1,dt*1.3)
    POS[f]=p; SZ[f]=z; NOT[f]=1.0 if state=='notice' else 0.0
NOT=np.convolve(NOT,np.ones(9)/9,'same')
print('lanterns lit at',{k:round(v,2) for k,v in TON.items()})
def fly_pos(f): return POS[f],SZ[f],NOT[f]
# ---------------- lantern glass glow from the footage itself
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
WARM=np.array([60,140,255],np.float32); HOT=np.array([150,215,255],np.float32)
def inten(k,t):
    if k not in TON or t<TON[k]: return 0.0
    a=t-TON[k]
    if a<0.6: return [0.25,0.05,0.5,0.15,0.75,0.6,0.9,0.85,1.0,1.0,1.0,1.0][min(11,int(a/0.05))]
    return 1.0+0.025*math.sin(t*7+len(k))
def glass_mask(f,p,s,gray):
    """trapezoid of lantern glass below the cap, keeping only the lighter (glass) pixels so the dark frame stays dark."""
    x0,y0=int(p[0]-34*s-2),int(p[1]-24*s-2); x1,y1=int(p[0]+34*s+3),int(p[1]+50*s+3)
    x0,y0=max(0,x0),max(0,y0); x1,y1=min(W,x1),min(H,y1)
    if x1<=x0 or y1<=y0: return None
    sub_y,sub_x=yy[y0:y1,x0:x1],xx[y0:y1,x0:x1]
    v=np.clip((sub_y-(p[1]-18*s))/(62*s),0,1)                       # 0 at the top of the glass, 1 at the bottom
    half=(27-12*v)*s
    inside=(np.abs(sub_x-p[0])<half)&(sub_y>p[1]-18*s)&(sub_y<p[1]+44*s)
    g=gray[y0:y1,x0:x1]
    loc=np.median(g[inside]) if inside.any() else 0.3
    glassy=np.clip((g-loc*0.75)/(loc*0.5+1e-3),0,1)                  # lighter glass pane pixels
    m=inside.astype(np.float32)*glassy
    m=cv2.GaussianBlur(m,(0,0),max(0.6,0.8*s))
    hot=np.exp(-(((sub_x-p[0])/(9*s+1))**2+((sub_y-(p[1]+4*s))/(10*s+1))**2))   # the mantle near the top
    return (y0,y1,x0,x1,m,hot*inside,(1-0.35*v))
TW,TH=120,300
def make_room():
    yy_,xx_=np.mgrid[0:TH,0:TW].astype(np.float32)
    room=np.zeros((TH,TW,3),np.float32)
    lamp=np.exp(-(((xx_-TW*0.8)/(TW*0.7))**2+((yy_-TH*0.42)/(TH*0.45))**2))
    room+=(0.22+0.78*lamp)[...,None]*np.array([105,170,240],np.float32)
    room+=np.exp(-(((xx_-TW*0.86)/(TW*0.12))**2+((yy_-TH*0.30)/(TH*0.06))**2))[...,None]*np.array([150,215,255],np.float32)*0.6   # the lamp itself
    room*=(0.75+0.25*(yy_/TH))[...,None]
    cur=np.clip((TW*0.16-xx_)/6,0,1)+0.15                         # curtain on the left edge, half drawn
    room*=(1-0.55*np.clip(cur,0,1))[...,None]
    sil=cv2.imread(O+'capy_tea.jpg',cv2.IMREAD_GRAYSCALE)
    m=(sil<128).astype(np.uint8); ys,xs=np.where(m>0); m=m[ys.min():ys.max()+1,xs.min():xs.max()+1]
    cw=int(TW*0.98); ch=int(m.shape[0]*cw/m.shape[1])
    m=cv2.resize(m.astype(np.float32),(cw,ch),interpolation=cv2.INTER_AREA)
    mask=np.zeros((TH,TW),np.float32); y1=TH-6; x0=int(TW*0.14)
    x0=max(0,x0-int(TW*0.12)); mask[max(0,y1-ch):y1,x0:x0+cw]=m[max(0,ch-y1):ch,:min(cw,TW-x0)]
    mask=cv2.GaussianBlur(mask,(0,0),0.8)
    room=room*(1-0.9*mask[...,None])+mask[...,None]*np.array([20,22,30],np.float32)
    sill=(yy_>TH-8).astype(np.float32); room=room*(1-0.6*sill[...,None])
    return room
ROOM=make_room()
def draw_window(img,f,t,gray,glowsrc):
    I=inten('WIN',t)
    if I<=0 or not all(f in lamps[k] for k in WINK): return 0.0
    q=np.array([lamps[k][f] for k in WINK],np.float32)
    x0,y0=np.floor(q.min(0)).astype(int)-2; x1,y1=np.ceil(q.max(0)).astype(int)+3
    x0,y0=max(0,x0),max(0,y0); x1,y1=min(W,x1),min(H,y1)
    if x1-x0<3 or y1-y0<3: return 0.0
    Hm=cv2.getPerspectiveTransform(np.float32([[0,0],[TW,0],[TW,TH],[0,TH]]),q-np.float32([x0,y0]))
    tex=cv2.warpPerspective(ROOM,Hm,(x1-x0,y1-y0))
    cov=cv2.warpPerspective(np.ones((TH,TW),np.float32),Hm,(x1-x0,y1-y0))
    g=gray[y0:y1,x0:x1]; med=np.median(g[cov>0.5]) if (cov>0.5).any() else 0.2
    glass=np.clip((g-med*0.7)/(med*0.45+1e-3),0,1)*cov                  # keep the dark iron bars dark
    glass=cv2.GaussianBlur(glass,(0,0),0.5)
    reg=img[y0:y1,x0:x1]
    img[y0:y1,x0:x1]=reg*(1-glass[...,None]*I)+tex*glass[...,None]*I*0.85
    glowsrc[y0:y1,x0:x1]+=glass*I*0.35
    return I
def frame(f):
    t=f/FPS; src=FR[f]; img=src.astype(np.float32)
    gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    hl=np.clip((gray-0.25)/0.5,0,1)
    img*=0.80
    illum=np.zeros((H,W),np.float32); refl=np.zeros((H,W),np.float32); glowsrc=np.zeros((H,W),np.float32)
    lit=[]
    for k,d in lamps.items():
        if k.startswith('W'): continue
        I=inten(k,t)
        if I<=0: continue
        p=lpf(k,f); dv=p-VP; s=scl(p)
        if np.linalg.norm(dv)>2500: continue
        dd=(xx-p[0])**2+(yy-(p[1]+10*s))**2
        illum+=I*(1.3*np.exp(-dd/(2*(105*s)**2))+0.45*np.exp(-dd/(2*(320*s)**2)))
        g=VP+np.array([dv[0]*0.92,-0.67*dv[1]])
        illum+=I*0.85*np.exp(-(((xx-g[0])/(150*s))**2+((yy-g[1])/(45*s))**2))
        ry=VP[1]-2.33*dv[1]
        band=np.exp(-((xx-g[0])/(20*s+2))**2)*np.clip((yy-g[1]+10*s)/(30*s+1),0,1)*np.exp(-np.clip(yy-ry,0,None)/(160*s+20))
        refl+=I*band
        gm=glass_mask(f,p,s,gray) if f in d else None
        if gm: lit.append((I,gm))
    img+=illum[...,None]*WARM*(0.25+gray*1.6)[...,None]*0.55
    img+=(refl*(0.10+hl*0.9))[...,None]*np.array([35,115,255],np.float32)*0.75
    for I,(y0,y1,x0,x1,m,hot,vg) in lit:                 # the glass itself glows warm, frame bars stay dark
        reg=img[y0:y1,x0:x1]
        glow=(m*vg)[...,None]*WARM*1.15*I+(hot*m)[...,None]*HOT*0.9*I
        img[y0:y1,x0:x1]=reg*(1-0.6*m[...,None]*I)+glow
        glowsrc[y0:y1,x0:x1]+=m*I*0.9+hot*m*I
    Iw=draw_window(img,f,t,gray,glowsrc)
    if Iw>0 and 'WIN' in lamps:
        pw=lpf('WIN',f); sw=scl(pw)
        dd=(xx-pw[0])**2+(yy-pw[1])**2
        img+=(Iw*0.55*np.exp(-dd/(2*(140*sw)**2)))[...,None]*WARM*(0.25+gray*1.6)[...,None]*0.5
    small=cv2.resize(glowsrc,(W//4,H//4),interpolation=cv2.INTER_AREA)
    bloom=cv2.resize(cv2.GaussianBlur(small,(0,0),3),(W,H))*1.3+cv2.resize(cv2.GaussianBlur(small,(0,0),10),(W,H))*1.1
    img+=bloom[...,None]*WARM*0.6
    # firefly: a soft green-gold body that pulses, brighter when it notices something
    fp,sz,notice=fly_pos(f)
    pul=(0.75+0.25*math.sin(t*5.5))*(1+0.6*notice)
    dd=(xx-fp[0])**2+(yy-fp[1])**2
    img+=np.exp(-dd/(2*(4.2*sz+1.3)**2))[...,None]*np.array([150,255,225],np.float32)*pul
    img+=np.exp(-dd/(2*(22*sz+4)**2))[...,None]*np.array([50,210,190],np.float32)*0.42*pul
    return np.clip(img,0,255).astype(np.uint8)
if __name__=='__main__':
    if sys.argv[1]=='still':
        for v in sys.argv[2].split(','): cv2.imwrite(O+f'l2_{v}.jpg',frame(int(float(v)*FPS)))
        sys.exit()
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-vf','scale=1080:1920:flags=lanczos','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for f in range(N): enc.stdin.write(frame(f).tobytes())
    enc.stdin.close(); enc.wait(); print('done')
