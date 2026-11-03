"""Guess-the-animal Short: a ball bounces round a pixel grid and paints the animal in.
Usage: python3 guess.py art.jpg NAME out.mp4 [still SEC,SEC]
Env: HEXR (hex radius px), GW GH GY0 (grid area), K, CROP x0,y0,x1,y1, SEC, SEED, SPD, B0 B1 BP (brush), GAP, MUSIC, MSTART
"""
import os,sys,subprocess,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from sklearn.cluster import KMeans
art,NAME,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
STILL=[float(x) for x in sys.argv[5].split(',')] if len(sys.argv)>5 and sys.argv[4]=='still' else None
E=os.environ.get; COLS=int(E('COLS',44)); ROWS=int(E('ROWS',56)); K=int(E('K',10)); SEC=float(E('SEC',52)); SEED=int(E('SEED',3))
W,H,FPS=1080,1920,24; R=float(E('HEXR',6.5)); GW=int(E('GW',960)); GH=int(E('GH',1080)); GX0=(W-GW)//2; GY0=int(E('GY0',380)); CELL=R*2.2
from scipy.spatial import cKDTree
hw=np.sqrt(3)*R; hh=1.5*R; ncol=int(GW/hw)+2; nrow=int(GH/hh)+2
cs=np.array([((c+0.5*(r%2))*hw+hw*0.25, r*hh+R*0.6) for r in range(nrow) for c in range(ncol)]); cs=cs[(cs[:,0]<GW-hw*0.2)&(cs[:,1]<GH-R*0.2)]
tree=cKDTree(cs); N=len(cs)
yy,xx=np.mgrid[0:GH,0:GW]; d,ix=tree.query(np.c_[xx.ravel(),yy.ravel()],k=2); idmap=ix[:,0].reshape(GH,GW); edge=((d[:,1]-d[:,0])<float(E('GAP',1.3))).reshape(GH,GW)
im=Image.open(art).convert('RGB')
if E('CROP'): im=im.crop(tuple(int(v) for v in E('CROP').split(',')))
from PIL import ImageFilter
im=im.filter(ImageFilter.UnsharpMask(radius=3,percent=160,threshold=2)).resize((GW,GH),Image.LANCZOS)
blur=cv2.GaussianBlur(np.array(im),(0,0),R*0.55)
small=blur[np.clip(cs[:,1].astype(int),0,GH-1),np.clip(cs[:,0].astype(int),0,GW-1)].astype(np.float32)
km=KMeans(K,n_init=4,random_state=1).fit(small); pal=km.cluster_centers_; lab=km.labels_
bgi=int(np.argmax(pal.sum(1)-pal.std(1)*2)); pal[bgi]=(38,38,44)
isbg=lab==bgi; grid=pal[lab].astype(np.uint8)
UNP=np.array((38,38,44),np.uint8); BGC=(18,18,22)
rng=np.random.default_rng(SEED)
BR=float(E('BALL',9)); TOUCH=BR+R*0.9
G=float(E('GRAV',2600)); REST=float(E('REST',0.80)); WREST=0.9; FRIC=0.985   # px/s^2, bounce restitution, wall restitution, floor grip per bounce
# --- real bouncing: each ball is released with a start position/velocity (chosen to cross unpainted hexes), then pure physics:
#     constant gravity, energy-losing bounces off floor and side walls, rolls to rest. No steering, no time-warp.
SDT=1/480; painted=np.zeros(N,bool); ptime=np.full(N,-1.0); bounces=[]; balls=[]
animal=np.where(~isbg)[0]; tot_animal=len(animal); isun0=~isbg
def step(p,v,dt):
    v=v+np.array([0,G])*dt; p=p+v*dt; hit=None; spd=0
    if p[0]<BR: p[0]=BR; spd=abs(v[0]); v[0]=abs(v[0])*WREST; hit=0
    elif p[0]>GW-BR: p[0]=GW-BR; spd=abs(v[0]); v[0]=-abs(v[0])*WREST; hit=0
    if p[1]>GH-BR:
        p[1]=GH-BR; spd=abs(v[1]); v[1]=-v[1]*REST; v[0]*=FRIC; hit=1
        if abs(v[1])<90: v[1]=0
    return p,v,hit,spd
def plan(painted_now):
    isun=~painted_now&isunan; best=None
    for _ in range(int(E('CAND',70))):
        p=np.array([rng.uniform(BR,GW-BR),rng.uniform(BR,GH*0.55)]); v=np.array([rng.choice([-1,1])*rng.uniform(120,760),rng.uniform(-900,200)])
        seen=set(); dt=1/120
        for k in range(int(9/dt)):
            p,v,h,_=step(p,v,dt)
            if k%2==0:
                for i in tree.query_ball_point(p,TOUCH):
                    if isun[i]: seen.add(i)
            if p[1]>GH-BR-1 and abs(v[1])<1 and abs(v[0])<20: break
        if best is None or len(seen)>best[0]: best=(len(seen),p.copy(),v.copy())
    return best
isunan=np.zeros(N,bool); isunan[animal]=True
t=0.0; active=[]; last_launch=-99; CAD=float(E('CAD',3.0)); stop=False
def launch(t):
    global rng
    p0=np.array([rng.uniform(GW*0.15,GW*0.85),BR+4]) if not balls else None
    # re-run candidate search but keep the *start state*: recompute by regenerating with the same rng draw
    st=None; bestn=-1; isun=~painted&isunan
    for _ in range(max(1,int(float(E('C0',1))+float(E('C1',40))*min(1,t/float(E('TREF',70)))**float(E('CP',2))))):
        p=np.array([rng.uniform(BR,GW-BR),rng.uniform(BR,GH*0.5)]); v=np.array([rng.choice([-1,1])*rng.uniform(120,760),rng.uniform(-900,200)])
        p1,v1=p.copy(),v.copy(); seen=set(); dt=1/120
        for k in range(int(9/dt)):
            p1,v1,h,_=step(p1,v1,dt)
            if k%2==0:
                for i in tree.query_ball_point(p1,TOUCH):
                    if isun[i]: seen.add(i)
            if p1[1]>GH-BR-1 and abs(v1[1])<1 and abs(v1[0])<20: break
        if len(seen)>bestn: bestn=len(seen); st=(p,v)
    b={'t0':t,'p':st[0].copy(),'v':st[1].copy(),'pos':[],'vel':[],'rest':None}; return b
while t<float(E('MAXT',120)):
    if (t-last_launch>=float(E('CAD0',6))*(float(E('CAD1',0.25))/float(E('CAD0',6)))**min(1,t/float(E('TREF',70))) or not active) and (painted&isunan).sum()<(1-float(E('TAIL',0.04)))*tot_animal:
        b=launch(t); active.append(b); balls.append(b); last_launch=t
    for b in list(active):
        b['p'],b['v'],h,spd=step(b['p'],b['v'],SDT)
        b['pos'].append(b['p'].copy()); b['vel'].append(b['v'].copy())
        if h is not None and spd>150: bounces.append((t,h,b['p'][0]/GW,min(1,spd/2200)))
        if b['p'][1]>=GH-BR-0.5 and abs(b['v'][1])<1: b['v'][0]*=1-1.6*SDT   # rolling drag
        for i in tree.query_ball_point(b['p'],TOUCH):
            if not painted[i]: painted[i]=True; ptime[i]=t
        if b['p'][1]>=GH-BR-0.5 and abs(b['v'][1])<1 and abs(b['v'][0])<14:
            b['t1']=t; active.remove(b)
    t+=SDT
    if not active and (painted&isunan).sum()>=(1-float(E('TAIL',0.04)))*tot_animal: break
for b in active: b['t1']=t
T_end=t; print('balls',len(balls),'sim end',round(T_end,1),'s; bounces',len(bounces),flush=True)
NF=int(max(float(E('SEC',0)) or 0,T_end)*FPS); extra=int(2.0*FPS)
left=np.where(~painted)[0]; rng.shuffle(left)
fo=np.full(N,10**9)
for i in range(N):
    if painted[i]: fo[i]=int(ptime[i]*FPS)
for k,i in enumerate(left): fo[i]=NF+int(k*extra/max(1,len(left)))
TOT=NF+extra+int(float(E('HOLD',3.4))*FPS)
bounces=[(tb,ax,x,v) for tb,ax,x,v in bounces]
cov=lambda q:((fo<=int(NF*q))&~isbg).sum()/tot_animal
print('coverage at 25/50/75% time:',[round(cov(q),2) for q in (.25,.5,.75)],flush=True)
F=lambda s,w='Bold': ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-%s.ttf'%w,s)
def text_c(d,xy,t_,font,fill):
    w=d.textlength(t_,font=font); d.text((xy[0]-w/2,xy[1]),t_,font=font,fill=fill)
TINTS=[(255,190,90),(255,140,170),(110,220,210),(190,160,255)]; CREAM=(250,236,210); GOLD=(255,214,120); DIM=(34,37,62)
TITLE=E('TITLE',"who\u2019s getting cosy tonight?")
ART=E('ARTICLE','an' if NAME[0].lower() in 'aeiou' else 'a')
# background: dusk gradient, faint stars, warm glow behind the panel
yy_=np.linspace(0,1,H)[:,None,None]; top=np.array((20,22,50)); bot=np.array((66,40,64)); bgimg=(top*(1-yy_)+bot*yy_)*np.ones((1,W,1))
cx_,cy_=W/2,GY0+GH/2; gy,gx=np.mgrid[0:H,0:W]; bgimg+=np.exp(-(((gx-cx_)/700)**2+((gy-cy_)/800)**2))[...,None]*np.array((60,34,18))
for _ in range(110):
    x_,y_=int(rng.integers(0,W)),int(rng.integers(0,int(H*0.55))); br=rng.uniform(.25,.8); cv2.circle(bgimg,(x_,y_),int(rng.integers(1,3)),tuple(float(c)*br for c in (255,240,210)),-1,cv2.LINE_AA)
BG=np.clip(bgimg,0,255).astype(np.uint8)
RAD=44; m=np.zeros((GH,GW),np.uint8); cv2.rectangle(m,(RAD,0),(GW-RAD,GH),255,-1); cv2.rectangle(m,(0,RAD),(GW,GH-RAD),255,-1)
for cx2,cy2 in ((RAD,RAD),(GW-RAD,RAD),(RAD,GH-RAD),(GW-RAD,GH-RAD)): cv2.circle(m,(cx2,cy2),RAD,255,-1,cv2.LINE_AA)
mask=(cv2.GaussianBlur(m,(0,0),1.2)/255.0)[...,None]
glowm=np.zeros((H,W),np.float32); glowm[GY0:GY0+GH,GX0:GX0+GW]=m/255.0; glowm=cv2.GaussianBlur(glowm,(0,0),26)[...,None]*np.array((255,190,110))*0.35
emask=np.where(edge,0.55,1.0)[...,None]
def frame(f):
    age=(f-fo).astype(np.float32); st=np.where((fo<=f)[:,None],grid,np.array(DIM,np.uint8)).astype(np.float32)
    fl=np.clip(1-age/14,0,1)*(fo<=f)*0.9; st=st*(1-fl[:,None])+np.array(GOLD,np.float32)*fl[:,None]
    st=np.where(isbg[:,None],np.array(DIM,np.float32),st)
    big=st[idmap]*emask
    canvas=BG.astype(np.float32)+glowm
    reg=canvas[GY0:GY0+GH,GX0:GX0+GW]; canvas[GY0:GY0+GH,GX0:GX0+GW]=reg*(1-mask)+big*mask
    tt=f/FPS
    for bl in balls:
        if tt<bl['t0'] or tt>bl['t1']+0.6: continue
        k=min(len(bl['pos'])-1,int((tt-bl['t0'])/SDT)); bx,by=bl['pos'][k]; vx,vy=bl['vel'][k]; fade=1.0 if tt<=bl['t1'] else max(0,1-(tt-bl['t1'])/0.6)
        px,py=int(GX0+bx),int(GY0+by); r0=int(BR*5); y0,y1,x0,x1=max(0,py-r0),min(H,py+r0),max(0,px-r0),min(W,px+r0)
        lay=np.zeros((y1-y0,x1-x0,3),np.float32); cv2.circle(lay,(px-x0,py-y0),int(BR*1.8),TINTS[balls.index(bl)%4],-1,cv2.LINE_AA); lay=cv2.GaussianBlur(lay,(0,0),BR*1.3)*1.5*fade
        canvas[y0:y1,x0:x1]+=lay
        sp=min(1,np.hypot(vx,vy)/2500); ang=np.degrees(np.arctan2(vy,vx)); ax_=(int(BR*(1+0.45*sp)),int(BR*(1-0.2*sp)))
        near=(by>GH-BR-2.5 and abs(vy)<600)  # squashed while in contact with the floor
        if near: ax_=(int(BR*1.25),int(BR*0.78)); ang=0
        for col,sc in (((255,244,214),1.0),((255,255,255),0.55)):
            cv2.ellipse(canvas,(px,py),(max(2,int(ax_[0]*sc)),max(2,int(ax_[1]*sc))),ang,0,360,tuple(c*fade for c in col),-1,cv2.LINE_AA)
    p=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)); d=ImageDraw.Draw(p)
    pct=int(100*((fo<=f)&~isbg).sum()/tot_animal); done=f>=NF+extra
    text_c(d,(W//2,120),TITLE,F(60,'Light'),CREAM); text_c(d,(W//2,212),'guess in the comments',F(34,'Light'),(200,184,170))
    shown=100 if done else min(pct,99)
    text_c(d,(W//2,GY0+GH+26),f'{shown}%',F(52,'Bold'),GOLD)
    bw=560; bx0=(W-bw)//2; by0=GY0+GH+108; d.rounded_rectangle((bx0,by0,bx0+bw,by0+10),5,fill=(60,52,84)); 
    if shown>0: d.rounded_rectangle((bx0,by0,bx0+max(10,int(bw*shown/100)),by0+10),5,fill=GOLD)
    if not done: text_c(d,(W//2,GY0+GH+140),'can you name it before halfway?' if pct<50 else 'still guessing?',F(32,'Light'),(214,198,180))
    else:
        k=min(1,(f-(NF+extra))/12); sz=int(80+10*np.sin(k*np.pi)); t2=f"it\u2019s {ART} {NAME.lower()}"
        gl=Image.new('RGB',p.size,(0,0,0)); ImageDraw.Draw(gl).text((W//2-d.textlength(t2,font=F(sz))/2,GY0+GH+138),t2,font=F(sz),fill=(255,170,60)); gl=gl.filter(ImageFilter.GaussianBlur(14))
        p=Image.fromarray(np.clip(np.array(p).astype(np.int16)+(np.array(gl)*0.6*k).astype(np.int16),0,255).astype(np.uint8)); d=ImageDraw.Draw(p)
        text_c(d,(W//2,GY0+GH+138),t2,F(sz),(255,236,190))
    text_c(d,(W//2,GY0+GH+275),'z o n e l y t i m e',F(28,'Light'),(150,134,150))
    return np.array(p)
if STILL:
    for s_ in STILL: Image.fromarray(frame(int(s_*FPS))).save(OUT.replace('.mp4',f'_{s_}.png'))
    sys.exit()
# audio: pentatonic plinks on bounces over a quiet cleared track
SR=44100; dur=TOT/FPS; aud=np.zeros(int(dur*SR)+SR,np.float32)
pent=[0,2,4,7,9,12,14,16]; base=392.0
for i,(t,ax,xr,vol) in enumerate(bounces):
    fr=base*2**(pent[min(len(pent)-1,int(xr*len(pent)))]/12); n=int(0.5*SR); tt=np.arange(n)/SR
    s=(np.sin(2*np.pi*fr*tt)+.3*np.sin(2*np.pi*fr*2.01*tt))*np.exp(-tt*7)*(0.06+0.22*vol); i0=int(t*SR); aud[i0:i0+n]+=s[:max(0,len(aud)-i0)][:n] if i0+n<=len(aud) else s[:len(aud)-i0]
tr=(NF+extra)/FPS
for j,sem in enumerate((0,4,7,12,16)):
    fr=base*2**(sem/12); n=int(1.4*SR); tt=np.arange(n)/SR; s2=np.sin(2*np.pi*fr*tt)*np.exp(-tt*2.6)*0.2; i0=int((tr+j*0.09)*SR); aud[i0:i0+n]+=s2[:max(0,len(aud)-i0)][:n]
if E('MUSIC'):
    m=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',E('MSTART','20'),'-t',str(dur),'-i',E('MUSIC'),'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32)
    aud[:len(m)]+=m*0.55
aud=np.clip(aud/ max(1,np.abs(aud).max())*0.9,-1,1)
open(OUT+'.raw','wb').write(aud.tobytes())
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-f','f32le','-ar',str(SR),'-ac','1','-i',OUT+'.raw','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-af','loudnorm=I=-14:TP=-1.5','-shortest','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for f in range(TOT): enc.stdin.write(frame(f).tobytes())
enc.stdin.close(); enc.wait(); os.remove(OUT+'.raw'); print('done',OUT)
