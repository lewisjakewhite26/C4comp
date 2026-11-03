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
BR=float(E('BALL',22)); TOUCH=BR+R*0.9; G=float(E('GRAV',1000)); VT=float(E('VT',480))
# --- firefly-ball: thrown in gravity arcs aimed at unpainted hexes, bounces off the walls/floor, paints EVERY hex it touches
SDT=1/240; t=0.0; pos=np.array([GW*0.2,GH*0.9]); painted=np.zeros(N,bool); ptime=np.full(N,-1.0); path=[]; bounces=[]; cvh=[]
animal=np.where(~isbg)[0]; tot_animal=len(animal)
def aim(p):
    un=animal[~painted[animal]]
    if len(un)==0: return np.array([GW/2,GH/2])
    cand=rng.choice(un,min(len(un),60),replace=False); isun=(~painted&~isbg)
    sc=[isun[tree.query_ball_point(cs[c],TOUCH*2.2)].sum() for c in cand]; T=cs[cand[int(np.argmax(sc))]]; dist=np.hypot(*(T-p)); tau=float(np.clip(dist/VT,0.45,1.5))
    return (T-p)/tau-0.5*np.array([0,G])*tau
vel=aim(pos); nb=0
while t<float(E('MAXT',140)):
    vel=vel+np.array([0,G])*SDT; pos=pos+vel*SDT; hit=None
    if pos[0]<BR: pos[0]=BR; hit=0
    elif pos[0]>GW-BR: pos[0]=GW-BR; hit=0
    if pos[1]>GH-BR: pos[1]=GH-BR; hit=1
    elif pos[1]<BR: pos[1]=BR; hit=1
    if hit is not None: bounces.append((t,hit,pos[0]/GW)); vel=aim(pos); nb+=1
    for i in tree.query_ball_point(pos,TOUCH):
        if not painted[i]: painted[i]=True; ptime[i]=t
    path.append(pos.copy()); cvh.append(int((painted&~isbg).sum())); t+=SDT
    if (painted&~isbg).sum()>=(1-float(E('TAIL',0.03)))*tot_animal: break
T_end=t; path=np.array(path); cvh=np.array(cvh)/tot_animal; cmax=cvh[-1]
NF=int(SEC*FPS); extra=int(1.6*FPS); WL=float(E('WLIN',0.15))
us=np.linspace(0,1,NF+1); s_cov=np.searchsorted(cvh,us*cmax)*SDT; s_cov=np.minimum(s_cov,T_end); S=WL*us*T_end+(1-WL)*s_cov   # sim time shown at video fraction u
print('S at u=.25/.5/.75/1:',[round(S[int(NF*q)],1) for q in (.25,.5,.75,1)],end=' ');print('cells',N,'sim end',round(T_end,1),'s; bounces',len(bounces),flush=True)
tofr=lambda ts: int(np.searchsorted(S,ts))   # first frame showing sim time ts
left=np.where(~painted)[0]; rng.shuffle(left)
fo=np.full(N,10**9)
for i in range(N):
    if painted[i]: fo[i]=tofr(ptime[i])
for k,i in enumerate(left): fo[i]=NF+int(k*extra/max(1,len(left)))
TOT=NF+extra+int(float(E('HOLD',3.4))*FPS)
bounces=[(tofr(tb)/FPS,ax,x) for tb,ax,x in bounces]
cov=lambda q:((fo<=int(NF*q))&~isbg).sum()/tot_animal
print('coverage at 25/50/75% time:',[round(cov(q),2) for q in (.25,.5,.75)],flush=True)
pf=lambda f: path[min(len(path)-1,int(S[min(f,NF)]/SDT))]
F=lambda s,w='Bold': ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-%s.ttf'%w,s)
def text_c(d,xy,t_,font,fill):
    w=d.textlength(t_,font=font); d.text((xy[0]-w/2,xy[1]),t_,font=font,fill=fill)
CREAM=(250,236,210); GOLD=(255,214,120); DIM=(34,37,62)
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
    st=np.where(isbg[:,None],np.where((fo<=f)[:,None],np.array(DIM,np.float32)*0.8+fl[:,None]*np.array((90,70,30),np.float32),np.array(DIM,np.float32)),st)
    big=st[idmap]*emask
    canvas=BG.astype(np.float32)+glowm
    reg=canvas[GY0:GY0+GH,GX0:GX0+GW]; canvas[GY0:GY0+GH,GX0:GX0+GW]=reg*(1-mask)+big*mask
    if f<NF:
        bx,by=pf(f); px,py=int(GX0+bx),int(GY0+by); r0=int(BR*5); y0,y1,x0,x1=max(0,py-r0),min(H,py+r0),max(0,px-r0),min(W,px+r0)
        lay=np.zeros((y1-y0,x1-x0,3),np.float32); cv2.circle(lay,(px-x0,py-y0),int(BR*1.6),(255,190,90),-1,cv2.LINE_AA); lay=cv2.GaussianBlur(lay,(0,0),BR*1.1)*1.3
        canvas[y0:y1,x0:x1]+=lay; cv2.circle(canvas,(px,py),int(BR),(255,244,214),-1,cv2.LINE_AA); cv2.circle(canvas,(px,py),int(BR*0.55),(255,255,255),-1,cv2.LINE_AA)
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
for i,(t,ax,xr) in enumerate(bounces):
    fr=base*2**(pent[min(len(pent)-1,int(xr*len(pent)))]/12); n=int(0.5*SR); tt=np.arange(n)/SR
    s=(np.sin(2*np.pi*fr*tt)+.3*np.sin(2*np.pi*fr*2.01*tt))*np.exp(-tt*7)*0.22; i0=int(t*SR); aud[i0:i0+n]+=s[:max(0,len(aud)-i0)][:n] if i0+n<=len(aud) else s[:len(aud)-i0]
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
