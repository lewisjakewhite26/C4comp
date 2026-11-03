"""Guess-the-animal Short: a ball bounces round a pixel grid and paints the animal in.
Usage: python3 guess.py art.jpg NAME out.mp4 [still SEC,SEC]
Env: HEXR (hex radius px), GW GH GY0 (grid area), K, CROP x0,y0,x1,y1, SEC, SEED, SPD, B0 B1 BP (brush), GAP, MUSIC, MSTART
"""
import os,sys,subprocess,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from sklearn.cluster import KMeans
art,NAME,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
STILL=[float(x) for x in sys.argv[5].split(',')] if len(sys.argv)>5 and sys.argv[4]=='still' else None
E=os.environ.get; COLS=int(E('COLS',44)); ROWS=int(E('ROWS',56)); K=int(E('K',10)); SEC=float(E('SEC',52)); SEED=int(E('SEED',3))
W,H,FPS=1080,1920,24; R=float(E('HEXR',6.5)); GW=int(E('GW',960)); GH=int(E('GH',1080)); GX0=(W-GW)//2; GY0=int(E('GY0',560)); CELL=R*2.2
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
NF=int(SEC*FPS); SUB=6; pos=np.array([GW*0.31,GH*0.18]); ang=0.9; spd=float(E('SPD',720))
vel=np.array([np.cos(ang),np.sin(ang)])*spd; bounces=[]; painted=np.zeros(N,bool); order=[]
B0,B1=float(E('B0',1.0)),float(E('B1',2.5)); brush=lambda t: (B0+B1*(t/SEC)**float(E('BP',1.5)))*CELL
cov=[]; ballpath=[]
for f in range(NF):
    for s_ in range(SUB):
        dt=1/FPS/SUB; pos=pos+vel*dt
        for ax,lim in ((0,GW),(1,GH)):
            if pos[ax]<8: pos[ax]=8; vel[ax]=abs(vel[ax]); bounces.append(((f+s_/SUB)/FPS,ax))
            if pos[ax]>lim-8: pos[ax]=lim-8; vel[ax]=-abs(vel[ax]); bounces.append(((f+s_/SUB)/FPS,ax))
        for i in tree.query_ball_point(pos,brush(f/FPS)):
            if not painted[i]: painted[i]=True; order.append((f,i))
    ballpath.append(pos.copy()); cov.append((painted&~isbg).sum()/(~isbg).sum())
left=[i for i in range(N) if not painted[i]]; rng.shuffle(left); extra=int(1.8*FPS)
for k,i in enumerate(left): order.append((NF+int(k*extra/max(1,len(left))),i))
TOT=NF+extra+int(float(E('HOLD',3.2))*FPS)
fo=np.full(N,10**9)
for f,i in order: fo[i]=f
print('cells',N,'coverage at 25/50/75% time:',[round(cov[int(NF*q)-1],2) for q in (.25,.5,.75)],'leftover',len(left),flush=True)
F=lambda s,b=True: ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-%s.ttf'%('Bold' if b else 'Regular'),s)
def text_c(d,xy,t,font,fill):
    w=d.textlength(t,font=font); d.text((xy[0]-w/2,xy[1]),t,font=font,fill=fill)
TITLE=E('TITLE','Guess the Animal Before It\u2019s Revealed!')
emask=(~edge)[...,None]
def frame(f):
    st=np.where((fo<=f)[:,None],grid,UNP)
    big=(st[idmap]*np.where(emask,1.0,0.45)).astype(np.uint8)
    canvas=np.zeros((H,W,3),np.uint8); canvas[:]=BGC; canvas[GY0:GY0+GH,GX0:GX0+GW]=big
    if f<NF:
        bx,by=ballpath[f]; cv2.circle(canvas,(int(GX0+bx),int(GY0+by)),11,(255,255,255),-1,cv2.LINE_AA)
    cv2.rectangle(canvas,(GX0-3,GY0-3),(GX0+GW+2,GY0+GH+2),(150,150,160),2)
    p=Image.fromarray(canvas); d=ImageDraw.Draw(p)
    pct=int(100*((fo<=f)&~isbg).sum()/(~isbg).sum())
    text_c(d,(W//2,200),TITLE,F(50),(255,255,255)); text_c(d,(W//2,285),'comment your guess before the end',F(32,False),(190,190,200))
    done=f>=NF+extra
    text_c(d,(W//2,GY0+GH+34),f'{100 if done else min(pct,99)}%',F(46),(255,255,255))
    if not done: text_c(d,(W//2,GY0+GH+120),'Did you guess it before 50%?',F(34,False),(190,190,200))
    else:
        k=min(1,(f-(NF+extra))/10); sz=int(96+14*np.sin(k*np.pi))
        for dx in (-3,0,3):
            for dy in (-3,0,3): text_c(d,(W//2+dx,GY0+GH+95+dy),NAME.upper()+'!',F(sz),(60,50,20))
        text_c(d,(W//2,GY0+GH+95),NAME.upper()+'!',F(sz),(255,224,70))
    return np.array(p)
if STILL:
    for s in STILL: Image.fromarray(frame(int(s*FPS))).save(OUT.replace('.mp4',f'_{s}.png'))
    sys.exit()
# audio: pentatonic plinks on bounces over a quiet cleared track
SR=44100; dur=TOT/FPS; aud=np.zeros(int(dur*SR)+SR,np.float32)
pent=[0,2,4,7,9,12,14,16]; base=392.0
for i,(t,ax) in enumerate(bounces):
    fr=base*2**(pent[(i*3+ax)%len(pent)]/12); n=int(0.5*SR); tt=np.arange(n)/SR
    s=(np.sin(2*np.pi*fr*tt)+.3*np.sin(2*np.pi*fr*2.01*tt))*np.exp(-tt*7)*0.22; i0=int(t*SR); aud[i0:i0+n]+=s[:max(0,len(aud)-i0)][:n] if i0+n<=len(aud) else s[:len(aud)-i0]
if E('MUSIC'):
    m=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',E('MSTART','20'),'-t',str(dur),'-i',E('MUSIC'),'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32)
    aud[:len(m)]+=m*0.55
aud=np.clip(aud/ max(1,np.abs(aud).max())*0.9,-1,1)
open(OUT+'.raw','wb').write(aud.tobytes())
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-f','f32le','-ar',str(SR),'-ac','1','-i',OUT+'.raw','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-af','loudnorm=I=-14:TP=-1.5','-shortest','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for f in range(TOT): enc.stdin.write(frame(f).tobytes())
enc.stdin.close(); enc.wait(); os.remove(OUT+'.raw'); print('done',OUT)
