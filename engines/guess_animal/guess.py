"""Guess-the-animal Short: a ball bounces round a pixel grid and paints the animal in.
Usage: python3 guess.py art.jpg NAME out.mp4 [still SEC,SEC]
Env: COLS ROWS (grid), K (palette size), CROP x0,y0,x1,y1 (of the art), SEC (length), SEED, MUSIC, BG (art background rgb)
"""
import os,sys,subprocess,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from sklearn.cluster import KMeans
art,NAME,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
STILL=[float(x) for x in sys.argv[5].split(',')] if len(sys.argv)>5 and sys.argv[4]=='still' else None
E=os.environ.get; COLS=int(E('COLS',44)); ROWS=int(E('ROWS',56)); K=int(E('K',10)); SEC=float(E('SEC',52)); SEED=int(E('SEED',3))
W,H,FPS=1080,1920,24; CELL=int(E('CELL',22)); GW,GH=COLS*CELL,ROWS*CELL; GX0=(W-GW)//2; GY0=int(E('GY0',520))
im=Image.open(art).convert('RGB')
if E('CROP'): im=im.crop(tuple(int(v) for v in E('CROP').split(',')))
from PIL import ImageFilter
im=im.filter(ImageFilter.UnsharpMask(radius=3,percent=160,threshold=2))
small=np.array(im.resize((COLS,ROWS),Image.LANCZOS)).reshape(-1,3).astype(np.float32)
km=KMeans(K,n_init=4,random_state=1).fit(small); pal=km.cluster_centers_; lab=km.labels_
# brightest/least-saturated cluster is the paper background: make it a warm cream
bgi=int(np.argmax(pal.sum(1)-pal.std(1)*2)); pal[bgi]=(38,38,44)
isbg=(lab==bgi).reshape(ROWS,COLS)
grid=pal[lab].reshape(ROWS,COLS,3).astype(np.uint8)
UNP=np.array((38,38,44),np.uint8); BGC=(18,18,22)
rng=np.random.default_rng(SEED)
# ball path: billiard in the grid, irrational-ish slope, brush grows over time
NF=int(SEC*FPS); SUB=6; pos=np.array([GW*0.31,GH*0.18]); ang=0.9; spd=float(E('SPD',1500))  # px/s
vel=np.array([np.cos(ang),np.sin(ang)])*spd; bounces=[]; painted=np.zeros((ROWS,COLS),bool); order=[]
B0,B1=float(E('B0',1.0)),float(E('B1',1.8)); brush=lambda t: B0+B1*(t/SEC)**float(E('BP',1.5))
cov=[]; trail=[]; ballpath=[]
for f in range(NF):
    for s in range(SUB):
        dt=1/FPS/SUB; pos=pos+vel*dt
        for ax,lim in ((0,GW),(1,GH)):
            if pos[ax]<8: pos[ax]=8; vel[ax]=abs(vel[ax]); bounces.append(((f+s/SUB)/FPS,ax))
            if pos[ax]>lim-8: pos[ax]=lim-8; vel[ax]=-abs(vel[ax]); bounces.append(((f+s/SUB)/FPS,ax))
        r=brush(f/FPS); cx,cy=pos[0]/CELL,pos[1]/CELL
        y0,y1,x0,x1=max(0,int(cy-r)),min(ROWS,int(cy+r)+1),max(0,int(cx-r)),min(COLS,int(cx+r)+1)
        for yy in range(y0,y1):
            for xx in range(x0,x1):
                if (xx+.5-cx)**2+(yy+.5-cy)**2<=r*r and not painted[yy,xx]: painted[yy,xx]=True; order.append((f,yy,xx))
    ballpath.append(pos.copy()); cov.append((painted&~isbg).sum()/(~isbg).sum())
# sweep any leftovers in the last 2 s so it always finishes at 100%
left=[(yy,xx) for yy in range(ROWS) for xx in range(COLS) if not painted[yy,xx]]
rng.shuffle(left); n0=NF; extra=int(1.8*FPS)
for i,(yy,xx) in enumerate(left): order.append((n0+int(i*extra/max(1,len(left))),yy,xx))
TOT=NF+extra+int(float(E('HOLD',3.2))*FPS)
fo=np.full((ROWS,COLS),10**9); 
for f,yy,xx in order: fo[yy,xx]=f
print('coverage at 25/50/75% time:',[round(cov[int(NF*q)-1],2) for q in (.25,.5,.75)],'leftover',len(left),flush=True)
F=lambda s,b=True: ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-%s.ttf'%('Bold' if b else 'Regular'),s)
def text_c(d,xy,t,font,fill):
    w=d.textlength(t,font=font); d.text((xy[0]-w/2,xy[1]),t,font=font,fill=fill)
TITLE=E('TITLE','Guess the Animal Before It’s Revealed!')
def frame(f):
    gridimg=np.where((fo<=f)[...,None],grid,UNP)
    canvas=np.zeros((H,W,3),np.uint8); canvas[:]=BGC
    big=np.kron(gridimg,np.ones((CELL,CELL,1),np.uint8))
    # soft cell gaps
    big[::CELL,:]=big[::CELL,:]//2; big[:,::CELL]=big[:,::CELL]//2
    canvas[GY0:GY0+GH,GX0:GX0+GW]=big
    if f<NF:
        bx,by=ballpath[f]; cv2.circle(canvas,(int(GX0+bx),int(GY0+by)),11,(255,255,255),-1,cv2.LINE_AA)
    cv2.rectangle(canvas,(GX0-3,GY0-3),(GX0+GW+2,GY0+GH+2),(150,150,160),2)
    p=Image.fromarray(canvas); d=ImageDraw.Draw(p)
    pct=int(100*((fo<=f)&~isbg).sum()/(~isbg).sum()); 
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
    Image.fromarray(grid).resize((COLS*8,ROWS*8),Image.NEAREST).save(OUT.replace('.mp4','_grid.png')); sys.exit()
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
