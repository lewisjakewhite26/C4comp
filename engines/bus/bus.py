import sys, os, math, subprocess
import numpy as np, cv2
O=os.path.dirname(os.path.abspath(__file__))+'/'
W,H,FPS=1080,1920,30
PLATE=sys.argv[2] if len(sys.argv)>2 else '/mnt/user-data/uploads/3087318-uhd_3840_2160_30fps.mp4'
SS=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
DUR=float(sys.argv[4]) if len(sys.argv)>4 else 5.0

# ---------------- foreground: front-seat image, keyed
fr=cv2.imread(O+'front.jpg').astype(np.float32)
fr=cv2.resize(fr,(W,H),interpolation=cv2.INTER_AREA)
B,G,R=fr[...,0],fr[...,1],fr[...,2]
M=np.clip(((np.minimum(R,B)-G)-90)/50,0,1)               # 1 = window glass
M=cv2.GaussianBlur(M,(0,0),0.6)
ys,xs=np.where(M>0.5); WY0,WY1=ys.min()-4,ys.max()+4
# despill the fringe
sp=(M>0.01)&(M<0.99)
fg=fr.copy(); fg[...,0]=np.where(sp,np.minimum(B,G+20),B); fg[...,2]=np.where(sp,np.minimum(R,G+35),R)
# how much each interior pixel picks up light from outside (near the window, and the ceiling)
dist=cv2.distanceTransform((M<0.5).astype(np.uint8),cv2.DIST_L2,5)
SPILL=(np.exp(-dist/180.0)*(1-M)).astype(np.float32)

# ---------------- rain on the glass: static drops + slow trickles
rng=np.random.RandomState(3)
GH,GW=WY1-WY0,W
drops=[]
for _ in range(650):
    r=rng.choice([2.0,3.0,4.5,6.0,8.0,11.0],p=[.28,.25,.2,.14,.08,.05])
    drops.append((rng.uniform(0,GW),rng.uniform(0,GH),r*rng.uniform(.8,1.2),rng.uniform(0.7,1.3),rng.uniform(0,6.3)))
def drop_layer(extra):
    off=np.zeros((GH,GW,2),np.float32); a=np.zeros((GH,GW),np.float32); hl=np.zeros((GH,GW),np.float32); rim=np.zeros((GH,GW),np.float32)
    for dd in drops+extra:
        x,y,r=dd[:3]; asp=dd[3] if len(dd)>3 else 1.25; ph=dd[4] if len(dd)>4 else 0.0
        x0,x1=int(max(0,x-r-2)),int(min(GW,x+r+3)); y0,y1=int(max(0,y-r*1.6-2)),int(min(GH,y+r*1.6+3))
        if x1<=x0 or y1<=y0: continue
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        dx=(xx-x)/r; dy=(yy-y)/(r*asp); ang=np.arctan2(dy,dx)
        d2=(dx*dx+dy*dy)*(1+0.18*np.sin(3*ang+ph)+0.10*np.sin(5*ang+2*ph))
        ins=np.clip((1-d2)*r,0,1)
        # a drop is a little lens: it shows the scene flipped and shrunk
        off[y0:y1,x0:x1,0]+=ins*(-dx*r*2.2); off[y0:y1,x0:x1,1]+=ins*(-dy*r*2.2-r*1.5)
        a[y0:y1,x0:x1]=np.maximum(a[y0:y1,x0:x1],ins)
        hl[y0:y1,x0:x1]+=ins*np.exp(-((dx+0.35)**2+(dy+0.45)**2)/0.03)
        rim[y0:y1,x0:x1]=np.maximum(rim[y0:y1,x0:x1],ins*np.clip((d2-0.55)/0.45,0,1))
    return off,a,hl,rim
STATIC=drop_layer([])
TR=[(rng.uniform(0,GW),rng.uniform(-GH,GH),rng.uniform(2.5,4.5),rng.uniform(25,70)) for _ in range(14)]
gx,gy=np.meshgrid(np.arange(GW,dtype=np.float32),np.arange(GH,dtype=np.float32))
def glass(bg,t):
    extra=[]; streak=np.zeros((GH,GW),np.float32)
    for x0,y0,r,v in TR:
        y=(y0+v*t)%(GH*1.4)-GH*0.2; x=x0+3*math.sin(y*0.02+x0)
        extra.append((x,y,r))
        cv2.line(streak,(int(x),int(max(0,y-v*2.5))),(int(x),int(y)),0.35,max(1,int(r*0.6)),cv2.LINE_AA)
    off,a,hl,rim=STATIC
    if extra:
        o2,a2,h2,r2=drop_layer(extra); off=off+o2; a=np.maximum(a,a2); hl=hl+h2; rim=np.maximum(rim,r2)
    soft=cv2.GaussianBlur(bg,(0,0),1.2)
    ref=cv2.remap(soft,gx+off[...,0],gy+off[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    out=bg*(1-a[...,None])+(ref*1.05+6)*a[...,None]
    out*=(1-0.28*rim)[...,None]
    out+=np.clip(hl,0,1)[...,None]*np.array([235,235,240],np.float32)*0.22
    out=out*(1-streak[...,None]*0.5)+cv2.GaussianBlur(bg,(0,0),3)*streak[...,None]*0.5
    # light mist on the glass
    fog=(0.10+0.18*np.clip((gy/GH-0.55)/0.45,0,1))[...,None]
    return out*(1-fog)+cv2.GaussianBlur(bg,(0,0),10)*fog+8

# ---------------- plate: crop to the window's shape, slight night grade
def plate_frames():
    p=subprocess.Popen(['ffmpeg','-v','error','-ss',str(SS),'-i',PLATE,'-t',str(DUR+0.2),
        '-vf',f'fps={FPS},crop=ih*{W}/{GH}:ih,scale={W}:{GH}:flags=lanczos','-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    n=W*GH*3
    while True:
        b=p.stdout.read(n)
        if len(b)<n: break
        yield np.frombuffer(b,np.uint8).reshape(GH,W,3).astype(np.float32)
GRADE=np.array([1.06,0.98,0.92],np.float32)
def frame(bg,t):
    bg=cv2.GaussianBlur(bg,(0,0),3.0)*GRADE*0.74
    g=glass(bg,t)
    full=np.zeros((H,W,3),np.float32); full[WY0:WY1]=g
    img=fg*(1-M[...,None])+full*M[...,None]
    # interior picks up the street's colour
    avg=cv2.resize(bg,(1,1),interpolation=cv2.INTER_AREA)[0,0]
    img+=SPILL[...,None]*(avg-avg.mean()*0.6)[None,None,:]*0.35
    # bloom from street lights through the glass
    br=np.clip(img-170,0,None); sm=cv2.resize(br,(W//6,H//6),interpolation=cv2.INTER_AREA)
    img+=cv2.resize(cv2.GaussianBlur(sm,(0,0),3),(W,H))*0.6
    img+=np.random.randn(H//2,W//2).astype(np.float32).repeat(2,0).repeat(2,1)[...,None]*2.5
    return np.clip(img,0,255).astype(np.uint8)

if __name__=='__main__' and sys.argv[1]=='still':
    bg=next(plate_frames()); cv2.imwrite(O+'still.jpg',cv2.resize(frame(bg,1.0),(540,960))); cv2.imwrite(O+'still_full.png',frame(bg,1.0)); sys.exit()
if __name__=='__main__':
    out=O+sys.argv[1]
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for i,bg in enumerate(plate_frames()):
        if i>=int(DUR*FPS): break
        f=frame(bg,i/FPS)
        if i==int(DUR*FPS*0.5): cv2.imwrite(O+'mid.png',f)
        enc.stdin.write(f.tobytes())
    enc.stdin.close(); enc.wait()
