import numpy as np,cv2,subprocess,math
W,H,FPS,DUR=1080,1920,24,44
NF=FPS*DUR
src=subprocess.Popen(['ffmpeg','-v','error','-i','preview_pb.mp4','-vf',"scale=3413:1920:flags=lanczos,crop=w=1080:h=1920:x=(iw-1080)*(0.30+0.28*n/1056):y=0",'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
dst=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','20','-preset','medium','-pix_fmt','yuv420p','tiktok_boats.mp4'],stdin=subprocess.PIPE)
rng=np.random.RandomState(5)
def sprite(r):
    n=int(r*2+6); y,x=np.mgrid[:n,:n]-(n-1)/2; d=np.hypot(x,y)/r
    s=np.clip((1.0-d)/0.25,0,1)*0.55+np.exp(-((d-0.92)/0.08)**2)*0.45   # soft disc with brighter rim
    return cv2.GaussianBlur(s.astype(np.float32),(0,0),max(0.8,r*0.12))
N=110; B=[]
for i in range(N):
    z=rng.rand()                        # depth 0 far .. 1 near
    r=5+z**1.6*38
    B.append(dict(sp=sprite(r),r=r,x=rng.rand()*W,y=rng.rand()*H,vx=rng.uniform(-6,10)*(0.4+z),vy=-rng.uniform(5,22)*(0.4+z),
        a=0.25+0.6*z*rng.rand()+0.15,k=rng.randint(4,14),ph=rng.rand()*6.28,col=np.array([rng.uniform(70,110),rng.uniform(170,205),255],np.float32)/255))
fs=W*H*3; f=0
while f<NF:
    buf=src.stdout.read(fs)
    if len(buf)<fs: break
    img=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)
    t=f/FPS; lay=np.zeros((H,W,3),np.float32)
    for b in B:
        x=(b['x']+b['vx']*t+14*math.sin(0.3*t+b['ph']))%(W+120)-60
        y=(b['y']+b['vy']*t)%(H+120)-60
        al=b['a']*(0.65+0.35*math.sin(b['k']*0.18*t+b['ph']))
        sp=b['sp']; n=sp.shape[0]; x0=int(x-n/2); y0=int(y-n/2)
        xa,ya=max(0,x0),max(0,y0); xb,yb=min(W,x0+n),min(H,y0+n)
        if xb<=xa or yb<=ya: continue
        s=sp[ya-y0:yb-y0,xa-x0:xb-x0,None]
        lay[ya:yb,xa:xb]+=s*al*b['col']*255*0.9
    out=255-(255-img)*(255-np.clip(lay,0,255))/255   # screen blend
    dst.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes()); f+=1
dst.stdin.close(); dst.wait(); print(f)
