# Overlay extra soft rain on the loop. Every layer scrolls a whole number of tile-heights per loop, so the loop stays seamless.
import cv2, numpy as np, subprocess, sys
src,out=sys.argv[1],sys.argv[2]
c=cv2.VideoCapture(src);F=[]
while True:
    ok,f=c.read()
    if not ok: break
    F.append(f)
n=len(F); H,W=F[0].shape[:2]; TS=540
def tile(cnt,ln,seed,th,ang,blur):
    r=np.random.RandomState(seed); t=np.zeros((TS,TS),np.float32)
    for _ in range(cnt):
        x,y=r.randint(0,TS,2); l=r.randint(ln//2,ln)
        for ox in (-TS,0,TS):
            for oy in (-TS,0,TS):
                cv2.line(t,(x+ox,y+oy),(x+ox+int(l*ang),y+oy+l),r.uniform(0.35,1),th,cv2.LINE_AA)
    return cv2.GaussianBlur(t,(0,0),blur)
# far: fine and dense, mid, near: few, long and soft (out of focus)
L=[(tile(520,26,1,1,0.04,0.6),5,0.20),(tile(200,48,2,2,0.04,0.9),8,0.24),(tile(45,110,3,3,0.04,2.2),12,0.20)]
yy=np.linspace(0,1,H,dtype=np.float32)[:,None]
fade=(0.55+0.45*yy)                                    # a touch heavier lower in frame
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-',
    '-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
for i,f in enumerate(F):
    acc=np.zeros((H,W),np.float32)
    for tl,wraps,a in L:
        oy=int(round(i/n*wraps*TS))%TS; ox=int(round(i/n*wraps*TS*0.04))%TS
        big=np.tile(np.roll(np.roll(tl,oy,0),ox,1),(H//TS+1,W//TS+1))[:H,:W]
        acc+=big*a
    img=f.astype(np.float32)
    img=img+(acc*fade)[...,None]*np.array([235,225,215],np.float32)*0.6
    p.stdin.write(np.clip(img,0,255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
