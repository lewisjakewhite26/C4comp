# Seamless loop: blend the last N frames into the first N with optical-flow morphing, then upscale to 1080p.
import cv2, numpy as np, subprocess, sys
src,out,N=sys.argv[1],sys.argv[2],int(sys.argv[3])
c=cv2.VideoCapture(src);F=[]
while True:
    ok,f=c.read()
    if not ok: break
    F.append(f)
n=len(F); H,W=F[0].shape[:2]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
g=lambda f: cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
def warp(img,fx,fy,s): return cv2.remap(img,gx-s*fx,gy-s*fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
seq=[f.astype(np.float32) for f in F[N:n-N]]
for k in range(N):
    E=F[n-N+k]; B=F[k]; u=(k+0.5)/N; a=u*u*(3-2*u)
    Feb=cv2.GaussianBlur(dis.calc(g(E),g(B),None),(0,0),4); Fbe=cv2.GaussianBlur(dis.calc(g(B),g(E),None),(0,0),4)
    seq.append(warp(E.astype(np.float32),Feb[...,0],Feb[...,1],a)*(1-a)+warp(B.astype(np.float32),Fbe[...,0],Fbe[...,1],1-a)*a)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-',
    '-vf','scale=1920:1080:flags=lanczos,unsharp=5:5:0.4','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
for f in seq: p.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print('loop frames',len(seq),'seconds',len(seq)/24)
