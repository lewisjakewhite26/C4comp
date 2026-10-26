import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess, sys
D=ZT+'jf/'
W,H=1280,720
N=int(sys.argv[1]) if len(sys.argv)>1 else 96      # morph length in frames
# stream the chained sequence: keep only the first and last N frames
import collections
counts=[int(cv2.VideoCapture(D+f'c{i}.mp4').get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(1,9)]
n=sum(counts); print('frames',n)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-','-c:v','libx264','-crf','12','-preset','fast','-pix_fmt','yuv420p',D+'loopM.mp4'],stdin=subprocess.PIPE)
first=[]; last=collections.deque(maxlen=N); idx=0
for i in range(1,9):
    c=cv2.VideoCapture(D+f'c{i}.mp4')
    while True:
        ok,f=c.read()
        if not ok: break
        if idx<N: first.append(f)
        elif idx<n-N: p.stdin.write(f.tobytes())
        else: last.append(f)
        idx+=1
print('read',idx)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
def gray(f): return cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
def warp(img,fx,fy,s):
    return cv2.remap(img,gx-s*fx,gy-s*fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
morph=[]
for k in range(N):
    E=last[k]; B=first[k]
    u=(k+0.5)/N; a=u*u*(3-2*u)
    ge,gb=gray(E),gray(B)
    Feb=dis.calc(ge,gb,None)          # E -> B
    Fbe=dis.calc(gb,ge,None)          # B -> E
    Feb=cv2.GaussianBlur(Feb,(0,0),6); Fbe=cv2.GaussianBlur(Fbe,(0,0),6)
    Ew=warp(E.astype(np.float32),Feb[...,0],Feb[...,1],a)
    Bw=warp(B.astype(np.float32),Fbe[...,0],Fbe[...,1],1-a)
    dip=1-0.62*np.sin(np.pi*u)**2
    # dim the bright subjects more than the dark water, like drifting out of the light
    out=(Ew*(1-a)+Bw*a)
    lum=out.mean(2,keepdims=True)/255
    out=out*(dip+(1-dip)*np.clip(1-lum*3,0,1))
    p.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
