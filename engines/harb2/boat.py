import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np, subprocess, sys
D=ZT+''
X0,X1=265,955            # crop around the boat (600 px wide)
OW,OH=1080,1920
def portrait(f):
    c=f[:,X0:X1].astype(np.float32)                  # 600x720
    s=OW/c.shape[1]; h=int(round(c.shape[0]*s))      # 1080x1296
    body=cv2.resize(c,(OW,h),interpolation=cv2.INTER_LANCZOS4)
    top=OH-h
    S_=body[:330]                                                     # clean sky band above the mast
    parts=[]; k=0
    while sum(p.shape[0] for p in parts)<top:
        parts.append(cv2.flip(S_,0) if k%2==0 else S_); k+=1
    mir=np.vstack(parts[::-1])[-top:]                                  # mirror-tile the sky upward
    mir=cv2.GaussianBlur(mir,(0,0),1.2)
    g=np.linspace(0.80,1.0,top,dtype=np.float32)[:,None,None]          # deeper violet higher up
    hue=np.linspace(1,0,top,dtype=np.float32)[:,None,None]
    mir=mir*g*(1-0.10*hue)+np.array([60,20,10],np.float32)*0.10*hue*0
    out=np.zeros((OH,OW,3),np.float32); out[:top]=mir; out[top:]=body
    return out
if __name__=='__main__':
    c=cv2.VideoCapture(D+'harb2/m1.mp4'); ok,f=c.read()
    cv2.imwrite(D+'harb2/bp.png',np.clip(portrait(f),0,255).astype(np.uint8))
