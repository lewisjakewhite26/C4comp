import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math
sys.path.insert(0,ZT+'jf')
sys.path.insert(0,ZT+'lava')
import numpy as np, cv2, lava, tim
W,H=1920,1080; CX,CY=W//2,H//2; R=330
TXT=np.array([250,240,255],np.float32)
def digits_only(img,t,total,glowcol):
    left=max(0,math.ceil(total-t-1e-6))
    a,mul,_=tim.digits(left)
    roi=img[tim.RY0:tim.RY0+tim.RH,tim.RX0:tim.RX0+tim.RW]
    glow=cv2.GaussianBlur(a[...,0],(0,0),12)[...,None]*np.array(glowcol,np.float32)
    roi=roi*mul+glow*0.45; roi=roi*(1-a)+a*TXT
    img[tim.RY0:tim.RY0+tim.RH,tim.RX0:tim.RX0+tim.RW]=roi; return img
def style1(img,t,total):           # thin soft ring, recoloured
    frac=max(0.0,1-t/total); ring=np.zeros((H,W),np.float32); tr=np.zeros((H,W),np.float32)
    cv2.ellipse(ring,(CX*4,CY*4),(R*4,R*4),-90,0,360*frac,1.0,16,cv2.LINE_AA,2); cv2.circle(tr,(CX*4,CY*4),R*4,0.12,16,cv2.LINE_AA,2)
    img=img+np.maximum(ring,tr)[...,None]*np.array([240,200,255],np.float32)*0.75+cv2.GaussianBlur(ring,(0,0),10)[...,None]*np.array([220,90,255],np.float32)*0.5
    return digits_only(img,t,total,(200,60,230))
def style2(img,t,total):           # glossy wax tube that drains
    frac=max(0.0,1-t/total)
    tube=np.zeros((H,W),np.float32); full=np.zeros((H,W),np.float32)
    cv2.ellipse(full,(CX*4,CY*4),(R*4,R*4),-90,0,360,1.0,26*4,cv2.LINE_AA,2)
    cv2.ellipse(tube,(CX*4,CY*4),(R*4,R*4),-90,0,360*frac,1.0,26*4,cv2.LINE_AA,2)
    tube=cv2.GaussianBlur(tube,(0,0),1.2)
    Y,X=np.mgrid[0:H,0:W].astype(np.float32); d=np.sqrt((X-CX)**2+(Y-CY)**2)-R
    across=np.clip(1-np.abs(d)/26,0,1)                              # 1 at tube centre
    ang=(np.arctan2(X-CX,-(Y-CY))%(2*np.pi))/(2*np.pi)
    col=np.array([170,70,255],np.float32)*(1-ang[...,None])+np.array([255,80,175],np.float32)*ang[...,None]
    shade_=(0.55+0.45*across)[...,None]
    hl=np.exp(-((d+8)/5)**2)*0.9                                    # inner highlight line
    glass=cv2.GaussianBlur(full,(0,0),1.2)*0.10
    img=img*(1-tube[...,None]*0.95)+tube[...,None]*(col*shade_+hl[...,None]*np.array([255,235,255],np.float32))
    img=img+glass[...,None]*np.array([255,220,255],np.float32)+cv2.GaussianBlur(tube,(0,0),14)[...,None]*np.array([200,60,230],np.float32)*0.35
    return digits_only(img,t,total,(200,60,230))
def style3(img,t,total):           # 60 glowing beads that go out one by one
    frac=max(0.0,1-t/total); lay=np.zeros((H,W),np.float32); n=60
    for i in range(n):
        a=2*np.pi*i/n; x=CX+R*math.sin(a); y=CY-R*math.cos(a)
        on=1.0 if i/n<frac else 0.18
        cv2.circle(lay,(int(x*4),int(y*4)),int(9*4),on,-1,cv2.LINE_AA,2)
    img=img+lay[...,None]*np.array([245,205,255],np.float32)*0.8+cv2.GaussianBlur(lay,(0,0),9)[...,None]*np.array([220,80,255],np.float32)*0.6
    return digits_only(img,t,total,(200,60,230))
base=lava.shade(20.0)
outs=[]
for k,fn in enumerate((style1,style2,style3)):
    im=fn(base.copy(),150,180); im=np.clip(im,0,255).astype(np.uint8)
    cv2.putText(im,str(k+1),(40,90),cv2.FONT_HERSHEY_SIMPLEX,2.5,(255,255,255),5,cv2.LINE_AA)
    outs.append(im)
cv2.imwrite('rings.png',np.vstack([np.hstack([cv2.resize(outs[0],(960,540)),cv2.resize(outs[1],(960,540))]),np.hstack([cv2.resize(outs[2],(960,540)),np.zeros((540,960,3),np.uint8)])]))

Y,X=np.mgrid[0:H,0:W].astype(np.float32)
DD=np.sqrt((X-CX)**2+(Y-CY)**2); ANG=(np.arctan2(X-CX,-(Y-CY))%(2*np.pi))
YN=(Y/H)[...,None]
def style4(img,t,total):            # splodgy wax tube
    frac=max(0.0,1-t/total); A=ANG
    # organic thickness + slight radial wander, both periodic around the ring and slowly drifting
    wv=(0.30*np.sin(3*A+0.40*t)+0.22*np.sin(7*A-0.65*t+1.3)+0.14*np.sin(13*A+0.9*t+2.1)+0.08*np.sin(23*A-1.3*t))
    w=40*(1+0.5*np.clip(wv,-0.55,1.2))
    # splodges: a few fat bulges that wobble in place
    for ak,ph in ((0.6,0.0),(1.9,1.7),(3.1,3.3),(4.4,0.9),(5.5,2.5)):
        da=np.angle(np.exp(1j*(A-ak)))
        w=w+22*(0.75+0.25*math.sin(0.8*t+ph))*np.exp(-(da/0.11)**2)
    off=5*np.sin(5*A+0.3*t+0.7)+3*np.sin(11*A-0.5*t)
    d=DD-R-off
    across=np.clip(1-np.abs(d)/w,0,1)
    m=np.clip(across*w/2.0,0,1)                                    # anti-aliased edge
    end=2*np.pi*frac
    ang_ok=np.clip((end-A)*R/3.0,0,1)*np.clip((A-0.10)*R/3.0,0,1)    # between a rounded start and the current time
    m*=ang_ok
    # rounded drip at the leading end
    ex,ey=CX+R*math.sin(end),CY-R*math.cos(end)
    eb=np.sqrt((X-ex)**2+(Y-ey)**2)
    rb=50+6*math.sin(t*1.7)
    capm=np.clip((rb-eb)/1.5,0,1) if frac>0.002 else 0*eb
    cap_across=np.clip(1-eb/rb,0,1)
    sx0,sy0=CX+R*math.sin(0.10),CY-R*math.cos(0.10)
    sb=np.sqrt((X-sx0)**2+(Y-sy0)**2); rs_=44
    sm=np.clip((rs_-sb)/1.5,0,1) if frac>0.02 else 0*sb
    capm=np.maximum(capm,sm); cap_across=np.maximum(cap_across*(capm>0),np.clip(1-sb/rs_,0,1)*sm)
    m=np.maximum(m,capm); acr=np.maximum(across*ang_ok,cap_across*capm)
    dome=np.sqrt(acr)
    col=np.array([255,70,170],np.float32)*(1-YN)+np.array([180,60,255],np.float32)*YN   # same palette as the wax
    gy_,gx_=np.gradient(cv2.GaussianBlur(dome,(0,0),2))
    nx,ny=-gx_*30,-gy_*30; nz=np.ones_like(nx); nl=np.sqrt(nx*nx+ny*ny+1); nx/=nl; ny/=nl; nz/=nl
    L=np.array([-0.45,-0.55,0.70]); L/=np.linalg.norm(L); Hh=np.array([L[0],L[1],L[2]+1]); Hh/=np.linalg.norm(Hh)
    diff=np.clip(nx*L[0]+ny*L[1]+nz*L[2],0,1); spec=np.clip(nx*Hh[0]+ny*Hh[1]+nz*Hh[2],0,1)**35
    wax=col*(0.45+0.5*diff[...,None]+0.25*dome[...,None])+spec[...,None]*np.array([255,240,255],np.float32)*0.9
    # faint empty track
    tr=np.clip(1-np.abs(DD-R)/22,0,1)*0.08
    img=img+tr[...,None]*np.array([255,210,255],np.float32)
    img=img*(1-m[...,None])+wax*m[...,None]
    img=img+cv2.GaussianBlur(m,(0,0),16)[...,None]*np.array([200,60,230],np.float32)*0.35*(1-m[...,None])
    return digits_only(img,t,total,(200,60,230))
ims=[np.clip(style4(lava.shade(20.0+k*4).copy(),tt,180),0,255).astype(np.uint8) for k,tt in enumerate((40,150))]
cv2.imwrite('ring4.png',np.hstack([cv2.resize(i,(960,540)) for i in ims])); cv2.imwrite('ring4_full.jpg',ims[0],[cv2.IMWRITE_JPEG_QUALITY,92])
