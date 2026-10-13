import sys, math, subprocess, colorsys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
W,H=1920,1080; CELL=6; GW,GH=W//CELL,H//CELL; FPS=30; COUNT=60.0; TOT=66.0; NF=int(TOT*FPS)
GHm=np.load('GH.npy',mmap_mode='r'); GJm=np.load('GJ.npy',mmap_mode='r')
pcnt=np.load('pcnt.npy'); poff=np.concatenate([[0],np.cumsum(pcnt)]); pall=np.load('pall.npy',mmap_mode='r'); dall=np.load('dall.npy',mmap_mode='r')
# palette (BGR float)
pal=np.zeros((256,3),np.float32)
for i in range(1,251):
    r,g,b=colorsys.hsv_to_rgb(i/250.0,0.62,1.0); pal[i]=(b*255,g*255,r*255)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
g=np.linspace(0,1,H)[:,None,None]
bg=(np.array([34,14,12],np.float32)*(1-g)+np.array([60,24,26],np.float32)*g)*np.ones((H,W,3),np.float32)   # deep indigo (BGR)
# bead tile
tile=np.zeros((CELL,CELL),np.float32)
c=(CELL-1)/2.0
for j in range(CELL):
    for i in range(CELL):
        d=math.hypot(i-c,j-c)/(CELL/2.0); a=np.clip((1.0-d)*3.0,0,1)
        hl=max(0,1-math.hypot(i-c+1.1,j-c+1.1)/2.6)
        tile[j,i]=a*(0.78+0.5*hl)
alpha_t=np.clip(np.where(tile>0,1,0),0,1).astype(np.float32)
TILE=np.tile(tile,(GH,GW)); TILEA=np.tile(np.clip(tile*1.6,0,1),(GH,GW))
VIG=1-0.32*np.clip((np.hypot((xx-W/2)/(W/2),(yy-H/2)/(H/2))-0.55)/0.8,0,1)**1.6
rng=np.random.default_rng(5); GRAIN=[rng.normal(0,1.0,(H,W,1)).astype(np.float32) for _ in range(5)]
low=(68,120)
lyy,lxx=np.mgrid[0:low[0],0:low[1]].astype(np.float32)
# --- timer overlay (same style as other countdowns) ---
FONT=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',230); FONT_S=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',150)
def sstep(a,b,x): x=min(1,max(0,(x-a)/(b-a))); return x*x*(3-2*x)
_c={}
def layer(txt):
    if txt in _c: return _c[txt]
    im=Image.new('L',(1300,360),0); d=ImageDraw.Draw(im)
    if ':' in txt:
        cell=130;cw=70;tot=cell*4+cw;x0=(1300-tot)/2
        pos=[x0,x0+cell,x0+2*cell+cw,x0+3*cell+cw]
        for ch,px in zip(txt.replace(':',''),pos): d.text((px+cell/2,185),ch,font=FONT,fill=255,anchor='mm')
        d.text((x0+2*cell+cw/2,175),':',font=FONT,fill=255,anchor='mm')
    else: d.text((650,185),txt,font=FONT_S,fill=255,anchor='mm')
    a=np.array(im,np.float32)/255.0
    y_,x_=np.mgrid[0:360,0:1300].astype(np.float32)
    back=0.5*np.exp(-(((x_-650)/470)**2+((y_-185)/120)**2))
    sh=np.clip(np.maximum(np.clip(cv2.GaussianBlur(a,(0,0),26)*1.1,0,0.8),back),0,0.88)
    _c[txt]=(a,sh); return _c[txt]
def overlay(big,t):
    rem=int(math.ceil(COUNT-t-1e-9))
    if t>=COUNT: txt='Time’s up'; k=min(1,(t-COUNT)/1.5)
    else: txt='%02d:%02d'%(rem//60,rem%60); k=1
    a,sh=layer(txt)
    warm=sstep(0,1,(10-(COUNT-t))/3.0) if t<COUNT else 1.0
    col=np.array([255,255,255],np.float32)*(1-warm)+np.array([120,196,255],np.float32)*warm
    y0,x0=40,W//2-650
    roi=big[y0:y0+360,x0:x0+1300]; roi*=(1-sh)[...,None]; roi+=(a*0.95*k)[...,None]*col[None,None,:]
    p=min(1,t/COUNT); bx0,bx1,by=W//2-420,W//2+420,1010
    seg=big[by-10:by+10]
    b=np.zeros((20,W,3),np.float32); cv2.line(b,(bx0,10),(bx1,10),(255,255,255),5,cv2.LINE_AA); seg*=1-0.30*(b[...,:1]/255.0); seg+=b*0.18
    f=np.zeros((20,W,3),np.float32); cv2.line(f,(bx0,10),(int(bx0+(bx1-bx0)*p),10),(255,255,255),5,cv2.LINE_AA); seg*=1-0.45*(f[...,:1]/255.0); seg+=f*0.62
    if t>=COUNT:
        kk=(t-COUNT)/3.0; w_=0.14*math.sin(math.pi*min(1,kk)) if kk<1 else 0
        big*=(1-w_); big+=np.array([190,230,255],np.float32)*w_
    if t<1.2: big*=sstep(0,1.2,t)
    if t>COUNT+4: big*=1-sstep(COUNT+4,COUNT+6,t)
    return big
def render(f):
    t=f/FPS
    # animated background glow field (low-res)
    fld=0.5+0.5*np.sin(lxx/17.0+t*0.35)*np.sin(lyy/11.0-t*0.27)+0.35*np.sin((lxx+lyy)/23.0+t*0.2)
    fld=np.clip(fld,0,1.4)[...,None]
    tint=np.array([90,40,30],np.float32)*np.clip(0.5+0.5*np.sin(t*0.3+lxx[...,None]/40.0),0,1)+np.array([40,70,20],np.float32)*(1-np.clip(0.5+0.5*np.sin(t*0.3+lxx[...,None]/40.0),0,1))
    glowbg=cv2.resize((fld*tint*0.5).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)
    out=bg+glowbg
    # airborne beads
    lay=np.zeros((H,W,3),np.float32)
    P=pall[poff[f]:poff[f+1]]
    for x,y,vx,vy,h in P:
        col=pal[int(h)]
        sp=math.hypot(vx,vy)
        if sp>40: cv2.line(lay,(int(x-vx*0.045),int(y-vy*0.045)),(int(x),int(y)),(float(col[0])*0.35,float(col[1])*0.35,float(col[2])*0.35),2,cv2.LINE_AA)
        cv2.circle(lay,(int(x),int(y)),3,(float(col[0]),float(col[1]),float(col[2])),-1,cv2.LINE_AA)
    Dd=dall[f]
    for x,y,vx,vy,h,r in Dd:
        col=pal[int(h)]*0.9; tw=0.6+0.4*math.sin(t*3+x*0.05)
        cv2.circle(lay,(int(x),int(y)),max(1,int(round(r))),(float(col[0]*tw),float(col[1]*tw),float(col[2]*tw)),-1,cv2.LINE_AA)
    sm=cv2.resize(lay,(W//4,H//4),interpolation=cv2.INTER_AREA); sm=cv2.GaussianBlur(sm,(0,0),5)
    out+=cv2.resize(sm,(W,H),interpolation=cv2.INTER_CUBIC)*1.4+lay
    # settled pile
    hg=np.asarray(GHm[f]); jg=np.asarray(GJm[f])
    occ=hg>0
    if occ.any():
        depth=(0.62+0.38*(1-np.arange(GH,dtype=np.float32)/GH))[:,None,None]
        colg=pal[hg]*(0.9+0.2*(jg.astype(np.float32)/255.0))[...,None]*depth
        rim=np.zeros_like(occ); rim[1:]=occ[1:]&~occ[:-1]; rim[0]=occ[0]
        rim2=np.zeros_like(occ); rim2[2:]=occ[2:]&~occ[:-2]
        colg=colg*np.where(rim,1.7,np.where(rim2,1.3,1.0))[...,None]
        # soft bloom from pile
        bl=cv2.GaussianBlur(cv2.resize(colg*occ[...,None],(GW//2,GH//2),interpolation=cv2.INTER_AREA),(0,0),3)
        out+=cv2.resize(bl,(W,H),interpolation=cv2.INTER_CUBIC)*0.30
        big=cv2.resize(colg,(W,H),interpolation=cv2.INTER_NEAREST)
        A=cv2.resize(occ.astype(np.float32),(W,H),interpolation=cv2.INTER_NEAREST)
        bead=big*TILE[...,None]
        dark=big*0.10
        pile=bead*TILEA[...,None]+dark*(1-TILEA[...,None])
        out=out*(1-A[...,None])+pile*A[...,None]
    out*=VIG[...,None]
    out=overlay(out,t)
    out+=GRAIN[f%5]
    return np.clip(out,0,255).astype(np.uint8)
def worker(f0,f1,name):
    pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(FPS),'-i','-','-c:v','libx264','-preset','veryfast','-crf','16','-pix_fmt','yuv420p','-g','60','-x264-params','threads=1',name],stdin=subprocess.PIPE)
    for i in range(f0,f1): pr.stdin.write(render(i).tobytes())
    pr.stdin.close(); pr.wait(); print('done',name,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='test':
        import time
        for s in sys.argv[2:]:
            t0=time.time(); cv2.imwrite('t_%s.png'%s,render(int(float(s)*FPS))); print(s,round(time.time()-t0,2),flush=True)
    else: worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
