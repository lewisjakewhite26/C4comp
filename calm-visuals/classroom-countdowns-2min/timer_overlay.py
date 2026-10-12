import sys, math, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
W,H=1920,1080; FPS=24; COUNT=120.0; TOT=126.0; NF=int(TOT*FPS)
src,out=sys.argv[1],sys.argv[2]
FONT=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',230)
FONT_S=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',150)
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
    a=np.array(im,np.float32)/255.0; sh=np.clip(cv2.GaussianBlur(a,(0,0),26)*1.1,0,0.8)
    yy,xx=np.mgrid[0:360,0:1300].astype(np.float32); back=0.42*np.exp(-(((xx-650)/470)**2+((yy-185)/120)**2)); sh=np.clip(np.maximum(sh,back),0,0.85)
    _c[txt]=(a,sh); return _c[txt]
rng=np.random.default_rng(1)
def overlay(big,t):
    rem=int(math.ceil(COUNT-t-1e-9))
    if t>=COUNT: txt='Time’s up'; k=min(1,(t-COUNT)/1.5)
    else: txt='%02d:%02d'%(rem//60,rem%60); k=1
    a,sh=layer(txt)
    warm=sstep(0,1,(12-(COUNT-t))/3.0) if t<COUNT else 1.0
    col=np.array([255,255,255],np.float32)*(1-warm)+np.array([120,196,255],np.float32)*warm   # BGR amber
    y0,x0=40,W//2-650
    roi=big[y0:y0+360,x0:x0+1300]
    roi*=(1-sh)[...,None]; roi+=(a*0.92*k)[...,None]*col[None,None,:]
    p=min(1,t/COUNT); bx0,bx1,by=W//2-420,W//2+420,1010
    ov=np.zeros((20,W,3),np.float32); y=by-10
    seg=big[y:y+20]
    base=np.zeros((20,W,3),np.float32); cv2.line(base,(bx0,10),(bx1,10),(255,255,255),5,cv2.LINE_AA); seg+=base*0.14
    fil=np.zeros((20,W,3),np.float32); cv2.line(fil,(bx0,10),(int(bx0+(bx1-bx0)*p),10),(255,255,255),5,cv2.LINE_AA); seg+=fil*0.55
    if t>=COUNT:
        kk=(t-COUNT)/3.0; w_=0.16*math.sin(math.pi*min(1,kk)) if kk<1 else 0
        big*=(1-w_); big+=np.array([190,230,255],np.float32)*w_
    if t<1.5: big*=sstep(0,1.5,t)
    if t>COUNT+4: big*=1-sstep(COUNT+4,COUNT+6,t)
    return big
dec=subprocess.Popen(['ffmpeg','-v','error','-i',src,'-frames:v',str(NF),'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-g','48',out],stdin=subprocess.PIPE)
fs=W*H*3
for i in range(NF):
    b=dec.stdout.read(fs)
    if len(b)<fs: break
    big=np.frombuffer(b,np.uint8).reshape(H,W,3).astype(np.float32)
    big=overlay(big,i/FPS)
    enc.stdin.write(np.clip(big,0,255).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait(); print('done',out)
