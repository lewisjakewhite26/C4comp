import numpy as np, cv2, sys, subprocess, colorsys, math
from PIL import Image,ImageDraw,ImageFont,ImageFilter
cv2.setNumThreads(1)
FPS=30; TS=3.0; W,H=1920,1080; R0=0.0100; N=520; AR=16/9; CEN=np.array([AR/2,0.44])
POS=None; HUE=np.load('hue.npy')
def col(h): return np.array(colorsys.hsv_to_rgb(float(h),0.80,1.0))[::-1]*255
C=np.array([col(h) for h in HUE],np.float32)           # BGR 0..255
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
vg=1-0.42*(((xx/W-0.5)*2)**2*0.7+((yy/H-0.5)*2)**2*0.55)
BG=(np.stack([16+10*(1-yy/H),6+3*(1-yy/H),6+3*(1-yy/H)],2)*vg[...,None]).astype(np.float32)
SF=0.915
def simframe(f):
    t=f/FPS
    if t<TS: return f
    fs=int(TS*FPS+(f-TS*FPS)*SF)
    if t>TS+300: fs=int(TS*FPS+300*FPS*SF+(f-(TS+300)*FPS))
    return min(fs,9359)
def ease(x): x=min(max(x,0),1); return x*x*(3-2*x)
def frame(pos,t):
    P=pos.astype(np.float32).copy()
    te=TS+300
    if t>te:
        a=0.05*(t-te)*ease((t-te)/3.0)
        d=P-CEN; ca,sa=math.cos(a),math.sin(a); P=CEN+np.stack([d[:,0]*ca-d[:,1]*sa,d[:,0]*sa+d[:,1]*ca],1)
    pulse=1.0+(0.18*math.sin(2*math.pi*(t-te)/4.0)*ease((t-te)/2.0) if t>te else 0)
    img=BG.copy(); g=np.zeros((H//4,W//4,3),np.uint8)
    px=P*H; r=R0*H
    for i in range(N):
        x,y=int(px[i,0]),int(px[i,1]); c=C[i]
        cv2.circle(g,(x//4,y//4),int(r*2.3/4),tuple(float(v*0.55*pulse) for v in c),-1,cv2.LINE_AA)
    for i in range(N):
        x,y=px[i]; c=C[i]; ri=r
        cv2.circle(img,(int(x),int(y)),int(round(ri)),tuple(float(v*0.50) for v in c),-1,cv2.LINE_AA)
        cv2.circle(img,(int(x-ri*0.10),int(y-ri*0.14)),int(round(ri*0.82)),tuple(float(v*0.92) for v in c),-1,cv2.LINE_AA)
        cv2.circle(img,(int(x-ri*0.28),int(y-ri*0.32)),max(1,int(round(ri*0.38))),tuple(float(min(255,v*0.55+120)) for v in c),-1,cv2.LINE_AA)
    g=cv2.GaussianBlur(g,(0,0),5); gu=cv2.resize(g,(W,H),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out=np.clip(img+gu*0.95,0,255)
    return out.astype(np.uint8)
FD='/usr/share/fonts/truetype/google-fonts/'
FT=ImageFont.truetype(FD+'Poppins-Bold.ttf',150); FS=ImageFont.truetype(FD+'Poppins-Medium.ttf',54); FN=ImageFont.truetype(FD+'Poppins-Bold.ttf',120)
def overlay(img,t):
    im=Image.fromarray(img[...,::-1]).convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
    if t<TS+0.4:
        a=min(1,t/0.4)*(1 if t<TS-0.4 else max(0,1-(t-(TS-0.4))/0.8))
        if a>0:
            bd=Image.new('RGBA',im.size,(0,0,0,0)); ImageDraw.Draw(bd).rounded_rectangle((300,360,1620,680),60,fill=(0,0,8,int(200*a))); im=Image.alpha_composite(im,bd.filter(ImageFilter.GaussianBlur(45)))
            for s,f,y,c in (("5 Minute Timer",FT,470,(255,255,255)),("calm colour sorting",FS,590,(255,200,235))):
                d.text((963,y+5),s,font=f,fill=(0,0,0,int(170*a)),anchor='mm'); d.text((960,y),s,font=f,fill=c+(int(255*a),),anchor='mm')
    rem=max(0.0,300-(t-TS)) if t>=TS else 300.0
    sec=int(math.ceil(rem-1e-9)); txt=f"{sec//60}:{sec%60:02d}"
    ta=min(1,max(0,(t-TS+0.6)/0.6)) if t<TS else 1.0
    al=int(235*ta)
    if al>0:
        d.text((963,988),txt,font=FN,fill=(0,0,0,int(150*ta)),anchor='mm'); d.text((960,985),txt,font=FN,fill=(255,255,255,al),anchor='mm')
    prog=min(1,max(0,(t-TS)/300.0))
    d.rectangle((160,1058,1760,1062),fill=(255,255,255,int(35*ta)))
    d.rectangle((160,1058,160+int(1600*prog),1062),fill=(255,150,235,int(150*ta)))
    return np.asarray(Image.alpha_composite(im,ov).convert('RGB'))[...,::-1]
if __name__=='__main__':
    POS=np.load(sys.argv[1] if sys.argv[1].endswith('.npy') else 'pos.npy',mmap_mode='r')
    if sys.argv[1]=='test':
        POS=np.load('pos.npy',mmap_mode='r'); ts=[100,150,200,230,250,270,285,295,299,302,303,308]
        fr=[cv2.resize(overlay(frame(np.array(POS[simframe(int(t*FPS))]),t),t),(640,360)) for t in ts]
        cv2.imwrite('sheet.png',np.vstack([np.hstack(fr[i:i+4]) for i in range(0,12,4)]))
    else:
        a,b,OUT=int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
        p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',OUT],stdin=subprocess.PIPE)
        for f in range(a,b):
            p.stdin.write(np.ascontiguousarray(overlay(frame(np.array(POS[simframe(f)]),f/FPS),f/FPS)).tobytes())
        p.stdin.close(); p.wait()
