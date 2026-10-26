import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys, math, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
O=ZT+'wall2/'
W,H,FPS=1080,1920,24
T=float(os.environ.get('T','10'))
TW,TH,FP=661,1389,139                     # tile size, floor pitch

# ---------------- facade tile (seamless crop, windows start at floor phase 4)
src=cv2.imread(O+'facade.png').astype(np.float32)
tile=np.roll(src[25:25+TH,142:142+TW],8,axis=0)
lum=tile.mean(2)
# panes: (x0,x1,top,bot) relative to each floor's window top
PANES=[(17,51,1,59),(64,98,1,59),(102,134,1,59),(149,183,1,59),(218,242,16,59),(250,273,16,59),
       (301,331,2,58),(338,366,2,58),(391,421,2,58),(428,457,2,58),(486,510,16,60),(519,543,16,60),(556,576,18,58),(582,602,18,58)]
NF=TH//FP
# original lit panes -> per-tile on/off; darkened version replaces them when off
ph=(np.arange(TH)%FP)[:,None]
band=(ph>=3)&(ph<=68)
litm=((lum>95)&band).astype(np.uint8)
litm=cv2.dilate(litm,np.ones((5,5),np.uint8))
ncomp,lab=cv2.connectedComponents(litm)
comps=[(lab==k).astype(np.float32) for k in range(1,ncomp) if (lab==k).sum()>120]
dark=tile.copy()
glass=np.array([62,44,30],np.float32)
lb=cv2.GaussianBlur(lum,(0,0),3)
struct=np.clip(lum/np.maximum(lb,1),0.7,1.25)
for c in comps:
    m=cv2.GaussianBlur(c,(0,0),1.0)[...,None]
    dark=dark*(1-m)+(glass*struct[...,None]*(0.8+0.4*np.random.RandomState(len(comps)).rand()))*m
# texture of curtains etc. used inside our own lit panes
S2=np.clip(lum/np.maximum(cv2.GaussianBlur(lum,(0,0),6),1),0.55,1.35)

def hsh(*a):
    x=0
    for k,v in enumerate(a): x=x ^ (int(v)*[73856093,19349663,83492791,2654435761][k%4])
    x=int(x) & 0xffffffffffff
    x=((x ^ (x>>13))*1274126177) & 0xffffffff; x=x ^ (x>>16)
    return (x & 0xffffff)/0x1000000

# ---------------- mip levels
LEV=[]
for L in range(7):
    tw=max(4,round(TW/2**L)); th=max(4,round(TH/2**L))
    it=cv2.INTER_AREA if L else cv2.INTER_LINEAR
    rz=lambda a: cv2.resize(a,(tw,th),interpolation=it) if L else a
    LEV.append(dict(tw=tw,th=th,sx=tw/TW,sy=th/TH,base=rz(dark),lit=rz(tile),comps=[rz(c) for c in comps],S=rz(S2)))

LC=[np.array(c,np.float32) for c in ([120,185,255],[150,205,255],[200,225,255],[245,235,225],[100,160,240])]
def pane_patch(lv,x0,x1,y0,y1,col,amt):
    sx,sy=lv['sx'],lv['sy']
    a,b,c,d=int(round(x0*sx)),max(int(round(x1*sx)),int(round(x0*sx))+1),int(round(y0*sy)),max(int(round(y1*sy)),int(round(y0*sy))+1)
    return a,b,c,d

def mosaic(k,cxw,cyw,t):
    L=0
    while L<6 and k*2**(L+1)<=1.0: L+=1
    lv=LEV[L]; tw,th=lv['tw'],lv['th']
    ksx=k/lv['sx']; ksy=k/lv['sy']                      # screen px per mosaic px
    wx0=cxw-W/(2*k)-2/k; wx1=cxw+W/(2*k)+2/k; wy0=cyw-H/(2*k)-2/k; wy1=cyw+H/(2*k)+2/k
    i0,i1=int(math.floor(wx0/TW)),int(math.floor(wx1/TW)); j0,j1=int(math.floor(wy0/TH)),int(math.floor(wy1/TH))
    ni,nj=i1-i0+1,j1-j0+1
    M=np.tile(lv['base'],(nj,ni,1))
    diff=lv['lit']-lv['base']
    for tj in range(nj):
        for ti in range(ni):
            I,J=i0+ti,j0+tj
            y,x=tj*th,ti*tw
            keep=np.zeros((th,tw),np.float32)
            for ci,cm in enumerate(lv['comps']):
                if hsh(I,J,ci,1)<0.35 and not (I==0 and J==0 and False): keep=np.maximum(keep,cm)
            if keep.any(): M[y:y+th,x:x+tw]+=diff*keep[...,None]
            for f in range(NF):
                for pi,(px0,px1,pt,pb) in enumerate(PANES):
                    r=hsh(I,J,f*20+pi,2)
                    if I==0 and J==0 and f==5 and pi in (6,7): continue      # the cat window
                    on=r<0.075
                    tog=hsh(I,J,f*20+pi,3)*T*1.5
                    if hsh(I,J,f*20+pi,4)<0.12 and t>tog: on=not on
                    if not on: continue
                    col=LC[int(hsh(I,J,f*20+pi,5)*len(LC))]*(0.55+0.5*hsh(I,J,f*20+pi,6))
                    if hsh(I,J,f*20+pi,7)<0.18:                          # TV
                        fl=0.65+0.35*math.sin(t*6.1+r*50)*math.sin(t*2.3+r*90)
                        col=np.array([175,125,80],np.float32)*fl
                    fy=f*FP+4
                    a,b,c,d=pane_patch(lv,px0,px1,fy+pt,fy+pb,col,1)
                    if b<=a or d<=c: continue
                    S=lv['S'][c:d,a:b]
                    hh,ww=d-c,b-a
                    seed=hsh(I,J,f*20+pi,8)
                    xw=np.linspace(px0,px1,ww,dtype=np.float32)[None,:]; yn=np.linspace(0,1,hh,dtype=np.float32)[:,None]
                    xn=(xw-px0)/max(1,px1-px0)
                    folds=0.80+0.20*np.sin(xw*(1.1+seed)+seed*30)
                    covL=0.35*hsh(I,J,f*20+pi,9); covR=1-0.35*hsh(I,J,f*20+pi,10)
                    cur=np.where((xn<covL)|(xn>covR),0.45,1.0)
                    lamp_x=0.2+0.6*seed
                    g=0.55+0.55*np.exp(-((xn-lamp_x)**2/0.12+(yn-0.35)**2/0.25))
                    floor=np.where(yn>0.80+0.1*seed,0.55,1.0)
                    em=col*0.82*(g*folds*cur*floor*S)[...,None]
                    M[y+c:y+d,x+a:x+b]=em
    # map to screen
    ox=(i0*TW)*lv['sx']; oy=(j0*TH)*lv['sy']               # mosaic origin in level px of world
    if ksx<0.999: M=cv2.GaussianBlur(M,(0,0),0.42/ksx)
    else: M=cv2.GaussianBlur(M,(0,0),0.55)                    # slight softness when magnified
    A=np.float32([[ksx,0,W/2-(cxw*lv['sx']-ox)*ksx],[0,ksy,H/2-(cyw*lv['sy']-oy)*ksy]])
    return cv2.warpAffine(M,A,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

# ---------------- cat window photo
cat=cv2.imread(O+'catwin.png').astype(np.float32)
CF=(140,100,1125,935)                       # window frame in the photo
BAYW=(297,697,371,760)                      # bay C, floor 5, tile (0,0) in world px
mx=(BAYW[2]-BAYW[0])/(CF[2]-CF[0]); my=(BAYW[3]-BAYW[1])/(CF[3]-CF[1])
CXW=(BAYW[0]+BAYW[2])/2; CYW=(BAYW[1]+BAYW[3])/2
cy_,cx_=np.mgrid[0:1264,0:1264].astype(np.float32)
inside=((cx_>CF[0]+15)&(cx_<CF[2]-15)&(cy_>CF[1]+15)&(cy_<CF[3]-15)).astype(np.float32)
inside=cv2.GaussianBlur(inside,(0,0),6)
def dist_out():
    dx=np.maximum(np.maximum(CF[0]-cx_,cx_-CF[2]),0); dy=np.maximum(np.maximum(CF[1]-cy_,cy_-CF[3]),0)
    return np.sqrt(dx*dx+dy*dy)
DOUT=dist_out()
EDGE=np.minimum(np.minimum(cx_,1263-cx_),np.minimum(cy_,1263-cy_))
WINM=np.clip((10-DOUT)/10,0,1).astype(np.float32)
OUTM=(np.clip(EDGE/70.0,0,1)**1.5).astype(np.float32)
rng=np.random.RandomState(4); KEYS=rng.rand(400)
def tv(t):
    # flicker: slow drift + scene changes every ~1.5 s + fast shimmer
    s=t*0.67; i=int(s); u=s-i; u=u*u*(3-2*u)
    lvl=KEYS[i%400]*(1-u)+KEYS[(i+1)%400]*u
    b=0.70+0.45*lvl+0.06*math.sin(t*23)+0.04*math.sin(t*37+1)
    hue=KEYS[(i*7)%400]*(1-u)+KEYS[(i*7+7)%400]*u
    col=np.array([1.0+0.08*(hue-0.5),1.0,1.0-0.20*(hue-0.5)],np.float32)
    return b,col
def cat_layer(k,t):
    sx=mx*k; sy=my*k
    b,col=tv(t)
    gk=1+0.9*float(np.clip((math.log(8.0)-math.log(k))/math.log(8.0),0,1))   # keep the cat's window glowing as it shrinks
    img=cat*(1+inside*(b*gk-1))[...,None]*(1+inside[...,None]*(col-1))
    # how much of the photo's own wall we keep: all of it early, just the window later
    lk=math.log(k)
    fr=float(np.clip((lk-math.log(6.0))/(math.log(14.0)-math.log(6.0)),0,1)); fr=fr*fr*(3-2*fr)
    # the photo's wall dissolves in place (no shrinking edge); the window itself always stays
    a=np.maximum(WINM,OUTM*fr)
    A=np.float32([[sx,0,W/2-((CF[0]+CF[2])/2)*sx],[0,sy,H/2-((CF[1]+CF[3])/2)*sy]])
    if sx<0.999:
        img=cv2.GaussianBlur(img,(0,0),0.42/sx); a=cv2.GaussianBlur(a,(0,0),0.42/sx)
    im=cv2.warpAffine(img,A,(W,H),flags=cv2.INTER_LINEAR,borderValue=0)
    al=cv2.warpAffine(a,A,(W,H),flags=cv2.INTER_LINEAR,borderValue=0)
    return im,al

# ---------------- zoom curve
K0,K1=21.0,0.24
def kat(t):
    x=np.clip(t/T,0,1)
    e=x**1.5                                   # gentle start, steady pull later
    return math.exp(math.log(K0)+(math.log(K1)-math.log(K0))*e)

# ---------------- rain + grain + text
def rain_tile(n,ln,seed,th,ang):
    r=np.random.RandomState(seed); tl=np.zeros((768,768),np.float32)
    for _ in range(n):
        x,y=r.randint(0,768,2); l=r.randint(ln//2,ln)
        cv2.line(tl,(x,y),(x+int(l*ang),y+l),r.uniform(0.3,1),th,cv2.LINE_AA)
    return np.tile(tl,(2,2))
RT=[(rain_tile(260,34,1,1,0.06),768/0.9,0.10),(rain_tile(90,70,2,1,0.06),768/0.55,0.14),(rain_tile(25,120,3,2,0.06),768/0.33,0.16)]
def rain(t):
    acc=np.zeros((H,W),np.float32)
    for tl,sp,a in RT:
        oy=int(t*sp)%768; ox=int(t*sp*0.06)%768
        blk=np.roll(np.roll(tl,oy,0),ox,1)[:768,:768]
        acc+=np.tile(blk,(3,2))[:H,:W]*a
    return acc
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Medium.otf',50)
im=Image.new('L',(W,H),0); dd=ImageDraw.Draw(im)
for k_,l in enumerate(["somewhere in this city,","someone's cat is up late"]): dd.text((W/2,290+k_*66),l,font=FB,fill=255,anchor='mm')
TXT=np.asarray(im,np.float32)/255; TSH=cv2.GaussianBlur(TXT,(0,0),7)
GR=np.random.RandomState(7)

def frame(f):
    t=f/FPS; k=kat(t)
    fac=mosaic(k,CXW,CYW,t)
    if k>1.5:                                   # the far wall sits out of focus while we're close
        fac=cv2.GaussianBlur(fac,(0,0),min(14,0.45*(k-1.5)))
    cim,al=cat_layer(k,t)
    img=fac*(1-al[...,None])+cim*al[...,None]
    # glow from bright windows
    br=np.clip(img-150,0,None)
    sm=cv2.resize(br,(W//6,H//6),interpolation=cv2.INTER_AREA)
    img+=cv2.resize(cv2.GaussianBlur(sm,(0,0),3),(W,H))*0.8
    X=np.linspace(-1,1,W,dtype=np.float32)[None,:]; Y=np.linspace(-1,1,H,dtype=np.float32)[:,None]
    img*=(1-0.32*(X**2*0.7+Y**2*0.5))[...,None]
    img+=rain(t)[...,None]*np.array([190,185,175],np.float32)*0.5
    img+=GR.randn(H//2,W//2).astype(np.float32).repeat(2,0).repeat(2,1)[...,None]*3.2
    img=img*(1-TSH[...,None]*0.5)
    img=img*(1-TXT[...,None])+TXT[...,None]*np.array([242,242,248],np.float32)
    return np.clip(img,0,255).astype(np.uint8)

if __name__=='__main__':
    if sys.argv[1]=='still':
        for tt in sys.argv[2].split(','): cv2.imwrite(O+f's_{tt}.png',frame(int(float(tt)*FPS)))
    else:
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
            '-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',O+'v.mp4'],stdin=subprocess.PIPE)
        for f in range(int(T*FPS)): p.stdin.write(frame(f).tobytes())
        p.stdin.close(); p.wait()
