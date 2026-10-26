import cv2, numpy as np
clean=cv2.imread('base_clean.png'); H,W=clean.shape[:2]; s=W/1998.0
g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY).astype(np.float32)
dark=255-cv2.GaussianBlur(g,(0,0),0.7)
def P(pts): return (np.array(pts,np.float32)*s).astype(np.int32)
# window mask pieces
m=np.zeros((H,W),np.uint8)
for p in [[(342,118),(733,118),(733,818),(342,818)],[(778,112),(1192,112),(1192,818),(778,818)],[(1247,118),(1612,118),(1612,818),(1247,818)],[(0,60),(128,100),(128,862),(0,800)]]:
    cv2.fillPoly(m,[P(p)],255)
# mid bulb interior
cx,cy=985,186
region=np.zeros((H,W),np.uint8); cv2.ellipse(region,(cx,cy),(60,64),0,0,360,255,-1); region[:140]=0
mk=np.zeros((H+2,W+2),np.uint8); mk[1:-1,1:-1]=(region==0).astype(np.uint8)
img=cv2.GaussianBlur(clean,(0,0),1.2)
cv2.floodFill(img.copy(),mk,(cx,cy),(0,0,0),(7,7,7),(7,7,7),4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
G0=((mk[1:-1,1:-1]==255)).astype(np.uint8)*255
G0=cv2.morphologyEx(G0,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
G0[:150]=0  # glass below the socket only
ys,xs=np.where(G0>0); x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
gx=cv2.Sobel(g,cv2.CV_32F,1,0); gy=cv2.Sobel(g,cv2.CV_32F,0,1); mag=np.sqrt(gx*gx+gy*gy).astype(np.float32)
tpl=mag[y0:y1,x0:x1+1]
def refine(Gs):
    ring0=lambda G: (G>0)&(cv2.erode(G,np.ones((3,3),np.uint8))==0)
    best=(-1,None,None)
    for e in (0,1,2,3):
        Ge=cv2.erode(Gs,np.ones((3,3),np.uint8),iterations=e) if e else Gs
        for dy in range(-6,7):
            for dx in range(-6,7):
                Gt=np.roll(np.roll(Ge,dy,0),dx,1)
                r=ring0(Gt)
                sc=dark[r].mean()
                if sc>best[0]: best=(sc,(dx,dy,e),Gt)
    return best
allb=np.zeros((H,W),np.uint8)
dbg=[]
cands={'mid':(0,0)}
for name,(sx0,sx1) in {'left':(380,620),'mid':(0,0),'right':(1330,1560)}.items():
    if name=='mid': Gs=G0.copy()
    else:
        res=cv2.matchTemplate(mag[100:330,sx0:sx1],tpl,cv2.TM_CCOEFF_NORMED)
        _,mv,_,ml=cv2.minMaxLoc(res)
        ox=sx0+ml[0]-x0; oy=100+ml[1]-y0
        Gs=cv2.warpAffine(G0,np.float32([[1,0,ox],[0,1,oy]]),(W,H))
    sc,par,Gt=refine(Gs)
    print(name,'score',round(float(sc),1),'dx,dy,erode',par)
    Gt=cv2.dilate(Gt,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))   # include the outline stroke
    ys,xs=np.where(Gt>0); bx=(xs.min()+xs.max())//2
    sock=np.zeros((H,W),np.uint8)
    cv2.fillPoly(sock,[np.array([(bx-13,98),(bx+13,98),(bx+15,125),(bx+21,133),(bx+21,153),(bx-21,153),(bx-21,133),(bx-15,125)],np.int32)],255)
    u=cv2.bitwise_or(Gt,sock)
    u=cv2.morphologyEx(u,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    allb=cv2.bitwise_or(allb,u)
ex=allb.copy()
cv2.fillPoly(ex,[P([(1478,748),(1615,742),(1615,822),(1478,822)])],255)
hsv=cv2.cvtColor(clean,cv2.COLOR_BGR2HSV)
fern=((hsv[...,0]>28)&(hsv[...,0]<90)&(hsv[...,1]>70)).astype(np.uint8)*255
fm=np.zeros((H,W),np.uint8); a,b,c,d=int(690*s),int(905*s),int(600*s),int(880*s)
fm[a:b,c:d]=fern[a:b,c:d]; fm=cv2.dilate(fm,np.ones((5,5),np.uint8),iterations=2)
ex=cv2.bitwise_or(ex,fm)
m=cv2.bitwise_and(m,cv2.bitwise_not(ex)); m=cv2.erode(m,np.ones((3,3),np.uint8))
np.save('wm.npy',cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),1.0))
np.save('bulbmask.npy',cv2.GaussianBlur(allb.astype(np.float32)/255,(0,0),0.9))
# debug: boundary drawn on 3x crops
rows=[]
for bx in (494,985,1442):
    crop=clean[100:260,bx-90:bx+90].copy()
    cr=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    mc=cv2.resize(allb[100:260,bx-90:bx+90],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
    cnt,_=cv2.findContours(mc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cr,cnt,-1,(255,0,255),1); rows.append(cr)
cv2.imwrite('bulbov2.png',np.hstack(rows))
