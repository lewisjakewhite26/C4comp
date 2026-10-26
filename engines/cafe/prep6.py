import cv2, numpy as np
clean=cv2.imread('base_clean.png'); H,W=clean.shape[:2]; s=W/1998.0
def P(pts): return (np.array(pts,np.float32)*s).astype(np.int32)
m=np.zeros((H,W),np.uint8)
for p in [[(342,118),(733,118),(733,818),(342,818)],[(778,112),(1192,112),(1192,818),(778,818)],[(1247,118),(1612,118),(1612,818),(1247,818)],[(0,60),(128,100),(128,862),(0,800)]]:
    cv2.fillPoly(m,[P(p)],255)
ex=np.zeros((H,W),np.uint8)
g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
bulbs=np.zeros((H,W),np.uint8)
cx,cy=985,186
region=np.zeros((H,W),np.uint8); cv2.ellipse(region,(cx,cy),(60,64),0,0,360,255,-1); region[:140]=0
mk=np.zeros((H+2,W+2),np.uint8); mk[1:-1,1:-1]=(region==0).astype(np.uint8)
img=cv2.GaussianBlur(clean,(0,0),1.2)
cv2.floodFill(img.copy(),mk,(cx,cy),(0,0,0),(7,7,7),(7,7,7),4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
got=((mk[1:-1,1:-1]==255)).astype(np.uint8)*255
got=cv2.morphologyEx(got,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
got=cv2.dilate(got,np.ones((3,3),np.uint8),iterations=3)
cv2.rectangle(got,(cx-12,98),(cx+12,150),255,-1)
ys,xs=np.where(got>0); x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
gx=cv2.Sobel(cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY),cv2.CV_32F,1,0); gy=cv2.Sobel(cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY),cv2.CV_32F,0,1)
mag=np.sqrt(gx*gx+gy*gy).astype(np.float32)
tpl=mag[y0+30:y1,x0:x1+1]  # glass part only
bulbs=got.copy()
for name,(sx0,sx1) in {'left':(380,620),'right':(1330,1560)}.items():
    res=cv2.matchTemplate(mag[100:330,sx0:sx1],tpl,cv2.TM_CCOEFF_NORMED)
    _,mv,_,ml=cv2.minMaxLoc(res)
    ox=sx0+ml[0]-x0; oy=100+ml[1]-(y0+30)
    print(name,'match',round(mv,3),'shift',ox,oy)
    M=np.float32([[1,0,ox],[0,1,oy]])
    bulbs=cv2.bitwise_or(bulbs,cv2.warpAffine(got,M,(W,H)))
ex=bulbs.copy()
cv2.fillPoly(ex,[P([(1478,748),(1615,742),(1615,822),(1478,822)])],255)
hsv=cv2.cvtColor(clean,cv2.COLOR_BGR2HSV)
fern=((hsv[...,0]>28)&(hsv[...,0]<90)&(hsv[...,1]>70)).astype(np.uint8)*255
fm=np.zeros((H,W),np.uint8); a,b,c,d=int(690*s),int(905*s),int(600*s),int(880*s)
fm[a:b,c:d]=fern[a:b,c:d]; fm=cv2.dilate(fm,np.ones((5,5),np.uint8),iterations=2)
ex=cv2.bitwise_or(ex,fm)
m=cv2.bitwise_and(m,cv2.bitwise_not(ex)); m=cv2.erode(m,np.ones((3,3),np.uint8))
np.save('wm.npy',cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),1.6))
np.save('bulbmask.npy',cv2.GaussianBlur(bulbs.astype(np.float32)/255,(0,0),2.0))
ov=clean.copy(); ov[bulbs>0]=(0.4*ov[bulbs>0]+0.6*np.array([255,0,255])).astype(np.uint8)
cv2.imwrite('bulbov.png',np.hstack([ov[90:260,420:580],ov[90:260,910:1070],ov[90:260,1365:1525]]))
