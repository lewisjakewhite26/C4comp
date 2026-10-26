import cv2, numpy as np, json
base=cv2.imread('base.png'); H,W=base.shape[:2]
s=W/1998.0
def P(pts): return (np.array(pts,np.float32)*s).astype(np.int32)
# --- remove painted steam
roi=(860,712,1060,892)  # x0,y0,x1,y1 in B coords
x0,y0,x1,y1=roi
f=base.astype(np.float32)
mn=f.min(axis=2)
dog=cv2.GaussianBlur(mn,(0,0),2.0)-cv2.GaussianBlur(mn,(0,0),14.0)
sm=np.zeros((H,W),np.uint8)
sm[y0:y1,x0:x1]=(dog[y0:y1,x0:x1]>7).astype(np.uint8)*255
# keep away from the mug itself (mug rim ~ y>=893)
sm[890:,:]=0
sm=cv2.dilate(sm,np.ones((5,5),np.uint8),iterations=2)
clean=cv2.inpaint(base,sm,7,cv2.INPAINT_TELEA)
cv2.imwrite('base_clean.png',clean)
cv2.imwrite('steam_mask.png',sm)
# --- window mask
m=np.zeros((H,W),np.uint8)
for p in [[(342,118),(733,118),(733,818),(342,818)],[(778,112),(1192,112),(1192,818),(778,818)],[(1247,118),(1612,118),(1612,818),(1247,818)],[(0,60),(128,100),(128,862),(0,800)]]:
    cv2.fillPoly(m,[P(p)],255)
# exclusions: bulbs, chair, fern
ex=np.zeros((H,W),np.uint8)
for cx,cy in [(497,170),(987,164),(1442,170)]:
    cv2.ellipse(ex,(int(cx*s),int(cy*s)),(int(50*s),int(68*s)),0,0,360,255,-1)
    cv2.rectangle(ex,(int((cx-14)*s),int(98*s)),(int((cx+14)*s),int(125*s)),255,-1)
cv2.fillPoly(ex,[P([(1478,748),(1615,742),(1615,822),(1478,822)])],255)
hsv=cv2.cvtColor(clean,cv2.COLOR_BGR2HSV)
fern=((hsv[...,0]>28)&(hsv[...,0]<90)&(hsv[...,1]>70)).astype(np.uint8)*255
fm=np.zeros((H,W),np.uint8); fm[int(690*s):int(905*s),int(600*s):int(880*s)]=fern[int(690*s):int(905*s),int(600*s):int(880*s)]
fm=cv2.dilate(fm,np.ones((5,5),np.uint8),iterations=2)
ex=cv2.bitwise_or(ex,fm)
m=cv2.bitwise_and(m,cv2.bitwise_not(ex))
m=cv2.erode(m,np.ones((3,3),np.uint8))
wm=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),1.6)
np.save('wm.npy',wm)
bulbs=np.zeros((H,W),np.uint8)
for cx,cy in [(497,170),(987,164),(1442,170)]:
    cv2.ellipse(bulbs,(int(cx*s),int(cy*s)),(int(46*s),int(60*s)),0,0,360,255,-1)
np.save('bulbmask.npy',cv2.GaussianBlur(bulbs.astype(np.float32)/255,(0,0),3))
# fairy light points
g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
strip=np.zeros_like(g); strip[20:125,240:1720]=255; strip[60:340,235:290]=255
pts=((g>238)&(strip>0)).astype(np.uint8)
n,lab,st,cen=cv2.connectedComponentsWithStats(pts)
fl=[(float(cen[i][0]),float(cen[i][1])) for i in range(1,n) if 2<=st[i][4]<=260]
json.dump(fl,open('fairy.json','w'))
print('fairy',len(fl))
ov=clean.copy(); ov[wm>0.5]=(0.55*ov[wm>0.5]+0.45*np.array([255,0,255])).astype(np.uint8)
for x,y in fl: cv2.circle(ov,(int(x),int(y)),5,(0,255,0),1)
cv2.imwrite('ov2.png',ov)
cv2.imwrite('crop_steam.png',np.hstack([base[690:910,820:1100],clean[690:910,820:1100]]))
