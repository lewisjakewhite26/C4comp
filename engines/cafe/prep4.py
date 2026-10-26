import cv2, numpy as np
base=cv2.imread('base.png'); H,W=base.shape[:2]
f=base.astype(np.float32)
bmr=f[...,0]-f[...,2]
m=np.zeros((H,W),np.uint8)
up=np.array([(872,819),(868,790),(878,772),(900,762),(930,758),(965,760),(972,742),(984,722),(1010,716),(1042,716),(1048,738),(1032,748),(1018,762),(1022,790),(1014,812),(990,818),(960,819)],np.int32)
cv2.fillPoly(m,[up],255)
leaf=((bmr<-45)&(f.max(axis=2)<235)).astype(np.uint8)*255
leaf=cv2.dilate(leaf,np.ones((3,3),np.uint8))
m[:819][leaf[:819]>0]=0
lo=np.array([(872,818),(878,835),(872,860),(880,893),(945,893),(948,860),(942,838),(958,820)],np.int32)
cv2.fillPoly(m,[lo],255)
m=cv2.dilate(m,np.ones((3,3),np.uint8))
clean=cv2.inpaint(base,m,6,cv2.INPAINT_TELEA)
# blend soft: keep original outside mask feathered
cv2.imwrite('base_clean.png',clean); cv2.imwrite('steam_mask.png',m)
cv2.imwrite('crop_steam.png',np.hstack([base[690:910,820:1100],clean[690:910,820:1100]]))
