import cv2, numpy as np
base=cv2.imread('base.png'); H,W=base.shape[:2]
x0,y0,x1,y1=860,700,1080,892
f=base.astype(np.float32)
mn=f.min(axis=2)
dog=cv2.GaussianBlur(mn,(0,0),2.0)-cv2.GaussianBlur(mn,(0,0),16.0)
bmr=f[...,0]-f[...,2]
sm=np.zeros((H,W),np.uint8)
sel=(dog>5)&(bmr>-32)
sm[y0:y1,x0:x1]=sel[y0:y1,x0:x1].astype(np.uint8)*255
sm=cv2.morphologyEx(sm,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
sm=cv2.dilate(sm,np.ones((3,3),np.uint8),iterations=2)
clean=cv2.inpaint(base,sm,5,cv2.INPAINT_TELEA)
cv2.imwrite('base_clean.png',clean); cv2.imwrite('steam_mask.png',sm)
cv2.imwrite('crop_steam.png',np.hstack([base[690:910,820:1100],cv2.cvtColor(sm[690:910,820:1100],cv2.COLOR_GRAY2BGR),clean[690:910,820:1100]]))
