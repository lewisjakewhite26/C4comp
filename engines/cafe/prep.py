import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import cv2, numpy as np
src=cv2.imread(ZT+'uploads/0a1c7927-image.jpg')
W,H=1990,1119
base=cv2.resize(src,(W,H),interpolation=cv2.INTER_AREA)
cv2.imwrite('base.png',base)
s=W/1998.0
def P(pts): return (np.array(pts,np.float32)*s).astype(np.int32)
m=np.zeros((H,W),np.uint8)
# three panes (glass area) in displayed coords
panes=[[(342,118),(733,118),(733,818),(342,818)],
       [(778,112),(1192,112),(1192,818),(778,818)],
       [(1247,118),(1612,118),(1612,818),(1247,818)]]
for p in panes: cv2.fillPoly(m,[P(p)],255)
# left side window
cv2.fillPoly(m,[P([(0,60),(128,100),(128,862),(0,800)])],255)
cv2.imwrite('mask_raw.png',m)
ov=base.copy(); ov[m>0]=(0.5*ov[m>0]+0.5*np.array([255,0,255])).astype(np.uint8)
cv2.imwrite('ov.png',ov)
