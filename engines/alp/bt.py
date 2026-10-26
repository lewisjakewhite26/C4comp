import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np,cv2,math
W,H=1920,1080; TAU=2*math.pi
exec(open(ZT+'alp/bf.py').read())
bg=np.zeros((180,1000,3),np.float32); bg[:]=(150,200,240)
out=np.zeros((H,W,3),np.float32); out[:]=(150,200,240)
tiles=[]
for i,ph in enumerate(np.linspace(0,TAU,8,endpoint=False)):
    o=out.copy(); draw_bird(o,100,100,1.5,ph,1.0,1.0); tiles.append(o[30:170,0:200])
row=np.hstack(tiles)
o2=out.copy(); tiles2=[]
for i,ph in enumerate(np.linspace(0,TAU,4,endpoint=False)):
    o=out.copy(); draw_bird(o,100,100,1.5,ph,1.0,0.05); tiles2.append(o[30:170,0:200])
cv2.imwrite(ZT+'alp/bird_sheet.png',np.vstack([row[:, :800],row[:,800:],np.hstack(tiles2)]).astype(np.uint8))
