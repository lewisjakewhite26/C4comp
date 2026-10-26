import os,sys,numpy as np,cv2,json
i=sys.argv[1]
V={'1':'300,480','2':'350,440','3':'375,560'}[i]
P={'1':'[["e",6.5,0.6,0.4,300]]','2':'[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]','3':'[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'}[i]
os.environ.update(dict(SCENE=i,PRELIT='a',NOFF='1',CLIP=f'slow{i}.mp4',TRK=f'tr{i}.json',RT=f'rf{i}.json',SB=f'seedb{i}.json',WTRK='none',VP=V,PLAN=P))
sys.argv=['x','none']
import light6 as L
rows=[]
REAL=dict(L.TON)
for k in L.MODEL['lamps']:
    t0=max(0,REAL.get(k,0)*24)
    fr=[f for f in range(int(t0),L.N) if L.ubox(k,f) and L.uvis(L.ubox(k,f))>0.3]
    if not fr: continue
    picks=[fr[int(q*(len(fr)-1))] for q in (0.0,0.3,0.6,0.85,1.0)]
    tiles=[]
    for f in picks:
        b=L.ubox(k,f); lit=L.frame(f); base=np.clip(L.FR[f].astype(np.float32)*0.8,0,255).astype(np.uint8)
        cx,cy=int(np.clip((b[0]+b[2])/2,0,L.W-1)),int(np.clip((b[1]+b[3])/2,0,L.H-1)); r=int(max(b[2]-b[0],b[3]-b[1],20)*1.8)
        def crop(a):
            pa=np.pad(a,((r,r),(r,r),(0,0)),mode='constant'); c=pa[cy:cy+2*r,cx:cx+2*r]
            return cv2.resize(c,(230,230),interpolation=cv2.INTER_CUBIC)
        t=np.vstack([crop(base),crop(lit)]); cv2.putText(t,f'{k} f{f}',(4,14),0,0.45,(0,255,0),1); tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(f'chk6_{i}.jpg',np.vstack(rows)); print('ok',i,len(rows))
