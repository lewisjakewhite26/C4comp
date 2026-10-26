import os,sys,numpy as np,cv2
i=sys.argv[1]
import json
rf=json.load(open(f'rf{i}.json')); keys=','.join(rf.keys())
V={'1':'300,480','2':'350,440','3':'375,560'}[i]
P={'1':'[["e",6.5,0.6,0.4,300]]','2':'[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]','3':'[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'}[i]
os.environ.update(dict(PRELIT='a',NOFF='1',CLIP=f'slow{i}.mp4',TRK=f'tr{i}.json',RT=f'rf{i}.json',SB=f'seedb{i}.json',WTRK='none',VP=V,PLAN=P))
sys.argv=['x','none']
import light4 as L
REAL=dict(L.TON)
for k_ in L.lamps: L.TON[k_]=-5.0
rows=[]; rep=[]
for k,d in L.lamps.items():
    if k.startswith('W') or k not in L.TPL: continue
    fs=sorted(d)
    fs=[f for f in fs if True]
    # frames where the glass is fully inside the image
    ok=[]
    for f in fs:
        p=np.array(d[f]); tp,bt,hw=L.geo(k,p,f)
        if p[0]-hw>2 and p[0]+hw<L.W-2 and p[1]+tp>2 and p[1]+bt<L.H-2 and bt-tp>=10: ok.append(f)
    t0=REAL.get(k,0); ok=[f for f in ok if f/24>=t0-0.01]
    if not ok: continue
    picks=[ok[int(q*(len(ok)-1))] for q in (0.0,0.35,0.7,1.0)]
    tiles=[]
    for f in picks:
        p=np.array(d[f]); tp,bt,hw=L.geo(k,p,f)
        lit=L.frame(f); base=np.clip(L.FR[f].astype(np.float32)*0.80,0,255).astype(np.uint8)
        diff=np.abs(lit.astype(np.float32)-base.astype(np.float32)).max(2)
        yy,xx=np.mgrid[0:L.H,0:L.W]
        far=(np.abs(xx-p[0])>hw*2.2)|((yy<p[1]+tp-(bt-tp)*0.9)|(yy>p[1]+bt+(bt-tp)*0.9))
        # beyond ~2.2 half-widths sideways or ~0.9 glass-heights vertically
        far=((np.abs(xx-p[0])>hw*2.4)&(yy>p[1]+tp-(bt-tp)))|((yy<p[1]+tp-(bt-tp)*1.2)|(yy>p[1]+bt+(bt-tp)*1.2))
        near=(np.abs(xx-p[0])<hw*4)&(yy>p[1]+tp-(bt-tp)*2)&(yy<p[1]+bt+(bt-tp)*2)
        far&=near
        rep.append((k,f,round(float(diff[far].max()),1) if far.any() else 0,round(float(np.percentile(diff[far],99)),1) if far.any() else 0))
        r=int(max(hw*3.2,(bt-tp)*1.6,24)); cx,cy=int(p[0]),int(p[1]+(tp+bt)/2)
        def crop(a):
            pa=np.pad(a,((r,r),(r,r),(0,0)),mode='constant'); c=pa[cy:cy+2*r,cx:cx+2*r]
            return cv2.resize(c,(260,260),interpolation=cv2.INTER_CUBIC)
        t=np.vstack([crop(base),crop(lit)]); cv2.putText(t,f'{k} f{f}',(4,14),0,0.45,(0,255,0),1); tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(f'chk4_{i}.jpg',np.vstack(rows))
for r in rep: print(r)
