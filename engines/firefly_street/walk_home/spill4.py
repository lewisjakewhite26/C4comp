import os,sys,json,numpy as np
i=sys.argv[1]
rf=json.load(open(f'rf{i}.json'))
V={'1':'300,480','2':'350,440','3':'375,560'}[i]
P={'1':'[["e",6.5,0.6,0.4,300]]','2':'[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]','3':'[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'}[i]
os.environ.update(dict(PRELIT='a',NOFF='1',CLIP=f'slow{i}.mp4',TRK=f'tr{i}.json',RT=f'rf{i}.json',SB=f'seedb{i}.json',WTRK='none',VP=V,PLAN=P))
sys.argv=['x','none']
import light4 as L, cv2
REAL=dict(L.TON); worst=0
for k,d in L.lamps.items():
    if k.startswith('W') or k not in L.TPL: continue
    L.TON.clear(); L.TON[k]=-5.0
    fs=[f for f in sorted(d) if f/24>=REAL.get(k,0)]
    for f in fs[::12]:
        p=np.array(d[f]); tp,bt,hw=L.geo(k,p,f); gh=bt-tp
        lit=L.frame(f).astype(np.float32); base=np.clip(L.FR[f].astype(np.float32)*0.8,0,255)
        diff=np.abs(lit-base).max(2); yy,xx=np.mgrid[0:L.H,0:L.W]
        far=((np.abs(xx-p[0])>hw*2.0)|(yy<p[1]+tp-gh*0.8)|(yy>p[1]+bt+gh*0.8))&(np.abs(xx-p[0])<hw*5)&(yy>p[1]+tp-gh*3)&(yy<p[1]+bt+gh*3)
        if far.any(): worst=max(worst,float(diff[far].max()))
print('scene',i,'max change beyond 2 glass-widths/0.8 glass-heights from the glass (levels /255):',round(worst,1))
