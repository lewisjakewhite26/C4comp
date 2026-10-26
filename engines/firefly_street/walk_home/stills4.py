import os,sys,numpy as np,cv2
i=sys.argv[1]; frames=[int(x) for x in sys.argv[2].split(',')]
V={'1':'300,480','2':'350,440','3':'375,560'}[i]
P={'1':'[["e",6.5,0.6,0.4,300]]','2':'[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]','3':'[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'}[i]
os.environ.update(dict(PRELIT='a',CLIP=f'slow{i}.mp4',TRK=f'tr{i}.json',RT=f'rf{i}.json',SB=f'seedb{i}.json',WTRK='none',VP=V,PLAN=P))
sys.argv=['x','none']
import light4 as L
ims=[]
for f in frames:
    a=L.frame(f); cv2.putText(a,f's{i} t={f/24:.1f}s',(8,26),0,0.8,(0,255,0),2); ims.append(a)
cv2.imwrite(f'ap4_{i}.jpg',np.hstack(ims))
