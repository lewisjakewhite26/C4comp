import sys; sys.path.insert(0,'.')
import numpy as np, jelly2 as J
# fast coverage: simulate bells only (skip chains)
for j in []: pass
S=J.make()
for j in S:
    j.tent.step=lambda *a,**k:None
    if j.arms: j.arms.step=lambda *a,**k:None
t=-12.0; rows=[]
while t<620:
    for j in S: j.step(t,J.DT)
    t+=J.DT
    if abs(t*J.FPS/240-round(t*J.FPS/240))<1e-6:
        big=[j for j in S if j.R>=85 and -j.R*0.3<j.p[1]<J.H-j.R*0.4 and 0<j.p[0]<J.W]
        allv=[j for j in S if -j.R<j.p[1]<J.H and 0<j.p[0]<J.W]
        L=sum(1 for j in allv if j.p[0]<J.W/2)
        rows.append((round(t),len(big),L,len(allv)-L))
print(' '.join(f'{a}:{b}/{c}-{d}' for a,b,c,d in rows))
print('min big',min(r[1] for r in rows),'min left',min(r[2] for r in rows),'min right',min(r[3] for r in rows))
