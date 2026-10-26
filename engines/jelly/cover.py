import sys; sys.path.insert(0,'.')
import numpy as np
import jelly as J
best=None
r=np.random.RandomState(1)
for trial in range(400):
    yo=list(r.uniform(0,J.SPAN,len(J.J)))
    J.rng=np.random.RandomState(7); J.YOFF=yo; js=J.mk_jellies()
    ts=np.arange(0,620,2.0); worst=99; tot=0
    for t in ts:
        big=0; ys=[]
        for j in js:
            x,y=j.pos(t,J.SPAN)
            vis=(-j.R*0.3< y < J.H-j.R*0.5) and (-j.R<x<J.W+j.R)
            if vis and j.R>=85: big+=1
        worst=min(worst,big); tot+=big
    sc=worst*1000+tot
    if best is None or sc>best[0]: best=(sc,worst,tot/len(ts),yo)
print(best[1],best[2]); print([round(v) for v in best[3]])
