import numpy as np, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
P=np.load('pos.npy',mmap_mode='r'); hue=np.load('hue.npy'); FPS=30; R0=0.01
prev=None; ev=[]; last=-9
for f in range(0,P.shape[0],15):
    x=np.array(P[f],np.float32); t=f/FPS
    pairs=cKDTree(x).query_pairs(2.7*R0,output_type='ndarray')
    n=len(x); A=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n)) if len(pairs) else coo_matrix((n,n))
    nc,lab=connected_components(A,directed=False)
    sizes=np.bincount(lab)
    if prev is not None:
        pl,ps=prev
        for c in np.argsort(-sizes)[:12]:
            if sizes[c]<10: break
            m=lab==c; pm=np.bincount(pl[m],minlength=len(ps)); pc=pm.argmax(); grow=sizes[c]-ps[pc]
            if grow>=14 and t-last>=1.3:
                h=np.angle(np.exp(2j*np.pi*hue[m]).mean())/(2*np.pi)%1.0; ev.append((t,float(h),int(sizes[c]))); last=t; break
    prev=(lab,sizes)
ev=[(3.0+(t-3.0)/0.915,h,s) for t,h,s in ev if t<3.0+300*0.915]
json.dump(ev,open('events.json','w')); print(len(ev),ev[:12])
