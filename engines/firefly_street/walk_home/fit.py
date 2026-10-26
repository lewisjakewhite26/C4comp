# The camera glides straight forward, so a fixed lamp moves out from a vanishing point as  p(t) = V + D/(T - t).
# Fit that to the reliable middle of each track, then use the fitted path for the lamp's whole time on screen.
import json,sys,numpy as np
from scipy.optimize import least_squares
rt=json.load(open(sys.argv[1])); sb=json.load(open(sys.argv[2])); N=int(sys.argv[4]); W,H=720,1280
out={}
for k,d in rt.items():
    bw,bh=sb[k][3],sb[k][4]
    F=np.array(sorted(int(f) for f in d)); P=np.array([d[str(f)] for f in F])
    ok=(P[:,0]>bw*P[:,2]*1.2)&(P[:,0]<W-bw*P[:,2]*1.2)&(P[:,1]>bh*P[:,2]*1.2)&(P[:,1]<H-bh*P[:,2]*1.2)
    F,P=F[ok],P[ok]
    if len(F)<20: print(k,'too short'); continue
    f0=F[0]; s0=P[0,2]
    def model(q,f):
        vx,vy,dx,dy,T=q; u=1.0/(T-f); return vx+dx*u, vy+dy*u, u
    def res(q):
        x,y,u=model(q,F); return np.concatenate([x-P[:,0],y-P[:,1]])
    T0=F[-1]+ (F[-1]-F[0])*P[0,2]/max(1e-3,P[-1,2]-P[0,2])+1
    q0=[360,640,(P[0,0]-360)*(T0-F[0]),(P[0,1]-640)*(T0-F[0]),T0]
    r=least_squares(res,q0,loss='soft_l1',f_scale=3.0)
    q=r.x; x,y,u=model(q,F); err=np.median(np.hypot(x-P[:,0],y-P[:,1]))
    sref=np.median(P[:,2]/u)                      # scale = sref * u
    o={}
    for f in range(N):
        if q[4]-f<=1: break
        x,y,u=model(q,f); s=sref*u
        if s>8: break
        if -bw*s<x<W+bw*s and -bh*s<y<H+bh*s and s>0.15: o[f]=[float(x),float(y),float(s)]
    out[k]=o; fs=sorted(o); print(k,'fit err px',round(err,2),'frames',fs[0],'-',fs[-1],'scale',round(o[fs[0]][2],2),'->',round(o[fs[-1]][2],2))
json.dump(out,open(sys.argv[3],'w'))
