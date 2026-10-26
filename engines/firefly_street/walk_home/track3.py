# Robust lamp tracking: a cloud of feature points on the lamp, frame-to-frame similarity transform (RANSAC).
# Gives centre + scale for every frame the lamp is on screen. Seeds: {name: [ref_frame, cx, cy, half_w, half_h]}
import cv2, numpy as np, json, sys, os
O=os.path.dirname(os.path.abspath(__file__))+'/'
clip,seedf,out=sys.argv[1],sys.argv[2],sys.argv[3]
cap=cv2.VideoCapture(O+clip); F=[]
while True:
    ok,f=cap.read()
    if not ok: break
    F.append(cv2.createCLAHE(3.0,(8,8)).apply(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)))
N=len(F); H,W=F[0].shape
GR=[cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g,cv2.CV_32F,1,0),cv2.Sobel(g,cv2.CV_32F,0,1)),(0,0),1.0) for g in F]
def refine(f,cx,cy,s,tmpl,tw,th):
    """snap the predicted lamp position onto the lamp with gradient template matching at the current scale"""
    w,h=max(8,int(tw*s)),max(8,int(th*s))
    if w>W*0.9 or h>H*0.6: return cx,cy,1.0
    T=cv2.resize(tmpl,(w,h)); R=int(max(w,h)*0.25)+4
    x0,y0=int(cx-w/2-R),int(cy-h/2-R)
    if x0<0 or y0<0 or x0+w+2*R>W or y0+h+2*R>H: return cx,cy,-1   # lamp reaching the edge
    win=GR[f][y0:y0+h+2*R,x0:x0+w+2*R]
    m=cv2.matchTemplate(win,T,cv2.TM_CCOEFF_NORMED); _,mx,_,ml=cv2.minMaxLoc(m)
    if mx<0.45: return cx,cy,mx
    return x0+ml[0]+w/2, y0+ml[1]+h/2, mx
seeds=json.load(open(O+seedf)); res={}
lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
def feats(g,cx,cy,hw,hh):
    m=np.zeros_like(g); x0,y0,x1,y1=int(cx-hw),int(cy-hh),int(cx+hw),int(cy+hh)
    cv2.rectangle(m,(max(0,x0),max(0,y0)),(min(W-1,x1),min(H-1,y1)),255,-1)
    p=cv2.goodFeaturesToTrack(g,80,0.01,3,mask=m)
    return p if p is not None else np.zeros((0,1,2),np.float32)
for name,(f0,cx,cy,hw,hh) in seeds.items():
    tr={f0:(cx,cy,1.0)}
    tw,th=int(hw*2.4),int(hh*2.4)
    tmpl=GR[f0][int(cy-th/2):int(cy+th/2),int(cx-tw/2):int(cx+tw/2)].copy()
    for step in (1,-1):
        c=np.array([cx,cy],np.float64); s=1.0; f=f0; P=feats(F[f],cx,cy,hw,hh); bad=0
        while 0<=f+step<N:
            if len(P)<8: break
            Q,st,_=cv2.calcOpticalFlowPyrLK(F[f],F[f+step],P,None,**lk)
            B,st2,_=cv2.calcOpticalFlowPyrLK(F[f+step],F[f],Q,None,**lk)
            good=(st[:,0]==1)&(st2[:,0]==1)&(np.linalg.norm((B-P)[:,0],axis=1)<1.0)
            if good.sum()<6: break
            M,inl=cv2.estimateAffinePartial2D(P[good],Q[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
            if M is None or inl.sum()<6: break
            sc=float(np.hypot(M[0,0],M[1,0]))
            c=M[:,:2]@c+M[:,2]; s*=sc; f+=step
            rx,ry,sc_=refine(f,c[0],c[1],s,tmpl,tw,th)
            if sc_>=0.6: c=np.array([rx,ry])
            elif sc_>=0.45: c=0.5*c+0.5*np.array([rx,ry])
            bad=bad+1 if (0<=sc_<0.45) else 0
            if bad>=14: break                                  # lost the lamp for over half a second: stop
            if sc_>=0.7 and f%12==0:                            # refresh the template as the lamp grows and turns
                w_,h_=int(tw*s),int(th*s); x_,y_=int(c[0]-w_/2),int(c[1]-h_/2)
                if x_>=0 and y_>=0 and x_+w_<=W and y_+h_<=H and w_>8:
                    tmpl=cv2.resize(GR[f][y_:y_+h_,x_:x_+w_],(tw,th))
            ccx,ccy=c
            if not (0<ccx<W and 0<ccy<H): break                 # centre has left the frame
            tr[f]=(float(ccx),float(ccy),float(s))
            keep=good.copy(); keep[good]=inl[:,0].astype(bool)
            P=Q[keep].reshape(-1,1,2)
            if len(P)<25 or f%6==0:                                   # refresh features on the lamp as it grows
                P=feats(F[f],ccx,ccy,hw*s*0.9,hh*s*0.9)
    res[name]={int(k):v for k,v in sorted(tr.items())}
    fs=sorted(tr); print(name,'frames',fs[0],'-',fs[-1],'scale',round(tr[fs[0]][2],2),'->',round(tr[fs[-1]][2],2),flush=True)
json.dump(res,open(O+out,'w'))
