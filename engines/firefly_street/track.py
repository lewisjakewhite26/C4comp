# Track each lantern through the clip: seed by hand at one frame, follow forwards and backwards with pyramidal Lucas-Kanade.
import cv2, numpy as np, json, sys, os
O=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/'
cap=cv2.VideoCapture(O+os.environ.get('CLIP','clip1.mp4')); F=[]
while True:
    ok,f=cap.read()
    if not ok: break
    F.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
F=[cv2.createCLAHE(3.0,(8,8)).apply(g) for g in F]     # lift the dark lanterns out of the gloom
N=len(F)
SEEDS=json.load(open(O+os.environ.get('SEEDS','seeds.json')))                    # [[name, frame, x, y], ...]
lk=dict(winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
tracks={}
for name,f0,x,y in SEEDS:
    tr={f0:(x,y)}
    for step in (1,-1):
        p=np.array([[[x,y]]],np.float32); f=f0
        while 0<=f+step<N:
            q,st,err=cv2.calcOpticalFlowPyrLK(F[f],F[f+step],p,None,**lk)
            # forward-backward check to stop when the track goes bad
            b,st2,_=cv2.calcOpticalFlowPyrLK(F[f+step],F[f],q,None,**lk)
            if not st[0][0] or not st2[0][0] or np.linalg.norm(b-p)>1.5: break
            qx,qy=q[0,0]
            if not (2<qx<F[0].shape[1]-2 and 2<qy<F[0].shape[0]-2): break
            f+=step; p=q; tr[f]=(float(qx),float(qy))
    tracks[name]={int(k):v for k,v in tr.items()}
    print(name,'frames',min(tr),'-',max(tr))
json.dump(tracks,open(O+os.environ.get('TRK','tracks.json'),'w'))
# check sheet: tracks drawn on 4 frames
cap=cv2.VideoCapture(O+os.environ.get('CLIP','clip1.mp4')); C=[]
while True:
    ok,f=cap.read()
    if not ok: break
    C.append(f)
tiles=[]
for fi in (0,216,432,648):
    a=C[fi].copy()
    for name,tr in tracks.items():
        if fi in tr:
            x,y=tr[fi]; cv2.circle(a,(int(x),int(y)),9,(0,255,255),2); cv2.putText(a,name,(int(x)+10,int(y)),0,0.5,(0,255,255),1)
    tiles.append(cv2.resize(a,(360,640)))
cv2.imwrite(O+'trk.jpg',np.hstack(tiles))
