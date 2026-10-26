import cv2,numpy as np
def patchfill(img,unk,p=13,lam=0.0006,seed=0):
    """img BGR uint8, unk bool mask (True = to fill). Exemplar-based fill (Criminisi-lite)."""
    img=img.astype(np.float32).copy(); H,W=unk.shape; h=p//2
    K=~unk; K=K.copy()
    valid_src=cv2.erode(K.astype(np.uint8),np.ones((p,p),np.uint8))>0   # patch centres fully known
    yy,xx=np.mgrid[0:H,0:W]
    it=0
    while (~K).any():
        # boundary: unknown pixels adjacent to known
        kd=cv2.dilate(K.astype(np.uint8),np.ones((3,3),np.uint8))>0
        bd=(~K)&kd
        ys,xs=np.nonzero(bd)
        if len(ys)==0: break
        # confidence = fraction known in patch
        conf=cv2.boxFilter(K.astype(np.float32),-1,(p,p),normalize=True,borderType=cv2.BORDER_REPLICATE)
        c=conf[ys,xs]; i=np.argmax(c); cy,cx=ys[i],xs[i]
        y0,y1=max(0,cy-h),min(H,cy+h+1); x0,x1=max(0,cx-h),min(W,cx+h+1)
        tpl=img[y0:y1,x0:x1]; mk=K[y0:y1,x0:x1].astype(np.float32)
        mk3=np.repeat(mk[...,None],3,2)
        res=cv2.matchTemplate(img,tpl,cv2.TM_SQDIFF,mask=mk3)
        # res index (r,c) -> patch top-left; centre offset
        th,tw=tpl.shape[:2]
        cost=res/ max(1.0,mk.sum()*3)
        # distance penalty from target centre
        ry,rx=np.mgrid[0:cost.shape[0],0:cost.shape[1]]
        dist=np.hypot(ry+th/2-cy,rx+tw/2-cx)
        cost=cost+lam*dist**2/ (1.0)
        # only sources fully known
        vs=valid_src[th//2:th//2+cost.shape[0],tw//2:tw//2+cost.shape[1]]
        # fix shape if cropped template
        vs=cv2.erode(K.astype(np.uint8),np.ones((th,tw),np.uint8))[0:cost.shape[0],0:cost.shape[1]]>0
        cost[~vs]=1e12
        j=np.argmin(cost); sy,sx=np.unravel_index(j,cost.shape)
        src=img[sy:sy+th,sx:sx+tw]
        u=~K[y0:y1,x0:x1]
        img[y0:y1,x0:x1][u]=src[u]
        K[y0:y1,x0:x1]=True
        it+=1
    return np.clip(img,0,255).astype(np.uint8),it
