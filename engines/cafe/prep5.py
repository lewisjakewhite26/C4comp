import cv2, numpy as np
base=cv2.imread('base.png'); H,W=base.shape[:2]
f=base.astype(np.float32)
bmr=f[...,0]-f[...,2]
rng=np.random.default_rng(3)
out=f.copy()
def fillrows(poly,y0,y1,leafkeep):
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(poly,np.int32)],255)
    fill=np.zeros((H,W),np.float32)
    for y in range(y0,y1):
        xs=np.where(m[y]>0)[0]
        if len(xs)==0: continue
        xl,xr=xs.min(),xs.max()
        def samp(a,b):
            blk=f[y,a:b].copy(); ok=(bmr[y,a:b]>-45)
            return np.median(blk[ok],axis=0) if ok.sum()>=2 else np.median(blk,axis=0)
        L=samp(xl-14,xl-2); R=samp(xr+3,xr+15)
        t=np.linspace(0,1,xr-xl+1)[:,None]
        seg=L*(1-t)+R*t+rng.normal(0,1.2,(xr-xl+1,1))
        out[y,xl:xr+1]=seg
        fill[y,xl:xr+1]=1
    if leafkeep:
        leaf=((bmr<-45)&(f.max(axis=2)<235)).astype(np.uint8)*255
        leaf=cv2.dilate(leaf,np.ones((3,3),np.uint8))
        fill[leaf>0]=0
    return fill
up=[(872,819),(868,790),(878,772),(900,762),(930,758),(965,760),(972,742),(984,722),(1010,716),(1042,716),(1048,738),(1032,748),(1018,762),(1022,790),(1014,812),(990,818),(960,819)]
lo=[(872,820),(878,835),(872,860),(880,893),(968,893),(968,820)]
f1=fillrows(up,716,820,True)
med=cv2.medianBlur(np.clip(out,0,255).astype(np.uint8),13).astype(np.float32)
sel=(f1>0)
out[sel]=med[sel]; f2=fillrows(lo,820,893,False)
fm=np.clip(f1+f2,0,1); fm=cv2.GaussianBlur(fm,(0,0),2.5)
res=base.astype(np.float32)*(1-fm[...,None])+out*fm[...,None]
res=np.clip(res,0,255).astype(np.uint8)
cv2.imwrite('base_clean.png',res)
cv2.imwrite('crop_steam.png',np.hstack([base[690:910,820:1100],res[690:910,820:1100]]))
