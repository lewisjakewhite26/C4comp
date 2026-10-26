# Reiter snowflake model on a hex grid (axial coords). Saves freeze iteration per cell.
import numpy as np, sys, cv2, math
def grow(alpha,beta,gamma,N=221,stop=92,maxit=40000):
    c=N//2; I,J=np.mgrid[0:N,0:N]-c
    hd=np.maximum(np.maximum(abs(I),abs(J)),abs(I+J))
    edge=hd>=c-2
    s=np.full((N,N),beta); s[c,c]=1.0
    ft=np.full((N,N),-1,np.int32); ft[c,c]=0
    def nb(a):
        return (np.roll(a,1,0),np.roll(a,-1,0),np.roll(a,1,1),np.roll(a,-1,1),np.roll(np.roll(a,1,0),-1,1),np.roll(np.roll(a,-1,0),1,1))
    for it in range(1,maxit):
        fr=s>=1
        rec=fr.copy()
        for x in nb(fr): rec|=x
        u=np.where(rec,0,s); v=np.where(rec,s+gamma,0)
        avg=sum(nb(u))/6
        s=u+alpha/2*(avg-u)+v
        s[edge]=beta
        nf=(s>=1)&(ft<0); ft[nf]=it
        if nf.any() and hd[nf].max()>=stop: break
    return ft,s,it
def pic(ft,s,a=4):
    N=ft.shape[0]; c=N//2; img=np.zeros((900,900),np.float32)
    ii,jj=np.where(ft>=0)
    x=450+a*((ii-c)+(jj-c)/2); y=450+a*(math.sqrt(3)/2)*(jj-c)
    for X,Y,v in zip(x,y,s[ii,jj]): cv2.circle(img,(int(X*4),int(Y*4)),int(a*0.62*4),float(min(1,0.5+(v-1)*2)),-1,cv2.LINE_AA,2)
    return img
if __name__=='__main__':
    sets=[tuple(map(float,x.split(','))) for x in sys.argv[1:]]
    tiles=[]
    for p in sets:
        ft,s,it=grow(*p); n=(ft>=0).sum(); print(p,it,n,flush=True)
        im=(pic(ft,s)*255).astype(np.uint8); cv2.putText(im,str(p)+f' n={n}',(10,40),0,1.0,255,2); tiles.append(cv2.resize(im,(450,450)))
    g=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
    cv2.imwrite('reiter_grid.png',g)
