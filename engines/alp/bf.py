# ---- realistic bird (3D-projected, supersampled) and leaping fish (shaded, with spray) ----
def _proj(P,yaw=0.75,pit=0.42):
    x,y,z=P[...,0],P[...,1],P[...,2]
    x1=x*math.cos(yaw)+z*math.sin(yaw); z1=-x*math.sin(yaw)+z*math.cos(yaw)
    y1=y*math.cos(pit)-z1*math.sin(pit)
    return np.stack([x1,-y1],-1)
def _wing(side,phase,amp,L=26.0):
    n=26; s=np.linspace(0,1,n)
    # dihedral angle along span (tip lags the root)
    th=0.2+amp*np.sin(phase-1.25*s**1.3)
    dz=np.cos(th)*L/(n-1); dy=np.sin(th)*L/(n-1)
    z=np.concatenate([[0],np.cumsum(dz[1:])]); y=np.concatenate([[0],np.cumsum(dy[1:])])
    chord=11.5*(1-0.5*s)+1.2*np.sin(np.pi*s)*0.3
    xle=-2.6+4.2*s**1.4-1.4*s*np.sin(phase-0.6)*amp
    xte=xle+chord
    # primary feathers: finger-like notches on the outer 38% of the trailing edge
    for i in range(n):
        if s[i]>0.62:
            xte[i]+=3.2*(s[i]-0.62)/0.38*(0.5+0.5*(1 if i%3==0 else -0.6))
    P_le=np.stack([xle,y,side*z],-1); P_te=np.stack([xte,y-0.9*(1-s),side*z],-1)
    return np.concatenate([P_le,P_te[::-1]],0)
def draw_bird(out,x,y,sc,phase,alpha,glide=1.0):
    SS=6; half=int(34*sc)+4
    amp=0.30+0.52*glide
    sub=[]
    for dph in (-0.22,0.0,0.22):
        ph=phase+dph
        polys=[]
        far=_proj(_wing(-1,ph,amp)); near=_proj(_wing(1,ph,amp))
        # body ellipse + head + beak + tail
        a=np.linspace(0,2*np.pi,40,endpoint=False)
        body=_proj(np.stack([8.8*np.cos(a),0.2+3.1*np.sin(a),2.2*np.cos(a)*0],-1))
        head=_proj(np.stack([-9.6+2.7*np.cos(a),0.9+2.3*np.sin(a),0*a],-1))
        beak=_proj(np.array([[-11.8,0.5,0],[-14.6,1.3,0],[-11.9,1.8,0]]))
        tail=_proj(np.array([[6.0,0.2,0],[14.5,0.8,-3.6],[15.6,1.2,-1.2],[15.6,1.2,1.2],[14.5,0.8,3.6]]))
        sub.append([far,body,head,beak,tail,near])
    mask=np.zeros((2*half*SS,2*half*SS),np.float32)
    for polys in sub:
        m=np.zeros_like(mask)
        for p in polys:
            q=np.round(((p*sc)+half)*SS).astype(np.int32)
            cv2.fillPoly(m,[q],1.0,cv2.LINE_AA)
        mask+=m/len(sub)
    mk=cv2.resize(mask,(2*half,2*half),interpolation=cv2.INTER_AREA)
    xi,yi=int(round(x))-half,int(round(y))-half
    x0,y0=max(0,xi),max(0,yi); x1,y1=min(W,xi+2*half),min(H,yi+2*half)
    if x1<=x0 or y1<=y0: return
    mk=mk[y0-yi:y1-yi,x0-xi:x1-xi]*alpha
    roi=out[y0:y1,x0:x1]
    # dark silhouette with a hint of warm rim at the lower edge (sunlit underside)
    col=np.array([44,36,42],np.float32)
    vy=np.linspace(0,1,mk.shape[0],dtype=np.float32)[:,None,None]
    colv=col*(1-0.35*vy)+np.array([70,70,95],np.float32)*0.35*vy
    out[y0:y1,x0:x1]=roi*(1-mk[...,None])+colv*mk[...,None]
def birds(out,t):
    for t0,n,sc,by in BIRDS:
        dur=15.0
        if not (t0<=t<=t0+dur): continue
        u=(t-t0)/dur
        for i in range(n):
            uu=u-0.05*i
            if not 0<=uu<=1: continue
            x=1990-2090*uu+i*10
            y=by+14*math.sin(uu*TAU*1.1+i*1.3)+uu*26+i*18
            s_=sc*(0.9+0.22*uu)*(1 if i==0 else 0.88)
            al=min(1,uu*8,(1-uu)*8)*0.92
            fr=(uu*4.5+i*0.3)%1.0; glide=1.0 if fr<0.62 else 0.05+0.95*max(0,(0.82-fr)/0.2) if fr<0.82 else 0.05
            draw_bird(out,x,y,s_,TAU*3.0*(t-t0)+i*1.9,al,glide)
# ---------------- fish ----------------
_FR=np.random.RandomState(11)
_SPOTS=[(_FR.uniform(0.2,0.8),_FR.uniform(-0.8,-0.35),_FR.uniform(0.6,1.3)) for _ in range(26)]
def _spray(tc,cx,sgn,n,seed):
    r=np.random.RandomState(seed); L=[]
    for k in range(n):
        ang=math.radians(r.uniform(55,125)); sp=r.uniform(70,150)
        L.append((tc,cx+r.uniform(-4,4),sgn*r.uniform(0.2,1.0)*sp*math.cos(ang)*0.55,-sp*abs(math.sin(ang)),r.uniform(0.9,1.8)))
    return L
FISH_DUR=0.95
def fish_sprite(u,Lf,SS=6):
    # returns (rgba float image, anchor) of the fish body at animation phase u in [0,1]
    hw=int(Lf*0.75)+6; size=2*hw
    wag=0.55*math.sin(u*TAU*2.2+0.6)
    # spine angle: head heading from trajectory + body curvature
    vx=55.0; vy=-4*30*(1-2*u)/FISH_DUR
    a0=math.atan2(vy,vx)
    ns=48; s=np.linspace(0,1,ns)
    curv=0.9*math.sin(u*TAU*1.0+0.3)*(0.5)
    ang=a0+np.pi+(curv*(s-0.35))+wag*(s**2.2)    # from head (s=0) to tail; pointing backward
    xs=np.concatenate([[0],np.cumsum(np.cos(ang[1:])*Lf/(ns-1))]); ys=np.concatenate([[0],np.cumsum(np.sin(ang[1:])*Lf/(ns-1))])
    xs-=xs.mean(); ys-=ys.mean()
    nx=-np.gradient(ys); ny=np.gradient(xs); nn=np.hypot(nx,ny)+1e-6; nx/=nn; ny/=nn   # normal
    w=0.145*Lf*(np.sin(np.pi*np.clip(s,0,1)**0.72)**0.85)*(1-0.55*np.clip((s-0.78)/0.22,0,1)**1.2)+0.4
    top=np.stack([xs+nx*w,ys+ny*w],1); bot=np.stack([xs-nx*w,ys-ny*w],1)
    # caudal fin
    tx,ty=xs[-1],ys[-1]; tdx,tdy=np.cos(ang[-1]),np.sin(ang[-1]); tnx,tny=-tdy,tdx
    fl=0.2*Lf
    tail=np.array([[tx,ty],[tx+tdx*fl+tnx*fl*0.55,ty+tdy*fl+tny*fl*0.55],[tx+tdx*fl*0.62,ty+tdy*fl*0.62],[tx+tdx*fl-tnx*fl*0.55,ty+tdy*fl-tny*fl*0.55]])
    # dorsal & pectoral fins
    i=int(0.40*ns); dors=np.array([top[i],[top[i][0]+nx[i]*0.14*Lf+tdx*0.02*Lf,top[i][1]+ny[i]*0.14*Lf],top[i+6]])
    j=int(0.20*ns); pec=np.array([bot[j],[bot[j][0]-nx[j]*0.12*Lf+np.cos(ang[j])*0.11*Lf,bot[j][1]-ny[j]*0.12*Lf+np.sin(ang[j])*0.11*Lf],bot[j+4]])
    body_poly=np.concatenate([top,bot[::-1]],0)
    img=np.zeros((size*SS,size*SS),np.float32)
    def fp(poly,v,m):
        q=np.round((poly+hw)*SS).astype(np.int32); cv2.fillPoly(m,[q],v,cv2.LINE_AA)
    mb=np.zeros_like(img); fp(body_poly,1.0,mb)
    mf=np.zeros_like(img); fp(tail,1.0,mf); fp(dors,1.0,mf); fp(pec,1.0,mf)
    # per-pixel shading by across-body coordinate
    ys_,xs_=np.nonzero(mb>0.02)
    pts=np.stack([xs_/SS-hw,ys_/SS-hw],1)
    d=((pts[:,None,:]-np.stack([xs,ys],1)[None])**2).sum(2); k=d.argmin(1)
    rel=pts-np.stack([xs[k],ys[k]],1); rho=(rel[:,0]*nx[k]+rel[:,1]*ny[k])/(w[k]+1e-3)    # +1 toward 'top' normal side
    # which side is dorsal? choose the side that faces upward in image (smaller y)
    dorsal_sign=-1 if ny.mean()>0 else 1
    rho=np.clip(rho*dorsal_sign,-1,1)        # +1 = back, -1 = belly
    col=np.zeros((len(k),3),np.float32)
    back=np.array([58,78,66],np.float32); flank=np.array([172,184,190],np.float32); band=np.array([150,142,206],np.float32); belly=np.array([218,228,236],np.float32)
    t1=np.clip((0.55-rho)/0.5,0,1)[:,None]            # back -> flank
    col=back*(1-t1)+flank*t1
    t2=np.clip((0.1-rho)/0.9,0,1)[:,None]*np.clip(1.4-np.abs(rho+0.0)*3.2,0,1)[:,None]
    col=col*(1-0.55*t2)+band*0.55*t2
    tb=np.clip((-0.25-rho)/0.7,0,1)[:,None]; col=col*(1-tb)+belly*tb
    # spots on the back
    for sp,rp,rr in _SPOTS:
        ii=int(sp*(ns-1)); pass
    colimg=np.zeros((size*SS,size*SS,3),np.float32); colimg[ys_,xs_]=col
    # fins: translucent reddish brown
    colimg=np.where(mf[...,None]>0.02,np.where(mb[...,None]>0.02,colimg,np.array([95,110,150],np.float32)),colimg)
    # eye
    ex,ey=top[2][0]*0.4+bot[2][0]*0.6,top[2][1]*0.4+bot[2][1]*0.6
    cv2.circle(colimg,(int((ex+hw)*SS),int((ey+hw)*SS)),int(1.5*SS),(20,22,24),-1,cv2.LINE_AA)
    cv2.circle(colimg,(int((ex+hw-0.4)*SS),int((ey+hw-0.5)*SS)),int(0.6*SS),(235,240,245),-1,cv2.LINE_AA)
    alpha=np.clip(mb+mf*0.9,0,1)
    # wet specular streak along the back
    return cv2.resize(colimg,(size,size),interpolation=cv2.INTER_AREA),cv2.resize(alpha,(size,size),interpolation=cv2.INTER_AREA),hw
def fish(out,t):
    px,py=FISH_P; u=(t-FISH_T)/FISH_DUR
    x0,x1,y0,y1=int(px-110),int(px+160),int(py-120),int(py+50)
    roi=out[y0:y1,x0:x1]; ov=roi.copy().astype(np.float32)
    yw=py+4
    # ---- ripples (flattened rings, bright crest + dark trough)
    fx=np.zeros(roi.shape[:2],np.float32); dk=np.zeros_like(fx)
    for tc,cx in ((FISH_T,px-6),(FISH_T+FISH_DUR,px+46)):
        age=t-tc
        if 0<age<4.0:
            for j in range(4):
                a2=age-j*0.38
                if a2<=0: continue
                r=5+30*a2**0.9; al=0.6*(1-a2/4.0)**1.5
                m=np.zeros(roi.shape[:2],np.uint8); cv2.ellipse(m,(int(cx-x0),int(yw+2-y0)),(int(r),int(r*0.27)),0,0,360,255,1,cv2.LINE_AA)
                m2=np.zeros(roi.shape[:2],np.uint8); cv2.ellipse(m2,(int(cx-x0),int(yw+4-y0)),(int(r),int(r*0.27)),0,0,360,255,2,cv2.LINE_AA)
                fx=np.maximum(fx,m.astype(np.float32)/255*al); dk=np.maximum(dk,m2.astype(np.float32)/255*al*0.5)
    ov=ov*(1-dk[...,None]*0.30)
    # ---- fish body
    if 0<=u<=1:
        Lf=40.0
        img,al,hw=fish_sprite(u,Lf)
        fxp=px-6+52*u; fyp=py-30*4*u*(1-u)-2
        xi=int(round(fxp-hw))-x0; yi=int(round(fyp-hw))-y0
        # waterline clip: hide anything below the surface near take-off / landing
        h,w_=al.shape
        yy=(np.arange(h)[:,None]+yi+y0)
        vis=np.clip((yw-yy)/3.0,0,1)
        # only clip while near the surface (first / last 14%)
        if 0.14<u<0.86: vis=np.ones_like(vis)
        a_=al*vis
        xa,ya=max(0,xi),max(0,yi); xb,yb=min(ov.shape[1],xi+w_),min(ov.shape[0],yi+h)
        if xb>xa and yb>ya:
            aa=a_[ya-yi:yb-yi,xa-xi:xb-xi,None]; cc=img[ya-yi:yb-yi,xa-xi:xb-xi]
            ov[ya:yb,xa:xb]=ov[ya:yb,xa:xb]*(1-aa)+cc*aa
    # ---- spray: droplets (streaks) from take-off, landing, and dripping off the body
    drops=_spray(FISH_T,px-6,-1,16,1)+_spray(FISH_T+FISH_DUR,px+46,1,22,2)
    for k in range(14):
        tb=FISH_T+0.06+k*0.062
        uu=(tb-FISH_T)/FISH_DUR
        bx=px-6+52*uu; by=py-30*4*uu*(1-uu)
        drops.append((tb,bx,(k%3-1)*10,-8+(k%4)*3,1.0))
    for tb,bx,vx,vy,r in drops:
        age=t-tb
        if not (0<age<1.3): continue
        gx_=bx+vx*age; gy_=py-0+vy*age+0.5*230*age*age
        if tb<FISH_T+FISH_DUR*0.9 and tb>FISH_T+0.05: gy_=(py-30*4*((tb-FISH_T)/FISH_DUR)*(1-(tb-FISH_T)/FISH_DUR))+vy*age+0.5*230*age*age
        if gy_>yw+1: continue
        al=min(1,(1.3-age)/0.5)*0.95
        vxx=vx; vyy=vy+230*age; L=0.035
        p0=(int((gx_-x0)*8),int((gy_-y0)*8)); p1=(int((gx_-vxx*L-x0)*8),int((gy_-vyy*L-y0)*8))
        lay=ov.copy(); cv2.line(lay,p0,p1,(238,240,246),1 if r<1.3 else 2,cv2.LINE_AA,shift=3)
        ov=ov*(1-al)+lay*al if al>0 else ov
        # note: blends whole roi: cheap at this crop size
    # splash column at entry/exit (short-lived water sheet)
    for tc,cx in ((FISH_T,px-6),(FISH_T+FISH_DUR,px+46)):
        age=t-tc
        if 0<age<0.45:
            h_=int(16*math.sin(min(1,age/0.45)*math.pi)**0.8); wd=int(5+9*age/0.45)
            m=np.zeros(roi.shape[:2],np.uint8)
            pts=np.array([[cx-x0-wd,yw-y0],[cx-x0-wd*0.45,yw-y0-h_*0.8],[cx-x0,yw-y0-h_],[cx-x0+wd*0.45,yw-y0-h_*0.8],[cx-x0+wd,yw-y0]],np.int32)
            cv2.fillPoly(m,[pts],255,cv2.LINE_AA); m=cv2.GaussianBlur(m,(0,0),1.2).astype(np.float32)/255*0.7*(1-age/0.45)
            ov=ov*(1-m[...,None])+np.array([240,242,247],np.float32)*m[...,None]
    ov=ov*(1-fx[...,None]*0.75)+np.array([236,239,244],np.float32)*fx[...,None]*0.75
    out[y0:y1,x0:x1]=np.clip(ov,0,255)
