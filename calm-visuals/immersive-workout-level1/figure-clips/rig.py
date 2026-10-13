import json, numpy as np, scipy.sparse as sp, cv2
class Rig:
    def __init__(self, d='.'):
        V=[];F=[];cur=None;body=[]
        for ln in open(f'{d}/base.obj'):
            if ln.startswith('v '): V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('g '): cur=ln.split()[1]
            elif ln.startswith('f ') and cur=='body':
                F.append([int(t.split('/')[0])-1 for t in ln.split()[1:]])
        self.Vall=np.array(V); self.nb_v=13380
        self.V=self.Vall[:self.nb_v]
        self.F=[f for f in F if max(f)<self.nb_v]
        sk=json.load(open(f'{d}/default.mhskel')); jt=sk['joints']
        self.names=sorted(sk['bones']); self.idx={n:i for i,n in enumerate(self.names)}
        self.parent=[self.idx.get(sk['bones'][n]['parent'],-1) if sk['bones'][n]['parent'] else -1 for n in self.names]
        self.head=np.array([self.Vall[jt[sk['bones'][n]['head']]].mean(0) for n in self.names])
        self.tail=np.array([self.Vall[jt[sk['bones'][n]['tail']]].mean(0) for n in self.names])
        w=json.load(open(f'{d}/default_weights.mhw'))['weights']
        r=[];c=[];v=[]
        for n,lst in w.items():
            if n not in self.idx: continue
            for vi,wt in lst:
                if vi<self.nb_v: r.append(vi);c.append(self.idx[n]);v.append(wt)
        W=sp.csr_matrix((v,(r,c)),shape=(self.nb_v,len(self.names)))
        s=np.asarray(W.sum(1)).ravel(); s[s==0]=1
        self.W=sp.diags(1/s)@W
        order=[];seen=set()
        def vis(i):
            if i in seen or i<0: return
            vis(self.parent[i]); seen.add(i); order.append(i)
        for i in range(len(self.names)): vis(i)
        self.order=order
    @staticmethod
    def rot(rx,ry,rz):
        rx,ry,rz=np.radians([rx,ry,rz])
        cx,sx,cy,sy,cz,sz=np.cos(rx),np.sin(rx),np.cos(ry),np.sin(ry),np.cos(rz),np.sin(rz)
        Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz@Ry@Rx
    def pose(self, rots, root_t=(0,0,0)):
        nb=len(self.names); Wm=[None]*nb
        for i in self.order:
            R=self.rot(*rots.get(self.names[i],(0,0,0)))
            h=self.head[i]
            A=np.eye(4); A[:3,:3]=R; A[:3,3]=h-R@h
            p=self.parent[i]
            Wm[i]=A if p<0 else Wm[p]@A
        Wm[self.idx['root']] = Wm[self.idx['root']]
        M=np.array([m[:3,:].reshape(-1) for m in Wm])      # nb x 12
        MV=self.W@M                                          # nv x 12
        MV=MV.reshape(-1,3,4)
        Vp=np.einsum('nij,nj->ni',MV[:,:,:3],self.V)+MV[:,:,3]
        return Vp+np.array(root_t)
    def raster(self, Vp, scale, cx, gy, W=1920, H=1080, yaw=0.0):
        P=Vp.copy()
        if yaw:
            c,s=np.cos(yaw),np.sin(yaw); x=P[:,0]*c+P[:,2]*s; z=-P[:,0]*s+P[:,2]*c; P[:,0]=x; P[:,2]=z
        X=cx-P[:,0]*scale; Y=gy-P[:,1]*scale   # mesh +x is the figure's left; mirror so the figure faces camera
        pts=np.stack([X,Y],1)
        M=np.zeros((H,W),np.uint8)
        polys=[np.round(pts[f]).astype(np.int32) for f in self.F]
        cv2.fillPoly(M,polys,255)
        return M
