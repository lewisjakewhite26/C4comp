import numpy as np, cv2, math
from rig import Rig
R = Rig('.')
SC = 38.0; GY = 940; CXs = 960
# which bone side is on screen-left? screen X = CX - x*SC -> larger +x is screen-left
_hl = R.head[R.idx['upperarm01.L']][0]
SIDE = {'SL': 'L', 'SR': 'R'} if _hl > 0 else {'SL': 'R', 'SR': 'L'}
SGN = {'L': 1, 'R': -1}   # +z raises the L arm, -z the R arm (verified), same for legs assumed
REST_ABD = 25.0

def arm(rots, side, abd=0, fwd=0, el_z=0, el_x=0, twist=0):
    b = SIDE[side]; s = SGN[b]
    rots[f'upperarm01.{b}'] = (-fwd, twist, s*(abd-REST_ABD))
    rots[f'lowerarm01.{b}'] = (-el_x, 0, s*el_z)
def leg(rots, side, abd=0, fwd=0, knee=0, foot=0):
    b = SIDE[side]; s = SGN[b]
    rots[f'upperleg01.{b}'] = (-fwd, 0, s*abd)
    rots[f'lowerleg01.{b}'] = (knee, 0, 0)
    rots[f'foot.{b}'] = (foot, 0, 0)
def spine(rots, lean=0, fwd=0, twist=0, share=4):
    for n in ('spine05', 'spine04', 'spine03', 'spine02', 'spine01')[:share]:
        rots[n] = (fwd/share, twist/share, lean/share)
def neck(rots, tilt=0, nod=0, turn=0):
    for n in ('neck01', 'neck02', 'neck03'):
        rots[n] = (nod/3, turn/3, tilt/3)

def build(rots, dx=0, jump=0, yaw=0.0):
    Vp = R.pose(rots)
    Vp = Vp.copy()
    Vp[:, 1] -= Vp[:, 1].min()
    Vp[:, 1] += jump/SC
    cx = Vp[:, 0].mean() if False else 0
    return Vp

def mask(Vp, dx=0, yaw=0.0):
    return R.raster(Vp, SC, CXs+dx, GY, yaw=yaw)

def outline(M, ph=0.0, th=5):
    Mb = cv2.GaussianBlur(cv2.dilate(M, np.ones((3, 3), np.uint8)), (0, 0), 4)
    _, Mb = cv2.threshold(Mb, 120, 255, cv2.THRESH_BINARY)
    cs, hier = cv2.findContours(Mb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 40: continue
        p = c[:, 0, :].astype(np.float32)
        k = 11; ker = np.ones(k)/k
        pad = np.concatenate([p[-k:], p, p[:k]])
        px = np.convolve(pad[:, 0], ker, 'same')[k:-k]; py = np.convolve(pad[:, 1], ker, 'same')[k:-k]
        px = px+0.9*np.sin(py*0.025+ph)+0.4*np.sin(py*0.07+ph*1.7)
        py = py+0.9*np.sin(px*0.025+ph*1.3)+0.4*np.sin(px*0.06+ph*0.6)
        out.append(np.stack([px, py], 1).round().astype(np.int32).reshape(-1, 1, 2))
    return out

def render(rots, dx=0, jump=0, yaw=0.0, ph=0.0, col=(212, 227, 127), th=5, H=1080, W=1920):
    Vp = build(rots, dx, jump)
    M = mask(Vp, dx, yaw)
    L = np.zeros((H, W, 3), np.uint8)
    cv2.polylines(L, outline(M, ph), True, col, th, cv2.LINE_AA)
    return L

def glow(L):
    sm_ = cv2.resize(L, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(sm_, (0, 0), 2.5)*0.7+cv2.GaussianBlur(sm_, (0, 0), 8)*0.9
    return np.clip(L.astype(np.float32)+cv2.resize(g, (L.shape[1], L.shape[0])), 0, 255).astype(np.uint8)
