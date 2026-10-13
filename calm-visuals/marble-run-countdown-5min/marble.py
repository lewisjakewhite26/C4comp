import sys, math, subprocess, colorsys, numpy as np, cv2
sys.path.insert(0, '.')
W, H = 1920, 1080; FPS = 30; COUNT = 300.0; TOT = 306.0; NF = int(TOT*FPS)
# ---- overlay (copied from the blossom timer) ----
src = open('brender.py').read()
a = src.index("# --- timer overlay"); b = src.index("def render(f):")
ns = {}
exec("import math, numpy as np, cv2\nfrom PIL import Image, ImageDraw, ImageFont\nW,H=1920,1080\nCOUNT=300.0\n"+src[a:b], ns)
overlay = ns['overlay']
# ---- track ----
SPEED = 78.0                      # px/s along the track
def yfun(x): return 660+105*np.sin(x/430.0)+60*np.sin(x/190.0+1.0)+28*np.sin(x/83.0+2.0)
XS = np.arange(-1200, 26000, 2.0)
YS = yfun(XS)
seg = np.hypot(np.diff(XS), np.diff(YS)); S = np.concatenate([[0], np.cumsum(seg)])
S0 = float(np.interp(0, XS, S))
LTOT = SPEED*COUNT
def s_of(t):
    u = min(1.0, max(0.0, t/COUNT)); return LTOT*(u+0.012*math.sin(2*math.pi*3*u)/(2*math.pi*3))
def x_of_s(s): return float(np.interp(S0+s, S, XS))
def pos(s): x = x_of_s(s); return x, float(yfun(x))
def tangent(x): 
    dy = (yfun(x+1.0)-yfun(x-1.0))/2.0; n = math.hypot(1, dy); return 1/n, dy/n
XEND = x_of_s(LTOT)
rng = np.random.default_rng(3)
# parallax stars/bokeh
def layer_pts(n, seed, rmin, rmax):
    r = np.random.default_rng(seed); return r.uniform(0, 8000, n), r.uniform(0, H, n), r.uniform(rmin, rmax, n), r.uniform(0, 1, n)
L1 = layer_pts(260, 1, 1.0, 2.4); L2 = layer_pts(70, 2, 4, 12)
bg = np.zeros((H, W, 3), np.float32); g = np.linspace(0, 1, H)[:, None, None]
bg[:] = np.array([40, 22, 16], np.float32)*(1-g)+np.array([78, 44, 34], np.float32)*g   # deep navy (BGR)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = 1-0.35*np.clip((np.hypot((xx-W/2)/(W/2), (yy-H/2)/(H/2))-0.5)/0.8, 0, 1)**1.5
def hue_col(u, sat=0.62, val=1.0):
    r, g_, b = colorsys.hsv_to_rgb((0.55+0.9*u) % 1.0, sat, val); return (b*255, g_*255, r*255)
CAM_OFF = 620
def render(f):
    t = f/FPS; s = s_of(t); mx, my = pos(s)
    cx = mx-CAM_OFF
    out = bg.copy()
    lay = np.zeros((H, W, 3), np.float32)
    # parallax layers
    for (px, py, pr, ph), par, br in ((L1, 0.25, 0.55), (L2, 0.5, 0.16)):
        sx = (px-cx*par) % 8000-300
        m = (sx > -20) & (sx < W+20)
        for x, y, r, p in zip(sx[m], py[m], pr[m], ph[m]):
            tw = 0.6+0.4*math.sin(t*1.3+p*6.28)
            col = (200*br*tw, 215*br*tw, 255*br*tw)
            cv2.circle(lay, (int(x), int(y)), int(max(1, r)), col, -1, cv2.LINE_AA)
    # supports + floor
    x0 = cx-60; x1 = cx+W+60
    xs = np.arange(math.floor(x0/6)*6, x1, 6.0); ys = yfun(xs)
    pts = np.stack([xs-cx, ys], 1).astype(np.float32)
    passed = xs <= mx
    for xk in np.arange(math.floor(x0/150)*150, x1, 150.0):
        y = float(yfun(xk)); cv2.line(lay, (int(xk-cx), int(y+14)), (int(xk-cx), H), (62, 46, 36), 3, cv2.LINE_AA)
    cv2.line(lay, (0, 1062), (W, 1062), (90, 70, 56), 3, cv2.LINE_AA)
    # track: dark casing, steel rail, coloured passed section
    ip = pts.round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(lay, [ip], False, (30, 20, 14), 30, cv2.LINE_AA)
    cv2.polylines(lay, [ip], False, (104, 92, 82), 22, cv2.LINE_AA)
    for i in range(len(xs)-1):
        if passed[i]:
            u = min(1.0, max(0.0, (x_of_s(0)+ (xs[i]-x_of_s(0)))/max(1, XEND-x_of_s(0)) - x_of_s(0)/max(1, XEND-x_of_s(0))))
            u = (xs[i]-x_of_s(0))/(XEND-x_of_s(0))
            c = hue_col(min(1, max(0, u)), 0.75, 1.0)
            cv2.line(lay, tuple(ip[i, 0]), tuple(ip[i+1, 0]), c, 20, cv2.LINE_AA)
    cv2.polylines(lay, [(pts+np.array([0, -5], np.float32)).round().astype(np.int32).reshape(-1, 1, 2)], False, (190, 186, 182), 2, cv2.LINE_AA)
    # minute gates
    for k in range(0, 6):
        sg = SPEED*60*k
        gx, gy = pos(sg); gxs = gx-cx
        if -80 < gxs < W+80:
            lit = s >= sg
            col = (170, 235, 255) if lit else (110, 100, 92)
            cv2.line(lay, (int(gxs-34), int(gy-12)), (int(gxs-34), int(gy-150)), col, 8, cv2.LINE_AA); cv2.line(lay, (int(gxs+34), int(gy-12)), (int(gxs+34), int(gy-150)), col, 8, cv2.LINE_AA)
            cv2.line(lay, (int(gxs-34), int(gy-150)), (int(gxs+34), int(gy-150)), col, 8, cv2.LINE_AA)
            lab = "START" if k == 0 else str(5-k)
            if k == 5:
                for r_ in range(3):
                    for c_ in range(6):
                        if (r_+c_) % 2 == 0: cv2.rectangle(lay, (int(gxs-33+c_*11), int(gy-148+r_*11)), (int(gxs-23+c_*11), int(gy-138+r_*11)), (240, 240, 240), -1)
    # glow of passed track + marble
    mc = hue_col(min(1, s/LTOT), 0.35, 1.0)
    # marble
    R = 40
    mxs = int(mx-cx); mys = int(my-11-R+2)
    tx, ty = tangent(mx)
    cv2.circle(lay, (mxs, mys), R+3, (18, 12, 8), -1, cv2.LINE_AA)
    ang = s/R
    base = np.array(hue_col((s/LTOT*3.0) % 1.0, 0.5, 0.95), np.float32)
    yy2, xx2 = np.mgrid[-R:R+1, -R:R+1].astype(np.float32); d = np.hypot(xx2, yy2); inside = d <= R
    z = np.sqrt(np.clip(R*R-d*d, 0, None))/R
    nx, ny = xx2/R, yy2/R
    # rotating swirl: stripes bending around the sphere
    ph = (np.arctan2(ny, z+1e-6)+ang)*3.0
    stripe = 0.5+0.5*np.sin(ph*2.0+nx*1.5)
    light = np.clip(0.55+0.55*(-nx*0.4-ny*0.6+z*0.7), 0.2, 1.3)
    colr = (base[None, None, :]*(0.55+0.45*stripe[..., None])+np.array([255, 255, 255], np.float32)*0.12*(1-stripe[..., None]))*light[..., None]
    hl = np.exp(-(((nx+0.35)**2+(ny+0.4)**2)/0.02))[..., None]*255*0.9
    colr = colr+hl
    sl = lay[mys-R:mys+R+1, mxs-R:mxs+R+1]
    if sl.shape[:2] == colr.shape[:2]:
        sl[inside] = np.clip(colr[inside], 0, 255)
    # shadow / glow
    sm = cv2.resize(lay, (W//4, H//4), interpolation=cv2.INTER_AREA)
    gl = cv2.GaussianBlur(sm, (0, 0), 4)*0.4+cv2.GaussianBlur(sm, (0, 0), 14)*0.45
    out += lay+cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)
    out *= VIG[..., None]
    out = overlay(out, t)
    # end confetti
    if t > COUNT:
        k = t-COUNT; r2 = np.random.default_rng(11)
        for i in range(70):
            a_ = r2.uniform(0, 6.283); v = r2.uniform(60, 330); x = mxs+math.cos(a_)*v*k*0.7; y = mys-80+math.sin(a_)*v*k*0.7+40*k*k
            al = max(0, 1-k/3.5)
            c = tuple(max(180.0, q) for q in hue_col(r2.random(), 0.55, 1.0))
            cv2.circle(out, (int(x), int(y)), 4, tuple(float(q*al) for q in c), -1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)
def worker(f0, f1, name):
    pr = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', '-g', '60', '-x264-params', 'threads=1', name], stdin=subprocess.PIPE)
    for i in range(f0, f1): pr.stdin.write(render(i).tobytes())
    pr.stdin.close(); pr.wait()
if __name__ == '__main__':
    if sys.argv[1] == 'test':
        import time
        tiles = []
        for s_ in sys.argv[2:]:
            t0 = time.time(); fr = render(int(float(s_)*FPS)); print(s_, round(time.time()-t0, 2), flush=True); tiles.append(cv2.resize(fr, (800, 450)))
        while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite('mchk.png', np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    else: worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
