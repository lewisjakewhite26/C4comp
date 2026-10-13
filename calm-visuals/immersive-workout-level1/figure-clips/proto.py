import sys, math, random, subprocess; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np, cv2
from moves3d import *
import wo
from wo import put_text, pop, TXT
W, H = 1920, 1080; TEAL = (212, 227, 127); DIMc = (110, 118, 66)
GYF = GY+30
MV = [("REACH UP HIGH!", "Touch the stars", m_reach, 'reach'),
      ("MARCH ON THE SPOT!", "Lift those knees!", m_march, 'march'),
      ("STAR JUMPS!", "Arms and legs wide!", m_star, 'star'),
      ("STAMP YOUR FEET!", "Stomp, stomp, stomp!", m_stamp, 'stamp'),
      ("PUNCH THE AIR!", "Left, right, left, right!", m_punch, 'punch')]
INTRO, MOVE = 5.0, 9.0
def sc(c, k): return tuple(int(x*max(0, min(1, k))) for x in c)

def fx_layer(kind, t, M, bbox):
    L = np.zeros((H, W, 3), np.uint8)
    x0, y0, x1, y1 = bbox
    cxm = (x0+x1)//2
    if kind == 'reach':
        u = 0.5-0.5*math.cos(2*math.pi*t/3.0)
        r = random.Random(5)
        for i in range(30):
            sx = r.randint(120, 1800); sy = r.randint(190, 520); ph = r.random()*6
            b = 0.15+0.85*max(0, u-0.35)/0.65
            b *= 0.6+0.4*math.sin(t*4+ph)
            n = int(6+10*b)
            c = sc(TEAL, b)
            cv2.line(L, (sx-n, sy), (sx+n, sy), c, 3, cv2.LINE_AA); cv2.line(L, (sx, sy-n), (sx, sy+n), c, 3, cv2.LINE_AA)
            if b > 0.7: cv2.circle(L, (sx, sy), 4, c, -1, cv2.LINE_AA)
        for i in range(10):   # rising light motes
            ph = ((t/2.0)+i/10) % 1.0
            xx = 300+ i*150+ 20*math.sin(i+t); yy = 900-ph*700
            cv2.circle(L, (int(xx), int(yy)), 4, sc(DIMc, math.sin(math.pi*ph)*u*1.5), -1, cv2.LINE_AA)
    elif kind in ('march', 'star', 'stamp'):
        # scrolling floor ticks
        sp = {'march': 260, 'star': 0, 'stamp': 0}[kind]
        if sp:
            for i in range(14):
                x = (i*180-(t*sp)) % (W+360)-180
                cv2.line(L, (int(x), GYF+22), (int(x+70), GYF+22), sc(DIMc, 0.9), 4, cv2.LINE_AA)
        cv2.line(L, (260, GYF), (1660, GYF), sc(DIMc, 0.6), 4, cv2.LINE_AA)
        if kind == 'march':
            a = sn(t, 1.0)
            for ph0, side in ((0.25, -1), (0.75, 1)):   # dust puffs at footfalls
                age = ((t % 1.0)-ph0) % 1.0
                if age < 0.35:
                    k = age/0.35
                    for j in range(5):
                        cv2.circle(L, (int(cxm+side*60+side*j*14*k*2), int(GYF-8-12*k-j*5*k)), int(6+8*k), sc(DIMc, 1-k), 3, cv2.LINE_AA)
        if kind == 'star':
            c = t % 1.0
            age = (c-0.0) % 1.0
            if age < 0.45:
                k = age/0.45
                cv2.ellipse(L, (cxm, GYF), (int(80+420*k), int(14+40*k)), 0, 0, 360, sc(TEAL, 1-k), 5, cv2.LINE_AA)
            ap = abs(c-0.5)
            if ap < 0.22:
                k = 1-ap/0.22; hy = y0+60
                for a in range(0, 360, 24):
                    d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                    p1 = np.array([cxm, hy])+d*(120+60*(1-k)); p2 = p1+d*(30+40*k)
                    cv2.line(L, tuple(p1.astype(int)), tuple(p2.astype(int)), sc(TEAL, k), 4, cv2.LINE_AA)
        if kind == 'stamp':
            for off, side in ((0.45, -1), (0.95, 1)):
                age = ((t-off) % 1.0)
                if age < 0.7:
                    k = age/0.7; r = random.Random(int(t-off)*2+(side > 0))
                    fx = cxm+side*55
                    cv2.ellipse(L, (int(fx), GYF), (int(40+260*min(1, k*2)), int(8+24*min(1, k*2))), 0, 0, 360, sc(TEAL, 1-k), 5, cv2.LINE_AA)
                    for j in range(7):
                        ang = math.radians(-170+j*27+r.randint(-6, 6)); ln = (110+r.randint(0, 100))*min(1, k*3)
                        px, py = fx, GYF+2
                        for s in range(5):
                            ang2 = ang+r.uniform(-0.35, 0.35); nx, ny = px+math.cos(ang2)*ln/5, py+math.sin(ang2)*ln/5*0.45+ (4 if s else 0)
                            if ny < GYF-2: ny = GYF-2+(s)
                            cv2.line(L, (int(px), int(py)), (int(nx), int(ny)), sc(TEAL, 1-k*0.9), 3, cv2.LINE_AA); px, py = nx, ny
    elif kind == 'punch':
        a = sn(t, 1.2)
        for sg in (1, -1):   # peak of each punch: left punch when a>0 side
            ang = (t/1.2) % 1.0
            peak = 0.18 if sg == 1 else 0.68
            age = ((ang-peak) % 1.0)*1.2
            if age < 0.4:
                k = age/0.4; fxp = (x0+20) if sg == 1 else (x1-20); fy = y0+70
                for j in range(3):
                    rr = int(30+110*k+j*26)
                    cv2.ellipse(L, (fxp, fy), (rr, rr), 0, (200 if sg == 1 else -20)-40, (200 if sg == 1 else -20)+40, sc(TEAL, (1-k)*(1-j*0.25)), 6, cv2.LINE_AA)
    return L

def frame(seg, t):
    # seg: -1 intro, 0..4 move
    Lf = np.zeros((H, W, 3), np.uint8)
    if seg < 0:
        r, dx, j, yw = rest(t); r2 = {}; 
    else:
        title, cue, fn, kind = MV[seg]
        r, dx, j, yw = fn(t); 
        if t < 1.2: r, dx, j, yw = mixd(rest(0), (r, dx, j, yw), sm((t-0.2)/1.0))
    Vp = build(r, dx, j); M = mask(Vp, dx, yw)
    Mb = cv2.GaussianBlur(cv2.dilate(M, np.ones((3, 3), np.uint8)), (0, 0), 4); _, Mb = cv2.threshold(Mb, 120, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(Mb > 0)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (900, 300, 1020, 900)
    ph = 6.9*int(t*8)
    fig = np.zeros((H, W, 3), np.uint8)
    fig[Mb > 0] = (30, 35, 18)       # faint fill
    cv2.polylines(fig, outline(M, ph), True, TEAL, 5, cv2.LINE_AA)
    fxl = fx_layer(MV[seg][3], t, Mb, bbox) if seg >= 0 else np.zeros_like(fig)
    comp = np.maximum(fig, fxl)
    g = cv2.resize(comp, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32)
    gl = cv2.GaussianBlur(g, (0, 0), 2.5)*0.7+cv2.GaussianBlur(g, (0, 0), 8)*0.9
    out = comp.astype(np.float32)+cv2.resize(gl, (W, H))
    if seg < 0:
        if t < 2.0: put_text(out, "GET READY!", 190, 130, pop((t-0.1)/0.5))
        else:
            n = int(t-2.0); put_text(out, ["3", "2", "1"][min(2, n)], 230, 170, pop(((t-2.0)-n)/0.35))
        dur = INTRO
    else:
        title, cue = MV[seg][0], MV[seg][1]
        put_text(out, title, 135, 100, pop((t-0.1)/0.5)*(1+0.035*max(0, 1-((t-0.6) % 0.5)/0.16) if t > 0.6 else 1), maxw=1450)
        put_text(out, cue if t < MOVE-3 else ["KEEP GOING!", "BRILLIANT!", "NICE ONE!", "FANTASTIC!", "WELL DONE!"][seg], 56, 1008, pop((t-0.5)/0.5) if t < MOVE-3 else pop((t-(MOVE-3))/0.4), col=(210, 235, 150), weight='SemiBold')
        x0_, x1_, y_ = 660, 1260, 1060
        cv2.line(out, (x0_, y_), (x1_, y_), (60, 60, 40), 8, cv2.LINE_AA); cv2.line(out, (x0_, y_), (int(x0_+600*t/MOVE), y_), TEAL, 8, cv2.LINE_AA)
        dur = MOVE
    put_text(out, "ZONELYTIME", 36, 1060, 1.0, col=(120, 140, 85), cx=170, weight='Bold', maxw=400)
    put_text(out, "LEVEL 1", 34, 34, 1.0, col=(170, 190, 120), cx=1790, weight='Bold', maxw=300)
    f = min(1.0, t/0.25)*min(1.0, (dur-t)/0.2)
    return np.clip(out*f, 0, 255).astype(np.uint8)

def render_seg(seg, out):
    dur = INTRO if seg < 0 else MOVE
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-g', '30', out], stdin=subprocess.PIPE)
    for k in range(int(dur*30)): p.stdin.write(frame(seg, k/30).tobytes())
    p.stdin.close(); p.wait()
if __name__ == '__main__':
    if sys.argv[1] == 'seg': render_seg(int(sys.argv[2]), sys.argv[3])
    else:
        tiles = [frame(s, tt) for s, tt in ((0, 2.2), (1, 0.25), (2, 0.5), (3, 0.5), (4, 0.2), (2, 0.05))]
        tiles = [cv2.resize(x, (640, 360)) for x in tiles]
        cv2.imwrite('proto.png', np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
