"""Fast seamless 15-minute café loop. Everything is periodic over T=900 s."""
import cv2, numpy as np, json, sys, subprocess, math, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)
base = cv2.imread('base_clean.png').astype(np.float32)   # BGR, 1990x1119
H, W = base.shape[:2]
wm = np.load('wm.npy').astype(np.float32)[..., None]
bmask = np.load('bulbmask.npy').astype(np.float32)
fairy = json.load(open('fairy.json'))
FPS = 24
T = 900.0
NF = int(T * FPS)
OW, OH = 1920, 1080

def bgr(rgb): return np.array(rgb[::-1], np.float32)
def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def keyinterp(keys, p):
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]; t1, v1 = keys[i + 1]
        if p <= t1:
            u = smooth((p - t0) / (t1 - t0))
            return np.array(v0, np.float32) * (1 - u) + np.array(v1, np.float32) * u
    return np.array(keys[-1][1], np.float32)
def kscalar(keys, p): return float(keyinterp([(t, (v,)) for t, v in keys], p)[0])

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def gauss2(cx, cy, sx, sy):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) * .5)

# ---- cyclic colour keys over phase 0..1 (golden, sunset, dusk, night, pre-dawn, dawn, golden) ----
WIN_MULT = [(0, (1, 1, 1)), (.17, (1.04, .78, .66)), (.33, (.62, .50, .70)), (.50, (.26, .34, .64)), (.72, (.30, .40, .70)), (.88, (.95, .72, .70)), (1, (1, 1, 1))]
INT_MULT = [(0, (1, 1, 1)), (.17, (.99, .88, .80)), (.33, (.80, .68, .70)), (.50, (.58, .54, .72)), (.72, (.62, .58, .76)), (.88, (.90, .80, .80)), (1, (1, 1, 1))]
SKY_TOP = [(0, (0, 0, 0)), (.17, (10, 0, 18)), (.33, (22, 10, 42)), (.50, (4, 12, 54)), (.72, (8, 16, 56)), (.88, (18, 6, 30)), (1, (0, 0, 0))]
SKY_HOR = [(0, (0, 0, 0)), (.17, (70, 28, 0)), (.33, (60, 22, 30)), (.50, (14, 8, 26)), (.72, (22, 14, 40)), (.88, (80, 40, 20)), (1, (0, 0, 0))]
LAMP_L = [(0, .30), (.17, .40), (.33, .80), (.50, .95), (.72, .90), (.88, .55), (1, .30)]
POST_E = [(0, .10), (.17, .30), (.33, .80), (.50, 1.0), (.72, .95), (.88, .50), (1, .10)]
DARK = [(0, 0), (.17, .35), (.33, .75), (.50, 1), (.72, .9), (.88, .4), (1, 0)]
gtop = np.clip(1 - (yy - 112) / 450, 0, 1) ** 1.5
ghor = np.exp(-((yy - 520) / 120) ** 2)
sky_top_map = (gtop * wm[..., 0]).astype(np.float32)[..., None]
sky_hor_map = (ghor * wm[..., 0]).astype(np.float32)[..., None]

def sun_state(ph):
    if ph <= .5: y = 300 + 240 * smooth(ph / .32)
    else: y = 540 - 240 * smooth((ph - .78) / .22)
    i = 0.0
    if ph < .46:
        i = (.35 + .6 * math.sin(math.pi * min(1.0, ph / .34))) * (1 - smooth((ph - .34) / .06))
    elif ph > .70:
        i = (.35 + .6 * math.sin(math.pi * min(1.0, (ph - .74) / .26))) * smooth((ph - .70) / .06) if ph >= .74 else .35 * smooth((ph - .70) / .06)
    return y, i

# ---- lamps ----
BULBS = [(494, 190), (985, 184), (1442, 191)]
bulbmap = []
for cx, cy in BULBS:
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    bulbmap.append((.55 * np.exp(-d2 / (2 * 30 ** 2)) + .30 * np.exp(-d2 / (2 * 150 ** 2)) + .10 * np.exp(-d2 / (2 * 340 ** 2))).astype(np.float32))
bulbmap_sum = sum(bulbmap)
bulb_sel = [(bmask * ((xx < 740) if i == 0 else ((xx >= 740) & (xx < 1215)) if i == 1 else (xx >= 1215))).astype(np.float32) for i in range(3)]
RB = 320
broi = []
for k, (cx, cy) in enumerate(BULBS):
    x0, x1 = max(0, cx - RB), min(W, cx + RB); y0, y1 = max(0, cy - RB), min(H, cy + RB)
    d2 = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2
    g = (.55 * np.exp(-d2 / (2 * 30 ** 2)) + .30 * np.exp(-d2 / (2 * 150 ** 2))).astype(np.float32)
    broi.append((x0, x1, y0, y1, g, bulb_sel[k][y0:y1, x0:x1].copy()))
LAMP_COL = bgr((255, 205, 125)) / 255.0
FAIRY_COL = bgr((255, 218, 150))
rng = np.random.default_rng(2)
DIPS = []
tt = 9.0
while tt < 890:
    DIPS.append((tt, int(rng.integers(0, 3)))); tt += rng.uniform(22, 55)
def flick(t, k):
    ph = k * 1.7
    n = .5 * math.sin(2 * math.pi * 495 * t / T + ph) + .3 * math.sin(2 * math.pi * 1233 * t / T + 2 * ph) + .2 * math.sin(2 * math.pi * 2790 * t / T + 3 * ph)
    f = 1 + .07 * n
    for c, kk in DIPS:
        if kk == k and abs(t - c) < 1.0:
            f *= 1 - .22 * math.exp(-((t - c) / .09) ** 2)
            f += .07 * math.exp(-((t - c - .22) / .12) ** 2)
    return f

def gk(sig, r):
    a = np.arange(-r, r + 1, dtype=np.float32)
    return np.exp(-(a[None, :] ** 2 + a[:, None] ** 2) / (2 * sig * sig))
FK_CORE = gk(3.0, 24); FK_HALO = gk(9.0, 24)
fr = np.random.default_rng(5)
fphase = fr.uniform(0, 6.28, len(fairy)); fm = fr.integers(315, 765, len(fairy))
def fairy_add(img, t, dusk_boost):
    for i, (fx, fy) in enumerate(fairy):
        x = int(round(fx)); y = int(round(fy)); r = 24
        x0, y0 = max(0, x - r), max(0, y - r); x1, y1 = min(W, x + r + 1), min(H, y + r + 1)
        if x1 <= x0 or y1 <= y0: continue
        tw = .75 + .25 * math.sin(2 * math.pi * fm[i] * t / T + fphase[i])
        ker = (40 * FK_CORE + 14 * FK_HALO)[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)] * tw
        roi = img[y0:y1, x0:x1]
        f = roi.astype(np.float32) + ker[..., None] * FAIRY_COL / 255.0 * dusk_boost
        roi[:] = np.clip(f, 0, 255).astype(np.uint8)

# ---- outside lamp posts ----
LP = [(612, 262, 1.0), (838, 392, .55)]
lpmap = (sum(s * (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 14 ** 2)) * .7 + np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 60 ** 2)) * .35) for cx, cy, s in LP).astype(np.float32) * wm[..., 0])[..., None]
vig = (1 - .24 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2)[..., None].astype(np.float32)
dither = np.random.default_rng(77).normal(0, .9, (H, W, 1)).astype(np.float32)

# ---- steam (periodic: NS divides T/DT) ----
SR = np.random.default_rng(11)
NS = 400
s_x0 = 905 + SR.normal(0, 7, NS); s_v = 34 * (.85 + .3 * SR.random(NS)); s_ph2 = SR.uniform(0, 6.28, NS)
STEAM_DT = .09; STEAM_L = 4.6
SK = {}
def skern(sig):
    s = int(round(sig * 2)) / 2
    if s not in SK:
        r = int(s * 3) + 1; a = np.arange(-r, r + 1, dtype=np.float32)
        SK[s] = (np.exp(-(a[None, :] ** 2 + a[:, None] ** 2) / (2 * s * s)), r)
    return SK[s]
SX0, SY0, SX1, SY1 = 800, 600, 1120, 910
def splat(canvas, k, x, y, amp, r):
    x = int(round(x)); y = int(round(y)); x0, y0 = x - r, y - r
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(2 * r + 1, canvas.shape[1] - x0), min(2 * r + 1, canvas.shape[0] - y0)
    if ex <= sx0 or ey <= sy0: return
    canvas[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex] += amp * k[sy0:ey, sx0:ex]
def steam_alpha(t):
    cv = np.zeros((SY1 - SY0, SX1 - SX0), np.float32)
    k0 = int(math.floor((t - STEAM_L) / STEAM_DT)); k1 = int(math.floor(t / STEAM_DT))
    for k in range(k0, k1 + 1):
        age = t - k * STEAM_DT
        if age < 0 or age > STEAM_L: continue
        i = k % NS; u = age / STEAM_L; kt = (k * STEAM_DT) % T
        phk = 1.1 * math.sin(2 * math.pi * 170 * kt / T) + .9 * math.sin(2 * math.pi * 310 * kt / T + 1.3)
        amp = 3 + 30 * u ** 1.2
        x = 905 + (s_x0[i] - 905) * .3 + amp * math.sin(2 * math.pi * age / 2.4 + phk) + 12 * u
        y = 892 - s_v[i] * age * (1 + .05 * age)
        sig = 3.2 + 11 * u ** .9
        a = .16 * (math.sin(math.pi * u ** .7) ** 1.3) * min(1.0, age / .35)
        ker, r = skern(sig); splat(cv, ker, x - SX0, y - SY0, a, r)
    return cv2.GaussianBlur(np.clip(cv, 0, .8), (0, 0), 2.0)

# ---- leaves (all periodic) ----
LR = np.random.default_rng(21)
NL = 16
l_x0 = LR.uniform(360, 1590, NL); l_y0 = LR.uniform(0, 740, NL)
l_n = LR.integers(30, 53, NL); l_v = l_n * 740.0 / T
l_m = LR.integers(180, 450, NL); l_ph = LR.uniform(0, 6.28, NL); l_ph2 = LR.uniform(0, 6.28, NL); l_ph3 = LR.uniform(0, 6.28, NL)
l_m2 = LR.integers(20, 60, NL); l_sz = LR.uniform(6.5, 12, NL)
l_col = [(232, 110, 40), (210, 70, 35), (242, 172, 62), (190, 90, 40), (225, 130, 50)]
def draw_leaves(img, t, mW):
    for j in range(NL):
        y = 100 + (l_y0[j] + l_v[j] * t) % 740
        x = l_x0[j] + 22 * math.sin(2 * math.pi * l_m[j] * t / T + l_ph[j]) + 12 * math.sin(2 * math.pi * t / T + l_ph3[j])
        ang = 35 * math.sin(2 * math.pi * l_m[j] * t / T + l_ph[j] + 1.2) + 25 * math.sin(2 * math.pi * l_m2[j] * t / T + l_ph2[j])
        a = float(l_sz[j]); r = int(a) + 4
        x0, y0 = int(x) - r, int(y) - r
        if x0 < 0 or y0 < 0 or x0 + 2 * r + 1 > W or y0 + 2 * r + 1 > H: continue
        al = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
        cv2.ellipse(al, (r, r), (int(a), max(2, int(a * .45))), ang, 0, 360, 255, -1, cv2.LINE_AA)
        w = (al.astype(np.float32) / 255.0)[..., None] * wm[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
        if w.max() <= 0: continue
        c = l_col[j % len(l_col)]; col = bgr(c) * mW
        roi = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
        roi[:] = np.clip(roi.astype(np.float32) * (1 - w) + col * w, 0, 255).astype(np.uint8)

# ---- dust ----
DR = np.random.default_rng(31)
ND = 30
d_x = DR.uniform(300, 1700, ND); d_y = DR.uniform(160, 800, ND); d_ph = DR.uniform(0, 6.28, ND); d_n = DR.integers(50, 100, ND)
DK = gk(1.7, 7)
def dust_add(img, t):
    for j in range(ND):
        per = T / d_n[j]
        x = d_x[j] + 16 * math.sin(2 * math.pi * t / per + d_ph[j])
        y = d_y[j] + 11 * math.cos(2 * math.pi * t / per + d_ph[j])
        b = .5 + .5 * math.sin(2 * math.pi * 2 * d_n[j] * t / T + d_ph[j] * 2)
        xi, yi = int(round(x)), int(round(y)); r = 7
        x0, y0 = xi - r, yi - r
        if x0 < 0 or y0 < 0 or x0 + 15 > W or y0 + 15 > H: continue
        roi = img[y0:y0 + 15, x0:x0 + 15]
        f = roi.astype(np.float32) + (55 * b) * DK[..., None] * bgr((255, 232, 195)) / 255.0
        roi[:] = np.clip(f, 0, 255).astype(np.uint8)

# ---- slow layer (once per second) ----
_cache = {'sec': None, 'img': None}
def slow_layer(sec):
    ph = (sec % T) / T
    mW = bgr(keyinterp(WIN_MULT, ph)); mI = bgr(keyinterp(INT_MULT, ph))
    skyT = bgr(keyinterp(SKY_TOP, ph)); skyH = bgr(keyinterp(SKY_HOR, ph))
    sy, si = sun_state(ph)
    sunmap = (gauss2(990, sy, 230, 130) * wm[..., 0])[..., None]
    winG = base * mW + sky_top_map * skyT + sky_hor_map * skyH + sunmap * si * 85 * bgr((255, 160, 80)) / 255.0
    winG += lpmap * kscalar(POST_E, ph) * 150 * bgr((255, 200, 110)) / 255.0
    intr = base * mI
    img = intr + wm * (winG - intr)
    L = kscalar(LAMP_L, ph)
    img += bulbmap_sum[..., None] * LAMP_COL * 255.0 * L * .55
    img *= vig
    img += dither
    return np.clip(img, 0, 255).astype(np.uint8)

def frame(fi):
    t = fi / FPS
    sec = int(t)
    if _cache['sec'] != sec:
        _cache['sec'] = sec; _cache['img'] = slow_layer(sec)
    ph = (t % T) / T
    img = _cache['img'].copy()
    mW = bgr(keyinterp(WIN_MULT, ph)); L = kscalar(LAMP_L, ph); dk = kscalar(DARK, ph)
    draw_leaves(img, t, mW)
    for k in range(3):
        fk = flick(t, k)
        x0, x1, y0, y1, g, sel = broi[k]
        roi = img[y0:y1, x0:x1]
        f = roi.astype(np.float32)
        f *= (1 + (fk - 1) * .9 * sel[..., None])
        f += (g * (fk - 1))[..., None] * LAMP_COL * 255.0 * L * .55
        roi[:] = np.clip(f, 0, 255).astype(np.uint8)
    fairy_add(img, t, .7 + .7 * dk)
    a = steam_alpha(t)
    sc = bgr(tuple(np.array((252, 246, 238)) * (1 - dk * .8) + np.array((226, 204, 190)) * dk * .8))
    roi = img[SY0:SY1, SX0:SX1]
    roi[:] = np.clip(roi.astype(np.float32) * (1 - a[..., None]) + sc * a[..., None], 0, 255).astype(np.uint8)
    dust_add(img, t)
    return cv2.resize(img, (OW, OH), interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'still':
        for s in sys.argv[2:]:
            t0 = time.time(); fr_ = frame(int(float(s) * FPS)); print(s, round(time.time() - t0, 3), 's')
            cv2.imwrite(f'loop_{int(float(s))}.png', fr_)
    elif m == 'time':
        t0 = time.time()
        for fi in range(2400, 2448): frame(fi)
        print('per frame', round((time.time() - t0) / 48, 4))
    elif m == 'loopcheck':
        a = frame(0).astype(np.float32); b = frame(NF).astype(np.float32); c = frame(NF - 1).astype(np.float32)
        print('t=0 vs t=900:', round(float(np.abs(a - b).mean()), 4), ' last vs first:', round(float(np.abs(a - c).mean()), 4))
    else:
        a, b, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        pr = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-tune', 'animation', '-g', '48', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for fi in range(a, b):
            pr.stdin.write(frame(fi).tobytes())
        pr.stdin.close(); pr.wait()
