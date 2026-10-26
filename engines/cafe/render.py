import cv2, numpy as np, json, sys, subprocess, math, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)
base = cv2.imread('base_clean.png').astype(np.float32)   # BGR
H, W = base.shape[:2]
wm = np.load('wm.npy')[..., None].astype(np.float32)
bm = np.load('bulbmask.npy').astype(np.float32)
fairy = json.load(open('fairy.json'))
FPS = 24
DUR = float(os.environ.get('DUR', 30))
N = int(DUR * FPS)
OW, OH = 1920, 1080

def bgr(rgb): return np.array(rgb[::-1], np.float32)
def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def keyinterp(keys, p):
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]; t1, v1 = keys[i + 1]
        if p <= t1:
            u = smooth((p - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return np.array(v0, np.float32) * (1 - u) + np.array(v1, np.float32) * u
    return np.array(keys[-1][1], np.float32)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def gauss2(cx, cy, sx, sy):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) * 0.5)

# ---- colour keyframes (RGB) ----
WIN_MULT = [(0, (1, 1, 1)), (.40, (1.04, .78, .66)), (.72, (.62, .50, .70)), (1, (.30, .38, .68))]
INT_MULT = [(0, (1, 1, 1)), (.40, (.99, .88, .80)), (.72, (.80, .68, .70)), (1, (.60, .56, .74))]
SKY_TOP = [(0, (0, 0, 0)), (.40, (10, 0, 18)), (.72, (22, 10, 42)), (1, (6, 14, 56))]
SKY_HOR = [(0, (0, 0, 0)), (.40, (70, 28, 0)), (.72, (60, 22, 30)), (1, (18, 10, 30))]
gtop = np.clip(1 - (yy - 112) / 450, 0, 1) ** 1.5
ghor = np.exp(-((yy - 520) / 120) ** 2)
sky_top_map = (gtop * wm[..., 0]).astype(np.float32)
sky_hor_map = (ghor * wm[..., 0]).astype(np.float32)

# ---- lights ----
BULBS = [(494, 190), (985, 184), (1442, 191)]
bulbmap = []
for cx, cy in BULBS:
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    bulbmap.append((.55 * np.exp(-d2 / (2 * 30 ** 2)) + .30 * np.exp(-d2 / (2 * 150 ** 2)) + .10 * np.exp(-d2 / (2 * 340 ** 2))).astype(np.float32))
bulb_sel = [(bm * ((xx < 740) if i == 0 else ((xx >= 740) & (xx < 1215)) if i == 1 else (xx >= 1215))).astype(np.float32)[..., None] for i in range(3)]
LAMP_COL = bgr((255, 205, 125)) / 255.0
FAIRY_COL = bgr((255, 218, 150))
DIPS = [(3.6, 0), (9.8, 2), (15.2, 1), (21.4, 0), (26.5, 2)]
def flick(t, k):
    ph = k * 1.7
    n = .5 * math.sin(2 * math.pi * .55 * t + ph) + .3 * math.sin(2 * math.pi * 1.37 * t + 2 * ph) + .2 * math.sin(2 * math.pi * 3.1 * t + 3 * ph)
    f = 1 + .07 * n
    for c, kk in DIPS:
        if kk == k:
            f *= 1 - .22 * math.exp(-((t - c) / .09) ** 2)
            f += .07 * math.exp(-((t - c - .22) / .12) ** 2)
    return f

def gk(sig, r):
    a = np.arange(-r, r + 1, dtype=np.float32)
    return np.exp(-(a[None, :] ** 2 + a[:, None] ** 2) / (2 * sig * sig))
FK_CORE = gk(3.0, 24); FK_HALO = gk(9.0, 24)
frng = np.random.default_rng(5)
fphase = frng.uniform(0, 6.28, len(fairy)); ffreq = frng.uniform(.35, .85, len(fairy))
def splat(canvas, k, x, y, amp, r):
    x = int(round(x)); y = int(round(y))
    x0, y0 = x - r, y - r
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(2 * r + 1, canvas.shape[1] - x0), min(2 * r + 1, canvas.shape[0] - y0)
    if ex <= sx0 or ey <= sy0: return
    canvas[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex] += amp * k[sy0:ey, sx0:ex]

# outside lamp posts
LP = [(612, 262, 1.0), (838, 392, .55)]
lpmap = sum(s * (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 14 ** 2)) * .7 + np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 60 ** 2)) * .35) for cx, cy, s in LP).astype(np.float32) * wm[..., 0]
# sun glow in sky
sunmask = wm[..., 0]

# ---- steam ----
SR = np.random.default_rng(11)
NS = 400
s_x0 = 905 + SR.normal(0, 7, NS); s_ph = SR.uniform(0, 6.28, NS); s_v = 34 * (.85 + .3 * SR.random(NS)); s_ph2 = SR.uniform(0, 6.28, NS)
STEAM_DT = .09; STEAM_L = 4.6
SK = {}
def skern(sig):
    s = int(round(sig * 2)) / 2
    if s not in SK:
        r = int(s * 3) + 1; a = np.arange(-r, r + 1, dtype=np.float32)
        SK[s] = (np.exp(-(a[None, :] ** 2 + a[:, None] ** 2) / (2 * s * s)), r)
    return SK[s]
SX0, SY0, SX1, SY1 = 800, 600, 1120, 910
def steam_alpha(t):
    cv = np.zeros((SY1 - SY0, SX1 - SX0), np.float32)
    k0 = int(math.floor((t - STEAM_L) / STEAM_DT)); k1 = int(math.floor(t / STEAM_DT))
    for k in range(k0, k1 + 1):
        age = t - k * STEAM_DT
        if age < 0 or age > STEAM_L: continue
        i = k % NS; u = age / STEAM_L
        kt = k * STEAM_DT
        phk = 1.1 * math.sin(2 * math.pi * kt / 5.3) + .9 * math.sin(2 * math.pi * kt / 2.9 + 1.3)
        amp = 3 + 30 * u ** 1.2
        x = s_x0[i] * .0 + 905 + (s_x0[i] - 905) * .3 + amp * math.sin(2 * math.pi * age / 2.4 + phk) + 12 * u
        y = 892 - s_v[i] * age * (1 + .05 * age)
        sig = 3.2 + 11 * u ** .9
        a = .16 * (math.sin(math.pi * u ** .7) ** 1.3) * min(1.0, age / .35)
        ker, r = skern(sig)
        splat(cv, ker, x - SX0, y - SY0, a, r)
    return np.clip(cv, 0, .8)

# ---- leaves ----
LR = np.random.default_rng(21)
NL = 16
l_x0 = LR.uniform(360, 1590, NL); l_y0 = LR.uniform(0, 730, NL); l_v = LR.uniform(22, 44, NL)
l_f = LR.uniform(.2, .5, NL); l_ph = LR.uniform(0, 6.28, NL); l_sz = LR.uniform(6.5, 12, NL); l_spin = LR.uniform(-40, 40, NL)
l_col = [(232, 110, 40), (210, 70, 35), (242, 172, 62), (190, 90, 40), (225, 130, 50)]
def draw_leaves(t):
    layer = np.zeros((H, W, 3), np.uint8); al = np.zeros((H, W), np.uint8)
    for j in range(NL):
        y = 100 + (l_y0[j] + l_v[j] * t) % 740
        x = l_x0[j] + 7 * t + 22 * math.sin(2 * math.pi * l_f[j] * t + l_ph[j])
        ang = 35 * math.sin(2 * math.pi * l_f[j] * t + l_ph[j] + 1.2) + l_spin[j] * t * .4
        a = l_sz[j]; c = l_col[j % len(l_col)]
        cv2.ellipse(layer, (int(x), int(y)), (int(a), max(2, int(a * .45))), ang, 0, 360, (c[2], c[1], c[0]), -1, cv2.LINE_AA)
        cv2.ellipse(al, (int(x), int(y)), (int(a), max(2, int(a * .45))), ang, 0, 360, 255, -1, cv2.LINE_AA)
    return layer, al

# ---- dust ----
DR = np.random.default_rng(31)
ND = 30
d_x = DR.uniform(300, 1700, ND); d_y = DR.uniform(160, 800, ND); d_ph = DR.uniform(0, 6.28, ND); d_per = DR.uniform(9, 17, ND)
DK = gk(1.7, 7)

vig = (1 - .24 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2)[..., None].astype(np.float32)
grng = np.random.default_rng(99)

def frame(fi):
    t = fi / FPS
    p = smooth(t / DUR)
    mW = bgr(keyinterp(WIN_MULT, p)); mI = bgr(keyinterp(INT_MULT, p))
    skyT = bgr(keyinterp(SKY_TOP, p)); skyH = bgr(keyinterp(SKY_HOR, p))
    # leaves into window layer
    layer, al = draw_leaves(t)
    la = (al.astype(np.float32) / 255.0)[..., None] * wm
    win = base * (1 - la) + layer.astype(np.float32) * la
    # sun glow: sinks and warms then fades
    sy = 300 + 240 * p
    sun_i = .95 * (math.sin(math.pi * min(1.0, p / .88)) ** 1.2)
    sunmap = gauss2(990, sy, 230, 130) * sunmask
    winG = win * mW + sky_top_map[..., None] * skyT + sky_hor_map[..., None] * skyH + sunmap[..., None] * sun_i * 85 * bgr((255, 160, 80)) / 255.0
    # outside lamp posts come on
    e = .10 + .95 * smooth((p - .35) / .6)
    winG += lpmap[..., None] * e * 150 * bgr((255, 200, 110)) / 255.0 * (1 + .03 * math.sin(2 * math.pi * 1.1 * t))
    intr = base * mI
    img = intr + wm * (winG - intr)
    # lamps
    L = .30 + .65 * smooth((p - .1) / .85)
    lm = np.zeros((H, W), np.float32)
    for k in range(3):
        fk = flick(t + k * .013, k)
        lm += bulbmap[k] * fk
        img *= 1 + (fk - 1) * .9 * bulb_sel[k]
    img += lm[..., None] * LAMP_COL * 255.0 * L * .55
    # fairy lights
    fc = np.zeros((H, W), np.float32)
    for i, (fx, fy) in enumerate(fairy):
        tw = .75 + .25 * math.sin(2 * math.pi * ffreq[i] * t + fphase[i])
        splat(fc, FK_CORE, fx, fy, 40 * tw, 24)
        splat(fc, FK_HALO, fx, fy, 14 * tw, 24)
    img += fc[..., None] * FAIRY_COL / 255.0 * (.7 + .7 * smooth(p))
    # steam
    a = steam_alpha(t)
    a = cv2.GaussianBlur(a, (0, 0), 2.0)
    sc = bgr(tuple(np.array((252, 246, 238)) * (1 - p) + np.array((226, 204, 190)) * p))
    roi = img[SY0:SY1, SX0:SX1]
    img[SY0:SY1, SX0:SX1] = roi * (1 - a[..., None]) + sc * a[..., None]
    # dust
    dc = np.zeros((H, W), np.float32)
    for j in range(ND):
        x = d_x[j] + 16 * math.sin(2 * math.pi * t / d_per[j] + d_ph[j]) + 3 * t
        y = d_y[j] + 11 * math.cos(2 * math.pi * t / (d_per[j] * 1.3) + d_ph[j]) - 2.2 * t
        b = .5 + .5 * math.sin(2 * math.pi * t / (d_per[j] * .6) + d_ph[j] * 2)
        splat(dc, DK, x, y, 55 * b, 7)
    img += dc[..., None] * bgr((255, 232, 195)) / 255.0
    img *= vig
    # camera drift
    z = 1 + .03 * (t / DUR)
    sc_ = 0.9655 * z
    cx = W / 2 + 14 * math.sin(2 * math.pi * t / 40) + 10 * (t / DUR)
    cy = H / 2 + 6 * math.cos(2 * math.pi * t / 33) - 5 * (t / DUR)
    M = np.array([[sc_, 0, OW / 2 - sc_ * cx], [0, sc_, OH / 2 - sc_ * cy]], np.float32)
    out = cv2.warpAffine(np.clip(img, 0, 255).astype(np.uint8), M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    n = grng.normal(0, 1.6, (OH, OW, 1)).astype(np.float32)
    return np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    if sys.argv[1] == 'still':
        import time
        for f in map(float, sys.argv[2:]):
            t0 = time.time(); fr = frame(int(f * FPS)); print(f, round(time.time() - t0, 2), 's')
            cv2.imwrite(f'still_{int(f)}.png', fr)
    else:
        a, b, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        pr = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for fi in range(a, b):
            pr.stdin.write(frame(fi).tobytes())
        pr.stdin.close(); pr.wait()
