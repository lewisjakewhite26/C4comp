"""Zonelytime immersive workout - Level 1 (KS1). Teal line-art figure on black.
Usage: python3 wo.py seg <idx> <out.mp4>   (idx 0=intro, 1..20 moves, 21=outro)
       python3 wo.py test <out.png>        (contact sheet)
"""
import numpy as np, cv2, math, sys, subprocess, random
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
TEAL = (212, 227, 127)      # BGR for #7FE3D4
DIM = (110, 118, 66)
TXT = (245, 255, 235)       # BGR near-white teal
S = 0.85
TORSO, NECK, HR, SH, HH = 170*S, 18*S, 46*S, 62*S, 38*S
UA, FA, TH, SHN = 105*S, 100*S, 135*S, 135*S
GROUND, CX = 935, 960
LEVEL = "LEVEL 1"
INTRO, MOVE, OUTRO = 8.0, 14.0, 12.0

def sn(t, per, ph=0.0): return math.sin(2*math.pi*(t/per+ph))
def sm(x):
    x = min(1.0, max(0.0, x)); return x*x*(3-2*x)
def lerp(a, b, k): return a + (b-a)*k

# ---------------- pose ----------------
REST_A = (10, 1, 6, 1)
REST_L = (6, 1, 2, 1)
def P(dx=0, jump=0, lean=0, ws=1, sh=(0, 0, 0, 0), aL=REST_A, aR=None, lL=REST_L, lR=None):
    aR = aL if aR is None else aR
    lR = lL if lR is None else lR
    return np.array([dx, jump, lean, ws, *sh, *aL, *aR, *lL, *lR], dtype=float)
REST = P()
def mix(p, q, k): return p*(1-k)+q*k
def T(a, b, k): return tuple(lerp(x, y, k) for x, y in zip(a, b))

def m_reach(t):
    u = sm(0.5-0.5*math.cos(2*math.pi*t/3.0)*1.0)
    u = 0.5-0.5*math.cos(2*math.pi*t/3.0)
    a = (12+168*u*0.97, 1, 12+168*u*0.97, 1)
    return P(jump=16*u, lean=0.04*sn(t, 6), aL=a, lL=(6, 1, 2, 1))
def m_shake(t):
    f = sn(t, 0.30)
    aL = (35+4*f, 1, 45+55*f, 1); aR = (35-4*f, 1, 45-55*f, 1)
    return P(dx=3*f, aL=aL, aR=aR, jump=3*abs(f))
def m_wiggle(t):
    a = sn(t, 1.0)
    return P(dx=45*a, lean=0.14*sn(t, 1.0, .25), aL=(40+35*a, 1, 40+45*a, 1), aR=(40-35*a, 1, 40-45*a, 1),
             lL=(8+14*max(0, a), 1, 3, 1), lR=(8+14*max(0, -a), 1, 3, 1), jump=4*abs(sn(t, 0.5)))
def m_roll(t):
    ph = 2*math.pi*t/2.6
    return P(dx=6*math.sin(ph), sh=(10*math.cos(ph), -18*math.sin(ph), -10*math.cos(ph), 18*math.sin(ph)),
             aL=(14, 1, 10, 1), lean=0.03*math.sin(ph))
def _lift(c):  # smooth 0..1..0 within cycle c in [0,1)
    return sm(c/0.25) if c < 0.5 else 1-sm((c-0.5)/0.25) if c < 0.75 else 0
def m_march(t):
    a = sn(t, 1.0)
    lf, rf = max(0, a), max(0, -a)
    return P(jump=0, lL=(5, 1-0.58*lf, 3, 1), lR=(5, 1-0.58*rf, 3, 1),
             aL=(18, 1-0.3*rf, 20+60*rf, 1-0.3*rf), aR=(18, 1-0.3*lf, 20+60*lf, 1-0.3*lf),
             lean=0.04*a, dx=10*a)
def m_stamp(t):
    c = (t % 1.0)
    def fl(c0):
        d = (c-c0) % 1.0
        if d < 0.40: return sm(d/0.35)
        if d < 0.48: return max(0.0, 1-(d-0.40)/0.08)
        return 0.0
    lf, rf = fl(0.0), fl(0.5)
    return P(lL=(5, 1-0.62*lf, 3, 1), lR=(5, 1-0.62*rf, 3, 1),
             aL=(22+30*rf, 1, 40+50*rf, 1), aR=(22+30*lf, 1, 40+50*lf, 1),
             dx=14*(rf-lf), jump=0)
def m_clap(t):
    h = sm(((t % 2.0)-0.0)/0.01) if False else (1.0 if (t % 2.0) < 1.0 else 0.0)
    cyc = t % 2.0
    h = sm(cyc/0.2) * (1-sm((cyc-1.0)/0.2)) if cyc < 1.2 else 0.0
    o = 0.5+0.5*math.cos(2*math.pi*t/0.5)
    hi = T((163, 1, 163, 1), (130, 1, 135, 1), o)
    lo = T((22, 1, -38, 1), (30, 1, 40, 1), o)
    return P(aL=T(lo, hi, h), jump=0, lean=0)
def m_frog(t):
    c = t % 1.6
    if c < 0.55: q, j, up = 1, 0, 0
    elif c < 0.9: k = (c-0.55)/0.35; q, j, up = 1-sm(k), 0, sm(k)
    elif c < 1.2: k = (c-0.9)/0.3; q, j, up = 0, 85*math.sin(math.pi*k), 1
    else: k = (c-1.2)/0.4; q, j, up = sm(k), 0, 1-sm(k)
    legs = T((7, 1, 3, 1), (74, 0.95, 2, 1), q)
    arms = T((28, 1, 12, 1), (168, 1, 168, 1), up)
    if q == 0 and up == 1: legs = (7, 1, 3, 1)
    return P(jump=j, lL=legs, aL=T((26, 1, 20, 1), arms, 1.0 if up > 0.01 else 0))
def m_star(t):
    c = t % 1.0
    o = math.sin(math.pi*c)**2
    return P(jump=60*o, aL=(12+155*o, 1, 12+155*o, 1), lL=(6+30*o, 1, 4+30*o, 1))
def m_run(t):
    a = sn(t, 0.5)
    lf, rf = max(0, a), max(0, -a)
    return P(jump=10*abs(a), lean=0.05, lL=(4, 1-0.6*lf, 2, 1), lR=(4, 1-0.6*rf, 2, 1),
             aL=(30, 1-0.4*rf, 100, 1-0.45*rf), aR=(30, 1-0.4*lf, 100, 1-0.45*lf), dx=6*a)
def m_giant(t):
    a = sn(t, 2.4)
    la, ra = max(0, a), max(0, -a)
    return P(dx=-70*a, lean=0.12*a, aL=(50, 1, 60, 1),
             lL=(6+42*la, 1, 4+10*la, 1), lR=(6+42*ra, 1, 4+10*ra, 1))
def m_hop(t):
    left = (t % 4.0) < 2.0
    j = 38*abs(math.sin(math.pi*t*2.0/1.0*0.5*2/2*1.0*1.0))
    j = 38*abs(math.sin(2*math.pi*t/1.0*0.5))
    raised = (5, 0.5, -8, 1)
    sw = 0.0
    c = t % 2.0
    if c < 0.3: sw = sm(c/0.3)
    stand = (4, 1, 2, 1)
    if left:  # hop on left, right leg raised
        return P(jump=j, lL=stand, lR=raised, aL=(55, 1, 60, 1), dx=-6)
    return P(jump=j, lR=stand, lL=raised, aL=(55, 1, 60, 1), dx=6)
def m_punch(t):
    a = sn(t, 1.2)
    k = 0.5+0.5*math.tanh(3*a)
    ext = (125, 1, 128, 1); guard = (22, 1, 160, 1)
    return P(aL=T(guard, ext, k), aR=T(ext, guard, k), lean=0.08*(2*k-1),
             lL=(14, 1, 4, 1), jump=0)
def m_squash(t):
    c = t % 4.0
    if c < 1.6: q = sm(c/1.6)
    elif c < 2.0: q = 1
    elif c < 3.6: q = 1-sm((c-2.0)/1.6)
    else: q = 0
    tall = 1-q if c > 2.0 else 0
    legs = T((6, 1, 2, 1), (46, 0.9, 2, 0.95), q)
    arms = T((12, 1, 8, 1), (28, 1, 150, 1), q)
    if c >= 2.0:
        arms = T(arms, (165, 1, 168, 1), tall*sm((c-2.0)/1.0))
    return P(lL=legs, aL=arms, jump=18*tall*sm((c-2.8)/0.6) if c > 2.8 and c < 3.6 else 0)
def m_ninja(t):
    a = sn(t, 1.8)
    lf, rf = max(0, a), max(0, -a)
    return P(dx=22*sn(t, 3.6), lean=0.05*a,
             lL=(34, 0.85-0.15*lf, 4, 1-0.3*lf), lR=(34, 0.85-0.15*rf, 4, 1-0.3*rf),
             aL=(40, 1, 140, 1))
def m_spin(t):
    ph = 2*math.pi*t/4.0
    ws = max(0.3, abs(math.cos(ph)))
    return P(ws=ws, aL=(90, 1, 90, 1), lL=(8, 1, 3, 1), jump=0)
def m_balance(t):
    rL = sm(t/1.2)*(1-sm((t-6.3)/0.8)); rR = sm((t-7.0)/1.2)
    arms = sm((t-1.0)/1.0)
    w = 0.015*math.sin(t*2.4)
    raised = lambda r: T((4, 1, 2, 1), (80, 0.97, -86, 1), r)
    ar = T(REST_A, (163, 1, 165, 1), arms)
    return P(lean=w, aL=ar, lL=raised(rL), lR=raised(rR), dx=-12*rL+12*rR)
def m_circle(t):
    ph = 2*math.pi*t/3.0
    th = 90+42*math.sin(ph); k = 0.82+0.18*math.cos(ph)
    return P(aL=(th, k, th, k), lL=(8, 1, 3, 1), lean=0.02*math.sin(ph))
def m_breath(t):
    c = t % 8.0
    b = sm(c/4.0) if c < 4 else 1-sm((c-4.0)/4.0)
    return P(aL=(12+152*b, 1, 12+156*b, 1), jump=16*b)
def m_shakeout(t):
    f1, f2, f3 = sn(t, 0.27), sn(t, 0.33, .3), sn(t, 0.41, .6)
    return P(dx=14*f1, lean=0.08*f2, aL=(45+40*f1, 1, 50+40*f2, 1), aR=(45-40*f3, 1, 50-40*f1, 1),
             lL=(8+14*f2, 1, 3, 1), lR=(8-10*f3, 1, 3, 1), jump=8*abs(f3))
def m_cheer(t):
    c = t % 1.0
    o = math.sin(math.pi*c)
    return P(jump=45*o, aL=(150+10*o, 1, 160, 1), lL=(8+12*o, 1, 3, 1))

# ---------------- moves table ----------------
# (title, cue, pose, effects)
MOVES = [
 ("REACH UP HIGH!", "Stretch your fingers to the ceiling", m_reach, ["rise"]),
 ("SHAKE YOUR HANDS!", "Fast and floppy!", m_shake, ["spark"]),
 ("WIGGLE YOUR BODY!", "Wiggle everything!", m_wiggle, ["spark"]),
 ("ROLL YOUR SHOULDERS!", "Round and round, nice and slow", m_roll, ["calm"]),
 ("MARCH ON THE SPOT!", "Lift those knees!", m_march, ["beat"]),
 ("STAMP YOUR FEET!", "Stomp, stomp, stomp!", m_stamp, ["stamp"]),
 ("CLAP HIGH, CLAP LOW!", "Two claps up, two claps down", m_clap, ["clap"]),
 ("JUMP LIKE A FROG!", "Crouch low, then leap!", m_frog, ["land"]),
 ("STAR JUMPS!", "Arms and legs wide!", m_star, ["spark", "land2"]),
 ("RUN ON THE SPOT!", "Quick feet!", m_run, ["speed"]),
 ("TAKE GIANT STEPS!", "Big, big steps side to side", m_giant, ["beat"]),
 ("HOP, HOP, HOP!", "Left foot, then right foot", m_hop, ["land3"]),
 ("PUNCH THE AIR!", "Left, right, left, right!", m_punch, ["speed"]),
 ("SQUASH DOWN, GROW TALL!", "Tiny like a mouse, tall like a giraffe", m_squash, ["rise"]),
 ("TIPTOE LIKE A NINJA!", "Quiet, slow and sneaky", m_ninja, ["calm"]),
 ("SPIN SLOWLY ROUND!", "Turn round and round, nice and steady", m_spin, ["orbit"]),
 ("BALANCE ON ONE LEG!", "Stand still like a tree", m_balance, ["breath"]),
 ("BIG SLOW ARM CIRCLES!", "Nice and smooth", m_circle, ["calm"]),
 ("BIG BREATHS", "", m_breath, ["breath"]),
 ("SHAKE IT ALL OUT!", "Shake your arms, legs and head!", m_shakeout, ["spark"]),
]
SECTIONS = {1: "WAKE UP", 5: "GET MOVING", 13: "POWER UP", 17: "COOL DOWN", 20: "SHAKE IT OUT"}
ENCOURAGE = ["KEEP GOING!", "BRILLIANT!", "YOU'RE DOING GREAT!", "NICE ONE!", "FANTASTIC!", "WELL DONE!"]

def section_of(i):
    s = "WAKE UP"
    for k in sorted(SECTIONS):
        if i >= k: s = SECTIONS[k]
    return s

def cue_for(i, t):
    title, cue, *_ = MOVES[i-1]
    if i == 19:
        return "BREATHE IN..." if (t % 8.0) < 4 else "BREATHE OUT..."
    if i == 17 and t > 7.0: return "Now the other leg!"
    if t > MOVE-4.2: return ENCOURAGE[(i*3) % len(ENCOURAGE)]
    return cue

# ---------------- skeleton ----------------
def vec(theta, length, s, ws):
    a = math.radians(theta)
    return np.array([s*math.sin(a)*length*ws, math.cos(a)*length])

def fk(p):
    dx, jump, lean, ws, sdLx, sdLy, sdRx, sdRy = p[:8]
    arms = {-1: p[8:12], 1: p[12:16]}; legs = {-1: p[16:20], 1: p[20:24]}
    up = np.array([math.sin(lean), -math.cos(lean)]); perp = np.array([math.cos(lean), math.sin(lean)])
    hc = np.array([CX+dx, 0.0]); sc = hc+up*TORSO
    sd = {-1: np.array([sdLx, sdLy]), 1: np.array([sdRx, sdRy])}
    J = {'hc': hc, 'sc': sc}
    for s in (-1, 1):
        sh = sc+s*perp*np.array([SH*ws, SH])+sd[s]
        hp = hc+s*perp*np.array([HH*ws, HH])
        a = arms[s]; l = legs[s]
        el = sh+vec(a[0], UA, s, ws)*a[1]; ha = el+vec(a[2], FA, s, ws)*a[3]
        kn = hp+vec(l[0], TH, s, ws)*l[1]; ft = kn+vec(l[2], SHN, s, ws)*l[3]
        J.update({f'sh{s}': sh, f'hp{s}': hp, f'el{s}': el, f'ha{s}': ha, f'kn{s}': kn, f'ft{s}': ft})
    J['head'] = sc+up*(NECK+HR)
    off = GROUND-max(J['ft-1'][1], J['ft1'][1])-jump
    for k in J: J[k] = J[k]+np.array([0, off])
    J['ws'] = ws
    return J

def IP(v): return (int(round(v[0])), int(round(v[1])))

def draw_fig(L, J, col=TEAL, th=15):
    def seg(a, b, t=th):
        cv2.line(L, IP(a), IP(b), col, t, cv2.LINE_AA)
        cv2.circle(L, IP(a), t//2, col, -1, cv2.LINE_AA); cv2.circle(L, IP(b), t//2, col, -1, cv2.LINE_AA)
    seg(J['hc'], J['sc'], th+2); seg(J['sh-1'], J['sh1']); seg(J['hp-1'], J['hp1'])
    for s in (-1, 1):
        seg(J[f'sh{s}'], J[f'el{s}']); seg(J[f'el{s}'], J[f'ha{s}'])
        seg(J[f'hp{s}'], J[f'kn{s}']); seg(J[f'kn{s}'], J[f'ft{s}'])
        seg(J[f'ft{s}'], J[f'ft{s}']+np.array([s*26*S*max(0.3, J['ws']), 0]), th-3)
        cv2.circle(L, IP(J[f'ha{s}']), 10, col, -1, cv2.LINE_AA)
    hd = J['head']
    cv2.circle(L, IP(hd), int(HR), col, th-3, cv2.LINE_AA)
    ex = 15*max(0.35, J['ws'])
    for sx in (-1, 1): cv2.circle(L, IP(hd+np.array([sx*ex, -5])), 4, col, -1, cv2.LINE_AA)
    cv2.ellipse(L, IP(hd+np.array([0, 8])), (int(15*max(0.4, J['ws'])), 9), 0, 25, 155, col, 4, cv2.LINE_AA)

def scale(c, k): return tuple(int(x*max(0, min(1, k))) for x in c)

# ---------------- effects ----------------
def ring(L, c, age, dur=0.5, r0=24, r1=150, col=TEAL):
    if not (0 <= age < dur): return
    k = age/dur
    cv2.circle(L, IP(c), int(lerp(r0, r1, 1-(1-k)**2)), scale(col, 1-k), 5, cv2.LINE_AA)
    for a in range(0, 360, 45):
        r = lerp(r0+10, r1*0.9, 1-(1-k)**2)
        d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        cv2.line(L, IP(c+d*r), IP(c+d*(r+22*(1-k))), scale(col, 1-k), 5, cv2.LINE_AA)

def events(t, per, off):  # age since last event with period `per`, offset `off`
    return (t-off) % per

def draw_fx(L, fx, t, J, rng):
    for f in fx:
        if f == 'rise':
            for side in (-1, 1):
                for i in range(3):
                    ph = ((t/2.4)+i/3.0) % 1.0
                    y = 820-ph*520; x = CX+side*(380+30*i)
                    c = scale(DIM, math.sin(math.pi*ph))
                    pts = np.array([[x-34, y+16], [x, y-16], [x+34, y+16]], np.int32)
                    cv2.polylines(L, [pts], False, c, 8, cv2.LINE_AA)
        elif f == 'spark':
            r = random.Random(int(t*8))
            for i in range(14):
                x = r.randint(300, 1620); y = r.randint(330, 900)
                if abs(x-CX) < 170 and y > 330: continue
                a = 0.4+0.6*abs(math.sin(t*3+i))
                c = scale(DIM, a); n = r.randint(8, 16)
                cv2.line(L, (x-n, y), (x+n, y), c, 4, cv2.LINE_AA); cv2.line(L, (x, y-n), (x, y+n), c, 4, cv2.LINE_AA)
        elif f == 'speed':
            for i in range(9):
                ph = ((t*2.2)+i*0.37) % 1.0
                y = 420+((i*97) % 440); side = -1 if i % 2 else 1
                x = CX+side*(300+ph*420); n = 70+40*(i % 3)
                cv2.line(L, IP((x, y)), IP((x+side*n, y)), scale(DIM, math.sin(math.pi*ph)), 6, cv2.LINE_AA)
        elif f == 'calm':
            for i in range(2):
                ph = ((t/3.5)+i*0.5) % 1.0
                cv2.circle(L, IP((CX, GROUND-250)), int(260+260*ph), scale(DIM, (1-ph)*0.7), 3, cv2.LINE_AA)
        elif f == 'breath':
            if True:
                c = t % 8.0
                b = sm(c/4.0) if c < 4 else 1-sm((c-4.0)/4.0)
                cv2.circle(L, IP((CX, GROUND-270)), int(230+150*b), scale(DIM, 0.55+0.4*b), 5, cv2.LINE_AA)
                cv2.circle(L, IP((CX, GROUND-270)), int(200+120*b), scale(DIM, 0.25), 2, cv2.LINE_AA)
        elif f == 'orbit':
            ph = t*2*math.pi/4.0
            for i in range(14):
                a = ph*1.5+i*2*math.pi/14
                x = CX+430*math.cos(a); y = GROUND-12+30*math.sin(a)
                cv2.circle(L, IP((x, y)), 6, scale(DIM, 0.4+0.6*(math.sin(a)*0.5+0.5)), -1, cv2.LINE_AA)
        elif f == 'beat':
            ph = (t % 0.5)/0.5
            cv2.ellipse(L, IP((CX, GROUND+8)), (int(140+90*ph), int(18+12*ph)), 0, 0, 360, scale(DIM, 1-ph), 4, cv2.LINE_AA)
        elif f == 'stamp':
            for off, key in ((0.45, 'ft-1'), (0.95, 'ft1')):
                age = events(t, 1.0, off)
                ring(L, (J[key][0], GROUND), age, 0.45, 20, 150)
        elif f == 'clap':
            age = events(t, 0.5, 0.25)
            hy = (J['ha-1']+J['ha1'])/2
            ring(L, hy, age, 0.4, 20, 110)
        elif f in ('land', 'land2', 'land3'):
            per = {'land': 1.6, 'land2': 1.0, 'land3': 0.5}[f]
            off = {'land': 1.2, 'land2': 0.99, 'land3': 0.5}[f]
            age = events(t, per, off)
            ring(L, (J['hc'][0], GROUND), age, 0.45, 40, 200)
    return L

# ---------------- text ----------------
_cache = {}
def text_img(txt, size, maxw=1740, weight='ExtraBold'):
    key = (txt, size, weight)
    if key in _cache: return _cache[key]
    f = ImageFont.truetype(f'/usr/share/fonts/opentype/inter/Inter-{weight}.otf', size)
    while f.getlength(txt) > maxw and size > 30:
        size -= 4; f = ImageFont.truetype(f'/usr/share/fonts/opentype/inter/Inter-{weight}.otf', size)
    w = int(f.getlength(txt))+120; h = int(size*1.5)
    im = Image.new('L', (w, h), 0); ImageDraw.Draw(im).text((w//2, h//2), txt, font=f, fill=255, anchor='mm')
    m = np.asarray(im, np.float32)/255
    g = cv2.GaussianBlur(m, (0, 0), size*0.09)
    out = np.clip(m+0.55*g, 0, 1)
    _cache[key] = out
    return out

def put_text(frame, txt, size, cy, k=1.0, col=TXT, alpha=1.0, cx=CX, weight='ExtraBold', maxw=1740):
    if not txt: return
    m = text_img(txt, size, maxw, weight)
    if k != 1.0:
        m = cv2.resize(m, (max(2, int(m.shape[1]*k)), max(2, int(m.shape[0]*k))), interpolation=cv2.INTER_LINEAR)
    h, w = m.shape
    x0 = int(cx-w/2); y0 = int(cy-h/2)
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0+w), min(H, y0+h)
    if xb <= xa or yb <= ya: return
    sub = m[ya-y0:yb-y0, xa-x0:xb-x0, None]*alpha
    frame[ya:yb, xa:xb] += sub*np.array(col, np.float32)

def pop(x):  # ease-out-back 0..1
    x = min(1.0, max(0.0, x)); c1 = 1.70158; c3 = c1+1
    return 1+c3*(x-1)**3+c1*(x-1)**2

# ---------------- frame ----------------
def render_frame(kind, i, t):
    L = np.zeros((H, W, 3), np.uint8)
    rng = None
    if kind == 'intro':
        p = mix(REST, P(jump=4*abs(sn(t, 1.0))), 1.0); fx = ['calm'] if t < 4.5 else []
    elif kind == 'move':
        title, cue, fn, fx = MOVES[i-1]
        blend = sm((t-0.6)/1.2)
        p = mix(REST, fn(t), blend)
    else:
        title = "WELL DONE!"; fx = ['spark']
        p = mix(REST, m_cheer(t), sm((t-0.3)/0.8))
    J = fk(p)
    draw_fig(L, J)
    if kind == 'move' and i in (7,) and False: pass
    # ground
    cv2.line(L, (CX-420, GROUND+34), (CX+420, GROUND+34), DIM, 4, cv2.LINE_AA)
    draw_fx(L, fx, t, J, rng)
    # glow
    sm_ = cv2.resize(L, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(sm_, (0, 0), 3)*0.8+cv2.GaussianBlur(sm_, (0, 0), 10)*1.3
    frame = L.astype(np.float32)+cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    # text
    if kind == 'intro':
        if t < 4.5:
            put_text(frame, "GET READY!", 170, 130, pop((t-0.1)/0.5))
            put_text(frame, "Stand up and make some space", 60, 245, pop((t-0.4)/0.5), weight='SemiBold')
        else:
            n = int(t-4.5)  # 0:3 1:2 2:1 3:GO
            lab = ["3", "2", "1", "GO!"][min(3, n)]
            c = (t-4.5) - n
            put_text(frame, lab, 230, 175, pop(c/0.35)*(1 if lab != "GO!" else 1.0))
    elif kind == 'move':
        put_text(frame, title, 150, 140, pop((t-0.1)/0.5))
        cue_s = cue_for(i, t)
        ck = pop(((t % 1000) - 0.5)/0.5) if t < MOVE-4.2 else pop((t-(MOVE-4.2))/0.4)
        if i == 17 and t > 7.0: ck = pop((t-7.0)/0.4)
        if i == 19: ck = 1.0
        put_text(frame, cue_s, 58, 255, ck, col=(210, 235, 150), weight='SemiBold')
        # progress
        x0, x1, y = CX-300, CX+300, 1028
        cv2.line(frame, (x0, y), (x1, y), (60, 60, 40), 8, cv2.LINE_AA)
        cv2.line(frame, (x0, y), (int(x0+(x1-x0)*min(1, t/MOVE)), y), TEAL, 8, cv2.LINE_AA)
        put_text(frame, f"{i} / 20", 44, 1030, 1.0, col=(170, 190, 120), cx=1760, weight='SemiBold', maxw=300)
        put_text(frame, section_of(i), 40, 70, 1.0, col=(170, 190, 120), cx=190, weight='SemiBold', maxw=300) if False else None
    else:
        put_text(frame, "WELL DONE!", 220, 150, pop((t-0.2)/0.6))
        put_text(frame, "You are a superstar", 64, 275, pop((t-0.7)/0.5), col=(210, 235, 150), weight='SemiBold')
    if kind != 'outro' or True:
        put_text(frame, LEVEL, 34, 34, 1.0, col=(170, 190, 120), cx=1790, weight='Bold', maxw=300)
        put_text(frame, "ZONELYTIME", 36, 1030, 1.0, col=(120, 140, 85), cx=170, weight='Bold', maxw=400)
        if kind == 'move':
            put_text(frame, section_of(i), 34, 34, 1.0, col=(170, 190, 120), cx=190, weight='Bold', maxw=400)
    # fades
    dur = {'intro': INTRO, 'move': MOVE, 'outro': OUTRO}[kind]
    f = min(1.0, t/0.25) * (1.0 if kind != 'outro' else min(1.0, (dur-t)/1.0))
    if kind == 'move': f *= min(1.0, (dur-t)/0.2)
    if kind == 'intro': f *= min(1.0, (dur-t)/0.15)
    frame *= max(0.0, f)
    return np.clip(frame, 0, 255).astype(np.uint8)

def seg_info(idx):
    if idx == 0: return 'intro', 0, INTRO
    if idx == 21: return 'outro', 0, OUTRO
    return 'move', idx, MOVE

def render_seg(idx, out):
    kind, i, dur = seg_info(idx)
    n = int(dur*FPS)
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-maxrate', '3500k',
                          '-bufsize', '7000k', '-pix_fmt', 'yuv420p', '-g', '30', out], stdin=subprocess.PIPE)
    for k in range(n):
        p.stdin.write(render_frame(kind, i, k/FPS).tobytes())
    p.stdin.close(); p.wait()

def contact(out):
    tiles = []
    for idx in range(0, 22):
        kind, i, dur = seg_info(idx)
        t = 6.2 if idx == 0 else (3.0 + (0.35 if idx in (6, 7, 8, 9) else 0.1) if kind == 'move' else 3.0)
        if kind == 'move' and idx == 8: t = 4.2
        fr = render_frame(kind, i, t)
        tiles.append(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(tiles[r*4:(r+1)*4]) if len(tiles[r*4:(r+1)*4]) == 4 else np.hstack(tiles[r*4:]+[np.zeros_like(tiles[0])]*(4-len(tiles[r*4:]))) for r in range(6)]
    cv2.imwrite(out, np.vstack(rows))

if __name__ == '__main__':
    if sys.argv[1] == 'seg': render_seg(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'test': contact(sys.argv[2])
