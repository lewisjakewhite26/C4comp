import math, numpy as np
from pose import *
def sn(t, per, ph=0.0): return math.sin(2*math.pi*(t/per+ph))
def sm(x):
    x = min(1.0, max(0.0, x)); return x*x*(3-2*x)
def mixd(a, b, k):   # blend two result tuples (rots, dx, jump, yaw)
    ra, rb = a[0], b[0]; keys = set(ra)|set(rb); r = {}
    for n in keys:
        x = ra.get(n, (0, 0, 0)); y = rb.get(n, (0, 0, 0)); r[n] = tuple(x[i]*(1-k)+y[i]*k for i in range(3))
    return r, a[1]*(1-k)+b[1]*k, a[2]*(1-k)+b[2]*k, a[3]*(1-k)+b[3]*k
def root(r, tilt=0, twist=0, pitch=0): r['root'] = (pitch, twist, tilt)

def rest(t=0):
    r = {}; arm(r, 'SL', 8); arm(r, 'SR', 8); return r, 0, 0, 0
def m_reach(t):
    u = 0.5-0.5*math.cos(2*math.pi*t/3.0); r = {}
    a = 8+148*u; arm(r, 'SL', a); arm(r, 'SR', a); neck(r, nod=-8*u); return r, 0, 0, 0
def m_shake(t):
    f = sn(t, 0.30); r = {}
    arm(r, 'SL', 35+4*f, fwd=30, el_x=60+45*f); arm(r, 'SR', 35-4*f, fwd=30, el_x=60-45*f)
    return r, 3*f, 3*abs(f), 0
def m_wiggle(t):
    a = sn(t, 1.0); r = {}
    root(r, tilt=7*a, twist=14*sn(t, 1.0, .25)); spine(r, lean=-9*a, twist=-8*sn(t, 1.0, .25))
    arm(r, 'SL', 35+30*a, el_x=20); arm(r, 'SR', 35-30*a, el_x=20)
    leg(r, 'SL', 4+6*max(0, a), knee=0); leg(r, 'SR', 4+6*max(0, -a))
    return r, 40*a, 4*abs(sn(t, 0.5)), 0
def m_roll(t):
    ph = 2*math.pi*t/2.6; r = {}
    for side, sg in (('L', 1), ('R', -1)):
        r[f'clavicle.{side}'] = (14*math.sin(ph+(0 if side == 'L' else math.pi)), 0, sg*16*math.cos(ph+(0 if side == 'L' else math.pi)))
    arm(r, 'SL', 8); arm(r, 'SR', 8); neck(r, tilt=4*math.sin(ph)); return r, 0, 0, 0
def m_march(t):
    a = sn(t, 1.0); lf, rf = max(0, a), max(0, -a); r = {}
    leg(r, 'SL', 4, fwd=80*lf, knee=85*lf); leg(r, 'SR', 4, fwd=80*rf, knee=85*rf)
    arm(r, 'SL', 12, fwd=-35*a, el_x=60+30*lf); arm(r, 'SR', 12, fwd=35*a, el_x=60+30*rf)
    return r, 8*a, 0, 0
def m_stamp(t):
    c = t % 1.0
    def fl(c0):
        d = (c-c0) % 1.0
        if d < 0.40: return sm(d/0.35)
        if d < 0.48: return max(0.0, 1-(d-0.40)/0.08)
        return 0.0
    lf, rf = fl(0.0), fl(0.5); r = {}
    leg(r, 'SL', 4, fwd=85*lf, knee=80*lf); leg(r, 'SR', 4, fwd=85*rf, knee=80*rf)
    arm(r, 'SL', 25+25*rf, el_x=70*rf); arm(r, 'SR', 25+25*lf, el_x=70*lf)
    return r, 12*(rf-lf), 0, 0
def m_clap(t):
    cyc = t % 2.0
    h = sm(cyc/0.2)*(1-sm((cyc-1.0)/0.2)) if cyc < 1.2 else 0.0
    o = 0.5+0.5*math.cos(2*math.pi*t/0.5)
    r = {}
    hi = 168-26*o; lo_abd = 18+8*o
    # high claps: arms overhead; low: hands together in front of the belly
    ah = hi*h+lo_abd*(1-h)
    arm(r, 'SL', ah, fwd=(1-h)*40, el_x=(1-h)*(95-25*o), el_z=0)
    arm(r, 'SR', ah, fwd=(1-h)*40, el_x=(1-h)*(95-25*o))
    return r, 0, 0, 0
def m_frog(t):
    c = t % 1.6
    if c < 0.55: q, j, up = 1, 0, 0
    elif c < 0.9: k = (c-0.55)/0.35; q, j, up = 1-sm(k), 0, sm(k)
    elif c < 1.2: k = (c-0.9)/0.3; q, j, up = 0, 42*math.sin(math.pi*k), 1
    else: k = (c-1.2)/0.4; q, j, up = sm(k), 0, 1-sm(k)
    r = {}
    for s in ('SL', 'SR'): leg(r, s, 38*q, fwd=70*q, knee=120*q)
    spine(r, fwd=25*q)
    for s in ('SL', 'SR'): arm(r, s, lerp_(30, 150, up), fwd=40*q*(1-up), el_x=30*q*(1-up))
    return r, 0, j, 0
def lerp_(a, b, k): return a+(b-a)*k
def m_star(t):
    c = t % 1.0; o = math.sin(math.pi*c)**2; r = {}
    arm(r, 'SL', 8+148*o); arm(r, 'SR', 8+148*o)
    leg(r, 'SL', 3+30*o); leg(r, 'SR', 3+30*o)
    return r, 0, 30*o, 0
def m_run(t):
    a = sn(t, 0.5); lf, rf = max(0, a), max(0, -a); r = {}
    leg(r, 'SL', 3, fwd=60*lf, knee=100*lf); leg(r, 'SR', 3, fwd=60*rf, knee=100*rf)
    arm(r, 'SL', 20, fwd=-45*a, el_x=95); arm(r, 'SR', 20, fwd=45*a, el_x=95)
    spine(r, fwd=8); return r, 6*a, 10*abs(a), 0
def m_giant(t):
    a = sn(t, 2.4); la, ra = max(0, a), max(0, -a); r = {}
    leg(r, 'SL', 5+34*la); leg(r, 'SR', 5+34*ra)
    spine(r, lean=-8*a); arm(r, 'SL', 50, el_x=10); arm(r, 'SR', 50, el_x=10)
    return r, -70*a, 0, 0
def m_hop(t):
    left = (t % 4.0) < 2.0; j = 28*abs(math.sin(2*math.pi*t/1.0*0.5)); r = {}
    up, st = ('SR', 'SL') if left else ('SL', 'SR')
    leg(r, up, 6, fwd=55, knee=100); leg(r, st, 3)
    arm(r, 'SL', 55, el_x=20); arm(r, 'SR', 55, el_x=20)
    return r, (-6 if left else 6), j, 0
def m_punch(t):
    a = sn(t, 1.2); k = 0.5+0.5*math.tanh(3*a); r = {}
    arm(r, 'SL', lerp_(40, 150, k), fwd=0, el_z=lerp_(150, 0, k) if False else 0)
    arm(r, 'SL', lerp_(35, 140, k), el_x=lerp_(100, 5, k)); arm(r, 'SR', lerp_(140, 35, k), el_x=lerp_(5, 100, k))
    spine(r, lean=-8*(2*k-1), twist=10*(2*k-1)); leg(r, 'SL', 14); leg(r, 'SR', 14)
    return r, 0, 0, 0
def m_squash(t):
    c = t % 4.0
    if c < 1.6: q = sm(c/1.6)
    elif c < 2.0: q = 1
    elif c < 3.6: q = 1-sm((c-2.0)/1.6)
    else: q = 0
    tall = 1-q if c > 2.0 else 0; r = {}
    for s in ('SL', 'SR'):
        leg(r, s, 30*q, fwd=85*q, knee=135*q)
        arm(r, s, lerp_(8, 25, q)+ (150-8)*tall*sm((c-2.0)/1.0), fwd=40*q, el_x=110*q)
    spine(r, fwd=30*q)
    return r, 0, 0, 0
def m_ninja(t):
    a = sn(t, 1.8); lf, rf = max(0, a), max(0, -a); r = {}
    leg(r, 'SL', 14, fwd=40+25*lf, knee=70+20*lf); leg(r, 'SR', 14, fwd=40+25*rf, knee=70+20*rf)
    spine(r, fwd=18, lean=4*a)
    arm(r, 'SL', 70, el_z=100, el_x=0); arm(r, 'SR', 70, el_z=100)
    return r, 22*sn(t, 3.6), 0, 0
def m_spin(t):
    ph = 360*t/4.0; r = {}
    arm(r, 'SL', 90); arm(r, 'SR', 90); root(r, twist=ph)
    return r, 0, 0, 0
def m_balance(t):
    rL = sm(t/1.2)*(1-sm((t-6.3)/0.8)); rR = sm((t-7.0)/1.2); arms = sm((t-1.0)/1.0); r = {}
    leg(r, 'SL', 70*rL, fwd=45*rL, knee=125*rL); leg(r, 'SR', 70*rR, fwd=45*rR, knee=125*rR)
    a = lerp_(8, 150, arms); arm(r, 'SL', a); arm(r, 'SR', a)
    spine(r, lean=2*math.sin(t*2.4)); return r, -14*rL+14*rR, 0, 0
def m_circle(t):
    ph = 2*math.pi*t/3.0; r = {}
    abd = 95+50*math.sin(ph); fw = 40*math.cos(ph)
    arm(r, 'SL', abd, fwd=fw); arm(r, 'SR', abd, fwd=fw); return r, 0, 0, 0
def m_breath(t):
    c = t % 8.0; b = sm(c/4.0) if c < 4 else 1-sm((c-4.0)/4.0); r = {}
    a = 8+148*b; arm(r, 'SL', a); arm(r, 'SR', a); spine(r, fwd=-6*b); return r, 0, 0, 0
def m_shakeout(t):
    f1, f2, f3 = sn(t, 0.27), sn(t, 0.33, .3), sn(t, 0.41, .6); r = {}
    arm(r, 'SL', 45+40*f1, el_x=50+40*f2); arm(r, 'SR', 45-40*f3, el_x=50-40*f1)
    leg(r, 'SL', 6+8*f2, fwd=15+15*f1, knee=20+15*f1); leg(r, 'SR', 6-6*f3, fwd=15-15*f2, knee=20-15*f3)
    neck(r, tilt=10*f3, turn=15*f2); root(r, tilt=4*f1)
    return r, 14*f1, 8*abs(f3), 0
def m_cheer(t):
    c = t % 1.0; o = math.sin(math.pi*c); r = {}
    arm(r, 'SL', 142+10*o); arm(r, 'SR', 142+10*o); leg(r, 'SL', 6+10*o); leg(r, 'SR', 6+10*o)
    return r, 0, 26*o, 0
POSES = [m_reach, m_shake, m_wiggle, m_roll, m_march, m_stamp, m_clap, m_frog, m_star, m_run, m_giant, m_hop, m_punch, m_squash, m_ninja, m_spin, m_balance, m_circle, m_breath, m_shakeout]
