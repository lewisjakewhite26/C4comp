import numpy as np, subprocess, sys, wave
SR = 44100; DUR = 300.0; N = int(SR*DUR)
BPM = 120; BEAT = 60/BPM
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)

def add(buf, x, t, g=1.0):
    i = int(t*SR)
    if i >= N or i < 0: return
    j = min(N, i+len(x)); buf[i:j] += x[:j-i]*g
def both(x, t, g=1.0, pan=0.0):
    add(L, x, t, g*(1-max(0, pan))); add(R, x, t, g*(1+min(0, pan)))
def tt(d): return np.arange(int(d*SR))/SR
def env(d, a=0.005, r=0.1):
    n = int(d*SR); e = np.ones(n); a_ = max(1, int(a*SR)); e[:a_] = np.linspace(0, 1, a_)
    r_ = max(1, int(r*SR)); e[-r_:] *= np.linspace(1, 0, r_); return e
def midi(n): return 440*2**((n-69)/12)

def kick():
    t = tt(0.28); f = 48+110*np.exp(-t*28); ph = 2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-t*11)
def hat(d=0.06):
    t = tt(d); x = rng.standard_normal(len(t)); x = np.diff(x, prepend=0)  # high-passed
    return x*np.exp(-t*70)*0.5
def clap():
    t = tt(0.18); x = rng.standard_normal(len(t))
    x = np.convolve(x, np.ones(6)/6, 'same'); return x*(np.exp(-t*30)+0.6*np.exp(-((t-0.012)*180)**2))
def pluck(f, d=0.35):
    t = tt(d); x = sum(np.sin(2*np.pi*f*k*t)/k**1.4 for k in (1, 2, 3, 4))
    return x*np.exp(-t*9)*env(d, 0.002, 0.05)
def bass(f, d=0.45):
    t = tt(d); x = np.tanh(2.2*np.sin(2*np.pi*f*t)+0.4*np.sin(2*np.pi*2*f*t)); return x*np.exp(-t*4.5)*env(d, 0.004, 0.08)
def pad(freqs, d):
    t = tt(d); x = sum(np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*f*1.003*t) for f in freqs)/ (len(freqs)*1.5)
    a = np.minimum(1, t/0.8)*np.minimum(1, (d-t)/0.8); return x*a
def pop(f0=420, f1=1000):
    t = tt(0.22); f = f0+(f1-f0)*np.minimum(1, t/0.12); ph = 2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-t*16)*0.9
def beep(f, d=0.22):
    t = tt(d); return np.sin(2*np.pi*f*t)*env(d, 0.004, 0.08)
def bell(f, d=1.8):
    t = tt(d); return (np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*f*2.76*t)*np.exp(-t*3)+0.3*np.sin(2*np.pi*f*5.4*t)*np.exp(-t*6))*np.exp(-t*2.2)
def whoosh(d=0.35):
    t = tt(d); x = rng.standard_normal(len(t)); x = np.convolve(x, np.ones(30)/30, 'same')*6
    return x*np.sin(np.pi*t/d)**2

CH = [(48, [60, 64, 67]), (45, [57, 60, 64]), (41, [57, 60, 65]), (43, [55, 59, 62])]   # C Am F G
ARP = [0, 1, 2, 1, 0, 1, 2, 3]
def chord_at(bar): return CH[bar % 4]

# mode per time: 'soft' 'full' 'power' 'calm'
def mode(t):
    if t < 8: return 'calm'
    if t < 64: return 'soft'
    if t < 176: return 'full'
    if t < 232: return 'power'
    if t < 274: return 'calm'
    if t < 288: return 'power'
    return 'power'
K = kick(); H = hat(); CL = clap()
nbars = int(DUR/(4*BEAT))
for bar in range(nbars):
    t0 = bar*4*BEAT
    m = mode(t0); root, tri = chord_at(bar)
    # pad always
    both(pad([midi(n) for n in tri]+[midi(root+12)], 4*BEAT+0.4), t0, 0.07 if m != 'calm' else 0.11)
    for b in range(16):  # sixteenth grid
        t = t0+b*BEAT/4
        if t >= DUR-1.5: break
        m = mode(t)
        step = b
        if m in ('soft', 'full', 'power'):
            if step % 4 == 0: both(K, t, 0.55 if m == 'soft' else 0.8)
            if m != 'soft' and step in (4, 12): both(CL, t, 0.25)
            if step % 4 == 2: both(H, t, 0.12 if m == 'soft' else 0.16, pan=0.2)
            if m == 'power' and step % 2 == 1: both(H, t, 0.07, pan=-0.2)
        if m in ('full', 'power') and step in (0, 6, 8, 14):
            both(bass(midi(root)), t, 0.3)
        # arp on eighths
        if step % 2 == 0:
            k = ARP[(step//2) % 8]
            n = ([*tri, tri[0]+12])[k]+(12 if m == 'power' and (step//2) % 8 >= 4 else 0)+12
            g = 0.10 if m in ('calm',) else 0.14
            if m == 'calm' and step % 4: continue
            both(pluck(midi(n)), t, g, pan=0.35*np.sin(t))
# sfx
for i in range(20):
    t0 = 8+14*i
    both(pop(), t0+0.1, 0.35); both(whoosh(), t0-0.05, 0.12)
# clap move (7): t0=8+14*6=92
for k in range(28):
    both(CL, 92+0.25+k*0.5+1.0*0, 0.5, 0)
# stamp move (6): t0=78
for k in range(14):
    for off in (0.45, 0.95):
        both(K, 78+k+off, 0.9)
# intro 3-2-1-go
for k, f in enumerate((660, 660, 660)): both(beep(f), 4.5+k, 0.4)
both(beep(990, 0.5), 7.5, 0.45)
# outro
both(bell(midi(84), 2.5), 288.2, 0.35); both(bell(midi(88), 2.5), 288.5, 0.3); both(bell(midi(91), 3.5), 288.8, 0.3)
# fade out last 2 s
f = np.ones(N); n2 = int(2.5*SR); f[-n2:] = np.linspace(1, 0, n2)
L *= f; R *= f
# simple delay for width
d = int(0.18*SR); L[d:] += 0.18*R[:-d]; R[d:] += 0.18*L[:-d]
pk = max(abs(L).max(), abs(R).max()); L /= pk; R /= pk
L *= 0.85; R *= 0.85
st = (np.stack([L, R], 1)*32767).astype('<i2')
with wave.open('music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
print('ok')
