import sys, math, subprocess; sys.path.insert(0, '.')
import numpy as np, cv2
from moves3d import *
D = dict(aLa=8, aLf=0, aLz=0, aLx=0, aRa=8, aRf=0, aRz=0, aRx=0, lLa=3, lLf=0, lLk=0, lRa=3, lRf=0, lRk=0, sl=0, sf=0, st=0, rt=0, dx=0, jump=0, tw=0, nod=0, aLt=0, aRt=0)
def K(**k): d = dict(D); d.update(k); return d
def interp(frames, t):
    t = t % frames[-1][0]
    for (t0, a), (t1, b) in zip(frames, frames[1:]):
        if t0 <= t <= t1:
            u = sm((t-t0)/max(1e-6, t1-t0)); return {k: a[k]+(b[k]-a[k])*u for k in a}
    return frames[-1][1]
def to_pose(p):
    r = {}
    arm(r, 'SL', p['aLa'], fwd=p['aLf'], el_z=p['aLz'], el_x=p['aLx'], twist=SGN[SIDE['SL']]*p['aLt']); arm(r, 'SR', p['aRa'], fwd=p['aRf'], el_z=p['aRz'], el_x=p['aRx'], twist=SGN[SIDE['SR']]*p['aRt'])
    leg(r, 'SL', p['lLa'], fwd=p['lLf'], knee=p['lLk']); leg(r, 'SR', p['lRa'], fwd=p['lRf'], knee=p['lRk'])
    spine(r, lean=p['sl'], fwd=p['sf'], twist=p['st']); root(r, twist=p['rt']); neck(r, nod=p['nod'], turn=p['tw'])
    return r, p['dx'], p['jump'], 0
CR = dict(lLa=22, lLf=60, lLk=100, lRa=22, lRf=60, lRk=100, sf=22)
JUMP = [(0, K()), (0.25, K(aLa=25, aRa=25, aLf=-45, aRf=-45, **CR)), (0.42, K(aLa=150, aRa=150, lLa=4, lRa=4, jump=0, sf=-4)),
        (0.7, K(aLa=155, aRa=155, lLa=10, lRa=10, jump=58, lLf=20, lRf=20, lLk=40, lRk=40)),
        (0.98, K(aLa=40, aRa=40, aLf=-20, aRf=-20, lLa=18, lRa=18, lLf=45, lRf=45, lLk=70, lRk=70, sf=14)), (1.3, K())]
THROW = [(0, K(aLa=45, aLf=35, aLx=30)),
         (0.45, K(aRa=105, aRz=110, aLa=60, aLf=45, aLx=20, st=-18, sl=6, sf=-6, lLf=0, lRf=0, dx=14)),
         (0.62, K(aRa=45, aRf=120, aRx=10, aLa=20, aLf=-10, st=22, sf=20, lLf=30, lLk=25, dx=-10)),
         (0.9, K(aRa=15, aRf=60, aRx=25, aLa=18, st=12, sf=14, dx=-6)), (1.6, K(aLa=45, aLf=35, aLx=30))]
CHOP = [(0, K(aLa=10, aRa=10)), (0.5, K(aLa=150, aRa=150, sf=-12, lLa=8, lRa=8)), (0.8, K(aLa=150, aRa=150, sf=-14, lLa=8, lRa=8)),
        (0.98, K(aLa=20, aRa=20, aLf=70, aRf=70, aLt=-45, aRt=-45, sf=34, lLf=40, lLk=60, lRf=40, lRk=60, lLa=10, lRa=10)),
        (1.3, K(aLa=20, aRa=20, aLf=70, aRf=70, aLt=-45, aRt=-45, sf=32, lLf=40, lLk=60, lRf=40, lRk=60, lLa=10, lRa=10)), (1.9, K(aLa=10, aRa=10))]
DUCK = [(0, K()), (0.3, K(aLa=150, aRa=150, aLz=40, aRz=40, sf=36, lLa=26, lRa=26, lLf=80, lRf=80, lLk=130, lRk=130)),
        (1.4, K(aLa=150, aRa=150, aLz=40, aRz=40, sf=36, lLa=26, lRa=26, lLf=80, lRf=80, lLk=130, lRk=130, nod=15)), (1.9, K()), (2.4, K())]
def mir(p):
    q = dict(p)
    for a, b in (('aLt', 'aRt'), ('aLa', 'aRa'), ('aLf', 'aRf'), ('aLz', 'aRz'), ('aLx', 'aRx'), ('lLa', 'lRa'), ('lLf', 'lRf'), ('lLk', 'lRk')): q[a], q[b] = p[b], p[a]
    q['st'] = -p['st']; q['dx'] = -p['dx']; q['rt'] = -p['rt']; q['sl'] = -p['sl']; q['tw'] = -p['tw']
    return q
CLIPS = {'stamp': lambda t: m_stamp(t), 'jump': lambda t: to_pose(interp(JUMP, t)), 'throw_right': lambda t: to_pose(interp(THROW, t)),
         'throw_left': lambda t: to_pose(mir(interp(THROW, t))), 'chop': lambda t: to_pose(interp(CHOP, t)), 'duck': lambda t: to_pose(interp(DUCK, t))}
def frame(name, t):
    r, dx, j, yw = CLIPS[name](t)
    Vp = build(r, dx, j); M = mask(Vp, dx, yw)
    Mb = cv2.GaussianBlur(cv2.dilate(M, np.ones((3, 3), np.uint8)), (0, 0), 4); _, Mb = cv2.threshold(Mb, 120, 255, cv2.THRESH_BINARY)
    fig = np.zeros((1080, 1920, 3), np.uint8); fig[Mb > 0] = (30, 35, 18)
    cv2.polylines(fig, outline(M, 6.9*int(t*8)), True, (212, 227, 127), 5, cv2.LINE_AA)
    return glow(fig)
def render(name, out, dur=6.0):
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-g', '30', out], stdin=subprocess.PIPE)
    for k in range(int(dur*30)): p.stdin.write(frame(name, k/30).tobytes())
    p.stdin.close(); p.wait()
if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        rows = []
        for name, ts in (('jump', [0, .25, .42, .7, .98]), ('throw_right', [0, .45, .62, .9, 1.2]), ('chop', [0, .5, .98, 1.3, 1.7]), ('duck', [0, .3, .9, 1.6, 2.1])):
            rows.append(np.hstack([cv2.resize(frame(name, t)[100:980, 480:1440], (320, 293)) for t in ts]))
        cv2.imwrite('c5.png', np.vstack(rows))
    else: render(sys.argv[1], sys.argv[2])
