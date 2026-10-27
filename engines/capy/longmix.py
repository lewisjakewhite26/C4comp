# Long ambient video from a seamless loop: music tracks levelled to -16 LUFS and crossfaded, the loop repeated to
# match, and an optional variant loop (e.g. lightning) swapped in at chosen minutes. Writes the video and a tracklist.
#
#   python longmix.py <loop.mp4> <out.mp4> <tracks.txt> [--variant clip.mp4 --at 16,41,59] [--minutes 120]
#
# tracks.txt: one audio file per line, in play order. Lines after the first blank line are "closers": they always
# end the video, and filler tracks from above are used only until the closers bring it to --minutes.
# The loop and variant are encoded once, as identical closed GOPs, then joined with stream copy, so a 2-hour video
# takes minutes, not hours. Set FFMPEG/FFPROBE if ffmpeg isn't on PATH.
import numpy as np, subprocess, sys, os, re, argparse, tempfile
FF = os.environ.get('FFMPEG', 'ffmpeg'); FP = os.environ.get('FFPROBE', 'ffprobe'); SR = 48000; XF = 4.0; TARGET = -16.0

ap = argparse.ArgumentParser()
ap.add_argument('loop'); ap.add_argument('out'); ap.add_argument('tracks')
ap.add_argument('--variant'); ap.add_argument('--at', default='')
ap.add_argument('--minutes', type=float, default=120)
a = ap.parse_args()

def load(p):
    r = subprocess.run([FF, '-v', 'error', '-i', p, '-vn', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()
def lufs(x):
    r = subprocess.run([FF, '-nostats', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-af', 'ebur128', '-f', 'null', '-'],
                       input=x.tobytes(), capture_output=True).stderr.decode()
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS', r)[-1])
def title(p):
    t = re.sub(r'^\d+\s*-\s*', '', os.path.splitext(os.path.basename(p))[0])
    return {'Herons Rest': "Heron's Rest", 'Slow Tea Soft Rain': 'Slow Tea, Soft Rain'}.get(t, t)
def dur(p):
    return float(subprocess.run([FP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                                capture_output=True, text=True).stdout)

# Choose tracks: fillers in order until fillers + closers reach the target length
txt = open(a.tracks, encoding='utf-8').read().strip().split('\n\n')
fill = [l.strip() for l in txt[0].splitlines() if l.strip()]
close = [l.strip() for l in txt[1].splitlines() if l.strip()] if len(txt) > 1 else []
need = a.minutes * 60 - sum(dur(p) - XF for p in close)
play = []
for p in fill:
    if need <= 0: break
    play.append(p); need -= dur(p) - XF
play += close

# Audio: stream track by track, equal-power crossfades, soft limiter, AAC out
wav = os.path.join(tempfile.gettempdir(), 'longmix_audio.m4a')
enc = subprocess.Popen([FF, '-y', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-c:a', 'aac', '-b:a', '256k', wav], stdin=subprocess.PIPE)
n_xf = int(XF * SR); fin = np.sin(np.linspace(0, np.pi / 2, n_xf))[:, None]; fout = np.cos(np.linspace(0, np.pi / 2, n_xf))[:, None]
def limit(x, t=0.8):
    m = np.abs(x) > t
    x[m] = np.sign(x[m]) * (t + (1 - t) * np.tanh((np.abs(x[m]) - t) / (1 - t)))
    return x
tail = None; t = 0.0; log = []
for i, p in enumerate(play):
    x = load(p); x *= 10 ** ((TARGET - lufs(x)) / 20)
    if i == len(play) - 1:
        f = int(10 * SR); x[-f:] *= np.linspace(1, 0, f)[:, None]   # 10 s fade at the very end
    if tail is None:
        log.append((0.0, title(p)))
        x[:SR] *= np.linspace(0, 1, SR)[:, None]                     # 1 s fade in
    else:
        log.append((t - XF, title(p)))
        x[:n_xf] = x[:n_xf] * fin + tail * fout
    body = x if i == len(play) - 1 else x[:-n_xf]
    enc.stdin.write(limit(body).astype(np.float32).tobytes()); t += len(body) / SR
    tail = x[-n_xf:]
enc.stdin.close(); enc.wait()
total = t
print('audio %.1f min, %d tracks' % (total / 60, len(play)))

# Video: encode loop and variant once with identical settings, then concat with stream copy
tmp = tempfile.mkdtemp()
def unit(src, out):
    subprocess.run([FF, '-y', '-v', 'error', '-i', src, '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                    '-r', '24', '-g', '9999', '-keyint_min', '9999', '-sc_threshold', '0', '-bf', '2', '-x264-params', 'open-gop=0', out], check=True)
    return dur(out)
L = unit(a.loop, os.path.join(tmp, 'loop.mp4'))
slots = set()
if a.variant:
    unit(a.variant, os.path.join(tmp, 'var.mp4'))
    slots = {round(float(m) * 60 / L) for m in a.at.split(',') if m}
reps = int(np.ceil(total / L))
with open(os.path.join(tmp, 'list.txt'), 'w') as f:
    for r in range(reps): f.write("file '%s'\n" % ('var.mp4' if r in slots else 'loop.mp4'))
subprocess.run([FF, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp, 'list.txt'), '-i', wav,
                '-map', '0:v', '-map', '1:a', '-c', 'copy', '-t', '%.3f' % total, '-movflags', '+faststart', a.out], check=True)

hms = lambda s: '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)
tl = '\n'.join('%s %s' % (hms(s), n) for s, n in log)
open(os.path.splitext(a.out)[0] + '_tracklist.txt', 'w', encoding='utf-8').write(tl + '\n')
print(tl)
print('variant at', sorted(hms(r * L) for r in slots))
