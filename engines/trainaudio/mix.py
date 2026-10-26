import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import subprocess,numpy as np,re,sys,json
U=ZT+'uploads/'
TR=[('4a5b2975-akiko_shina-lofi-piano-quotdreamyquot-248626.mp3','Akiko Shina','Lofi Piano "Dreamy"'),
('96e768cd-surprising_media-guitar-in-the-forest-439961.mp3','Surprising Media','Guitar in the Forest'),
('4e913e10-good_b_music-golden-sea-long-8min-596828.mp3','Good B Music','Golden Sea (Long 8 min)'),
('5251f83f-tramp963-relaxing-piano-lo-fi-music-268029.mp3','Tramp963','Relaxing Piano Lo-Fi Music'),
('875e57ac-a1lhs-lofi-music-5-136397.mp3','A1lhs','Lofi Music 5'),
('15861cf8-tramp963-sunrise-meditation-lo-fi-music-267024.mp3','Tramp963','Sunrise Meditation Lo-Fi Music'),
('4fbb3a8c-pixelwhimsy-productivity-music-loop-calm-ambient-sound-for-focus-367143.mp3','Pixelwhimsy','Productivity Music Loop (Calm Ambient Sound for Focus)')]
SR=44100; XF=5*SR; TOTAL=3600*SR
def loud(p):
    r=subprocess.run(['ffmpeg','-nostats','-i',p,'-af','ebur128','-f','null','-'],capture_output=True,text=True).stderr
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS',r)[-1])
def load(p):
    I=loud(p); gain=-16-I
    flt=f'silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.25,areverse,silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.35,areverse,volume={gain:.2f}dB,alimiter=limit=0.84:level=disabled'
    raw=subprocess.run(['ffmpeg','-v','error','-i',p,'-af',flt,'-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).reshape(-1,2).copy(),I
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-c:a','libmp3lame','-b:a','192k','train_bridge_1h.mp3'],stdin=subprocess.PIPE)
fi=np.linspace(0,1,XF,dtype=np.float32); cin=np.sin(fi*np.pi/2)[:,None]; cout=np.cos(fi*np.pi/2)[:,None]
written=0; tail=None; tl=[]
for i,(f,a,t) in enumerate(TR):
    x,I=load(U+f); print(a,t,len(x)/SR,I,file=sys.stderr)
    if i==0:
        n=3*SR; x[:n]*=np.linspace(0,1,n,dtype=np.float32)[:,None]
        tl.append((0.0,a,t))
    else:
        head=x[:XF]*cin; mix=tail*cout+head; enc.stdin.write(mix.tobytes()); written+=XF
        x=x[XF:]; tl.append(((written-XF)/SR,a,t))
    tail_len=XF
    body=x[:-tail_len]; tail=x[-tail_len:]
    if written+len(body)>=TOTAL-tail_len:   # last track: cut at TOTAL with fade
        rem=TOTAL-written; seg=np.concatenate([body,tail])[:rem].copy(); n=8*SR; seg[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]
        enc.stdin.write(seg.tobytes()); written+=len(seg); tail=None; break
    enc.stdin.write(body.tobytes()); written+=len(body)
if tail is not None:
    n=8*SR; tail=tail.copy(); tail[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]; enc.stdin.write(tail.tobytes()); written+=len(tail)
enc.stdin.close(); enc.wait()
print('total',written/SR,file=sys.stderr)
def ts(s): s=int(s); return f'{s//3600}:{s%3600//60:02d}:{s%60:02d}'
with open('tracklist.md','w') as o:
    o.write('# Train Bridge Lofi - 1 hour mix: track list\n\nMusic from Pixabay (Pixabay Content Licence). Titles taken from file names; check each against its Pixabay page.\n\n')
    for s,a,t in tl: o.write(f'{ts(s)} {a} - {t}\n')
    o.write('\nCredits:\n'+''.join(f'- {a} (Pixabay)\n' for a in dict.fromkeys(a for _,a,_ in tl)))
