import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import subprocess, numpy as np, json, random, re, sys, os
U=ZT+'uploads/'
TR=[ # file, artist, title
('85694843-zephiramusic-tokyo-lofi-604786.mp3','Zephira Music','Tokyo Lofi'),
('0f28134e-alex-morgan-chillhop-jazz-coffee-shop-552792.mp3','Alex Morgan','Chillhop Jazz Coffee Shop'),
('1a4685f1-alex-morgan-lofi-restaurant-568157.mp3','Alex Morgan','Lofi Restaurant'),
('0cb5a255-alex-morgan-lofi-coffee-shop-568150.mp3','Alex Morgan','Lofi Coffee Shop'),
('5d45c758-prettyjohn1-lofi-lofi-music-587176.mp3','prettyjohn1','Lofi Music'),
('d86e9fa5-zephiramusic-lofi-reel-588906.mp3','Zephira Music','Lofi Reel'),
('0f68b728-xethrocc-lofi-chillzone-162827.mp3','xethrocc','Lofi Chillzone'),
('1b5f60b2-viacheslavstarostin-lofi-lofi-music-427237.mp3','Viacheslav Starostin','Lofi Music'),
('6024ed9d-zephiramusic-abstract-lofi-602638.mp3','Zephira Music','Abstract Lofi'),
('923b79de-tunetank-chill-cafe-lofi-beat-348477.mp3','TuneTank','Chill Cafe Lofi Beat'),
('d767439b-denis-pavlov-music-cozy-relax-jazz-podcast-coffee-ambiance-271621.mp3','Denis Pavlov Music','Cozy Relax Jazz Podcast Coffee Ambiance'),
('947b1153-denis-pavlov-music-coffee-jazz-podcast-music-256133.mp3','Denis Pavlov Music','Coffee Jazz Podcast Music'),
('d0f7d963-tunetank-relaxed-lofi-cafe-music-349015.mp3','TuneTank','Relaxed Lofi Cafe Music'),
]
SR=44100; XF=5.0; TARGET=-16.0; TOTAL=10800.0
def dur(f):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',U+f]).decode())
def loud(f):
    r=subprocess.run(['ffmpeg','-nostats','-i',U+f,'-af','ebur128','-f','null','-'],capture_output=True,text=True).stderr
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS',r)[-1])
info=[]
for f,a,t in TR:
    d=dur(f); l=loud(f); info.append(dict(f=f,artist=a,title=t,dur=d,lufs=l,gain=TARGET-l)); print(a,t,round(d,1),l,flush=True)
long_=set(i for i,x in enumerate(info) if x['dur']>400)
rnd=random.Random(7); order=[]
for r in range(4):
    for _ in range(100000):
        o=list(range(len(info))); rnd.shuffle(o)
        seq=order[-1:]+o
        ok=all(info[seq[i]]['artist']!=info[seq[i+1]]['artist'] for i in range(len(seq)-1))
        ok=ok and all(not(seq[i] in long_ and seq[i+1] in long_) for i in range(len(seq)-1))
        ok=ok and not (order and set(o[:4])&set(order[-4:]))
        if ok: break
    order+=o
# timeline
starts=[];t=0.0
for k,i in enumerate(order):
    starts.append(t); t+=info[i]['dur']-XF
    if t+XF>=TOTAL+ 1: pass
n_used=0;t=0.0
for k,i in enumerate(order):
    if t>=TOTAL-30: break
    n_used=k+1; t=starts[k]+info[i]['dur']-XF if False else starts[k]+info[i]['dur']-XF
playlist=order
print('tracks in mix',len(playlist),flush=True)
def decode(i):
    x=info[i]
    af=f"silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.25,areverse,silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.35,areverse,volume={x['gain']:.2f}dB,alimiter=limit=0.84:level=disabled"
    p=subprocess.run(['ffmpeg','-v','error','-i',U+x['f'],'-af',af,'-f','s16le','-ar',str(SR),'-ac','2','-'],capture_output=True)
    return np.frombuffer(p.stdout,np.int16).reshape(-1,2)
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','s16le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled',
 '-c:a','libmp3lame','-b:a','192k','-id3v2_version','3','-metadata','title=Autumn Cafe Lofi - 3 Hour Mix','autumn_cafe_lofi_3h.mp3'],stdin=subprocess.PIPE)
nx=int(XF*SR); w=np.linspace(0,np.pi/2,nx,dtype=np.float32); cin=np.sin(w)[:,None]; cout=np.cos(w)[:,None]
total_samples=int(TOTAL*SR); written=0; tail=None; tl=[]; pos=0.0
def emit(a):
    global written
    a=np.clip(a,-32768,32767).astype(np.int16)
    room=total_samples-written
    if room<=0: return
    a=a[:room]
    enc.stdin.write(a.tobytes()); written+=len(a)
for k,i in enumerate(playlist):
    a=decode(i).astype(np.float32)
    if k==0:
        f=int(3*SR); a[:f]*=np.linspace(0,1,f,dtype=np.float32)[:,None]
        tl.append((0.0,i))
    else:
        head=a[:nx]; mixed=tail*cout+head*cin; emit(mixed); a=a[nx:]
        tl.append((written/SR-XF,i))
    if written+len(a)-nx>=total_samples-0 or k==len(playlist)-1:
        # last track: body to the end of the timeline with fade-out
        need=total_samples-written
        body=a[:need].copy(); fo=min(int(8*SR),len(body)); body[-fo:]*=np.linspace(1,0,fo,dtype=np.float32)[:,None]
        emit(body); break
    emit(a[:-nx]); tail=a[-nx:]
    print(k,info[i]['title'],round(written/SR,1),flush=True)
enc.stdin.close(); enc.wait()
def ts(s): s=int(round(s)); return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
lines=['# Autumn Cafe Lofi - 3 hour mix: track list\n','Music from Pixabay (Pixabay Content Licence). Titles taken from file names; check each against its Pixabay page.\n']
for s,i in tl:
    x=info[i]; lines.append(f"{ts(s)} {x['artist']} - {x['title']}")
lines.append('\nCredits:')
for a in sorted(set(info[i]['artist'] for _,i in tl)): lines.append(f"- {a} (Pixabay)")
open('tracklist.md','w').write('\n'.join(lines)+'\n')
json.dump([(s,info[i]['artist'],info[i]['title']) for s,i in tl],open('timeline.json','w'))
print('done',written/SR,flush=True)
