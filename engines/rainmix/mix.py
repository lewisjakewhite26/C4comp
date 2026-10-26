import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import subprocess,numpy as np,re,sys,json
U=ZT+'uploads/'
TR=[('dc74c4bd-rain_on_paper_walls.mp3', 'Lyria (Gemini)', 'Rain on Paper Walls'), ('8f207ee4-settle_on_the_tatami.mp3', 'Lyria (Gemini)', 'Settle on the Tatami'), ('1b1be9eb-rain_on_the_temple_stone.mp3', 'Lyria (Gemini)', 'Rain on the Temple Stone'), ('96e768cd-surprising_media-guitar-in-the-forest-439961.mp3', 'Surprising Media', 'Guitar in the Forest'), ('17aa22aa-morning_breath_of_bamboo.mp3', 'Lyria (Gemini)', 'Morning Breath of Bamboo'), ('c49b54c8-beside_the_open_window.mp3', 'Lyria (Gemini)', 'Beside the Open Window'), ('f051871f-damp_earth_warming.mp3', 'Lyria (Gemini)', 'Damp Earth Warming'), ('4e913e10-good_b_music-golden-sea-long-8min-596828.mp3', 'Good B Music', 'Golden Sea (Long 8 min)'), ('60036c3f-paper_boats_in_twilight.mp3', 'Lyria (Gemini)', 'Paper Boats in Twilight'), ('8178cd0d-amber_light_through_glass.mp3', 'Lyria (Gemini)', 'Amber Light Through Glass'), ('04153ce9-morning_by_the_glass.mp3', 'Lyria (Gemini)', 'Morning by the Glass'), ('15861cf8-tramp963-sunrise-meditation-lo-fi-music-267024.mp3', 'Tramp963', 'Sunrise Meditation Lo-Fi Music'), ('604e10d9-rain_on_the_fire_escape.mp3', 'Lyria (Gemini)', 'Rain on the Fire Escape'), ('1f72b7ad-coffee_at_three.mp3', 'Lyria (Gemini)', 'Coffee at Three'), ('40cb32cf-october_after_midnight.mp3', 'Lyria (Gemini)', 'October After Midnight')]
SR=44100; XF=5*SR; TOTAL=3600*SR
def loud(p):
    r=subprocess.run(['ffmpeg','-nostats','-i',p,'-af','ebur128','-f','null','-'],capture_output=True,text=True).stderr
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS',r)[-1])
def load(p):
    I=loud(p); gain=-16-I
    flt=f'silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.25,areverse,silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.35,areverse,volume={gain:.2f}dB,alimiter=limit=0.84:level=disabled'
    raw=subprocess.run(['ffmpeg','-v','error','-i',p,'-af',flt,'-ar',str(SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).reshape(-1,2).copy(),I
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-c:a','libmp3lame','-b:a','192k','rain_house_1h_music.mp3'],stdin=subprocess.PIPE)
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
    tail=tail.copy(); n=len(tail); tail*=np.linspace(1,0,n,dtype=np.float32)[:,None]; enc.stdin.write(tail.tobytes()); written+=len(tail)
enc.stdin.close(); enc.wait()
print('total',written/SR,file=sys.stderr)
def ts(s): s=int(s); return f'{s//3600}:{s%3600//60:02d}:{s%60:02d}'
with open('tracklist.md','w') as o:
    o.write('# Rain House Lofi - 1 hour mix: track list\n\nPixabay tracks: Pixabay Content Licence (check each Pixabay page). Lyria (Gemini) tracks: AI-generated; tick YouTube altered/synthetic content disclosure.\n\n')
    for s,a,t in tl: o.write(f'{ts(s)} {a} - {t}\n')
    o.write('\nCredits:\n'+''.join(f'- {a}\n' for a in dict.fromkeys(a for _,a,_ in tl)))
