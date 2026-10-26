import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np,subprocess,re,os
SR=48000; D=ZT+''
RM=D+'rainmix/rain_house_1h_music.mp3'; NA=D+'nordic_audio/'
sec=lambda s: sum(int(a)*b for a,b in zip(s.split(':'),(3600,60,1)))
OLD={'Paper Boats in Twilight':(RM,'0:32:01','0:34:40'),'Beside the Open Window':(RM,'0:18:52','0:21:33'),'Damp Earth Warming':(RM,'0:21:33','0:24:15'),
     'Amber Light Through Glass':(RM,'0:34:40','0:37:29'),'Rain on Paper Walls':(RM,'0:00:00','0:02:41'),'Settle on the Tatami':(RM,'0:02:41','0:05:23'),
     'When the Valley Sleeps':(NA+'004176fa-when_the_valley_sleeps.mp3',None,None),'Where the Mist Lifts':(NA+'70247d4e-where_the_mist_lifts.mp3',None,None),
     "Beneath the Aurora's Glow":(NA+'33d0d108-beneath_the_aurora_s_glow.mp3',None,None),'Blue Hour at the Ridge':(NA+'6d982907-blue_hour_at_the_ridge.mp3',None,None)}
F={'Midnight by the Pond':'midnight_by_the_pond.wav','Lamplight on the Reeds':'l20_song1.wav','Rain on the Glass':'rain_on_the_glass.wav','Sleepy Pond':'lofi_pond_7min_sleepy_v2.wav',
   'Mist Over the Water':'track_40604.wav',"Heron's Rest":'l20_song2.wav','Streetlights Through the Rain':'streetlights_through_the_rain.wav','Slow Tea, Soft Rain':'combo_song2.wav',
   'Between the Raindrops':'between_the_raindrops.wav','The Last Lamp Burning':'the_last_lamp_burning.wav','Rainless Mist':'n03_rainless_mist.wav','Lantern on the Lily Pads':'track_40607.wav',
   'Slow Ripples':'l20_song3.wav','After the Last Train':'after_the_last_train.wav','Under the Willow':'combo_song3.wav','Midnight Pond Reflections':'midnight_pond_reflections.wav',
   'Mist Pond Dawn':'n01_mist_pond_dawn.wav','Rain Against the Glass':'rain_against_the_glass.wav','Under a Lantern Moon':'l20_song4.wav','Late Night Porch Light':'late_night_porch_light.wav',
   'Quiet Water':'combo_song4.wav','Velvet Pavement':'velvet_pavement.wav','Willow Hollow':'n02_willow_hollow.wav','Moonlight on the Water':'midnight_on_the_pond_1.wav',
   'Last Light on the Lily Pads':'l20_song5.wav','Ripples After Rain':'track_40608.wav','Drifting Off':'combo_song5.wav','Warm Light':'n04_warm_light.wav','Midnight on the Pond':'midnight_on_the_pond.wav'}
ORDER=['Midnight by the Pond','Lamplight on the Reeds','Rain on the Glass','Paper Boats in Twilight','Sleepy Pond','Mist Over the Water',"Heron's Rest",
 'Streetlights Through the Rain','When the Valley Sleeps','Slow Tea, Soft Rain','Between the Raindrops','The Last Lamp Burning','Rainless Mist','Beside the Open Window',
 'Lantern on the Lily Pads','Slow Ripples','After the Last Train','Damp Earth Warming','Under the Willow','Midnight Pond Reflections','Where the Mist Lifts',
 'Mist Pond Dawn','Rain Against the Glass','Under a Lantern Moon','Late Night Porch Light','Amber Light Through Glass','Quiet Water','Velvet Pavement',
 "Beneath the Aurora's Glow",'Willow Hollow','Moonlight on the Water','Last Light on the Lily Pads','Rain on Paper Walls','Ripples After Rain','Drifting Off',
 'Blue Hour at the Ridge','Warm Light','Settle on the Tatami','Midnight on the Pond']
assert len(ORDER)==len(set(ORDER))==39
SRC={n:'Lyria (Gemini)' for n in ORDER}
for n in ['Lamplight on the Reeds','Sleepy Pond',"Heron's Rest",'Slow Tea, Soft Rain','Rainless Mist','Slow Ripples','Under the Willow','Mist Pond Dawn','Under a Lantern Moon','Quiet Water','Willow Hollow','Last Light on the Lily Pads','Drifting Off','Warm Light']: SRC[n]='Meta AI'
def load(p,ss=None,t=None):
    a=['ffmpeg','-v','error']+(['-ss',str(ss)] if ss is not None else [])+(['-t',str(t)] if t else [])+['-i',p,'-ac','2','-ar',str(SR),'-f','f32le','-']
    return np.frombuffer(subprocess.run(a,capture_output=True).stdout,np.float32).reshape(-1,2).copy()
def lufs(x):
    r=subprocess.run(['ffmpeg','-nostats','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','ebur128','-f','null','-'],input=x.tobytes(),capture_output=True).stderr.decode()
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS',r)[-1])
def trim(x,thr=2e-3):
    a=np.abs(x).max(1); i=np.where(a>thr)[0]; return x[max(0,i[0]-int(0.05*SR)):i[-1]+int(0.2*SR)]
GAP=np.zeros((int(2.0*SR),2),np.float32)
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=false','-c:a','libmp3lame','-b:a','320k',D+'pondmusic/capy_2h_mix.mp3'],stdin=subprocess.PIPE)
t=0.0; log=[]; first=True
for name in ORDER:
    if name in OLD:
        p,a,b=OLD[name]
        x=load(p,sec(a)+6,sec(b)-sec(a)-12) if a else load(p)
        cut=a is not None
    else:
        x=load(F[name]); cut=False
    x=trim(x); x*=10**((-16-lufs(x))/20)
    fi=int((2.5 if cut else 0.4)*SR); x[:fi]*=np.linspace(0,1,fi,dtype=np.float32)[:,None]
    fo=int((4.0 if cut else 2.5)*SR); x[-fo:]*=(np.linspace(1,0,fo,dtype=np.float32)**1.6)[:,None]
    if not first: enc.stdin.write(GAP.tobytes()); t+=len(GAP)/SR
    first=False
    log.append((t,name,SRC[name])); enc.stdin.write(x.astype(np.float32).tobytes()); t+=len(x)/SR
enc.stdin.close(); enc.wait()
fmt=lambda s: f"{int(s)//3600}:{int(s)%3600//60:02d}:{int(s)%60:02d}"
tl='\n'.join(f"{fmt(a)} {n}" for a,n,_ in log)
open(D+'pondmusic/capy_2h_tracklist.txt','w').write(tl+'\n')
print('TOTAL',fmt(t)); print(tl)
