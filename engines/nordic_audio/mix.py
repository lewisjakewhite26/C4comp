import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, subprocess, re, sys
D=ZT+'nordic_audio/'
SR=48000
ORDER=['47a60da2-amber_light_on_the_ridge','c72efb95-sun_over_the_fjord','6d982907-blue_hour_at_the_ridge',
       '02f108b2-beneath_the_arctic_veil','33d0d108-beneath_the_aurora_s_glow','70247d4e-where_the_mist_lifts',
       '6025ee02-sunlight_on_the_pines','a117ab83-last_light_at_the_peak','8766f075-below_the_horizon',
       '004176fa-when_the_valley_sleeps','306f4b31-under_the_arctic_canopy']
TARGET=-16.0; XF=5.0; TOTAL=30*60.0; FADE_OUT=10.0
def lufs(path):
    r=subprocess.run(['ffmpeg','-nostats','-i',path,'-af','ebur128','-f','null','-'],capture_output=True,text=True).stderr
    return float(re.findall(r'I:\s+(-?[\d.]+) LUFS',r)[-1])
def load(path):
    raw=subprocess.run(['ffmpeg','-v','error','-i',path,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).reshape(-1,2).copy()
def trim_silence(x,thr=1e-3):
    a=np.abs(x).max(1); idx=np.where(a>thr)[0]
    return x[max(0,idx[0]-int(0.05*SR)):idx[-1]+int(0.3*SR)]
out=np.zeros((int((TOTAL+60)*SR),2),np.float32); pos=0; log=[]
xf=int(XF*SR); w=np.linspace(0,1,xf,dtype=np.float32)[:,None]
for i,name in enumerate(ORDER):
    p=D+name+'.mp3'; L=lufs(p); g=10**((TARGET-L)/20)
    x=trim_silence(load(p))*g
    if i==0:
        fi=int(3*SR); x[:fi]*=np.linspace(0,1,fi,dtype=np.float32)[:,None]
        out[:len(x)]+=x; start=0
    else:
        start=pos-xf
        x[:xf]*=np.sin(w*np.pi/2); out[start:start+xf]*=np.cos(w*np.pi/2)
        out[start:start+len(x)]+=x
    pos=start+len(x)
    log.append(f'{int(start/SR)//60}:{int(start/SR)%60:02d}  {name.split("-",1)[1].replace("_"," ")}  ({L:.1f} LUFS -> {TARGET})')
n=int(TOTAL*SR); out=out[:n]
fo=int(FADE_OUT*SR); out[-fo:]*=np.linspace(1,0,fo,dtype=np.float32)[:,None]**1.5
peak=np.abs(out).max(); print('peak before limit',round(float(20*np.log10(peak)),2),'dBFS; music ran to',round(pos/SR/60,2),'min')
open(D+'tracklist.txt','w').write('\n'.join(log)+'\n')
print('\n'.join(log))
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=false','-c:a','pcm_s16le',D+'nordic_music_30min.wav'],input=out.tobytes())
