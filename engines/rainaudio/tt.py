import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np,subprocess,sys
sys.path.insert(0,ZT+'rainaudio')
import rainfx as r
t0,t1=205.0,223.0
R,cur=r.build(t0,t1)
raw=subprocess.run(['ffmpeg','-v','error','-ss',str(t0),'-t',str(t1-t0),'-i',r.MUSIC,'-ar',str(r.SR),'-ac','2','-f','f32le','-'],capture_output=True).stdout
m=np.frombuffer(raw,np.float32).reshape(-1,2)
mix=np.zeros_like(cur); k=min(len(m),len(mix)); mix[:k]=m[:k]*0.75
mix=mix+cur*1.0
n=int(0.6*r.SR); mix[:n]*=np.linspace(0,1,n,dtype=np.float32)[:,None]; mix[-n:]*=np.linspace(1,0,n,dtype=np.float32)[:,None]
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(r.SR),'-ac','2','-i','-','-af','alimiter=limit=0.89:level=disabled','-c:a','aac','-b:a','192k','tt_audio.m4a'],stdin=subprocess.PIPE); p.stdin.write(mix.tobytes()); p.stdin.close(); p.wait()
