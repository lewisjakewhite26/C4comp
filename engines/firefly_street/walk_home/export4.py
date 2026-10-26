import os,sys,json,subprocess,numpy as np
i=sys.argv[1]
V={'1':'300,480','2':'350,440','3':'375,560'}[i]
P={'1':'[["e",6.5,0.6,0.4,300]]','2':'[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]','3':'[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'}[i]
os.environ.update(dict(PRELIT='a',NOFF='1',CLIP=f'slow{i}.mp4',TRK=f'tr{i}.json',RT=f'rf{i}.json',SB=f'seedb{i}.json',WTRK='none',VP=V,PLAN=P))
sys.argv=['x','none']
import light4 as L
N=L.N; out={'N':N,'W':L.W,'H':L.H,'lamps':{}}
for k,d in L.lamps.items():
    if k.startswith('W') or k not in L.TPL: continue
    fs=sorted(d); f0=fs[0]; arr=[]
    for f in range(f0,fs[-1]+1):
        p=L.lpf(k,f); tp,bt,hw=L.geo(k,p,f)
        arr.append([round(p[0]-hw,1),round(p[1]+tp,1),round(p[0]+hw,1),round(p[1]+bt,1)])
    lit=L.TON.get(k,None)
    out['lamps'][k]={'f0':f0,'box':arr,'lit':None if lit is None else int(round(lit*24))}
json.dump(out,open(f'placer/model{i}.json','w'),separators=(',',':'))
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{L.W}x{L.H}','-r','24','-i','-','-vf','scale=540:960:flags=area','-c:v','libx264','-crf','27','-preset','fast','-g','6','-pix_fmt','yuv420p','-movflags','+faststart',f'placer/scene{i}.mp4'],stdin=subprocess.PIPE)
for f in range(N):
    g=(L.FR[f].astype(np.float32)/255)**0.72*255
    enc.stdin.write(np.clip(g,0,255).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()
print('scene',i,'frames',N,{k:(v['f0'],v['f0']+len(v['box']),v['lit']) for k,v in out['lamps'].items()})
