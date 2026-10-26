import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, subprocess, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
D=ZT+'tt2/'
W,H=1080,1920; X0=660
VF=f"crop=608:1080:{X0}:0,scale=1080:1920:flags=lanczos,eq=gamma=1.12:brightness=0.02:saturation=1.08"
src=subprocess.Popen(['ffmpeg','-v','error','-ss','155.5','-t','21','-i',D+'p3.mp4','-vf',VF,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
# text layer
lay=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(lay)
F1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',92)
F2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',40)
for txt,f,y,col in (('THE LAST',F1,215,(250,244,228,255)),('TRAIN HOME',F1,320,(250,244,228,255)),('1 HOUR VERSION ON YOUTUBE',F2,1395,(255,214,150,255))):
    d.text((W//2,y),txt,font=f,fill=col,anchor='mm')
a=lay.split()[3]
sh=Image.new('RGBA',(W,H),(8,10,24,0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(9)).point(lambda v:int(min(255,v*1.3))))
txt=Image.alpha_composite(sh,lay)
T=np.asarray(txt,np.float32); TA=T[...,3:4]/255; TC=T[...,:3]
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','24','-i','-','-i',D+'night_audio.wav',
    '-map','0:v','-map','1:a','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
    '-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',D+'tiktok_last_train.mp4'],stdin=subprocess.PIPE)
fs=W*H*3; first=[]; XF=24; N=504
def out(img):
    o=img*(1-TA)+TC*TA
    enc.stdin.write(np.clip(o,0,255).astype(np.uint8).tobytes())
for i in range(N):
    buf=src.stdout.read(fs)
    if len(buf)<fs: print('short at',i); break
    f=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)
    if i<XF: first.append(f); continue
    if i>=N-XF:
        k=i-(N-XF); u=(k+0.5)/XF; u=u*u*(3-2*u)
        f=f*(1-u)+first[k]*u
    out(f)
enc.stdin.close(); enc.wait()
