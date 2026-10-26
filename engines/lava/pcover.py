import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
D=ZT+'lava/'
im=Image.open(D+'p_0.5.png').convert('RGB'); W,H=im.size
im=im.convert('RGBA'); lay=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(lay)
FB=ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',150)
FS=ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',44)
d.text((W/2,250),'2 MINUTE',font=FB,fill=(255,255,255,255),anchor='mm')
d.text((W/2,400),'TIMER',font=FB,fill=(255,170,235,255),anchor='mm')
d.text((W/2,1480),'calm lava lamp · no alarm',font=FS,fill=(255,220,245,240),anchor='mm')
a=lay.split()[3]
sh=Image.new('RGBA',(W,H),(25,0,30,0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(12)).point(lambda v:int(v*0.9)))
gl=Image.new('RGBA',(W,H),(230,80,220,0)); gl.putalpha(a.filter(ImageFilter.GaussianBlur(26)).point(lambda v:int(v*0.5)))
im=Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(im,sh),gl),lay)
im.convert('RGB').save(D+'lava_lamp_timer_2min_tiktok_cover.jpg',quality=92)
