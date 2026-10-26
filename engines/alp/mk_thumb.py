import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import numpy as np, cv2
from PIL import Image,ImageDraw,ImageFont
D=ZT+'alp/'
src=cv2.imread(D+'th_560.png').astype(np.float32)
sky=cv2.imread(D+'sky_alpha.png',0).astype(np.float32)/255
sky=cv2.GaussianBlur(sky,(0,0),3)[...,None]
# lift the aurora and sky so they read at small sizes
hsv=cv2.cvtColor(np.clip(src,0,255).astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[...,1]=np.clip(hsv[...,1]*(1+0.35*sky[...,0]),0,255)
hsv[...,2]=np.clip(hsv[...,2]*(1+0.45*sky[...,0]),0,255)
img=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32)
img=img*1.10+6
img=cv2.resize(np.clip(img,0,255).astype(np.uint8),(1280,720),interpolation=cv2.INTER_AREA)
im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
g=Image.new('L',(1280,720),0); d=ImageDraw.Draw(g)
for y in range(720):
    a=max(0,(y-430)/290); d.line([(0,y),(1280,y)],fill=int(215*a**1.25))
im=Image.composite(Image.new('RGB',(1280,720),(6,10,22)),im,g)
d=ImageDraw.Draw(im)
f1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',112)
f2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',46)
def txt(xy,s,f,fill,sh=6):
    x,y=xy
    for dx,dy in [(sh,sh),(sh,0),(0,sh),(-2,sh)]: d.text((x+dx,y+dy),s,font=f,fill=(3,6,14))
    d.text(xy,s,font=f,fill=fill)
txt((50,500),'NORTHERN LIGHTS',f1,(240,246,250))
txt((54,624),'NORDIC MEADOW  ·  LO-FI & NATURE',f2,(160,235,195),4)
im.save(D+'thumb_nordic.jpg',quality=92)
