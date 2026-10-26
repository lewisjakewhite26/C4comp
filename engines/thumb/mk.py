import os as _os; ZT=_os.environ.get('ZT_WORK',_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+'/')   # engines/ root; inputs live in sibling folders
import sys
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageEnhance
im=Image.open(ZT+'uploads/0a1c7927-image.jpg').convert('RGB')
im=im.resize((1280,int(1280*im.height/im.width)),Image.LANCZOS)
im=im.crop((0,(im.height-720)//2,1280,(im.height-720)//2+720))
im=ImageEnhance.Color(im).enhance(1.12); im=ImageEnhance.Contrast(im).enhance(1.06)
# dark gradient at bottom for text
g=Image.new('L',(1280,720),0); d=ImageDraw.Draw(g)
for y in range(720):
    a=max(0,(y-330)/390); d.line([(0,y),(1280,y)],fill=int(210*a**1.4))
dark=Image.new('RGB',(1280,720),(20,10,6)); im=Image.composite(dark,im,g)
d=ImageDraw.Draw(im)
f1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',128)
f2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',60)
def txt(xy,s,f,fill,sh=6):
    x,y=xy
    for dx,dy in [(sh,sh),(sh,0),(0,sh),(-2,sh)]: d.text((x+dx,y+dy),s,font=f,fill=(15,8,4))
    d.text(xy,s,font=f,fill=fill)
txt((60,468),'AUTUMN CAFÉ',f1,(255,236,205))
txt((64,610),'3 HOURS  ·  COZY LOFI  ·  STUDY',f2,(255,176,92),4)
im.save(sys.argv[1],quality=95)
