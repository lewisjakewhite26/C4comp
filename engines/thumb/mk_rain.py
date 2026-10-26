from PIL import Image,ImageDraw,ImageFont,ImageEnhance
im=Image.open('rain_still.png').convert('RGB').resize((1280,720),Image.LANCZOS)
im=ImageEnhance.Brightness(im).enhance(1.12); im=ImageEnhance.Color(im).enhance(1.12); im=ImageEnhance.Contrast(im).enhance(1.05)
g=Image.new('L',(1280,720),0); d=ImageDraw.Draw(g)
for y in range(720):
    a=max(0,(y-420)/300); d.line([(0,y),(1280,y)],fill=int(215*a**1.25))
im=Image.composite(Image.new('RGB',(1280,720),(10,18,16)),im,g)
d=ImageDraw.Draw(im)
f1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',112)
f2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',46)
def txt(xy,s,f,fill,sh=6):
    x,y=xy
    for dx,dy in [(sh,sh),(sh,0),(0,sh),(-2,sh)]: d.text((x+dx,y+dy),s,font=f,fill=(4,10,10))
    d.text(xy,s,font=f,fill=fill)
txt((50,500),'RAINY DAY',f1,(240,250,240))
txt((54,624),'JAPANESE HOUSE  ·  RAIN & LO-FI  ·  1 HOUR',f2,(170,225,170),4)
im.save('thumb_rain.jpg',quality=92)
