from PIL import Image,ImageDraw,ImageFont,ImageEnhance
im=Image.open('../train/d298.png').convert('RGB').resize((1280,720),Image.LANCZOS)
im=ImageEnhance.Brightness(im).enhance(1.18); im=ImageEnhance.Color(im).enhance(1.15); im=ImageEnhance.Contrast(im).enhance(1.06)
g=Image.new('L',(1280,720),0); d=ImageDraw.Draw(g)
for y in range(720):
    a=max(0,(y-440)/280); d.line([(0,y),(1280,y)],fill=int(200*a**1.3))
im=Image.composite(Image.new('RGB',(1280,720),(14,8,22)),im,g)
d=ImageDraw.Draw(im)
f1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',118)
f2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',50)
def txt(xy,s,f,fill,sh=6):
    x,y=xy
    for dx,dy in [(sh,sh),(sh,0),(0,sh),(-2,sh)]: d.text((x+dx,y+dy),s,font=f,fill=(5,6,16))
    d.text(xy,s,font=f,fill=fill)
txt((50,490),'TRAIN BRIDGE',f1,(255,240,215))
txt((54,618),'SUNSET TO NIGHT  ·  1 HOUR',f2,(255,176,120),4)
im.save('thumb_train2.jpg',quality=92)
