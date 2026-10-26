from PIL import Image,ImageDraw,ImageFont,ImageEnhance
im=Image.open('../train/n527.png').convert('RGB').resize((1280,720),Image.LANCZOS)
im=ImageEnhance.Brightness(im).enhance(1.18); im=ImageEnhance.Color(im).enhance(1.15); im=ImageEnhance.Contrast(im).enhance(1.06)
g=Image.new('L',(1280,720),0); d=ImageDraw.Draw(g)
for y in range(720):
    a=max(0,(y-470)/250); d.line([(0,y),(1280,y)],fill=int(200*a**1.3))
im=Image.composite(Image.new('RGB',(1280,720),(6,8,22)),im,g)
d=ImageDraw.Draw(im)
f1=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',118)
f2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',50)
def txt(xy,s,f,fill,sh=6):
    x,y=xy
    for dx,dy in [(sh,sh),(sh,0),(0,sh),(-2,sh)]: d.text((x+dx,y+dy),s,font=f,fill=(5,6,16))
    d.text(xy,s,font=f,fill=fill)
txt((50,548),'NIGHT TRAIN',f1,(255,232,170))
txt((54,662-6),'',f2,(255,255,255))
im.save('thumb_train.jpg',quality=92)
