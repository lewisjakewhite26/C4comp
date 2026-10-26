import cv2,numpy as np
im=cv2.imread('clean_v4.png'); H,W=im.shape[:2]
sky=[(600,8),(800,25),(1000,60),(1100,150),(1200,215),(1330,70),(1560,25),(330,70),(960,160),(1080,200),(1180,120),(900,110),(700,160),(450,40),(1400,190),(1020,230),(1250,100),(1600,40),(1625,75),(1590,95),(1450,10),(250,5),(160,8),(1000,5),(1700,3),(1680,53),(1670,62),(1665,45),(1690,48),(1655,85),(1640,100),(233,33),(267,67),(220,15),(260,20),(300,40),(280,95)]
gnd=[(560,210),(700,230),(800,262),(890,215),(1060,275),(380,190),(120,60),(1800,100),(1500,330),(1300,300),(430,170),(1150,300),(60,200),(620,300),(1700,60),(1880,60),(340,160),(230,230),(1080,260),(560,140),(900,300),(1250,260),(620,95),(600,130),(690,170),(1780,40),(1730,90),
(602,90),(616,100),(626,118),(640,126),(655,140),(668,152),(682,162),(700,178),(715,190),(595,80),(590,70),(650,150),(625,112),(167,40),(173,67),(187,100),(160,87),(173,20),(233,100),(247,110),(143,33),(157,63),(165,15),(185,45),(200,70),(215,90),(205,110),(225,120),(240,125),(140,20),(150,33),(127,10),(135,45),(120,25),(145,45),(110,5)]
mk=np.zeros((H,W),np.int32)
for x,y in sky: cv2.circle(mk,(x,y),3,1,-1)
for x,y in gnd: cv2.circle(mk,(x,y),3,2,-1)
mk[360:,:]=2
g=cv2.GaussianBlur(im,(0,0),1.0)
lab=cv2.cvtColor(g,cv2.COLOR_BGR2LAB).astype(np.float32)
gx=cv2.Sobel(lab,cv2.CV_32F,1,0); gy=cv2.Sobel(lab,cv2.CV_32F,0,1)
grad=np.sqrt((gx**2+gy**2).sum(2)); grad=np.clip(grad/grad.max()*255*3,0,255).astype(np.uint8)
ws=cv2.watershed(cv2.cvtColor(grad,cv2.COLOR_GRAY2BGR),mk.copy())
skym=(ws==1).astype(np.uint8)*255
# remove tiny islands / fill pinholes
n,lbl,st,_=cv2.connectedComponentsWithStats(skym); 
big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); skym=((lbl==big)*255).astype(np.uint8)
inv=255-skym; n,lbl,st,_=cv2.connectedComponentsWithStats(inv)
for i in range(1,n):
    if st[i,cv2.CC_STAT_AREA]<400: skym[lbl==i]=255
cv2.imwrite('sky_ws3.png',skym)
vis=im.copy(); edge=cv2.morphologyEx(skym,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
vis[edge]=(0,0,255)
for i,(a,b,c,d) in enumerate(((440,20,760,220),(1180,150,1480,260),(1480,0,1920,130),(0,0,480,130),(760,100,1260,260))):
    cv2.imwrite(f'edge{i}.png',cv2.resize(vis[b:d,a:c],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
cv2.imwrite('edge_all.png',vis[:380])
