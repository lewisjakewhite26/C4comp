cd /tmp/claude-0/-home-claude/f7d1ccd4-c824-5d8b-9f33-8948edecece0/scratchpad/meta
V1=300,480; V2=350,440; V3=375,560; K1=7.6; K2=8; K3=5.3
Q1='[["e",6.5,0.6,0.4,300]]'
Q2='[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]'
Q3='[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'
for i in 1 2 3; do
  eval V=\$V$i; eval P=\$Q$i; eval K=\$K$i
  KB=$K SCENE=$i PRELIT=a CLIP=slow$i.mp4 TRK=tr$i.json RT=rf$i.json SB=seedb$i.json WTRK=none VP=$V PLAN="$P" python3 light6.py p7_$i.mp4 > r7_$i.log 2>&1
done
ffmpeg -y -v error -i ../street/street_test7.mp4 -i p7_1.mp4 -i p7_2.mp4 -i p7_3.mp4 -loop 1 -t 9 -i ../street/hook.png -i ../citybed/city_night_bed.mp3 -filter_complex "[0:v][1:v]xfade=transition=fade:duration=1.5:offset=32.75[a];[a][2:v]xfade=transition=fade:duration=1.5:offset=65.5[b];[b][3:v]xfade=transition=fade:duration=1.5:offset=98.25[c];[c]fade=t=out:st=129.5:d=3[d];[4:v]format=rgba,fade=t=out:st=7:d=2:alpha=1[h];[d][h]overlay=0:0:shortest=0[v];[5:a]atrim=0:132.5,afade=t=out:st=129.5:d=3,loudnorm=I=-14:TP=-1.5:LRA=11[au]" -map "[v]" -map "[au]" -c:v libx264 -crf 22 -preset fast -pix_fmt yuv420p -c:a aac -b:a 192k -t 132.5 firefly_walk_home_v7.mp4 > r7_join.log 2>&1
touch r7.done
