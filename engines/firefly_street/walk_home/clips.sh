G1='{"a":[30,-20,73,27],"e":[400,-3,73,20]}'
P1='[["a",0.0,0.6,0.4,300],["e",17.0]]'
G2='{"a":[30,0,63,27],"b":[30,3,70,27],"c":[300,-10,43,20],"h":[300,-7,47,20],"e":[600,-7,33,17],"f":[650,-7,40,19]}'
P2='[["a",0.0,0.4,0.4,300],["b",0.0,0.3,0.3,300],["c",12.5],["h",17.5],["e",24.0],["f",28.5]]'
G3='{"a":[20,-13,87,47],"b":[30,-17,77,37],"c":[250,-10,63,30],"d":[300,-10,63,25],"e":[500,-7,53,22],"i":[650,-7,70,30]}'
P3='[["a",0.0,0.3,0.2,380],["b",0.0,0.3,0.2,380],["c",9.0],["d",14.5],["e",21.0],["i",26.0]]'
V1=300,480; V2=350,440; V3=375,560
go(){ i=$1; shift; eval G=\$G$i P=\$P$i V=\$V$i; CLIP=slow$i.mp4 TRK=tr$i.json WTRK=none VP=$V PLAN="$P" GEO="$G" python3 light.py "$@"; }
go2(){ i=$1; shift; eval P=\$P$i V=\$V$i; CLIP=slow$i.mp4 TRK=tr$i.json RT=rf$i.json SB=seedb$i.json WTRK=none VP=$V PLAN="$P" python3 light.py "$@"; }
# v4: first lamp of each scene is already lit at the cut (no lit->dark jump at the crossfade)
Q1='[["e",6.5,0.6,0.4,300]]'
Q2='[["b",0.0,0.3,0.3,300],["c",9.5],["h",16.0],["e",24.0],["f",28.5]]'
Q3='[["b",0.0,0.3,0.2,380],["c",7.5],["d",14.0],["e",21.0],["i",26.0]]'
go3(){ i=$1; shift; eval P=\$Q$i V=\$V$i; PRELIT=a CLIP=slow$i.mp4 TRK=tr$i.json RT=rf$i.json SB=seedb$i.json WTRK=none VP=$V PLAN="$P" python3 light.py "$@"; }
