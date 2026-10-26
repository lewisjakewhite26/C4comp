# Firefly walk home (v7, approved)

A firefly lights the lamps down a street, canal, pier and forest (~2:12, vertical).

1. `clips.sh` has the clip tables and per-scene plans. `track3.py` and `fit.py` give first-guess lamp tracks (dolly model p(t)=V+D/(T-t)); they were not accurate enough alone.
2. `export4.py` makes the model JSON and scene videos for `placer/index.html` (the Lamp Placer artifact). Place lamp boxes by hand there; saved corrections are in `fixes/`.
3. `light6.py` is the final renderer: soft-edged glass template, tight halo, no wide spill, warm floor pools. Settings are env vars (SCENE, PRELIT, KB, FXS, FYS, SHIFT, EDGE, HK1, HK2, DKF).
4. `run7.sh` renders the three scenes one at a time (parallel full-res renders run out of memory), crossfades at 32.75/65.5/98.25 s, loudnorm -14 LUFS, crf 22.
5. `light4.py`/`light5.py` are earlier versions; `chk*.py`, `spill4.py` and `stills*.py` are check tools.

Lesson: hand-placed boxes beat automatic tracking for lamps. Start with the placer.
