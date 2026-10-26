# Firefly street engine

A firefly drifts down a Veo street clip and lights each lantern by touching it. There's also a lit-window capybara cameo.

1. **Make the clip.** Generate it in Veo: a smooth, straight forward glide with every lamp unlit.
2. **Slow it down.** Slow it to a third of the speed:
   `ffmpeg -i clip.mp4 -an -vf "minterpolate=fps=72:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,setpts=3*PTS" -r 24 slow.mp4`
3. **Track the lanterns.** Write a seeds file (`[[name, frame, x, y], ...]`, one per lantern, at the lamp glass), then run:
   `CLIP=slow.mp4 SEEDS=seeds.json TRK=tracks.json python3 track.py`
   Check `trk.jpg` to make sure the tracks follow the lanterns.
4. **Track the window (optional).** Track the four corners of a window as W0–W3 (top-left, top-right, bottom-right, bottom-left), saved to `tracks_win.json`.
5. **Render.**
   `CLIP=slow.mp4 TRK=tracks.json python3 light2.py out.mp4` (or `still 5,10` for check frames)

**Settings in `light2.py`:**
- `PLAN`: `[lantern, time it decides to go, hover s, notice s, speed]`. `WIN` is the window.
- `RAMP`: how slow the opening is.
- `VP`: the vanishing point.
- `MERGE`: joins duplicate seeds of the same lantern.

**How it works:**
- The firefly is a small physics creature (steering, momentum and breeze) that lives close to the camera.
- Lit lanterns keep their light after they leave the frame.
- The lantern glass glows from the footage's own panes, so the frame and bars stay dark.
