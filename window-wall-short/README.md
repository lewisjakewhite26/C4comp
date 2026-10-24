# Window wall Short ("someone's cat is up late")

A 10-second TikTok/Short: starts close on a cat silhouette in a TV-lit window, then pulls back until the window is one light in an endless wall of council-block windows.

## Status (24 Oct 2026)
- Video approved by Lewis (version 3: the photo's wall dissolves into the building, no "closing in").
- **Still to do:** swap in a better rain recording (Lewis is finding one), then mix and export with the music on top.
- On-screen text: "somewhere in this city, / someone's cat is up late".

## Files
| File | What it is |
| --- | --- |
| `wall2.py` | Renderer. `python3 wall2.py still 0,2,5` for test frames, `python3 wall2.py video` for `v.mp4` (10 s, 1080x1920, 24fps). |
| `audio.py` | Mixes cleared music with a rain bed. See the usage line at the top. |
| `facade.png` | Tower block facade (generated in Canva). Tiled seamlessly from crop y 25–1414, x 142–803. |
| `catwin.png` | Cat-in-window photo (generated in Canva). Its window frame maps onto one facade bay. |
| `window_wall_v3_silent.mp4` | The approved render, no sound. |

## Finish it
```
python3 audio.py 10 <music.mp3> <start s> <rain.mp3> <start s> a.wav
ffmpeg -i window_wall_v3_silent.mp4 -i a.wav -map 0:v -map 1:a -c:v copy \
  -af loudnorm=I=-14:TP=-1.5:LRA=11 -ar 48000 -c:a aac -b:a 192k -shortest \
  -movflags +faststart window_wall_final.mp4
```
The last mix used the Lyria rain-house music from 3314 s ("Coffee at Three"). Music and rain recordings are not stored here.

## Settings worth knowing
- Zoom: `K0=21` → `K1=0.24` over `T=10` s, eased (`x**1.5`).
- Photo dissolve: between zoom 14 and 6 (`fr` in `cat_layer`).
- Lit windows: 7.5% of panes, 12% switch during the clip, 18% of lit ones are TV flicker.
- Needs numpy, opencv-python, Pillow, ffmpeg.
