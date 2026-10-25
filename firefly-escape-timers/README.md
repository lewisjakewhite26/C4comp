# Firefly Escape classroom timers

A firefly escapes ring by ring over a blurred rainy-window background; the last ring opens as the countdown hits 0:00.

| File | What |
| --- | --- |
| `firefly_escape_timer_10min.mp4` | 10-minute timer, 1920x1080, HEVC, music only (Lyria city-night tracks, see `music_tracklist.txt`) |
| `firefly_escape_timer_10min_thumb.jpg` | YouTube thumbnail 1280x720 |
| `firefly_timer.py` | Renderer. Pure physics (gravity, elastic bounces, constant-speed rings); rings unlock on a schedule and a seed search finds a run whose last escape lands on T. Needs `bgclip.mp4` (Gemini rainy window, not in repo) and the music bed. |

Settings for 10 min: `T=600 NR=18 S=1`. Resumable render in 13 chunks (`seg/`).
Sound: no bounce notes (they clashed with the music); a soft chime per escape and an end chord, each tuned to the key of the track playing at that moment.
Upload: tick altered/synthetic content; audience "not made for kids".
