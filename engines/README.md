# ZonelyTime engines

Every Python script used to make the ZonelyTime videos, saved so future videos can start from a working engine and only need tweaking.

## How to run

- Each folder is one engine. Inputs sit next to the script, or in a sibling folder.
- Paths resolve from `ZT_WORK`. If it isn't set, they resolve from this `engines/` folder. So `python3 tower/tower.py out.mp4` works from here once the inputs are in place.
- Most render scripts take `S=0.5` (the default) for a quick half-resolution test and `S=1` for full 1080p. For example:

  ```
  S=1 python3 tower/tower.py firefly_tower.mp4
  ```

- Many have `SIMONLY=1`, which runs only the physics and prints the timings. Use it for tuning pacing before rendering.
- `still t1,t2,...` renders check frames instead of a video, for example:

  ```
  python3 tower/tower.py still 2,20,58
  ```

- Needs Python 3 with numpy, opencv-python, pillow and scipy, plus ffmpeg.

## The engines

| Folder | Makes | Key settings to tweak |
|---|---|---|
| `tower/` | **Firefly tower Short.** A jar of 170 fireflies on a ledge. Each escape flies across and lights one flat in a tower block during a power cut. Realistic lights (2200–4800 K, curtains, blinds, some stutter on, one broken bulb), and the streetlight returns at the end. Inputs: `dark.png` and `lit.png`. | `GW0`/`GW1`/`GP` (lid gap widening = pacing), `pick_light()` (colour temperature mix), `BROKEN` (which flat flickers), `WIN` grid for a new building image |
| `jar/` | **Firefly jar Short.** Every firefly that escapes brings back two friends. `mock.py` makes look-development stills. | `GW`, `LIFT`, `SEED`, `CAP` |
| `escape/` | **Ring escape.** `firefly5.py` is the 15 s TikTok version, `firefly6.py` the 10-minute classroom timer (rings unlock on a schedule, seed search, resumable chunks) and `escape.py` the original ball escape. Input: `bgclip.mp4`. | `UNL` schedule, seed, colours |
| `firefly_street/walk_home/` | **Firefly walk home.** Hand-placed lamp boxes (Lamp Placer page) drive `light6.py`; see its README. | `KB`, `EDGE`, `FXS`, `SHIFT` |
| `crystal/` | **Snowflake Short.** Real snowfall, and flakes that touch the crystal freeze to it. `reiter.py` is the snowflake growth model. `crystal.py` was the first version, without misses. Input: `win.png`. | `R0`/`R1` (snowfall rate = pacing), `PAR` in `reiter.grow()` (snowflake shape), `X0` crop |
| `globe/` | Snow globe Short. **Abandoned:** the escape idea didn't make sense. | — |
| `capy/` | **Capybara loops.** `loop.py`/`loop_tall.py` use an optical-flow morph so a short AI clip loops seamlessly. `addrain.py` adds three seamless rain layers. | loop blend length, rain strength |
| `pondmusic/` | **2-hour music mix.** Each track is levelled to −16 LUFS, with pauses between tracks, a limiter, streaming output and a tracklist with timestamps. | track order list, pause length |
| `rainmix/` | One-hour music and rain mix (Lyria tracks). | track list |
| `rain2/` | `bed.py` turns a short rain recording into a long, non-repeating bed. | slice length, crossfade |
| `jelly/`, `jf/` | **Jellyfish timers** (1, 2, 5 and 10 minutes): the scene, particles, timer digits and seamless loop. | timer lengths in `run_timers.sh` |
| `lava/` | **Lava lamp timers** (2, 3, 5 and 10 minutes, plus a portrait TikTok version) and the ring timer overlay. | blob seeds, colours |
| `pomodoro/` | **Capy Pomodoro timer.** Renders a centred, transparent pink-to-teal gradient ring and white countdown over the project image, with a 25-minute focus countdown, a 3-second pause and a 5-minute break. | `FOCUS_SECONDS`, `PAUSE_SECONDS`, `BREAK_SECONDS` |
| `wall/`, `wall2/` | **Window wall Short** (London tower windows, cat and plant, TV glow). | facade tile, pane list |
| `tt2/`, `pb/` | TikTok edits: text layers, night sound effects, pan-and-crop from a landscape clip. | — |
| `alp/`, `aur/`, `cafe/`, `harbour/`, `harb2/`, `hut/`, `block/`, `bus/`, `rain/`, `train/` | **Scene loops for long YouTube videos:** Alpine cycle, aurora, autumn café day-night cycle, harbour and boat, snowy hut blizzard, tower block windows, bus window, rain house, train bridge. | — |
| `rainaudio/`, `trainaudio/`, `nordic_audio/`, `bus/mixa.py` | Sound design: rain sound effects tied to video events, train passes, music beds. | — |
| `thumb/` | Thumbnail makers. | text, crop |
| `citybed/` | `city_night_bed.mp3`: the Lyria-only city-night music bed (no birds or crickets) used by `tower/`. | — |

## Not included (keep it that way)

- **Pixabay audio:** never goes in this repo.
- **Big music files:** for example `pondmusic/*.wav` (34 MB+). Scripts look for them by name, so put them back in the same folder before running.
- **Rendered videos:** these live in their own project folders, not here.
- **Uploaded clips:** scripts that read uploads expect them in `uploads/`.

## Checks before upload

- Key detection (chroma plus Krumhansl profile) and the nature-sound check are in the playbook.
- Always render a few `still` frames and look at them before a full render.
