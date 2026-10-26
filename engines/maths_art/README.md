# Maths art engine

Glowing teal-to-pink line art on navy, drawn by maths.

- `maths.py`: the shared pieces.
  - Pattern generators: bouncing light in a circle (with or without gravity), spirograph, times-table string art.
  - `colourise()`: turns line density and draw order into the ZonelyTime glow.
  - `python3 maths.py look.png` renders a comparison sheet.
- `short.py`: **"join each dot to its double"** Short.
  - 240 dots, with lines drawn one by one into a cardioid (heart), then the multiplier morphs from ×2 to ×5.
  - Settings: `N`, `TA` (end of the drawing phase), `K1` (final multiplier), `HOOK1`/`HOOK2`. `NOTEG` adds pluck chimes; the default is 0, music only.
- `capy_fourier.py`: **"can maths draw a capybara?"** Short.
  - Fourier epicycles trace the outline of `capy_sil.jpg` in stages of 1, 2, 4, 8, 16, 32, 64 and 128 spinning circles, then the shape fills and the eye appears.
  - Any solid black-on-white silhouette works: set `SIL=` to the file, and `EYEX`/`EYEY` to the eye position in that image's pixels.
  - Stage loop times are in `SD`.

Both use `S=0.5` for a quick test or `S=1` for full 1080×1920, plus `still t1,t2` for check frames. Music comes from `citybed/city_night_bed.mp3`.
