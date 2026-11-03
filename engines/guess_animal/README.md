# Guess the animal

A glowing firefly ball flies in bouncing arcs (aimed at unpainted hexagons) and paints every hexagon it touches to reveal an animal, with a % counter, "guess before the end" title and a name reveal at 100%. Inspired by @everymergestudio's "Guess the Flag" format.

`python3 guess.py art.jpg NAME out.mp4` (add `still 8,16,26` for check frames). Hexagon cells. Art should be a single animal on a plain light background. Settings are listed at the top of `guess.py`.

Capybara settings: `CROP=39,734,1133,1958 K=16` (defaults for the rest) with art `refs/ideas/capybara_blanket_watercolour.jpg`.

Look: ZonelyTime style (dusk sky, warm gold glow, lower-case Poppins Light text, `z o n e l y t i m e` watermark). Title and wording are set in `guess.py` (`TITLE`, caption lines).

Art used so far: capybara `CROP=39,734,1133,1958` on `refs/ideas/capybara_blanket_watercolour.jpg`; otter `refs/ideas/otter_pad_960x1080.png` (the otter art padded to the grid shape; no CROP). Ball physics: gravity 2600 px/s2, restitution 0.8 (`REST`), many small balls, launch cadence speeds up over time (`CAD0`, `CAD1`, `TREF`).
