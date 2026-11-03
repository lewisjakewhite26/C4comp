# Guess the animal

A glowing firefly ball flies in bouncing arcs (aimed at unpainted hexagons) and paints every hexagon it touches to reveal an animal, with a % counter, "guess before the end" title and a name reveal at 100%. Inspired by @everymergestudio's "Guess the Flag" format.

`python3 guess.py art.jpg NAME out.mp4` (add `still 8,16,26` for check frames). Hexagon cells. Art should be a single animal on a plain light background. Settings are listed at the top of `guess.py`.

Capybara settings: `CROP=39,734,1133,1958 K=16` (defaults for the rest) with art `refs/ideas/capybara_blanket_watercolour.jpg`.

Look: ZonelyTime style (dusk sky, warm gold glow, lower-case Poppins Light text, `z o n e l y t i m e` watermark). Title and wording are set in `guess.py` (`TITLE`, caption lines).

Art used so far: capybara `CROP=39,734,1133,1958` on `refs/ideas/capybara_blanket_watercolour.jpg`; otter `refs/ideas/otter_pad_960x1080.png` (the otter art padded to the grid shape; no CROP). Ball physics: gravity 2600 px/s2, restitution 0.8 (`REST`), many small balls, launch cadence speeds up over time (`CAD0`, `CAD1`, `TREF`).
Red panda: `GH=820 GY0=420 K=16` with `refs/ideas/red_panda_pad_960x820.png` (wide art, shorter panel).

Ball motion modes (`MODE=`): `bounce` = gravity hoppers with energy-losing bounces (capybara); `bubble` = buoyant, wobbling bubbles that rise and pop (otter); `pool` = friction, elastic wall and ball-ball collisions, like pool balls (red panda). All balls stay inside the panel (walls on all four sides; the ball layer is clipped to the rounded panel).
