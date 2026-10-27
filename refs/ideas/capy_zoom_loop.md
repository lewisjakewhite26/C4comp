# Capybara infinite zoom loop (pond → dewdrop → mug → coffee drop → pond)

**Status:** PAUSED (27 Oct 2026). Supersedes the six-shot chain in `capy_zoom_prompts.md`.

## What Lewis wants

An endless zoom Short (9:16). The camera only ever pushes **forward**: from the capybara's pond into a dewdrop on a lily pad, into a cosy room, onto a coffee drop on the mug rim, through that drop and back out over the **same pond**, landing on the exact opening frame so the video loops forever. No pull-backs.

The old six-shot chain didn't loop because shot 6 pulled back to a living room.

## Tool: Google Flow (Veo)

- Using Flow because it takes a **first frame and a last frame** per shot, so every shot's ends can be pinned and the last shot can end on the first image.
- Checked Meta AI on desktop for Midjourney: Meta licensed Midjourney's tech and Vibes was reported to be powered by it at launch, but you can't pick Midjourney as a model and there's no first/last-frame control. Not useful for this.
- Gemini-only chaining (start frame only) kept drifting in style and starting point.

## Attempt 1 in Flow (27 Oct): what went wrong

Three 8 s clips (720×1280) reviewed:

1. **K1 → K2:** the push from the pond into the dewdrop worked, but the end morphed into a different room and a glossier, non-watercolour look.
2. **Mug clip:** didn't move at all, and came out photoreal (wrong style, different mug).
3. **Coffee-drop clip:** held still for about 6 s, then jumped to a capybara with no scarf in a different style.

Causes:
- **Keyframe styles didn't match** (watercolour pond vs photoreal mug vs glossy digital coffee drop), so Flow froze or jumped.
- **Drop shapes didn't match.** The dewdrop on the lily pad is a round dome bead; the K2 and coffee-drop images were hanging teardrops, so Flow had to morph between them.
- **First and last frames too far apart** for the model to find a path, so it holds then cuts.

## The fix: build each drop from the scene inside it

Make the destination image first, then wrap that exact image in a drop. Each video then only has to zoom, not invent.

Rules for every keyframe:
- Same watercolour style as K1 (attach K1 as the style reference every time).
- Drops are **round dome-shaped beads sitting on a surface**, never hanging teardrops.
- Capybara always has the teal scarf with one pink band.

### Keyframes (keep the existing K1; redo the rest in this order)

**K1, pond (start and end frame). Already made and approved.** For reference, the prompt:

```
This exact capybara from the reference images, same face, colours and teal scarf with one pink band, floating calmly in a misty pond at dusk in light rain, eyes half closed. Foreground: a large lily pad with one big sparkling dewdrop centred in the frame. Rain ripples, mist between dark reeds, cool blue-green light. Soft storybook illustration, watercolour lighting, cosy lofi mood. Vertical 9:16, no text.
```

**K3, the mug.** Make this first, with K1 attached as the style reference.

```
Same watercolour style as the reference image: soft hand-painted watercolour on textured paper, visible paper grain, muted colours, gentle lamplight. Not photoreal, not 3D. Close-up of a cream ceramic mug with a teal and pink zigzag band on a wooden side table, a little steam rising, a lamp and a rainy window softly blurred behind. On the rim, centred in the frame, sits one small round bead of dark glossy coffee, dome-shaped, not a hanging teardrop. Vertical 9:16, no text.
```

**K2, the dewdrop.** Attach K3, plus K1 for style.

```
Same watercolour style as the reference images. Extreme close-up of one round, dome-shaped dewdrop sitting on a green lily pad, filling almost the whole frame, with the green lily pad visible only at the edges. Inside the dewdrop, seen clearly through the water, is exactly the mug scene from the reference image, same composition, slightly curved by the water. Soft sparkles on the drop's surface. Not a hanging teardrop. Not photoreal. Vertical 9:16, no text.
```

**K4, the coffee drop.** Attach K1, plus K3 for style.

```
Same watercolour style as the reference images. Extreme close-up of one round, dome-shaped bead of dark glossy coffee sitting on the rim of a cream mug, filling almost the whole frame, with the cream glaze visible only at the edges. Inside the drop, seen clearly through it, is exactly the pond scene from the reference image: the capybara floating on its back in its teal scarf with a pink band, and the lily pad with its dewdrop. Slightly curved, warm amber tint at the edges. Not a hanging teardrop. Not photoreal. Vertical 9:16, no text.
```

### Videos (first frame → last frame, 8 s each, about 32 s per loop)

| Shot | First frame | Last frame |
|---|---|---|
| 1 | K1 | K2 |
| 2 | K2 | K3 |
| 3 | K3 | K4 |
| 4 | K4 | K1 |

**Shot 1: K1 → K2**

```
Macro dolly-in. The camera is already moving forward from the very first frame and never stops. It glides low over the rainy pond, past the sleeping capybara, down towards the lily pad, and closes in on the dewdrop until the drop fills the frame. The scene inside the drop becomes clearer as we approach. Same watercolour style throughout, no style change, no cuts.
```

**Shot 2: K2 → K3**

```
Macro dolly-in. The camera is already moving forward from the very first frame and never stops. It pushes straight into the dewdrop: the curved water and the lily pad edges slide past and out of frame, the curve straightens out, and the mug scene inside fills the whole screen. Same watercolour style throughout, no style change, no cuts.
```

**Shot 3: K3 → K4**

```
Macro dolly-in. The camera is already moving forward from the very first frame and never stops. Steam drifts as it moves towards the mug, then down to the small bead of coffee on the rim, until the coffee drop fills the frame and the pond scene inside it becomes clear. Same watercolour style throughout, no style change, no cuts.
```

**Shot 4: K4 → K1**

```
Macro dolly-in. The camera is already moving forward from the very first frame and never stops. It pushes straight into the coffee drop: the cream glaze and the drop's edges slide past and out of frame, the amber tint fades, the curve straightens out, and the rainy pond with the sleeping capybara fills the whole screen. Same watercolour style throughout, no style change, no cuts.
```

Shots 2 and 4 should be easy, because the scene inside each drop is already the next keyframe. Shots 1 and 3 are the long moves; if one freezes, regenerate and keep the best take.

## When it resumes

1. Lewis makes K3, then K2 and K4, in Flow with K1 attached as the style reference.
2. Lewis generates the four shots with first and last frames set.
3. Claude stitches them: evens out camera speed at the joins, checks the loop point (blends the last second into K1 if it doesn't land exactly), applies one colour grade, exports 1080×1920, and adds Lyria-only lofi music.
