# Pastel maze ball (TikTok / Shorts)

A glowing pastel ball rolls round a floating stone maze in space. The ball runs straight into a plate on every beat, hops, rebounds 90° and the plate flashes, throws sparkles and plays a soft marimba note, over a gentle pad and bass. 1080x1920, 30 fps, 64.5 s, no text or branding.
Original scene and music, inspired by the "ball and note plates" music-maze style of video. Not a copy of any specific video.

## Files
- `pastel_maze_ball_tiktok_1080x1920.mp4` final video
- `source/scene.py` maze path, plates and melody, `source/maze.py` renderer, `source/aud.py` music (hit times come from the same scene, so notes match the visuals exactly)
- Render: `python3 maze.py 0 968 a.mp4` and `python3 maze.py 968 1936 b.mp4`, then join and add the audio from `aud.py`.
