from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "capy-pomodoro"
BACKGROUND = PROJECT / "Gemini_Generated_Image_qd5160qd5160qd51.jpg"
OUTPUT = PROJECT / "capy-pomodoro-25-5.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 24
FOCUS_SECONDS = 25 * 60
PAUSE_SECONDS = 3
BREAK_SECONDS = 5 * 60
FINISH_SECONDS = 1
TOTAL_SECONDS = FOCUS_SECONDS + PAUSE_SECONDS + BREAK_SECONDS + FINISH_SECONDS

OVERLAY_WIDTH = 520
OVERLAY_HEIGHT = 520
OVERLAY_X = (WIDTH - OVERLAY_WIDTH) // 2
OVERLAY_Y = (HEIGHT - OVERLAY_HEIGHT) // 2
RING_CENTER = (OVERLAY_WIDTH // 2, OVERLAY_HEIGHT // 2)
RING_RADIUS = 195
RING_BOUNDS = (
    RING_CENTER[0] - RING_RADIUS,
    RING_CENTER[1] - RING_RADIUS,
    RING_CENTER[0] + RING_RADIUS,
    RING_CENTER[1] + RING_RADIUS,
)
RING_PINK = (236, 72, 153)
RING_TEAL = (20, 184, 166)


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def draw_timer_ring(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.ellipse(RING_BOUNDS, outline=(63, 63, 70, 180), width=5)

    angle_count = round(max(0.0, min(progress, 1.0)) * 360)
    points = []
    for angle in range(angle_count + 1):
        radians = math.radians(angle - 90)
        points.append(
            (
                RING_CENTER[0] + RING_RADIUS * math.cos(radians),
                RING_CENTER[1] + RING_RADIUS * math.sin(radians),
            )
        )

    for start, end in zip(points, points[1:]):
        x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        blend = max(0.0, min(1.0, (x + y) / (OVERLAY_WIDTH + OVERLAY_HEIGHT)))
        color = tuple(
            round(first + (last - first) * blend)
            for first, last in zip(RING_PINK, RING_TEAL)
        )
        draw.line((start, end), fill=(*color, 255), width=5)


def make_background() -> Image.Image:
    with Image.open(BACKGROUND) as source:
        image = ImageOps.fit(source.convert("RGB"), (WIDTH, HEIGHT))

    image = ImageEnhance.Brightness(image).enhance(0.78)
    image = ImageEnhance.Color(image).enhance(0.96)
    shade = Image.new("RGB", image.size, (8, 8, 10))
    return Image.blend(image, shade, 0.18).convert("RGBA")


def render_overlay(
    index: int,
    frames_dir: Path,
    time_font: ImageFont.FreeTypeFont,
) -> None:
    if index < FOCUS_SECONDS:
        remaining = FOCUS_SECONDS - index
        progress = remaining / FOCUS_SECONDS
    elif index < FOCUS_SECONDS + PAUSE_SECONDS:
        remaining = FOCUS_SECONDS + PAUSE_SECONDS - index
        progress = 1.0
    elif index < FOCUS_SECONDS + PAUSE_SECONDS + BREAK_SECONDS:
        remaining = FOCUS_SECONDS + PAUSE_SECONDS + BREAK_SECONDS - index
        progress = remaining / BREAK_SECONDS
    else:
        remaining = 0
        progress = 0.0

    minutes, seconds = divmod(remaining, 60)
    time_text = f"{minutes:02d}:{seconds:02d}"
    overlay = Image.new("RGBA", (OVERLAY_WIDTH, OVERLAY_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_timer_ring(draw, progress)
    draw.text(
        RING_CENTER,
        time_text,
        font=time_font,
        fill=(255, 255, 255, 255),
        anchor="mm",
    )
    overlay.save(frames_dir / f"frame_{index:05d}.png", optimize=True)


def main() -> None:
    if not BACKGROUND.is_file():
        raise FileNotFoundError(f"Background image not found: {BACKGROUND}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render the Pomodoro video.")

    fonts = Path(r"C:\Windows\Fonts")
    time_font = load_font(fonts / "segoeui.ttf", 106)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    background = make_background()

    with tempfile.TemporaryDirectory(prefix="capy-pomodoro-") as temp_dir:
        temp_path = Path(temp_dir)
        background_path = temp_path / "background.png"
        frames_dir = temp_path / "frames"
        frames_dir.mkdir()
        background.save(background_path, optimize=True)

        for index in range(TOTAL_SECONDS):
            render_overlay(index, frames_dir, time_font)

        frame_pattern = frames_dir / "frame_%05d.png"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(FPS),
                "-i",
                str(background_path),
                "-framerate",
                "1",
                "-i",
                str(frame_pattern),
                "-filter_complex",
                f"[1:v]fps={FPS}[overlay];"
                f"[0:v][overlay]overlay={OVERLAY_X}:{OVERLAY_Y}:shortest=1,"
                "format=yuv420p[v]",
                "-map",
                "[v]",
                "-an",
                "-frames:v",
                str(TOTAL_SECONDS * FPS),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(OUTPUT),
            ],
            check=True,
        )

    print(f"Rendered {TOTAL_SECONDS // 60}:{TOTAL_SECONDS % 60:02d} video: {OUTPUT}")


if __name__ == "__main__":
    main()
