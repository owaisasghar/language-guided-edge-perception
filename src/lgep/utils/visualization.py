"""Draw detections on images with Pillow (no OpenCV dependency in Milestone 1)."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from lgep.perception.types import DetectionResult

# Colour-blind-friendly palette (Okabe-Ito), cycled per phrase.
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
           (0, 114, 178), (213, 94, 0), (204, 121, 167), (255, 255, 255)]


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def color_for(phrase_id: int) -> tuple[int, int, int]:
    return PALETTE[phrase_id % len(PALETTE)]


def label_text(det, show_tokens: bool = True) -> str:
    """'phrase 0.65', plus the model's raw token label when it differs, e.g. 'red car 0.64 [white truck red car]'."""
    text = f"{det.phrase} {det.score:.2f}"
    if show_tokens and det.token_label and det.token_label != det.phrase:
        text += f" [{det.token_label}]"
    return text


def draw_detections(image: Image.Image, result: DetectionResult, header: str | None = None,
                    show_tokens: bool = True) -> Image.Image:
    """Return a copy of ``image`` with boxes, labels and an optional header strip."""
    img = image.convert("RGB").copy()
    w, h = img.size
    scale = max(w, h) / 1000
    line_w = max(2, round(3 * scale))
    font = _font(max(12, round(18 * scale)))
    draw = ImageDraw.Draw(img)

    # draw low scores first so confident boxes end up on top
    for det in sorted(result.detections, key=lambda d: d.score):
        x0, y0, x1, y1 = det.box_xyxy
        c = color_for(det.phrase_id)
        draw.rectangle([x0, y0, x1, y1], outline=c, width=line_w)
        text = label_text(det, show_tokens)
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y0 - th - 4 if y0 - th - 4 >= 0 else y0
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th + 4], fill=c)
        draw.text((x0 + 3, ty + 1 - tb[1]), text, fill=(0, 0, 0), font=font)

    if header:
        hfont = _font(max(12, round(16 * scale)))
        tb = draw.textbbox((0, 0), header, font=hfont)
        strip_h = tb[3] - tb[1] + 10
        canvas = Image.new("RGB", (w, h + strip_h), (20, 20, 20))
        canvas.paste(img, (0, strip_h))
        ImageDraw.Draw(canvas).text((6, 5 - tb[1]), header, fill=(255, 255, 255), font=hfont)
        img = canvas
    return img
