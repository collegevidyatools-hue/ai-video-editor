"""PIL drawing helpers: easing, fonts, cards, captions and simple icons."""
from functools import lru_cache
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/opentype/inter/"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# Palette picked to match the red/white set behind the speaker.
RED = (229, 56, 59)
NAVY = (11, 31, 58)
WHITE = (255, 255, 255)
YELLOW = (255, 214, 10)
GREEN = (34, 197, 94)
GREY = (90, 102, 120)


@lru_cache(None)
def font(weight="Black", size=60, display=True):
    name = ("InterDisplay-" if display else "Inter-") + weight + ".otf"
    return ImageFont.truetype(FONT_DIR + name, size)


# ---------------------------------------------------------------- easing

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_out_back(x, s=1.70158):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ease_in_out(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def envelope(t, start, end, fade_in=0.35, fade_out=0.25):
    """Return (in_progress, out_progress) both in 0..1 for an element visible on [start, end]."""
    if t < start or t > end + fade_out:
        return None
    return clamp((t - start) / fade_in), clamp((t - end) / fade_out)


# ---------------------------------------------------------------- primitives

def text_size(txt, f):
    l, t, r, b = f.getbbox(txt)
    return r - l, b - t, l, t


def shadowed(img, radius=18, offset=(0, 10), opacity=110):
    """Return a new RGBA image of img with a soft drop shadow (padded)."""
    pad = radius * 2
    W, H = img.width + pad * 2, img.height + pad * 2
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    a = img.split()[3].point(lambda v: v * opacity // 255)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 255))
    sh.putalpha(a)
    out.alpha_composite(sh, (pad + offset[0], pad + offset[1]))
    out = out.filter(ImageFilter.GaussianBlur(radius))
    out.alpha_composite(img, (pad, pad))
    return out


def pill(txt, f, fg=WHITE, bg=RED, padx=28, pady=16, radius=None):
    w, h, l, t = text_size(txt, f)
    W, H = w + padx * 2, h + pady * 2
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=radius or H // 2, fill=bg)
    d.text((padx - l, pady - t), txt, font=f, fill=fg)
    return im


def emoji(ch, size):
    """Render a colour emoji at roughly `size` px (Noto only rasterises at 109px)."""
    f = ImageFont.truetype(EMOJI_FONT, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)


def check_icon(size, color=GREEN):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, size - 1, size - 1), fill=color)
    w = max(3, size // 9)
    d.line([(size * .27, size * .52), (size * .44, size * .69), (size * .75, size * .34)], fill=WHITE, width=w, joint="curve")
    return im


def card(w, h, bg=WHITE, radius=36, border=None):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=bg, outline=border, width=4 if border else 0)
    return im


def transform(img, scale=1.0, alpha=1.0, rotate=0.0):
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, expand=True)
    if abs(scale - 1.0) > 1e-3:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        a = img.split()[3].point(lambda v: int(v * clamp(alpha)))
        img = img.copy()
        img.putalpha(a)
    return img


def paste_center(canvas, img, cx, cy):
    canvas.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


def place(canvas, img, cx, cy, t, start, end, style="up", fade_in=0.35, fade_out=0.25, rise=60):
    """Animate `img` in and out around centre (cx, cy). Returns True if drawn."""
    env = envelope(t, start, end, fade_in, fade_out)
    if env is None:
        return False
    pin, pout = env
    alpha = ease_out_cubic(pin) * (1 - ease_out_cubic(pout))
    if alpha <= 0.01:
        return False
    if style == "up":
        dy = (1 - ease_out_cubic(pin)) * rise - ease_out_cubic(pout) * rise * 0.5
        paste_center(canvas, transform(img, alpha=alpha), cx, cy + dy)
    elif style == "pop":
        s = 0.3 + 0.7 * ease_out_back(pin)
        s *= 1 - 0.15 * ease_out_cubic(pout)
        paste_center(canvas, transform(img, scale=s, alpha=alpha), cx, cy)
    elif style == "left":
        dx = -(1 - ease_out_cubic(pin)) * 400
        paste_center(canvas, transform(img, alpha=alpha), cx + dx, cy)
    elif style == "stamp":
        s = 2.2 - 1.2 * ease_out_cubic(pin)
        s *= 1 - 0.1 * ease_out_cubic(pout)
        paste_center(canvas, transform(img, scale=s, alpha=alpha), cx, cy)
    return True


# ---------------------------------------------------------------- captions

def caption_image(words, highlight, size=74, active=None):
    """Bold caption with thick outline; highlighted words in yellow."""
    f = font("Black", size)
    stroke = max(6, size // 9)
    space = f.getlength(" ") + stroke * 1.4
    widths = [f.getlength(w) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    H = int(size * 1.45) + stroke * 2
    im = Image.new("RGBA", (int(total) + stroke * 2 + 8, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = stroke + 4
    for w, wd in zip(words, widths):
        key = w.strip(",.!?'\"").lower()
        col = YELLOW if key in highlight else WHITE
        d.text((x, stroke), w, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
        x += wd + space
    return im
