"""IGNOU MCA explainer reel: cuts, captions, motion graphics, SFX and music.

Usage:  python3 projects/ignou-mca/edit.py <input.mov> <output_dir> [--preview]
"""
import json
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from editor import gfx, sfx  # noqa: E402
from editor.gfx import NAVY, RED, WHITE, YELLOW, GREEN, GREY, font  # noqa: E402

W, H, FPS = 1080, 1920, 30
SRC_W, SRC_H = 720, 1280
CAPTION_Y = 1440
TOP_Y = 250  # first row of on-screen graphics (clear of the platform's top UI)

# --------------------------------------------------------------------------
# Script. Times are in SOURCE seconds and come from the speech/pause analysis
# (energy < -38 dB for >= 0.35 s). The muted intro (0-8.3 s) and the muted
# retake (49.3-57.2 s, incl. the cut-off "jo aapko pr...") are dropped.
# Chunks are separated with "|" and become one caption each.
PHRASES = [
    (8.34, 14.74, "IGNOU Master of | Computer Applications | MCA ki complete | details sirf | 50 seconds mein"),
    (15.26, 20.64, "1 lakh rupaye | private college mein | waste karne se pehle | ye video last tak | zaroor dekhna"),
    (21.00, 24.70, "IGNOU ki eligibility | simple hai | kisi bhi recognised | university se"),
    (25.10, 29.58, "BCA, | MSc Computer Science, | IT ya kisi | bhi stream se"),
    (30.00, 32.80, "graduation complete | honi chahiye | with Mathematics"),
    (33.60, 39.12, "Koi entrance | exam nahi hai | Duration ki | baat karein toh | minimum 2 saal | aur maximum"),
    (39.58, 42.62, "4 saal ki duration | hoti hai, matlab IGNOU | aapko flexibility | deta hai"),
    (43.06, 46.78, "Poore program ki fees | lagbhag 50 se 52 | hazaar ho jaati hai"),
    (47.18, 49.28, "jo aap semester-wise | ya year-wise | bhi de sakte hain"),
    (57.16, 60.86, "Ab baat karte hain | placement ki | Placement ka sach | ye hai ki IGNOU ka"),
    (61.16, 66.60, "CPC placement drive | organise karta hai | but ye degree | zyada valuable | unke liye hai"),
    (67.02, 72.86, "jo software | development, | data science, | IT industry mein job | ya high-tech roles | ki taiyari"),
    (73.30, 74.78, "ke saath apni | master degree"),
    (75.24, 77.46, "complete karna | chahte hain"),
    (77.84, 84.16, "Maine complete | syllabus, | project guidelines | aur admission steps ki | ek PDF guide | banayi hai"),
    (84.66, 89.20, "jo aapko ek | comment mein milegi | aapko bas comment mein | \"MCA\" karna hai"),
    (89.74, 91.12, "guide aapke DM | mein pahunch jayegi"),
]
TAIL_END = 92.38  # hold the last frames for the CTA

HIGHLIGHT = {
    "ignou", "mca", "\"mca\"", "1", "lakh", "waste", "zaroor", "eligibility", "bca", "msc", "it",
    "mathematics", "entrance", "nahi", "2", "4", "saal", "flexibility", "50", "52", "hazaar",
    "semester-wise", "year-wise", "placement", "sach", "cpc", "valuable", "software", "development",
    "data", "science", "high-tech", "master", "degree", "syllabus", "project", "guidelines",
    "admission", "steps", "pdf", "guide", "dm",
}


# --------------------------------------------------------------------------
# Timeline

def build_segments():
    segs = []
    for i, (a, b, _) in enumerate(PHRASES):
        end = b + 0.10 if i < len(PHRASES) - 1 else TAIL_END
        s = [max(0.0, a - 0.08), end]
        if segs and s[0] - segs[-1][1] < 0.12:
            segs[-1][1] = s[1]
        else:
            segs.append(s)
    out, o = [], 0.0
    for a, b in segs:
        out.append((a, b, o))
        o += b - a
    return out, o


SEGMENTS, DURATION = build_segments()


def out_t(src):
    """Map a source time to output time (times inside a cut snap to the next kept frame)."""
    for a, b, o in SEGMENTS:
        if src < a:
            return o
        if src <= b:
            return o + src - a
    return DURATION


def seg_index(t):
    for i, (a, b, o) in enumerate(SEGMENTS):
        if t < o + (b - a):
            return i
    return len(SEGMENTS) - 1


# --------------------------------------------------------------------------
# Word timing: spread each phrase's characters over its *speech* frames.

def speech_mask(wav16k):
    a, sr = sf.read(wav16k, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    hop = int(0.02 * sr)
    n = len(a) // hop
    e = 20 * np.log10(np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    return e > -38, 0.02


def word_timings(mask, step):
    words = []  # (word, src_start, src_end, phrase_idx, chunk_idx)
    for pi, (a, b, text) in enumerate(PHRASES):
        frames = [i for i in range(int(a / step), int(b / step)) if mask[min(i, len(mask) - 1)]]
        if not frames:
            frames = list(range(int(a / step), int(b / step)))
        chunks = [c.split() for c in text.split("|")]
        flat = [(w, ci) for ci, c in enumerate(chunks) for w in c]
        weights = np.array([len(w) + 1.5 for w, _ in flat], float)
        cum = np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
        idx = lambda f: frames[min(len(frames) - 1, int(f * len(frames)))] * step
        for k, (w, ci) in enumerate(flat):
            words.append((w, idx(cum[k]), idx(cum[k + 1]) + step, pi, ci))
    return words


def build_captions(words):
    caps = []
    for pi in range(len(PHRASES)):
        pw = [w for w in words if w[3] == pi]
        for ci in sorted({w[4] for w in pw}):
            cw = [w for w in pw if w[4] == ci]
            caps.append([out_t(cw[0][1]), out_t(cw[-1][2]), [w[0] for w in cw]])
    for i in range(len(caps) - 1):  # no gaps/flicker between consecutive captions
        if caps[i + 1][0] - caps[i][1] < 0.6:
            caps[i][1] = caps[i + 1][0]
    caps[-1][1] = DURATION
    return caps


# --------------------------------------------------------------------------
# Graphics assets (cached)

@lru_cache(None)
def a_header(txt, bg=RED, fg=WHITE):
    return gfx.shadowed(gfx.pill(txt, font("Black", 56), fg=fg, bg=bg, padx=40, pady=20), 12, (0, 6), 90)


@lru_cache(None)
def a_row(txt, icon="check", size=46, w=900, bg=WHITE, fg=NAVY):
    f = font("Bold", size)
    tw, th, l, t = gfx.text_size(txt, f)
    ic = gfx.check_icon(int(size * 1.25)) if icon == "check" else (gfx.emoji(icon, int(size * 1.2)) if icon else None)
    padx, pady, gap = 26, 22, 20
    iw = ic.width + gap if ic else 0
    Wd = max(w or 0, tw + iw + padx * 2)
    Hd = max(th, ic.height if ic else 0) + pady * 2
    im = gfx.card(Wd, Hd, bg=bg, radius=28)
    if ic:
        im.alpha_composite(ic, (padx, (Hd - ic.height) // 2))
    tx = padx + iw if ic else (Wd - tw) // 2
    ImageDraw.Draw(im).text((tx - l, (Hd - th) // 2 - t), txt, font=f, fill=fg)
    return gfx.shadowed(im, 14, (0, 8), 80)


@lru_cache(None)
def a_hook_title():
    im = gfx.card(780, 250, bg=NAVY, radius=44)
    f = font("Black", 190)
    tw, th, l, t = gfx.text_size("IGNOU", f)
    ImageDraw.Draw(im).text(((780 - tw) // 2 - l, (250 - th) // 2 - t), "IGNOU", font=f, fill=WHITE)
    return gfx.shadowed(im, 22, (0, 12), 120)


@lru_cache(None)
def a_hook_mca():
    return gfx.shadowed(gfx.pill("MCA", font("Black", 120), bg=RED, padx=48, pady=14, radius=30), 16, (0, 8), 110)


@lru_cache(None)
def a_hook_sub():
    return gfx.shadowed(gfx.pill("Complete Details", font("ExtraBold", 54), fg=NAVY, bg=YELLOW, padx=34, pady=16), 12, (0, 6), 90)


@lru_cache(None)
def a_money(struck_frac=0):
    im = gfx.card(760, 300, radius=40)
    d = ImageDraw.Draw(im)
    f1, f2 = font("SemiBold", 44), font("Black", 140)
    d.text((48, 36), "Private college fees", font=f1, fill=GREY)
    tw, th, l, t = gfx.text_size("₹1,00,000", f2)
    d.text((48 - l, 120 - t), "₹1,00,000", font=f2, fill=NAVY)
    if struck_frac > 0:
        y = 120 + th // 2 + 6
        x1 = 40 + (tw + 30) * struck_frac
        d.line([(36, y + 10), (x1, y - 10)], fill=RED, width=16)
    return gfx.shadowed(im, 20, (0, 10), 110)


@lru_cache(None)
def a_stamp():
    f = font("Black", 92)
    lines = ["NO ENTRANCE", "EXAM"]
    sizes = [gfx.text_size(s, f) for s in lines]
    Wd = max(s[0] for s in sizes) + 100
    Hd = sum(s[1] for s in sizes) + 40 + 90
    im = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, Wd - 1, Hd - 1), radius=30, fill=(255, 255, 255, 235), outline=GREEN, width=14)
    d.rounded_rectangle((22, 22, Wd - 23, Hd - 23), radius=20, outline=GREEN, width=4)
    y = 45
    for s, (tw, th, l, t) in zip(lines, sizes):
        d.text(((Wd - tw) // 2 - l, y - t), s, font=f, fill=GREEN)
        y += th + 40
    return gfx.shadowed(im.rotate(-7, resample=Image.BICUBIC, expand=True), 16, (0, 8), 100)


def draw_duration_bar(canvas, t, t_show, t_min, t_max, t_end):
    env = gfx.envelope(t, t_show, t_end)
    if env is None:
        return
    pin, pout = env
    alpha = gfx.ease_out_cubic(pin) * (1 - gfx.ease_out_cubic(pout))
    Wd, Hd = 900, 270
    im = gfx.card(Wd, Hd, radius=36)
    d = ImageDraw.Draw(im)
    x0, x1, y = 60, Wd - 60, 120
    d.rounded_rectangle((x0, y - 16, x1, y + 16), radius=16, fill=(228, 232, 240))
    # fill to 2 years, then on to 4
    years = 2 * gfx.ease_in_out((t - t_min) / 0.6) + 2 * gfx.ease_in_out((t - t_max) / 0.8)
    if years > 0.02:
        d.rounded_rectangle((x0, y - 16, x0 + (x1 - x0) * years / 4, y + 16), radius=16, fill=RED)
    fl = font("Bold", 34)
    for k in range(5):
        x = x0 + (x1 - x0) * k / 4
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE if years < k else NAVY)
        tw = fl.getlength(f"{k}y")
        d.text((x - tw / 2, y + 32), f"{k}y", font=fl, fill=GREY)
    fb = font("Black", 46)
    if t >= t_min:
        d.text((x0 + (x1 - x0) * 2 / 4 - fb.getlength("MIN 2 YRS") / 2, 22), "MIN 2 YRS", font=fb, fill=NAVY)
    if t >= t_max:
        d.text((x1 - fb.getlength("MAX 4 YRS") + 10, 22), "MAX 4 YRS", font=fb, fill=RED)
    im = gfx.shadowed(im, 18, (0, 10), 100)
    dy = (1 - gfx.ease_out_cubic(pin)) * 60
    gfx.paste_center(canvas, gfx.transform(im, alpha=alpha), W / 2, TOP_Y + 260 + dy)


def draw_fee_counter(canvas, t, t_show, t_count, t_done, t_end):
    env = gfx.envelope(t, t_show, t_end)
    if env is None:
        return
    pin, pout = env
    alpha = gfx.ease_out_cubic(pin) * (1 - gfx.ease_out_cubic(pout))
    p = gfx.ease_out_cubic((t - t_count) / max(0.3, t_done - t_count))
    lo, hi = int(round(50000 * p, -2)), int(round(52000 * p, -2))
    fmt = lambda v: "₹" + f"{v:,}".replace(",", ",")
    txt = f"{fmt(lo)} – {fmt(hi)}"
    Wd, Hd = 920, 240
    im = gfx.card(Wd, Hd, radius=36)
    d = ImageDraw.Draw(im)
    f = font("Black", 92)
    tw, th, l, tt = gfx.text_size(txt, f)
    d.text(((Wd - tw) // 2 - l, 38 - tt), txt, font=f, fill=NAVY if p < 1 else GREEN)
    fs = font("SemiBold", 38)
    sub = "approx. for the full programme"
    d.text(((Wd - fs.getlength(sub)) // 2, 170), sub, font=fs, fill=GREY)
    im = gfx.shadowed(im, 18, (0, 10), 100)
    s = 1.0 + 0.06 * max(0.0, 1 - abs(t - t_done) / 0.15)  # little bump when it lands
    dy = (1 - gfx.ease_out_cubic(pin)) * 60
    gfx.paste_center(canvas, gfx.transform(im, scale=s, alpha=alpha), W / 2, TOP_Y + 250 + dy)


@lru_cache(None)
def a_comment_box(typed):
    Wd, Hd = 900, 150
    im = gfx.card(Wd, Hd, radius=75)
    d = ImageDraw.Draw(im)
    d.ellipse((24, 24, 126, 126), fill=(230, 233, 240))
    d.ellipse((58, 40, 92, 74), fill=(160, 168, 184))
    d.pieslice((40, 80, 110, 150), 180, 360, fill=(160, 168, 184))
    f = font("Bold", 52, display=False)
    if typed:
        d.text((156, 42), typed, font=f, fill=NAVY)
        x = 160 + f.getlength(typed)
    else:
        d.text((156, 46), "Add a comment…", font=font("Medium", 46, display=False), fill=(150, 158, 172))
        x = 156
    d.rounded_rectangle((x + 4, 40, x + 10, 110), radius=3, fill=RED)  # caret
    send = gfx.pill("Post", font("Bold", 40), bg=RED, padx=30, pady=16)
    im.alpha_composite(send, (Wd - send.width - 26, (Hd - send.height) // 2))
    return gfx.shadowed(im, 20, (0, 10), 110)


@lru_cache(None)
def a_pdf_title():
    im = gfx.card(860, 190, bg=NAVY, radius=40)
    ic = gfx.emoji("\U0001F4C4", 110)
    im.alpha_composite(ic, (44, (190 - ic.height) // 2))
    d = ImageDraw.Draw(im)
    d.text((190, 34), "FREE PDF GUIDE", font=font("Black", 70), fill=WHITE)
    d.text((192, 120), "IGNOU MCA • 2026", font=font("SemiBold", 38), fill=YELLOW)
    return gfx.shadowed(im, 20, (0, 10), 120)


@lru_cache(None)
def caption(words_key, size):
    return gfx.caption_image(list(words_key), HIGHLIGHT, size)


# --------------------------------------------------------------------------
# Cue sheet

def make_cues(words):
    def at(pi, word, nth=0):
        key = word.strip(',."').lower()
        hits = [w for w in words if w[3] == pi and w[0].strip(',."').lower() == key]
        return out_t(hits[nth][1])

    def end(pi):
        return out_t(PHRASES[pi][1])

    c = {}
    c["hook"] = (0.05, at(0, "MCA"), at(0, "complete"), end(0))
    c["money"] = (out_t(PHRASES[1][0]) - 0.05, at(1, "waste"), at(1, "video"), end(1))
    c["elig"] = (out_t(PHRASES[2][0]), at(2, "recognised"), at(3, "BCA,"), at(4, "Mathematics"), end(4))
    c["stamp"] = (at(5, "entrance"), at(5, "Duration") - 0.1)
    c["dur"] = (at(5, "Duration"), at(5, "2"), at(6, "4"), at(6, "flexibility"), end(6))
    c["fees"] = (out_t(PHRASES[7][0]), at(7, "lagbhag"), at(7, "hazaar"), at(8, "semester-wise"), DURATION_BEFORE_CUT)
    c["place"] = (out_t(PHRASES[9][0]), at(9, "sach"), at(10, "CPC"), at(10, "but"), at(10, "valuable"), end(10))
    c["roles"] = (at(11, "software"), at(11, "data"), at(11, "IT"), at(11, "high-tech"), at(12, "master"), end(13))
    c["pdf"] = (out_t(PHRASES[14][0]), at(14, "syllabus,"), at(14, "project"), at(14, "admission"), end(14))
    c["cta"] = (out_t(PHRASES[15][0]), at(15, "comment", 1), at(15, '"MCA"'), at(16, "DM"), DURATION)
    return c


DURATION_BEFORE_CUT = out_t(49.30)


def sfx_track(c):
    ev = []  # (time, name, gain)
    h = c["hook"]
    ev += [(0.0, "impact", 0.8), (0.0, "whoosh", 0.5), (h[1], "pop_hi", 0.6), (h[2], "pop", 0.5), (h[3] - 0.1, "whoosh_down", 0.35)]
    m = c["money"]
    ev += [(m[0], "whoosh", 0.45), (m[1], "wrong", 0.45), (m[2], "pop", 0.5)]
    e = c["elig"]
    ev += [(e[0], "whoosh", 0.45), (e[1], "pop", 0.55), (e[2], "pop", 0.55), (e[3], "pop", 0.55)]
    ev += [(c["stamp"][0], "stamp", 0.9)]
    d = c["dur"]
    ev += [(d[0], "whoosh", 0.45), (d[1], "tick", 0.6), (d[2], "tick", 0.6), (d[3], "pop_hi", 0.55)]
    f = c["fees"]
    ev += [(f[0], "whoosh", 0.45)]
    ev += [(f[0] + 0.1 + k * 0.09, "tick", 0.3) for k in range(int((f[2] - f[0] - 0.1) / 0.09))]
    ev += [(f[2], "cha_ching", 0.55), (f[3], "pop", 0.5)]
    p = c["place"]
    ev += [(p[0], "whoosh", 0.5), (p[2], "pop", 0.55), (p[3], "wrong", 0.3), (p[4], "pop_hi", 0.5)]
    r = c["roles"]
    ev += [(r[0], "whoosh", 0.45)] + [(x, "pop", 0.5) for x in r[1:4]] + [(r[4], "pop_hi", 0.5)]
    pd = c["pdf"]
    ev += [(pd[0] - 1.0, "riser", 0.3), (pd[0], "impact", 0.45), (pd[1], "pop", 0.5), (pd[2], "pop", 0.5), (pd[3], "pop", 0.5)]
    ct = c["cta"]
    ev += [(ct[0], "whoosh", 0.45), (ct[2], "typing", 0.6), (ct[3], "ding", 0.6)]
    # soft whoosh on every jump cut to glue the edit
    for a, b, o in SEGMENTS[1:]:
        if not any(abs(o - t) < 0.4 for t, _, _ in ev):
            ev.append((o - 0.15, "whoosh_down", 0.12))
    lib = {k: f() for k, f in sfx.LIBRARY.items()}
    out = np.zeros(int((DURATION + 2) * sfx.SR), np.float32)
    for t, name, g in ev:
        s = max(0, int(t * sfx.SR))
        x = lib[name]
        e_ = min(len(out), s + len(x))
        out[s:e_] += x[: e_ - s] * g
    return out[: int(DURATION * sfx.SR)], ev


# --------------------------------------------------------------------------
# Per-frame overlay

def overlay(t, c, caps):
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)

    # progress bar
    d.rectangle((0, 0, int(W * t / DURATION), 9), fill=RED)

    # hook
    h = c["hook"]
    gfx.place(cv, a_hook_title(), W / 2, TOP_Y + 130, t, h[0], h[3], "pop", 0.4)
    gfx.place(cv, a_hook_mca(), W / 2, TOP_Y + 335, t, h[1], h[3], "pop", 0.35)
    gfx.place(cv, a_hook_sub(), W / 2, TOP_Y + 480, t, h[2], h[3], "up", 0.3)

    # money
    m = c["money"]
    frac = round(gfx.ease_out_cubic((t - m[1]) / 0.35), 1) if t >= m[1] else 0
    gfx.place(cv, a_money(frac), W / 2, TOP_Y + 170, t, m[0], m[3], "pop")
    gfx.place(cv, a_header("Watch till the end ↓", bg=YELLOW, fg=NAVY), W / 2, TOP_Y + 400, t, m[2], m[3], "up")

    # eligibility
    e = c["elig"]
    gfx.place(cv, a_header("① ELIGIBILITY"), W / 2, TOP_Y, t, e[0], e[4], "pop")
    gfx.place(cv, a_row("Graduation: any recognised university", size=42), W / 2, TOP_Y + 150, t, e[1], e[4], "left")
    gfx.place(cv, a_row("BCA / CS / IT / any stream", size=44), W / 2, TOP_Y + 285, t, e[2], e[4], "left")
    gfx.place(cv, a_row("Mathematics is compulsory", size=44), W / 2, TOP_Y + 420, t, e[3], e[4], "left")

    # no-entrance stamp
    s = c["stamp"]
    gfx.place(cv, a_stamp(), W / 2, TOP_Y + 200, t, s[0], s[1], "stamp", 0.22, 0.2)

    # duration
    du = c["dur"]
    gfx.place(cv, a_header("② DURATION"), W / 2, TOP_Y, t, du[0], du[4], "pop")
    draw_duration_bar(cv, t, du[0] + 0.1, du[1], du[2], du[4])
    gfx.place(cv, a_row("Flexible — study at your pace", size=42, bg=GREEN, fg=WHITE, icon=None),
              W / 2, TOP_Y + 500, t, du[3], du[4], "pop")

    # fees
    f = c["fees"]
    gfx.place(cv, a_header("③ TOTAL FEES"), W / 2, TOP_Y, t, f[0], f[4], "pop")
    draw_fee_counter(cv, t, f[0] + 0.1, f[0] + 0.1, f[2], f[4])
    gfx.place(cv, a_row("Pay semester-wise or year-wise", size=42, bg=YELLOW, fg=NAVY, icon=None),
              W / 2, TOP_Y + 470, t, f[3], f[4], "pop")

    # placement
    p = c["place"]
    gfx.place(cv, a_header("④ PLACEMENT: THE TRUTH"), W / 2, TOP_Y, t, p[0], p[5], "pop")
    gfx.place(cv, a_row("IGNOU CPC organises placement drives", size=42), W / 2, TOP_Y + 155, t, p[2], p[5], "left")
    gfx.place(cv, a_row("But the degree matters most for…", size=40, bg=NAVY, fg=WHITE, icon=None),
              W / 2, TOP_Y + 300, t, p[3], p[5], "pop")

    # roles
    r = c["roles"]
    gfx.place(cv, a_header("BEST FOR", bg=NAVY), W / 2, TOP_Y, t, r[0], r[5], "pop")
    chips = [("\U0001F4BB", "Software Development"), ("\U0001F4CA", "Data Science"),
             ("\U0001F5A5", "IT Industry Jobs"), ("\U0001F680", "High-tech Roles")]
    times = [r[0], r[1], r[2], r[3]]
    for k, ((ic, txt), t0) in enumerate(zip(chips, times)):
        gfx.place(cv, a_row(txt, icon=ic, size=46, w=820), W / 2, TOP_Y + 140 + k * 122, t, t0, r[5], "left")
    gfx.place(cv, a_row("Job + Master's degree, together", size=40, bg=YELLOW, fg=NAVY, icon=None),
              W / 2, TOP_Y + 640, t, r[4], r[5], "pop")

    # pdf guide
    g = c["pdf"]
    gfx.place(cv, a_pdf_title(), W / 2, TOP_Y + 90, t, g[0], g[4], "pop")
    for k, (txt, t0) in enumerate([("Complete syllabus", g[1]), ("Project guidelines", g[2]), ("Admission steps", g[3])]):
        gfx.place(cv, a_row(txt, size=48, w=720), W / 2, TOP_Y + 275 + k * 130, t, t0, g[4], "left")

    # CTA
    ct = c["cta"]
    gfx.place(cv, a_header("Comment “MCA” ↓", bg=YELLOW, fg=NAVY), W / 2, TOP_Y + 20, t, ct[0], ct[4], "pop")
    if t >= ct[0]:
        typed = "MCA"[: max(0, min(3, int((t - ct[2]) / 0.13) + 1))] if t >= ct[2] else ""
        box = a_comment_box(typed)
        pulse = 1 + 0.03 * np.sin(2 * np.pi * 1.6 * max(0, t - ct[3])) if t > ct[3] else 1.0
        env = gfx.envelope(t, ct[0], ct[4])
        if env:
            a = gfx.ease_out_cubic(env[0])
            gfx.paste_center(cv, gfx.transform(box, scale=(0.6 + 0.4 * gfx.ease_out_back(env[0])) * pulse, alpha=a),
                             W / 2, TOP_Y + 200)
    gfx.place(cv, a_row("Guide lands in your DM", icon="\U0001F4E9", size=44, bg=GREEN, fg=WHITE),
              W / 2, TOP_Y + 370, t, ct[3], ct[4], "pop")

    # captions
    for c0, c1, words in caps:
        if c0 <= t < c1:
            size = 76
            img = caption(tuple(words), size)
            while img.width > W - 140 and size > 48:
                size -= 4
                img = caption(tuple(words), size)
            k = gfx.clamp((t - c0) / 0.12)
            sc = 0.82 + 0.18 * gfx.ease_out_back(k)
            gfx.paste_center(cv, gfx.transform(img, scale=sc), W / 2, CAPTION_Y)
            break
    return cv


# --------------------------------------------------------------------------
# Camera: alternate framing on every cut (hides jump cuts) + slow push-in,
# extra punch-in on key lines.

FACE_X, FACE_Y = 0.50, 0.43


def zoom_at(t, punches):
    i = seg_index(t)
    a, b, o = SEGMENTS[i]
    base = (1.0, 1.13)[i % 2]
    z = base + 0.035 * (t - o) / max(1.0, b - a)
    for p0, p1 in punches:
        if p0 <= t <= p1:
            z += 0.10 * gfx.ease_out_cubic((t - p0) / 0.18) * (1 - gfx.ease_out_cubic((t - p1 + 0.25) / 0.25))
    return z


def crop_box(z):
    cw, ch = SRC_W / z, SRC_H / z
    x0 = min(max(0, FACE_X * SRC_W - cw / 2), SRC_W - cw)
    y0 = min(max(0, FACE_Y * SRC_H - ch / 2), SRC_H - ch)
    return (x0, y0, x0 + cw, y0 + ch)


# --------------------------------------------------------------------------

def run(cmd, **kw):
    print("+", " ".join(cmd)[:200], flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    preview = "--preview" in sys.argv
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "work")
    os.makedirs(tmp, exist_ok=True)

    wav16 = os.path.join(tmp, "a16k.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav16])
    mask, step = speech_mask(wav16)
    words = word_timings(mask, step)
    caps = build_captions(words)
    cues = make_cues(words)
    json.dump({"segments": SEGMENTS, "duration": DURATION, "captions": caps, "cues": cues},
              open(os.path.join(outdir, "timeline.json"), "w"), indent=1)
    print(f"segments={len(SEGMENTS)} duration={DURATION:.2f}s", flush=True)

    # ---------------- audio
    n = len(SEGMENTS)
    parts = "".join(
        f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.012,afade=t=out:st={b - a - 0.015:.3f}:d=0.015[a{i}];"
        for i, (a, b, _) in enumerate(SEGMENTS))
    voice = os.path.join(tmp, "voice.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex",
         parts + "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1,"
         "highpass=f=80,lowpass=f=15000,"
         "equalizer=f=250:t=q:w=1.2:g=-2,equalizer=f=3500:t=q:w=1.0:g=2.5,"
         "acompressor=threshold=-22dB:ratio=3:attack=5:release=150:makeup=2,"
         "loudnorm=I=-15:TP=-2:LRA=8,aresample=48000[v]",
         "-map", "[v]", "-ac", "1", voice])
    fx, events = sfx_track(cues)
    sf.write(os.path.join(tmp, "sfx.wav"), fx, sfx.SR)
    sf.write(os.path.join(tmp, "music.wav"), sfx.music_bed(DURATION), sfx.SR)
    json.dump(events, open(os.path.join(outdir, "sfx_events.json"), "w"), indent=1)

    mixes = {
        "final": "[1:a]volume=0.55[m];[m][0:a]sidechaincompress=threshold=0.05:ratio=4:attack=30:release=500[md];"
                 "[0:a][md][2:a]amix=inputs=3:weights='1 1 0.55':normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[o]",
        "nomusic": "[0:a][2:a]amix=inputs=2:weights='1 0.55':normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[o]",
    }
    for name, fc in mixes.items():
        run(["ffmpeg", "-v", "error", "-y", "-i", voice, "-i", os.path.join(tmp, "music.wav"),
             "-i", os.path.join(tmp, "sfx.wav"), "-filter_complex", fc, "-map", "[o]",
             "-ar", "48000", "-ac", "2", os.path.join(tmp, f"mix_{name}.wav")])

    # ---------------- video
    # Decode the source once, sequentially; for each output frame pick the source
    # frame at the mapped time (keeps lip-sync exact across every cut).
    reader = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={FPS}", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    silent = os.path.join(tmp, "video.mp4")
    grade = "eq=contrast=1.05:saturation=1.10:gamma=1.02,unsharp=5:5:0.5,format=yuv420p"
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-vf", grade, "-c:v", "libx264", "-preset", "veryfast" if preview else "medium",
         "-crf", "23" if preview else "18", "-profile:v", "high", "-pix_fmt", "yuv420p", silent],
        stdin=subprocess.PIPE)

    punches = [(cues["money"][1] - 0.05, cues["money"][2]), (cues["stamp"][0] - 0.05, cues["stamp"][1]),
               (cues["place"][1] - 0.05, cues["place"][2]), (cues["cta"][2] - 0.05, cues["cta"][3])]
    fsize = SRC_W * SRC_H * 3
    src_idx, buf = -1, None
    for k in range(int(DURATION * FPS)):
        t = k / FPS
        a, b, o = SEGMENTS[seg_index(t)]
        want = int(round((a + t - o) * FPS))
        while src_idx < want:
            nxt = reader.stdout.read(fsize)
            if len(nxt) < fsize:
                break
            buf, src_idx = nxt, src_idx + 1
        frame = Image.frombuffer("RGB", (SRC_W, SRC_H), buf, "raw", "RGB", 0, 1)
        frame = frame.resize((W, H), Image.BICUBIC, box=crop_box(zoom_at(t, punches))).convert("RGBA")
        frame.alpha_composite(overlay(t, cues, caps))
        writer.stdin.write(frame.convert("RGB").tobytes())
        if k % 300 == 0:
            print(f"  frame {k} ({t:.1f}s)", flush=True)
    writer.stdin.close()
    writer.wait()
    reader.stdout.close()
    reader.kill()

    for name in mixes:
        out = os.path.join(outdir, f"ignou_mca_{name}.mp4")
        run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", os.path.join(tmp, f"mix_{name}.wav"),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", out])
    print("done", flush=True)


if __name__ == "__main__":
    main()
