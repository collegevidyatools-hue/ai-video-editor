"""Procedurally synthesized sound effects and a lo-fi music bed.

Everything is generated with numpy so no stock-audio downloads are needed.
All functions return mono float32 arrays at SR, peak-normalised to ~0.9.
"""
import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _norm(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return (x / m * peak).astype(np.float32)


def _fft_bandpass(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _onepole_lp(x, cutoff):
    """Time-varying one-pole low-pass; cutoff may be scalar or per-sample array."""
    cutoff = np.broadcast_to(cutoff, x.shape)
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def whoosh(dur=0.5, up=True):
    t = _t(dur)
    n = rng.standard_normal(len(t))
    sweep = np.linspace(400, 5000, len(t)) if up else np.linspace(5000, 400, len(t))
    y = _onepole_lp(n, sweep) - _onepole_lp(n, sweep * 0.25)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return _norm(y * env, 0.8)


def impact(dur=0.9):
    t = _t(dur)
    f = 45 + 90 * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    click = _fft_bandpass(rng.standard_normal(len(t)), 1500, 8000) * np.exp(-t * 60)
    return _norm(body + 0.35 * click)


def pop(dur=0.09, f0=500, f1=1100):
    t = _t(dur)
    f = np.linspace(f0, f1, len(t))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)
    return _norm(y, 0.7)


def tick(dur=0.02):
    t = _t(dur)
    y = np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 300)
    return _norm(y, 0.5)


def stamp(dur=0.35):
    t = _t(dur)
    thud = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-t * 40)) / SR) * np.exp(-t * 14)
    slap = _fft_bandpass(rng.standard_normal(len(t)), 200, 3000) * np.exp(-t * 35)
    return _norm(thud + 0.6 * slap)


def cha_ching(dur=1.1):
    t = _t(dur)
    y = np.zeros_like(t)
    # mechanical "cha"
    y += _fft_bandpass(rng.standard_normal(len(t)), 2000, 9000) * np.exp(-t * 30) * 0.5
    # bell "ching" after 90 ms, inharmonic partials
    tb = np.clip(t - 0.09, 0, None)
    gate = (t >= 0.09).astype(float)
    for f, a in ((2093, 1.0), (2637, 0.6), (3520, 0.45), (5274, 0.25)):
        y += gate * a * np.sin(2 * np.pi * f * tb) * np.exp(-tb * 5)
    return _norm(y)


def ding(dur=1.0):
    t = _t(dur)
    y = np.zeros_like(t)
    for start, f in ((0.0, 1318.5), (0.12, 1760.0)):
        tt = np.clip(t - start, 0, None)
        g = (t >= start).astype(float)
        y += g * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 6)
    return _norm(y, 0.7)


def wrong(dur=0.45):
    """Soft descending two-note 'nope'."""
    t = _t(dur)
    y = np.zeros_like(t)
    for start, f in ((0.0, 392.0), (0.16, 311.1)):
        tt = np.clip(t - start, 0, None)
        g = ((t >= start) & (t < start + 0.26)).astype(float)
        y += g * np.sign(np.sin(2 * np.pi * f * tt)) * 0.3 * np.exp(-tt * 7)
    return _norm(_onepole_lp(y, 2500), 0.6)


def typing(n=3, gap=0.12):
    dur = n * gap + 0.1
    y = np.zeros(int(dur * SR))
    for i in range(n):
        s = int((i * gap + rng.uniform(-0.02, 0.02) + 0.02) * SR)
        c = _fft_bandpass(rng.standard_normal(int(0.03 * SR)), 1500, 7000)
        c *= np.exp(-np.arange(len(c)) / SR * 180)
        y[s:s + len(c)] += c
    return _norm(y, 0.6)


def riser(dur=1.2):
    t = _t(dur)
    f = np.linspace(200, 1400, len(t))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    noise = _onepole_lp(rng.standard_normal(len(t)), np.linspace(500, 8000, len(t)))
    env = (t / dur) ** 2
    return _norm((tone + noise) * env, 0.7)


# ---------------------------------------------------------------- music bed

def _note(freq, dur, kind="keys"):
    t = _t(dur)
    if kind == "keys":  # soft electric-piano-ish
        y = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) + 0.08 * np.sin(6 * np.pi * freq * t)
        env = np.minimum(t / 0.01, 1) * np.exp(-t * 1.6)
        return y * env * (1 + 0.15 * np.sin(2 * np.pi * 5 * t))
    if kind == "bass":
        y = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(4 * np.pi * freq * t)
        return y * np.minimum(t / 0.01, 1) * np.exp(-t * 2.5)
    raise ValueError(kind)


def music_bed(total, bpm=88):
    """Mellow lo-fi loop: Fmaj7 - Em7 - Dm7 - Cmaj7, soft kick/snare/hats."""
    beat = 60 / bpm
    bar = 4 * beat
    out = np.zeros(int((total + 2) * SR))
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    chords = [(53, [57, 60, 64, 65]), (52, [55, 59, 62, 64]), (50, [53, 57, 60, 62]), (48, [52, 55, 59, 60])]

    def add(sig, at, gain):
        s = int(at * SR)
        e = min(len(out), s + len(sig))
        if s < len(out):
            out[s:e] += sig[: e - s] * gain

    kick = impact(0.4) * 0.9
    snare = _fft_bandpass(rng.standard_normal(int(0.2 * SR)), 300, 6000) * np.exp(-_t(0.2) * 25)
    hat = _fft_bandpass(rng.standard_normal(int(0.05 * SR)), 6000, 14000) * np.exp(-_t(0.05) * 90)

    n_bars = int(total / bar) + 2
    for b in range(n_bars):
        root, notes = chords[b % 4]
        t0 = b * bar
        for k, m in enumerate(notes):  # gently strummed chord, re-struck on beat 3
            add(_note(hz(m), bar, "keys"), t0 + k * 0.02, 0.12)
            add(_note(hz(m), bar / 2, "keys"), t0 + 2 * beat + k * 0.02, 0.06)
        add(_note(hz(root - 12), beat * 1.5, "bass"), t0, 0.35)
        add(_note(hz(root - 12), beat, "bass"), t0 + 2.5 * beat, 0.25)
        for q in range(4):
            if q in (0, 2):
                add(kick, t0 + q * beat, 0.45)
            else:
                add(snare, t0 + q * beat, 0.12)
            add(hat, t0 + q * beat, 0.06)
            add(hat, t0 + (q + 0.6) * beat, 0.04)  # swung 8ths
    out = _onepole_lp(out, 3200)  # warm, keeps it under the voice
    out = out[: int(total * SR)]
    fade = int(1.5 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return _norm(out, 0.8)


LIBRARY = {
    "whoosh": lambda: whoosh(0.5, True),
    "whoosh_down": lambda: whoosh(0.4, False),
    "impact": impact,
    "pop": pop,
    "pop_hi": lambda: pop(0.08, 800, 1600),
    "tick": tick,
    "stamp": stamp,
    "cha_ching": cha_ching,
    "ding": ding,
    "wrong": wrong,
    "typing": lambda: typing(3, 0.13),
    "riser": riser,
}
