"""Phrase-level Whisper transcription without torch or Hugging Face.

Uses sherpa-onnx Whisper ONNX exports (downloadable from GitHub releases:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) and decodes the
raw byte tokens itself, so Hindi/Devanagari output is not mangled.

Usage: python3 editor/transcribe.py <audio16k.wav> <phrases.json> [lang] [model_dir]
"""
import base64
import json
import re
import subprocess
import sys

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


class Whisper:
    def __init__(self, model_dir, size="small", lang="hi", task="transcribe"):
        m = f"{model_dir}/{size}-"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(m + "encoder.int8.onnx", so)
        self.dec = ort.InferenceSession(m + "decoder.int8.onnx", so)
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.L, self.C, self.D = (int(meta[k]) for k in ("n_text_layer", "n_text_ctx", "n_text_state"))
        self.n_mels = int(meta["n_mels"])
        g = lambda k: int(meta[k])
        self.sot, self.eot, self.blank, self.nots = g("sot"), g("eot"), g("blank_id"), g("no_timestamps")
        self.suppress = [g("no_timestamps"), g("sot"), g("no_speech"), g("translate")]
        langs = dict(zip(meta["all_language_codes"].split(","), map(int, meta["all_language_tokens"].split(","))))
        self.seq = list(map(int, meta["sot_sequence"].split(","))) + [self.nots]
        self.seq[1] = langs[lang]
        if task == "translate":
            self.seq[2] = g("translate")
            self.suppress.remove(g("translate"))
        self.tok = {}
        for line in open(m + "tokens.txt", encoding="utf-8"):
            p = line.rstrip("\n").split(" ")
            if len(p) == 2:
                self.tok[int(p[1])] = base64.b64decode(p[0])

    def _feats(self, w):
        o = knf.WhisperFeatureOptions()
        o.dim = self.n_mels
        f = knf.OnlineWhisperFbank(o)
        f.accept_waveform(16000, w)
        f.input_finished()
        x = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])
        x = np.log10(np.clip(x, 1e-10, None))
        x = (np.maximum(x, x.max() - 8.0) + 4) / 4
        x = np.pad(x, ((0, 1500), (0, 0)))
        if x.shape[0] > 3000:
            x = np.pad(x[:2950], ((0, 50), (0, 0)))
        return x.T[None].astype(np.float32)

    def __call__(self, wave, max_tokens=200):
        ck, cv = self.enc.run(None, {self.enc.get_inputs()[0].name: self._feats(wave)})
        sk = np.zeros((self.L, 1, self.C, self.D), np.float32)
        sv = sk.copy()
        off = np.zeros(1, np.int64)
        names = [i.name for i in self.dec.get_inputs()]

        def step(toks, sk, sv):
            lg, sk, sv = self.dec.run(None, dict(zip(names, [np.array([toks], np.int64), sk, sv, ck, cv, off])))[:3]
            return lg[0, -1], sk, sv

        lg, sk, sv = step(self.seq, sk, sv)
        off += len(self.seq)
        lg[self.eot] = lg[self.blank] = -np.inf
        out = []
        while len(out) < max_tokens:
            lg[self.suppress] = -np.inf
            t = int(lg.argmax())
            if t == self.eot:
                break
            out.append(t)
            lg, sk, sv = step([t], sk, sv)
            off += 1
        return b"".join(self.tok.get(i, b"") for i in out).decode("utf-8", "replace").strip()


def phrases(wav, noise_db=-32, min_pause=0.22, max_len=7.0):
    """Split on pauses using ffmpeg silencedetect, merge into <= max_len phrases."""
    log = subprocess.run(["ffmpeg", "-i", wav, "-af", f"silencedetect=n={noise_db}dB:d={min_pause}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    dur = sf.info(wav).duration
    speech, cur = [], 0.0
    for a, b in zip(st, en + [dur]):
        if a - cur > 0.15:
            speech.append([cur, a])
        cur = b
    if dur - cur > 0.15:
        speech.append([cur, dur])
    out = []
    for a, b in speech:
        if out and a - out[-1][1] < 0.35 and b - out[-1][0] < max_len:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


if __name__ == "__main__":
    wav, dst = sys.argv[1], sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "hi"
    model_dir = sys.argv[4] if len(sys.argv) > 4 else "models/sherpa-onnx-whisper-small"
    native, english = Whisper(model_dir, lang=lang), Whisper(model_dir, lang=lang, task="translate")
    audio, sr = sf.read(wav, dtype="float32")
    res = []
    for a, b in phrases(wav):
        seg = audio[int(max(0, a - 0.1) * sr):int((b + 0.1) * sr)]
        r = {"start": round(a, 2), "end": round(b, 2), "text": native(seg), "en": english(seg)}
        print(f"[{a:6.2f}-{b:6.2f}] {r['text']}  |  {r['en']}", flush=True)
        res.append(r)
    json.dump(res, open(dst, "w"), ensure_ascii=False, indent=1)
