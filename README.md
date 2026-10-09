# ai-video-editor

Scripted editing pipeline for talking-head explainer reels (9:16): transcription, jump-cut tightening,
punch-in zooms, motion-graphic cards, keyword-highlighted captions, synthesized SFX and a ducked music bed.
Everything runs locally with `ffmpeg` + Python, with no stock-asset downloads.

```
editor/
  transcribe.py   Whisper (ONNX) phrase transcription, Hindi-safe token decoding
  gfx.py          easing, cards, pills, captions, icons (PIL)
  sfx.py          numpy-synthesized whoosh / impact / pop / stamp / cha-ching / ding / typing / riser + lo-fi bed
projects/
  ignou-mca/edit.py   cut list, captions, cue sheet and renderer for the IGNOU MCA reel
```

## Setup

```bash
pip install numpy pillow soundfile onnxruntime kaldi-native-fbank
# Whisper model (GitHub release, ~640 MB):
mkdir -p models && curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2 | tar xj -C models
```

## Use

```bash
ffmpeg -i input.mov -ac 1 -ar 16000 a16k.wav
python3 editor/transcribe.py a16k.wav phrases.json hi      # review/clean the text, then put it in edit.py
python3 projects/ignou-mca/edit.py input.mov output/        # add --preview for a fast draft
```

Outputs `output/ignou_mca_final.mp4` (with music) and `output/ignou_mca_nomusic.mp4`
(voice + SFX only, for adding trending audio in-app), plus `timeline.json` / `sfx_events.json`.
