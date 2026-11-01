# voice-dub-pipeline

Source video → translated dub in a cloned voice → re-timed mouth.

```
s1 download ──> s2 separate ──> s3 transcribe ──> s4 translate ──> s5 tts ──> s6 mix + lipsync
 yt-dlp+ffmpeg    Demucs          WhisperX          LLM            GPT-SoVITS   ffmpeg+video-retalking
```

Every stage writes into one job folder and records its result in `manifest.json`,
so stages can run in different machines and environments. Each stage is skipped
automatically once it is marked done.

## Which environment runs what

The split is driven by hardware, not preference. **GitHub Codespaces has no GPU
machine types** (2–32 cores, up to 128 GB RAM), so the model-heavy stages cannot
run there.

| Environment | Stages | Why it fits |
|---|---|---|
| **Codespaces** | `s1` download, `s2` vocal separation | CPU work, no CUDA needed; up to 32 cores helps Demucs |
| **Google Colab (T4)** | `s3` transcribe, `s5` TTS, `s6` lip-sync | CUDA required; the fine-tuned GPT-SoVITS weights already live there |
| **Server + domain** | artifact store, job handoff, `s4` translation | Always on; survives Colab recycling and Codespace idle timeouts |

`s4` is a pure API call, so it runs anywhere; the server is the natural place
because it is the only environment that stays up between the other two.

A job started in Codespaces is completed in Colab by copying the job folder
across. The server exists to make that handoff durable instead of a manual
download and re-upload.

## Requirements

| Environment | Needs |
|---|---|
| Any | Python 3.10+, `ffmpeg` |
| Codespaces | nothing extra; `.devcontainer` installs `ffmpeg`, `yt-dlp`, `demucs` |
| Colab | `bash colab/bootstrap.sh` (adds `whisperx`), plus a GPT-SoVITS checkout with your weights |
| Server | `ffmpeg`; only for the optional artifact handoff |

## Quick start (Codespaces)

```bash
export SILICONFLOW_API_KEY=...            # never commit this
python3 -m dub init --job jobs/demo --url "https://www.youtube.com/watch?v=..." --language en --target zh
python3 -m dub run  --job jobs/demo --stages s1,s2
python3 -m dub run  --job jobs/demo --stages s4
python3 -m dub status --job jobs/demo
```

Then copy `jobs/demo` to Colab and continue with `s3,s5,s6`. Alternatively run
`s3` in Colab and bring the SRT back for `s4` here — the stages only meet
through files.

## Colab (GPU stages)

```python
!git clone https://github.com/samlaying/voice-dub-pipeline
%cd voice-dub-pipeline
!bash colab/bootstrap.sh
%env SILICONFLOW_API_KEY=...
!python3 -m dub init --job /content/jobs/demo --url "..." --language en --target zh
!python3 -m dub run --job /content/jobs/demo --stages s3
```

With GPT-SoVITS already restored (`setup.py` + weights + reference clip):

```python
%env GPT_SOVITS_DIR=/content/GPT-SoVITS
%env GPT_SOVITS_PYTHON=/content/miniforge3/envs/GPTSoVITS/bin/python
%env REF_AUDIO=/content/ref.wav
%env REF_TEXT=<transcript of the reference clip>
!python3 -m dub run --job /content/jobs/demo --stages s4,s5,s6
```

`s6` uses `VIDEO_RETALKING_DIR` when it is set; otherwise it keeps the original
picture and only replaces the audio track, so the job still produces a file.

## Job layout

```
jobs/<id>/
├── manifest.json          stage status, timings, errors
├── source.mp4
├── audio/original.wav     16 kHz mono, what the ASR models expect
├── separated/
│   ├── vocals.wav
│   └── no_vocals.wav      background music / ambience
├── transcripts/source.srt
├── translated/
│   ├── zh.srt             same cue count and timing as source.srt
│   └── zh.txt             plain text fed to TTS
├── tts/zh.wav
├── mix/final.wav          dub over background, loudness-normalised
└── out/final.mp4
```

## Environment variables

| Variable | Stage | Default / notes |
|---|---|---|
| `SILICONFLOW_API_KEY` | s4 | required |
| `SILICONFLOW_BASE_URL` | s4 | `https://api.siliconflow.cn/v1` |
| `SILICONFLOW_MODEL` | s4 | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `TRANSLATE_BATCH` | s4 | `20` cues per request |
| `YTDLP_COOKIES` | s1 | path to `cookies.txt`; required by some sites |
| `YTDLP_FORMAT` | s1 | `bv*[height<=1080]+ba/b[height<=1080]/b` |
| `DEMUCS_MODEL` | s2 | `htdemucs` |
| `WHISPERX_MODEL` / `WHISPERX_DEVICE` | s3 | `large-v2` / `cuda` |
| `GPT_SOVITS_DIR` | s5 | `/content/GPT-SoVITS` |
| `GPT_SOVITS_PYTHON` | s5 | interpreter that has `TTS_infer_pack` |
| `TTS_EXP` | s5 | weight prefix, e.g. `myvoice` |
| `SOVITS_PATH` / `GPT_PATH` | s5 | explicit weights, override discovery |
| `REF_AUDIO` / `REF_TEXT` | s5 | reference clip and its transcript |
| `TTS_TEMPERATURE` / `TTS_FRAGMENT_INTERVAL` | s5 | `1.0` / `0.3` |
| `BACKGROUND_GAIN` | s6 | `0.35`, how loud the original background stays |
| `VIDEO_RETALKING_DIR` | s6 | empty → audio replacement without lip-sync |

## Translation is cue-aligned

`s4` parses the SRT into cues, translates in batches, and rebuilds the file with
the original timecodes. It fails loudly if the model returns a different number
of cues rather than silently shifting subtitles. This is why the pipeline does
not use the common `grep -v '^[0-9]'` trick: that also deletes dialogue starting
with a digit and leaves the `-->` timing lines in the text.

## Site-specific caveats

- **Douyin** requires fresh cookies; `yt-dlp` refuses without them
  (`Fresh cookies (not necessarily logged in) are needed`). Export cookies from a
  logged-in browser, mount them as `YTDLP_COOKIES`, and expect the file to expire.
  A non-China egress IP makes this less reliable.
- YouTube and most other hosts work without cookies.

## Security

- `SILICONFLOW_API_KEY` is read from the environment only. Never commit it, and
  rotate it if it has ever been pasted into a chat, issue or log.
- `cookies.txt` is a login credential; `.gitignore` excludes it.
- Model weights are 80–150 MB each. GitHub rejects files over 100 MB without Git
  LFS, so keep them in the Colab VM or an object store — not in this repository.

## Content and consent

Use this on material you own or are licensed to localize. Cloning a voice and
re-timing a real person's mouth onto speech they never gave is impersonation
unless the person consented.
