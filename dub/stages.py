"""The six stages.

Each stage is a function of the job directory, reads its inputs from there and
writes its outputs back, then records success in the manifest. Stages are
independent of each other except through those files, which is what lets the
CPU stages run in Codespaces and the GPU stages run in Colab.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config, srt


def run(cmd, cwd=None, env=None, check=True, capture=False):
    printable = " ".join(str(part) for part in cmd)
    print(f"  $ {printable}", flush=True)
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture,
    )
    if check and result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        raise SystemExit(f"command failed ({result.returncode}): {printable}")
    return result


def require_tool(name):
    if shutil.which(name) is None:
        raise SystemExit(f"{name} not found on PATH; see README for the install step")


# ---------------------------------------------------------------- s1 download


def s1_download(job):
    """Pull the source video and strip a 16 kHz mono track for the ASR models."""
    require_tool("ffmpeg")
    url = job.url
    if not url:
        raise SystemExit("job has no source url; pass --url when initialising")

    settings = config.download()
    require_tool("yt-dlp")

    target = job.root / "source.%(ext)s"
    cmd = ["yt-dlp", "-o", str(target), "--no-playlist", "-f", settings["format"]]
    if settings["cookies"]:
        cmd += ["--cookies", settings["cookies"]]
    cmd.append(url)
    run(cmd)

    candidates = sorted(job.root.glob("source.*"))
    if not candidates:
        raise SystemExit("yt-dlp produced no file")
    video = candidates[0]
    os.replace(video, job.video)

    run(
        [
            "ffmpeg", "-y", "-i", job.video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            job.original_audio,
        ],
        capture=True,
    )
    return {"video": str(job.video), "audio": str(job.original_audio)}


# ---------------------------------------------------------------- s2 separate


def s2_separate(job):
    """Split vocals from background music with Demucs (CPU is fine, just slower)."""
    job.require("s1", "original audio")
    settings = config.demucs()
    out_dir = job.root / "separated"
    cmd = [
        "python3", "-m", "demucs", str(job.original_audio),
        "-n", settings["model"], "--two-stems=vocals", "-o", str(out_dir),
    ]
    if settings["jobs"] != "0":
        cmd += ["-j", settings["jobs"]]
    run(cmd)

    stem_dir = out_dir / settings["model"] / job.original_audio.stem
    vocals = stem_dir / "vocals.wav"
    background = stem_dir / "no_vocals.wav"
    if not vocals.exists():
        raise SystemExit(f"demucs output missing: {vocals}")
    shutil.copy2(vocals, job.vocals)
    shutil.copy2(background, job.background)
    return {"vocals": str(job.vocals), "background": str(job.background)}


# -------------------------------------------------------------- s3 transcribe


def s3_transcribe(job):
    """Word-timed transcription of the vocal stem (GPU strongly preferred)."""
    job.require("s2", "vocal stem")
    settings = config.whisperx()
    require_tool("whisperx")

    out_dir = job.root / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    language = job.data.get("source", {}).get("language", "en")
    run(
        [
            "whisperx", str(job.vocals),
            "--model", settings["model"],
            "--language", language,
            "--device", settings["device"],
            "--output_format", "srt",
            "--output_dir", str(out_dir),
        ]
    )

    produced = out_dir / f"{job.vocals.stem}.srt"
    if not produced.exists():
        matches = sorted(out_dir.glob("*.srt"))
        if not matches:
            raise SystemExit("whisperx produced no srt")
        produced = matches[0]
    os.replace(produced, job.source_srt)

    cues = srt.read(job.source_srt)
    return {"srt": str(job.source_srt), "cues": len(cues)}


# --------------------------------------------------------------- s4 translate


def _chat(messages, settings, temperature=0.2, max_tokens=4096):
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{settings['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings["timeout"]) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:400]
        raise SystemExit(f"translation API error {exc.code}: {detail}")
    return body["choices"][0]["message"]["content"]


def _strip_fence(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _translate_batch(cues, settings, source_lang, target_lang):
    numbered = "\n".join(f"{i + 1}. {cue['text']}" for i, cue in enumerate(cues))
    system = (
        "You translate subtitles. Keep exactly one output per input cue, in the same order. "
        "Never merge, split, drop or reorder cues. Preserve names, numbers and product names. "
        "Return only a JSON array of strings with no commentary."
    )
    user = (
        f"Translate these {len(cues)} subtitle cues from {source_lang} to {target_lang}.\n"
        f"Return a JSON array of exactly {len(cues)} strings.\n\n{numbered}"
    )
    raw = _strip_fence(_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], settings))
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(cues):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: one numbered line per cue, tolerant to stray prose.
    retry_user = (
        f"Translate each cue from {source_lang} to {target_lang}. "
        f"Output exactly {len(cues)} lines, each starting with '<n>|' where n is the cue number. "
        f"No other text.\n\n{numbered}"
    )
    raw = _strip_fence(_chat([{"role": "system", "content": system}, {"role": "user", "content": retry_user}], settings))
    out = {}
    for line in raw.splitlines():
        if "|" not in line:
            continue
        key, value = line.split("|", 1)
        key = key.strip().rstrip(".")
        if key.isdigit():
            out[int(key)] = value.strip()
    if len(out) != len(cues):
        raise SystemExit(f"translation returned {len(out)} of {len(cues)} cues; refusing to desync subtitles")
    return [out[i + 1] for i in range(len(cues))]


def s4_translate(job):
    """Translate the cues, preserving timing and cue count exactly."""
    job.require("s3", "transcript")
    settings = config.siliconflow()
    cues = srt.read(job.source_srt)
    if not cues:
        raise SystemExit(f"no cues parsed from {job.source_srt}")

    batch_size = int(os.environ.get("TRANSLATE_BATCH", "20"))
    source_lang = job.data.get("source", {}).get("language", "en")
    target_lang = job.data.get("source", {}).get("target_language", "zh")
    languages = {"en": "English", "zh": "Simplified Chinese", "ja": "Japanese"}
    source_name = languages.get(source_lang, source_lang)
    target_name = languages.get(target_lang, target_lang)

    translated = []
    for start in range(0, len(cues), batch_size):
        batch = cues[start:start + batch_size]
        print(f"  translating cues {start + 1}-{start + len(batch)} of {len(cues)}", flush=True)
        translated.extend(_translate_batch(batch, settings, source_name, target_name))

    result = srt.with_translations(cues, translated)
    srt.write(job.translated_srt, result)
    job.translated_text.write_text(srt.plain_text(result), encoding="utf-8")
    return {
        "srt": str(job.translated_srt),
        "text": str(job.translated_text),
        "cues": len(result),
        "model": settings["model"],
    }


# --------------------------------------------------------------------- s5 tts


def s5_tts(job):
    """Synthesize the translated text with the fine-tuned GPT-SoVITS voice (GPU)."""
    job.require("s4", "translated text")
    settings = config.tts()
    repo = Path(settings["repo"])
    if not repo.exists():
        raise SystemExit(f"GPT-SoVITS not found at {repo}; set GPT_SOVITS_DIR")

    helper = Path(__file__).with_name("gpt_sovits_infer.py")
    env = os.environ.copy()
    env.update(
        {
            "TTS_REPO": str(repo),
            "TTS_EXP": settings["experiment"],
            "TEXT_FILE": str(job.translated_text),
            "OUT": str(job.tts_wav),
            "REF_AUDIO": settings["ref_audio"],
            "REF_TEXT": settings["ref_text"],
            "TEMPERATURE": settings["temperature"],
            "FRAGMENT_INTERVAL": settings["fragment_interval"],
            "MPLBACKEND": "Agg",
        }
    )
    if settings["sovits"]:
        env["SOVITS_PATH"] = settings["sovits"]
    if settings["gpt"]:
        env["GPT_PATH"] = settings["gpt"]

    run([settings["python"], str(helper)], env=env)
    if not job.tts_wav.exists():
        raise SystemExit("TTS produced no audio")
    return {"wav": str(job.tts_wav)}


# ------------------------------------------------------------ s6 mix + lipsync


def s6_mix(job):
    """Blend the dub over the original background, then re-time the mouth (GPU)."""
    job.require("s5", "synthesized speech")
    job.require("s2", "background stem")
    require_tool("ffmpeg")

    background_gain = os.environ.get("BACKGROUND_GAIN", "0.35")
    run(
        [
            "ffmpeg", "-y", "-i", job.tts_wav, "-i", job.background,
            "-filter_complex",
            f"[0:a]aresample=44100[v];"
            f"[1:a]aresample=44100,volume={background_gain}[bg];"
            f"[v][bg]amix=inputs=2:duration=longest:normalize=0[mixed];"
            f"[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map", "[out]", job.mixed_wav,
        ],
        capture=True,
    )

    settings = config.lipsync()
    if settings["repo"] and Path(settings["repo"]).exists():
        run(
            [
                settings["python"], "inference.py",
                "--face", str(job.video),
                "--audio", str(job.mixed_wav),
                "--outfile", str(job.output_video),
            ],
            cwd=settings["repo"],
        )
        mode = "video-retalking"
    else:
        # No lipsync model available: keep the original picture and replace audio.
        run(
            [
                "ffmpeg", "-y", "-i", job.video, "-i", job.mixed_wav,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest",
                job.output_video,
            ],
            capture=True,
        )
        mode = "mux-only"
    return {"video": str(job.output_video), "mode": mode}


REGISTRY = {
    "s1": s1_download,
    "s2": s2_separate,
    "s3": s3_transcribe,
    "s4": s4_translate,
    "s5": s5_tts,
    "s6": s6_mix,
}
