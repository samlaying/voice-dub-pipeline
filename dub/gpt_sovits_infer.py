#!/usr/bin/env python3
"""Run one GPT-SoVITS synthesis for a whole text file.

This is the only file that must execute inside the GPT-SoVITS interpreter
(`GPT_SOVITS_PYTHON`), because it imports `TTS_infer_pack`. Stage s5 launches it
as a subprocess so the rest of the pipeline can run in any Python.

Environment:
  TTS_REPO            GPT-SoVITS checkout (default /content/GPT-SoVITS)
  TEXT_FILE           UTF-8 text to speak
  OUT                 output wav
  REF_AUDIO           reference clip used for timbre transfer
  REF_TEXT            transcript of the reference clip
  TTS_EXP             experiment prefix used to locate weights (default voice)
  SOVITS_PATH         explicit SoVITS weight, overrides discovery
  GPT_PATH            explicit GPT weight, overrides discovery
  TEMPERATURE, FRAGMENT_INTERVAL, TOP_K, TOP_P, SPEED, SPLIT_METHOD, SEED
"""

import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

REPO = os.environ.get("TTS_REPO", "/content/GPT-SoVITS")
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def pick_weights(exp):
    sovits = os.environ.get("SOVITS_PATH") or ""
    gpt = os.environ.get("GPT_PATH") or ""
    if not sovits:
        found = sorted(glob.glob(f"SoVITS_weights_v2/{exp}_e*_s*.pth"), key=natural_key)
        sovits = found[-1] if found else ""
    if not gpt:
        found = sorted(glob.glob(f"GPT_weights_v2/{exp}-e*.ckpt"), key=natural_key)
        gpt = found[-1] if found else ""
    if not sovits or not gpt:
        sys.exit(f"[tts] weights for experiment {exp!r} not found (set SOVITS_PATH / GPT_PATH)")
    return sovits, gpt


def main():
    text_file = os.environ.get("TEXT_FILE", "")
    out_path = os.environ.get("OUT", "/content/dub_tts.wav")
    ref_audio = os.environ.get("REF_AUDIO", "")
    prompt_text = os.environ.get("REF_TEXT", "")

    if not text_file or not os.path.exists(text_file):
        sys.exit(f"[tts] TEXT_FILE missing: {text_file!r}")
    text = open(text_file, encoding="utf-8").read().strip()
    if not text:
        sys.exit("[tts] TEXT_FILE is empty")
    if not ref_audio or not os.path.exists(ref_audio):
        sys.exit(f"[tts] reference audio missing: {ref_audio!r}")
    if not prompt_text:
        sys.exit("[tts] REF_TEXT is required alongside REF_AUDIO")

    sovits_path, gpt_path = pick_weights(os.environ.get("TTS_EXP", "voice"))
    print(f"[tts] sovits={sovits_path}", flush=True)
    print(f"[tts] gpt={gpt_path}", flush=True)
    print(f"[tts] ref={ref_audio}", flush=True)
    print(f"[tts] text={len(text)} chars", flush=True)

    try:
        import torch

        device, half = ("cuda", True) if torch.cuda.is_available() else ("cpu", False)
    except Exception:  # noqa: BLE001 - torch is present in this env, but be explicit
        device, half = "cpu", False

    cfg = {
        "custom": {
            "device": device,
            "is_half": half,
            "version": "v2",
            "t2s_weights_path": gpt_path,
            "vits_weights_path": sovits_path,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        }
    }
    pipeline = TTS(TTS_Config(cfg))
    inputs = {
        "text": text,
        "text_lang": os.environ.get("TEXT_LANG", "zh"),
        "ref_audio_path": ref_audio,
        "prompt_text": prompt_text,
        "prompt_lang": os.environ.get("PROMPT_LANG", "zh"),
        "top_k": int(os.environ.get("TOP_K", "15")),
        "top_p": float(os.environ.get("TOP_P", "1")),
        "temperature": float(os.environ.get("TEMPERATURE", "1.0")),
        "text_split_method": os.environ.get("SPLIT_METHOD", "cut5"),
        "batch_size": 1,
        "speed_factor": float(os.environ.get("SPEED", "1.0")),
        "fragment_interval": float(os.environ.get("FRAGMENT_INTERVAL", "0.3")),
        "seed": int(os.environ.get("SEED", "-1")),
        "repetition_penalty": float(os.environ.get("REPETITION_PENALTY", "1.35")),
        "parallel_infer": True,
        "split_bucket": True,
    }

    chunks, rate = [], None
    for sample_rate, audio in pipeline.run(inputs):
        rate = sample_rate
        if audio is not None and len(audio):
            chunks.append(audio)
    if not chunks:
        sys.exit("[tts] no audio produced")

    audio = np.concatenate(chunks)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio, rate)
    print(f"[tts] wrote {out_path} ({len(audio) / rate:.2f}s @ {rate}Hz)", flush=True)


if __name__ == "__main__":
    main()
