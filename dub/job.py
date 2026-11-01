"""Job directory plus manifest bookkeeping.

A job is a folder. Every stage reads its inputs and writes its outputs inside
that folder, and records the result in manifest.json, so a stage can be re-run
in another environment (Codespaces today, Colab in five minutes) without
recomputing what already succeeded.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("s1", "s2", "s3", "s4", "s5", "s6")
STAGE_NAMES = {
    "s1": "download",
    "s2": "separate",
    "s3": "transcribe",
    "s4": "translate",
    "s5": "tts",
    "s6": "mix + lipsync",
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.data = {}

    # -- lifecycle

    @classmethod
    def create(cls, root, url=None):
        job = cls(root)
        job.root.mkdir(parents=True, exist_ok=True)
        job.data = {
            "id": job.root.name,
            "created_at": _now(),
            "source": {"url": url or ""},
            "settings": {},
            "stages": {},
        }
        job.save()
        return job

    @classmethod
    def load(cls, root):
        job = cls(root)
        if not job.manifest_path.exists():
            raise SystemExit(f"no manifest at {job.manifest_path}; run `init` first")
        job.data = json.loads(job.manifest_path.read_text(encoding="utf-8"))
        return job

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # -- paths

    def path(self, *parts):
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def url(self):
        return self.data.get("source", {}).get("url", "")

    @property
    def video(self):
        return self.root / "source.mp4"

    @property
    def original_audio(self):
        return self.root / "audio" / "original.wav"

    @property
    def vocals(self):
        return self.root / "separated" / "vocals.wav"

    @property
    def background(self):
        return self.root / "separated" / "no_vocals.wav"

    @property
    def source_srt(self):
        return self.root / "transcripts" / "source.srt"

    @property
    def translated_srt(self):
        return self.root / "translated" / "zh.srt"

    @property
    def translated_text(self):
        return self.root / "translated" / "zh.txt"

    @property
    def tts_wav(self):
        return self.root / "tts" / "zh.wav"

    @property
    def mixed_wav(self):
        return self.root / "mix" / "final.wav"

    @property
    def output_video(self):
        return self.root / "out" / "final.mp4"

    # -- stage bookkeeping

    def done(self, stage):
        return self.data.get("stages", {}).get(stage, {}).get("status") == "done"

    def mark(self, stage, status, **extra):
        record = self.data.setdefault("stages", {}).setdefault(stage, {})
        record.update({"status": status, "updated_at": _now(), **extra})
        self.save()

    def require(self, stage, what):
        if not self.done(stage):
            raise SystemExit(f"{what} missing: stage {stage} has not finished for {self.root.name}")
