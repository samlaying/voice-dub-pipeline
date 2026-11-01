"""Environment-driven settings.

Nothing here is hardcoded on purpose: the same repository runs in three places
(GPU Colab, CPU Codespaces, the always-on server) and each one supplies its own
values through environment variables.
"""

import os


class ConfigError(SystemExit):
    pass


def get(name, default=None, required=False):
    value = os.environ.get(name) or default
    if required and not value:
        raise ConfigError(f"{name} is required but not set")
    return value


def siliconflow():
    """LLM translation settings (stage s4)."""
    return {
        "api_key": get("SILICONFLOW_API_KEY", required=True),
        "base_url": get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
        "model": get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
        "timeout": int(get("SILICONFLOW_TIMEOUT", "120")),
    }


def tts():
    """GPT-SoVITS settings (stage s5)."""
    return {
        "repo": get("GPT_SOVITS_DIR", "/content/GPT-SoVITS"),
        "python": get("GPT_SOVITS_PYTHON", "/content/miniforge3/envs/GPTSoVITS/bin/python"),
        "experiment": get("TTS_EXP", "voice"),
        "sovits": get("SOVITS_PATH", ""),
        "gpt": get("GPT_PATH", ""),
        "ref_audio": get("REF_AUDIO", ""),
        "ref_text": get("REF_TEXT", ""),
        "temperature": get("TTS_TEMPERATURE", "1.0"),
        "fragment_interval": get("TTS_FRAGMENT_INTERVAL", "0.3"),
    }


def lipsync():
    """video-retalking settings (stage s6)."""
    return {
        "repo": get("VIDEO_RETALKING_DIR", ""),
        "python": get("VIDEO_RETALKING_PYTHON", "python3"),
    }


def download():
    """yt-dlp settings (stage s1)."""
    return {
        "cookies": get("YTDLP_COOKIES", ""),
        "format": get("YTDLP_FORMAT", "bv*[height<=1080]+ba/b[height<=1080]/b"),
    }


def demucs():
    return {
        "model": get("DEMUCS_MODEL", "htdemucs"),
        "jobs": get("DEMUCS_JOBS", "0"),
    }


def whisperx():
    return {
        "model": get("WHISPERX_MODEL", "large-v2"),
        "device": get("WHISPERX_DEVICE", "cuda"),
    }
