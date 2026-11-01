#!/usr/bin/env bash
# Prepare a Colab VM for the GPU stages (s3 transcribe, s5 tts, s6 lipsync).
# Run from the repository root:  !bash colab/bootstrap.sh
set -euo pipefail

command -v ffmpeg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ffmpeg; }

python3 - <<'PY'
import torch, sys
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: no GPU. s5/s6 will be impractically slow on CPU.", file=sys.stderr)
PY

pip install -q "whisperx>=3.1.0"
python3 -c "import whisperx; print('whisperx ok')"
echo "[bootstrap] done"
