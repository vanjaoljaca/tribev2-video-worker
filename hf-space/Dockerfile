FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/tribev2-api/cache/huggingface \
    TRANSFORMERS_CACHE=/data/tribev2-api/cache/transformers \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    HF_HUB_HTTP_TIMEOUT=300 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    python-multipart==0.0.9 \
    pydantic==2.8.2 \
    eval_type_backport \
    huggingface_hub>=0.24.0 \
    yt-dlp \
    uv

RUN pip install --no-cache-dir "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
RUN python -m spacy download en_core_web_lg
RUN python - <<'PY'
from pathlib import Path
import site

replacements = {
    "'Step' | None": "tp.Any",
    '"Step" | None': "tp.Any",
}
patched_files = []
for site_dir in site.getsitepackages():
    exca_dir = Path(site_dir) / "exca"
    if not exca_dir.exists():
        continue
    for path in exca_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        patched = text
        for old, new in replacements.items():
            patched = patched.replace(old, new)
        if patched != text:
            path.write_text(patched, encoding="utf-8")
            patched_files.append(str(path))
print("patched_exca_annotations", patched_files)
PY

WORKDIR /app
COPY app.py /app/app.py
COPY start.py /app/start.py
COPY pilot-report.html /app/pilot-report.html

EXPOSE 7860
CMD ["python", "-u", "/app/start.py"]
