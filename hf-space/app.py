import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APP_ROOT = Path("/data/tribev2-api")
CACHE_DIR = APP_ROOT / "cache"
JOBS_DIR = APP_ROOT / "jobs"
MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "facebook/tribev2")
MODEL_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
MODEL = None
MOVIEPY_PATCHED = False
TQDM_PATCHED = False
SPACY_PATCHED = False

app = FastAPI(title="TRIBE v2 Remote Worker")


class TikTokJobRequest(BaseModel):
    account_url: str = "https://www.tiktok.com/@vanjao.plays"
    count: int = 12
    clip_seconds: float | None = 5.0
    include_text: bool = True


class UrlJobItem(BaseModel):
    url: str
    label: str | None = None
    clip_seconds: float | None = None


class UrlJobRequest(BaseModel):
    items: list[UrlJobItem]
    include_text: bool = True


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_status(job_id: str, **updates: Any) -> None:
    status_path = job_dir(job_id) / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    status.update(updates)
    status["updated_at"] = time.time()
    write_json(status_path, status)


def get_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        ensure_open_stdio()
        disable_hf_progress()
        try:
            import eval_type_backport  # noqa: F401
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "eval_type_backport"],
                check=True,
            )
        patch_vendor_sources()
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        from tribev2 import TribeModel

        patch_progress_logging()
        patch_text_extractor_dtype()

        MODEL = TribeModel.from_pretrained(MODEL_ID, cache_folder=str(CACHE_DIR))
        return MODEL


def ensure_open_stdio() -> None:
    for name, fd in (("stdout", 1), ("stderr", 2)):
        stream = getattr(sys, name)
        broken = getattr(stream, "closed", False)
        if not broken:
            try:
                stream.isatty()
            except Exception:
                broken = True
        if broken:
            setattr(sys, name, open(f"/proc/self/fd/{fd}", "w", buffering=1))


def disable_hf_progress() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def patch_progress_logging() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def quiet_tqdm(iterable=None, *args: Any, **kwargs: Any):
        return iterable if iterable is not None else []

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    try:
        import tribev2.demo_utils as demo_utils
        import tribev2.eventstransforms as eventstransforms

        demo_utils.tqdm = quiet_tqdm
        eventstransforms.tqdm = quiet_tqdm
    except Exception:
        pass


def patch_spacy_logging() -> None:
    global SPACY_PATCHED
    if SPACY_PATCHED:
        return
    try:
        import spacy.cli
    except Exception:
        return

    original_download = spacy.cli.download

    def quiet_download(*args: Any, **kwargs: Any):
        with open(os.devnull, "w") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                return original_download(*args, **kwargs)

    spacy.cli.download = quiet_download
    SPACY_PATCHED = True


def patch_vendor_sources() -> None:
    import site

    for site_dir in site.getsitepackages():
        exca_dir = Path(site_dir) / "exca"
        if exca_dir.exists():
            for path in exca_dir.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                patched = text.replace("'Step' | None", "tp.Any").replace(
                    '"Step" | None', "tp.Any"
                ).replace(
                    "del os.environ[x]",
                    "os.environ.pop(x, None)"
                )
                if patched != text:
                    path.write_text(patched, encoding="utf-8")

        neuralset_audio = Path(site_dir) / "neuralset" / "events" / "transforms" / "audio.py"
        if neuralset_audio.exists():
            text = neuralset_audio.read_text(encoding="utf-8")
            patched = text.replace(
                "audio.write_audiofile(audio_filepath)",
                "audio.write_audiofile(audio_filepath, logger=None)",
            )
            if patched != text:
                neuralset_audio.write_text(patched, encoding="utf-8")


def patch_moviepy_logging() -> None:
    global MOVIEPY_PATCHED
    if MOVIEPY_PATCHED:
        return
    try:
        from moviepy.audio.AudioClip import AudioClip
    except Exception:
        return

    original_write_audiofile = AudioClip.write_audiofile

    def quiet_write_audiofile(self, *args, **kwargs):
        kwargs.setdefault("logger", None)
        return original_write_audiofile(self, *args, **kwargs)

    AudioClip.write_audiofile = quiet_write_audiofile
    MOVIEPY_PATCHED = True


def patch_tqdm_logging() -> None:
    global TQDM_PATCHED
    if TQDM_PATCHED:
        return
    try:
        import tqdm as tqdm_module
    except Exception:
        return
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    original_tqdm = tqdm_module.tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    tqdm_module.tqdm = quiet_tqdm
    try:
        import tribev2.eventstransforms as eventstransforms

        eventstransforms.tqdm = quiet_tqdm
    except Exception:
        pass
    TQDM_PATCHED = True


def patch_text_extractor_dtype() -> None:
    try:
        from neuralset.extractors.text import HuggingFaceText
    except Exception:
        return

    original_load_model = HuggingFaceText._load_model

    if getattr(original_load_model, "_tribev2_dtype_patch", False):
        return

    def load_model_compatible_dtype(self, **kwargs: Any):
        if getattr(self, "model_name", "").lower().startswith("meta-llama/llama"):
            import torch

            kwargs["torch_dtype"] = torch.float16
        model = original_load_model(self, **kwargs)
        if getattr(self, "model_name", "").lower().startswith("meta-llama/llama"):
            import torch

            original_forward = model.forward
            try:
                target_param = next(model.parameters())
                target_dtype = target_param.dtype
                device_type = target_param.device.type
            except StopIteration:
                return model

            def forward_with_autocast(*args: Any, **forward_kwargs: Any):
                enabled = (
                    device_type == "cuda"
                    and target_dtype in (torch.float16, torch.bfloat16)
                )
                with torch.autocast(device_type, dtype=target_dtype, enabled=enabled):
                    return original_forward(*args, **forward_kwargs)

            model.forward = forward_with_autocast
        return model

    load_model_compatible_dtype._tribev2_dtype_patch = True
    HuggingFaceText._load_model = load_model_compatible_dtype


def get_video_events(input_video: Path, include_text: bool):
    if include_text:
        model = get_model()
        return model.get_events_dataframe(video_path=str(input_video))

    import pandas as pd
    from tribev2.demo_utils import get_audio_and_text_events

    event = {
        "type": "Video",
        "filepath": str(input_video),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    return get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)


def run_prediction(
    input_video: Path,
    output_dir: Path,
    job_id: str | None = None,
    include_text: bool = False,
) -> dict[str, Any]:
    ensure_open_stdio()
    disable_hf_progress()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = time.time()
    if job_id:
        update_status(job_id, stage="loading_model")
    model = get_model()
    timings["load_model_seconds"] = round(time.time() - started, 3)
    started = time.time()
    if job_id:
        update_status(job_id, stage="building_events")
    patch_moviepy_logging()
    patch_tqdm_logging()
    patch_spacy_logging()
    events = get_video_events(input_video, include_text=include_text)
    timings["events_seconds"] = round(time.time() - started, 3)
    if job_id:
        update_status(job_id, stage="events_ready", events_rows=int(len(events)))
    started = time.time()
    if job_id:
        update_status(job_id, stage="model_forward")
    preds, segments = model.predict(events=events)
    timings["predict_seconds"] = round(time.time() - started, 3)
    if job_id:
        update_status(job_id, stage="prediction_ready", prediction_shape=list(preds.shape))

    np.save(output_dir / "predictions.npy", preds)
    preview = prediction_preview(preds)
    write_json(output_dir / "preview.json", preview)
    events.to_json(output_dir / "events.json", orient="records", indent=2)
    summary = {
        "input": input_video.name,
        "predictions_shape": list(preds.shape),
        "segments": str(segments),
        "preview_path": "preview.json",
        "timings": timings,
        "modalities": "audio+video+text" if include_text else "audio+video",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=str(path.relative_to(src_dir)))


def prediction_preview(preds: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(preds)
    if arr.ndim != 2:
        return {"shape": list(arr.shape), "error": "expected 2D prediction matrix"}

    n_bins = 96
    if arr.shape[1] >= n_bins:
        trimmed = arr[:, : (arr.shape[1] // n_bins) * n_bins]
        binned = trimmed.reshape(arr.shape[0], n_bins, -1).mean(axis=2)
    else:
        binned = arr

    def rounded(values: np.ndarray) -> list[float]:
        return [round(float(v), 6) for v in values.tolist()]

    return {
        "shape": list(arr.shape),
        "timesteps": int(arr.shape[0]),
        "vertices": int(arr.shape[1]),
        "mean": rounded(arr.mean(axis=1)),
        "abs_mean": rounded(np.abs(arr).mean(axis=1)),
        "std": rounded(arr.std(axis=1)),
        "max": rounded(arr.max(axis=1)),
        "min": rounded(arr.min(axis=1)),
        "heatmap_bins": [[round(float(v), 5) for v in row] for row in binned.tolist()],
        "global_min": round(float(arr.min()), 6),
        "global_max": round(float(arr.max()), 6),
        "global_abs_mean": round(float(np.abs(arr).mean()), 6),
    }


def render_report(job_id: str, manifest: dict[str, Any], previews: list[dict[str, Any]]) -> str:
    payload = json.dumps({"job_id": job_id, "manifest": manifest, "previews": previews})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TRIBE v2 report {job_id}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f4f1ea;
    }}
    body {{ margin: 0; background: #101114; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; font-weight: 680; }}
    h2 {{ font-size: 16px; margin: 0 0 12px; font-weight: 650; overflow-wrap: anywhere; }}
    p {{ margin: 0; color: #b9b3a7; }}
    a {{ color: #7dd3fc; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #2c2f36; background: #17191f; border-radius: 8px; padding: 16px; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
    .metric {{ background: #20232b; border-radius: 6px; padding: 10px; }}
    .metric span {{ display: block; color: #999386; font-size: 12px; margin-bottom: 5px; }}
    .metric strong {{ font-size: 14px; }}
    canvas {{ width: 100%; height: 170px; background: #0f1116; border: 1px solid #2c2f36; border-radius: 6px; display: block; }}
    .heat {{ height: 128px; margin-top: 10px; image-rendering: pixelated; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }}
    code {{ color: #fcd34d; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>TRIBE v2 pilot report</h1>
      <p>Job <code>{job_id}</code>. Predicted cortical responses on fsaverage5 vertices.</p>
    </div>
    <div class="links">
      <a href="/jobs/{job_id}">status json</a>
      <a href="/jobs/{job_id}/download">download results</a>
    </div>
  </header>
  <section id="cards" class="grid"></section>
</main>
<script>
const DATA = {payload};
function lineChart(canvas, values, color) {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#30343e';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {{
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }}
  const min = Math.min(...values), max = Math.max(...values);
  const span = Math.max(1e-9, max - min);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {{
    const x = values.length === 1 ? 0 : (i / (values.length - 1)) * w;
    const y = h - ((v - min) / span) * (h - 14) - 7;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}
function heatmap(canvas, rows, globalMin, globalMax) {{
  const cols = rows[0]?.length || 1;
  const width = cols, height = rows.length || 1;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const span = Math.max(1e-9, globalMax - globalMin);
  for (let y = 0; y < height; y++) {{
    for (let x = 0; x < width; x++) {{
      const value = rows[y][x];
      const t = Math.max(0, Math.min(1, (value - globalMin) / span));
      const idx = (y * width + x) * 4;
      img.data[idx] = Math.round(40 + t * 215);
      img.data[idx + 1] = Math.round(80 + Math.sin(t * Math.PI) * 120);
      img.data[idx + 2] = Math.round(210 - t * 170);
      img.data[idx + 3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}
function render() {{
  const cards = document.getElementById('cards');
  DATA.previews.forEach((preview, index) => {{
    const summary = DATA.manifest.summaries[index] || {{}};
    const card = document.createElement('article');
    card.className = 'card';
    card.innerHTML = `
      <h2>${{summary.input || `stimulus ${{index + 1}}`}}</h2>
      <div class="meta">
        <div class="metric"><span>timesteps</span><strong>${{preview.timesteps}}</strong></div>
        <div class="metric"><span>vertices</span><strong>${{preview.vertices}}</strong></div>
        <div class="metric"><span>|mean|</span><strong>${{preview.global_abs_mean}}</strong></div>
      </div>
      <p>Per-second absolute activation energy</p>
      <canvas class="line"></canvas>
      <p style="margin-top:12px">Binned vertex activation heatmap</p>
      <canvas class="heat"></canvas>
    `;
    cards.appendChild(card);
    lineChart(card.querySelector('.line'), preview.abs_mean || [], '#f59e0b');
    heatmap(card.querySelector('.heat'), preview.heatmap_bins || [[]], preview.global_min, preview.global_max);
  }});
}}
render();
</script>
</body>
</html>"""


def download_tiktok_batch(target_dir: Path, account_url: str, count: int) -> list[Path]:
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--playlist-end",
        str(count),
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--no-overwrites",
        "-o",
        str(raw_dir / "%(upload_date)s_%(id)s.%(ext)s"),
        account_url,
    ]
    subprocess.run(cmd, check=True)
    return sorted(raw_dir.glob("*.mp4"))


def download_video_url(target_dir: Path, url: str, index: int) -> Path:
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--no-overwrites",
        "-o",
        str(raw_dir / f"{index:02d}_%(upload_date)s_%(id)s.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    videos = sorted(raw_dir.glob(f"{index:02d}_*.mp4"))
    if not videos:
        raise RuntimeError(f"yt-dlp did not download an MP4 for URL {url}")
    return videos[-1]


def trim_video(input_path: Path, clip_seconds: float | None) -> Path:
    if clip_seconds is None or clip_seconds <= 0:
        return input_path
    trimmed_dir = input_path.parent.parent / "trimmed"
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    output_path = trimmed_dir / f"{input_path.stem}_{clip_seconds:g}s.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-t",
        str(clip_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def process_tiktok_job(
    job_id: str,
    account_url: str,
    count: int,
    clip_seconds: float | None,
    include_text: bool,
) -> None:
    ensure_open_stdio()
    disable_hf_progress()
    root = job_dir(job_id)
    try:
        with JOB_LOCK:
            update_status(job_id, state="downloading", message="Downloading TikTok videos")
            videos = download_tiktok_batch(root, account_url, count)
            if not videos:
                raise RuntimeError("yt-dlp did not download any MP4 videos")

            summaries = []
            for index, video in enumerate(videos, start=1):
                stimulus = trim_video(video, clip_seconds)
                update_status(
                    job_id,
                    state="predicting",
                    message=f"Running TRIBE v2 on {stimulus.name}",
                    current=index,
                    total=len(videos),
                    clip_seconds=clip_seconds,
                    include_text=include_text,
                )
                out = root / "results" / stimulus.stem
                summary = run_prediction(
                    stimulus,
                    out,
                    job_id=job_id,
                    include_text=include_text,
                )
                summary["source_video"] = video.name
                summary["clip_seconds"] = clip_seconds
                summaries.append(summary)

            manifest = {"summaries": summaries}
            write_json(root / "results" / "manifest.json", manifest)
            previews = []
            for result_dir in sorted((root / "results").iterdir()):
                preview_path = result_dir / "preview.json"
                if preview_path.exists():
                    previews.append(read_json(preview_path))
            report = render_report(job_id, manifest, previews)
            (root / "results" / "report.html").write_text(report, encoding="utf-8")
            zip_path = root / "tribev2-results.zip"
            zip_dir(root / "results", zip_path)
            update_status(
                job_id,
                state="complete",
                message="Done",
                current=len(videos),
                total=len(videos),
                download_url=f"/jobs/{job_id}/download",
                report_url=f"/jobs/{job_id}/report",
            )
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            sys.stderr.write(tb + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        update_status(
            job_id,
            state="error",
            message=str(exc),
            error_type=type(exc).__name__,
            traceback=tb[-8000:],
        )


def process_url_job(
    job_id: str,
    items: list[dict[str, Any]],
    include_text: bool,
) -> None:
    ensure_open_stdio()
    disable_hf_progress()
    root = job_dir(job_id)
    try:
        with JOB_LOCK:
            summaries = []
            total = len(items)
            for index, item in enumerate(items, start=1):
                url = item["url"]
                label = item.get("label")
                clip_seconds = item.get("clip_seconds")
                update_status(
                    job_id,
                    state="downloading",
                    message=f"Downloading {label or url}",
                    current=index,
                    total=total,
                    include_text=include_text,
                )
                video = download_video_url(root, url, index)
                stimulus = trim_video(video, clip_seconds)
                update_status(
                    job_id,
                    state="predicting",
                    message=f"Running TRIBE v2 on {label or stimulus.name}",
                    current=index,
                    total=total,
                    clip_seconds=clip_seconds,
                    include_text=include_text,
                    source_url=url,
                    label=label,
                )
                out = root / "results" / stimulus.stem
                summary = run_prediction(
                    stimulus,
                    out,
                    job_id=job_id,
                    include_text=include_text,
                )
                summary["label"] = label
                summary["source_url"] = url
                summary["source_video"] = video.name
                summary["clip_seconds"] = clip_seconds
                summaries.append(summary)

            manifest = {"summaries": summaries}
            write_json(root / "results" / "manifest.json", manifest)
            previews = []
            for result_dir in sorted((root / "results").iterdir()):
                preview_path = result_dir / "preview.json"
                if preview_path.exists():
                    previews.append(read_json(preview_path))
            report = render_report(job_id, manifest, previews)
            (root / "results" / "report.html").write_text(report, encoding="utf-8")
            zip_path = root / "tribev2-results.zip"
            zip_dir(root / "results", zip_path)
            update_status(
                job_id,
                state="complete",
                message="Done",
                current=total,
                total=total,
                download_url=f"/jobs/{job_id}/download",
                report_url=f"/jobs/{job_id}/report",
            )
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            sys.stderr.write(tb + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        update_status(
            job_id,
            state="error",
            message=str(exc),
            error_type=type(exc).__name__,
            traceback=tb[-8000:],
        )


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "tribev2-remote-worker", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "model_id": MODEL_ID,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }


@app.get("/pilot-report")
def pilot_report() -> HTMLResponse:
    report_path = Path("/app/pilot-report.html")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="pilot report not bundled")
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)) -> JSONResponse:
    ensure_dirs()
    request_id = uuid.uuid4().hex
    root = JOBS_DIR / f"single-{request_id}"
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = root / f"input{suffix}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        summary = run_prediction(input_path, root / "results")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse({"job_id": f"single-{request_id}", "summary": summary})


@app.post("/jobs/tiktok")
def start_tiktok_job(payload: TikTokJobRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if payload.count < 1 or payload.count > 50:
        raise HTTPException(status_code=400, detail="count must be between 1 and 50")
    ensure_dirs()
    job_id = uuid.uuid4().hex
    root = job_dir(job_id)
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "status.json",
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Queued",
            "account_url": payload.account_url,
            "count": payload.count,
            "created_at": time.time(),
            "updated_at": time.time(),
            "clip_seconds": payload.clip_seconds,
            "include_text": payload.include_text,
        },
    )
    background_tasks.add_task(
        process_tiktok_job,
        job_id,
        payload.account_url,
        payload.count,
        payload.clip_seconds,
        payload.include_text,
    )
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.post("/jobs/urls")
def start_url_job(payload: UrlJobRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must not be empty")
    if len(payload.items) > 10:
        raise HTTPException(status_code=400, detail="items must contain at most 10 videos")
    ensure_dirs()
    job_id = uuid.uuid4().hex
    root = job_dir(job_id)
    root.mkdir(parents=True, exist_ok=True)
    items = [item.model_dump() for item in payload.items]
    write_json(
        root / "status.json",
        {
            "job_id": job_id,
            "state": "queued",
            "message": "Queued",
            "items": items,
            "count": len(items),
            "created_at": time.time(),
            "updated_at": time.time(),
            "include_text": payload.include_text,
        },
    )
    background_tasks.add_task(
        process_url_job,
        job_id,
        items,
        payload.include_text,
    )
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    status_path = job_dir(job_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return read_json(status_path)


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    zip_path = job_dir(job_id) / "tribev2-results.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="results zip not ready")
    return FileResponse(zip_path, filename=f"{job_id}-tribev2-results.zip")


@app.get("/jobs/{job_id}/report")
def job_report(job_id: str) -> HTMLResponse:
    report_path = job_dir(job_id) / "results" / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not ready")
    return HTMLResponse(report_path.read_text(encoding="utf-8"))
