# Vanjao TRIBE v2 Pilot

Remote Hugging Face Space worker for running `@vanjao.plays` TikTok clips
through Meta/Facebook Research TRIBE v2 with audio, video, and text events.

This repo contains the runnable worker code plus a small verified two-video
report. It intentionally does not include downloaded TikTok videos, HF tokens,
model weights, caches, or full raw prediction archives.

## Verified Pilot

- Account: `https://www.tiktok.com/@vanjao.plays`
- Job: `1e1dcedd43534c67abb28eacb4a8b727`
- Clips: 2
- Clip length: 5 seconds each
- Modalities: `audio+video+text`
- Output shape for both clips: `[6, 20484]`

The generated phone-friendly report lives in [`report/REPORT.md`](report/REPORT.md).

## Cost Notes

The successful text-enabled two-video job itself ran for about 4 minutes.
On Hugging Face A10G-large at `$1.50/hr`, that is about `$0.10` of GPU time.

The first-time setup/debug session used more wall time because it included
Space rebuilds, dependency fixes, LLaMA dtype fixes, and failed attempts. A redo
from this code should be much faster because the worker is already patched and
the Space cache can retain dependencies/models.

## Remote Worker

Worker code is in [`hf-space/`](hf-space/). It exposes:

- `POST /jobs/tiktok`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/report`
- `GET /jobs/{job_id}/download`

Example request:

```bash
curl -X POST https://<space-subdomain>.hf.space/jobs/tiktok \
  -H 'Content-Type: application/json' \
  -d '{"account_url":"https://www.tiktok.com/@vanjao.plays","count":2,"include_text":true,"clip_seconds":5}'
```

## Deploy To Hugging Face

Create a Docker Space, set `HF_TOKEN` as a secret with access to the gated
Meta LLaMA dependency, then upload:

```bash
cd hf-space
huggingface-cli upload <user>/<space-name> . . --repo-type space
```

Use a GPU for the run, then pause or downgrade when done:

```python
from huggingface_hub import HfApi

api = HfApi(token="...")
api.pause_space(repo_id="<user>/<space-name>")
```

## Visuals

The pilot analysis plots are in [`report/visuals/`](report/visuals/).

