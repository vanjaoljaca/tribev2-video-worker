# TRIBE v2 Video Worker

Remote Hugging Face Space worker for running short-form videos through
Meta/Facebook Research TRIBE v2 with audio, video, and text events.

This repo contains the runnable worker code and local analysis helpers.
It intentionally does not include downloaded social videos, HF tokens, model
weights, caches, or full raw prediction archives.

## Cost Notes

Short social-video runs are usually cheap once the Space is built and the model
is cached. On Hugging Face A10G-large at `$1.50/hr`, 10 minutes is about `$0.25`
of GPU time.

First-time setup/debug sessions cost more wall time because they include Space
rebuilds, dependency fixes, dtype fixes, and failed attempts. Redos from this
code should be faster because the worker has the dtype patch, explicit URL jobs,
single-job locking, and report flow.

## Remote Worker

Worker code is in [`hf-space/`](hf-space/). It exposes:

- `POST /jobs/urls`
- `POST /jobs/tiktok`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/report`
- `GET /jobs/{job_id}/download`

Example explicit URL request:

```bash
curl -X POST https://<space-subdomain>.hf.space/jobs/urls \
  -H 'Content-Type: application/json' \
  -d '{
    "include_text": true,
    "items": [
      {"url": "https://example.com/video-1", "label": "clip_one", "clip_seconds": null},
      {"url": "https://example.com/video-2", "label": "clip_two", "clip_seconds": 30}
    ]
  }'
```

Example account batch request:

```bash
curl -X POST https://<space-subdomain>.hf.space/jobs/tiktok \
  -H 'Content-Type: application/json' \
  -d '{"account_url":"https://www.tiktok.com/@some_account","count":10,"include_text":true,"clip_seconds":30}'
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

Analysis helpers are in [`scripts/`](scripts/), including a text-first sentence
activation view that renders local HTML from a worker result zip.
