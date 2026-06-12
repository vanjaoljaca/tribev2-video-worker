---
title: TRIBE v2 Video API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# TRIBE v2 Video API

Remote TRIBE v2 worker for short videos and TikTok batches.

Endpoints:

- `GET /health`
- `POST /predict_video` with multipart `file`
- `POST /jobs/tiktok` with JSON `{"account_url":"https://www.tiktok.com/@some_account","count":12}`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/download`

Requires a GPU Space and an `HF_TOKEN` secret with access to Meta's gated LLaMA 3.2 dependency.
