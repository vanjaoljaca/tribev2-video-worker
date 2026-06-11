from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse


APP_ROOT = Path("/app")
REPORT_PATH = APP_ROOT / "pilot-report.html"
RESULTS_PATH = APP_ROOT / "pilot-results.zip"

app = FastAPI(title="Vanjao TRIBE v2 Pilot Report")


@app.get("/")
def home():
    return RedirectResponse(url="/pilot-report")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/pilot-status")
def pilot_status():
    return JSONResponse(
        {
            "state": "complete",
            "account_url": "https://www.tiktok.com/@vanjao.plays",
            "count": 2,
            "clip_seconds": 5.0,
            "include_text": True,
            "modalities": "audio+video+text",
            "prediction_shape": [6, 20484],
            "report_url": "/pilot-report",
            "download_url": "/pilot-results",
        }
    )


@app.get("/pilot-report", response_class=HTMLResponse)
def pilot_report():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="pilot report not found")
    return HTMLResponse(REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/pilot-results")
def pilot_results():
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="pilot results not found")
    return FileResponse(
        RESULTS_PATH,
        media_type="application/zip",
        filename="vanjao-tribev2-pilot-results.zip",
    )
