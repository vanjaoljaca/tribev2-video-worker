#!/usr/bin/env python3
import argparse
import html
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


ROUGH_SECTIONS = [
    ("prefrontal / executive", "#2f6fbb"),
    ("motor / premotor", "#c46628"),
    ("somatosensory / parietal", "#2f8f5b"),
    ("temporal / auditory-language", "#7654a8"),
    ("occipital / visual", "#c8a021"),
    ("insula / salience-interoception", "#16858a"),
    ("limbic / medial", "#b23a7a"),
    ("association cortex", "#667085"),
]


def load_json(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def load_npy(zf, name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))


def result_dirs(zf):
    dirs = set()
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) == 2 and parts[1] == "predictions.npy":
            dirs.add(parts[0])
    return sorted(dirs)


def normalize(values):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values), lo, hi
    return (values - lo) / (hi - lo), lo, hi


def build_payload(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        manifest = load_json(zf, "manifest.json")
        dirname = result_dirs(zf)[0]
        preds = load_npy(zf, f"{dirname}/predictions.npy")
        events = load_json(zf, f"{dirname}/events.json")
        summary = load_json(zf, f"{dirname}/summary.json")
        item = manifest.get("summaries", [{}])[0]

    energy = np.mean(np.abs(preds), axis=1)
    signed = np.mean(preds, axis=1)
    diff = np.r_[0.0, np.diff(energy)]
    energy_norm, energy_lo, energy_hi = normalize(energy)
    signed_norm, signed_lo, signed_hi = normalize(np.abs(signed))

    vertices = preds.shape[1]
    split_points = np.linspace(0, vertices, len(ROUGH_SECTIONS) + 1, dtype=int)
    sections = []
    for idx, (start, stop) in enumerate(zip(split_points[:-1], split_points[1:])):
        values = np.mean(np.abs(preds[:, start:stop]), axis=1)
        movement = np.r_[0.0, np.diff(values)]
        normed, lo, hi = normalize(values)
        sections.append(
            {
                "name": ROUGH_SECTIONS[idx][0],
                "color": ROUGH_SECTIONS[idx][1],
                "values": values.tolist(),
                "movement": movement.tolist(),
                "norm": normed.tolist(),
                "min": lo,
                "max": hi,
            }
        )

    top_seconds = []
    order = list(np.argsort(energy)[::-1])
    for idx in order:
        sec = int(idx)
        if all(abs(sec - other["second"]) >= 5 for other in top_seconds):
            top_seconds.append(
                {
                    "second": sec,
                    "energy": float(energy[idx]),
                    "signed": float(signed[idx]),
                    "jump": float(diff[idx]),
                }
            )
        if len(top_seconds) >= 8:
            break

    text_events = [
        event
        for event in events
        if event.get("type") in {"Sentence", "Word"} and event.get("text")
    ]

    return {
        "label": item.get("label") or Path(summary.get("input", dirname)).stem,
        "source_url": item.get("source_url"),
        "input": item.get("input") or summary.get("input"),
        "clip_seconds": item.get("clip_seconds"),
        "shape": list(preds.shape),
        "timings": item.get("timings") or summary.get("timings"),
        "energy": energy.tolist(),
        "energy_norm": energy_norm.tolist(),
        "signed": signed.tolist(),
        "signed_abs_norm": signed_norm.tolist(),
        "jump": diff.tolist(),
        "energy_min": energy_lo,
        "energy_max": energy_hi,
        "signed_abs_min": signed_lo,
        "signed_abs_max": signed_hi,
        "sections": sections,
        "top_seconds": top_seconds,
        "text_event_count": len(text_events),
    }


def render_page(payload, video_name, out_dir):
    payload_json = html.escape(json.dumps(payload), quote=False)
    title = html.escape(payload["label"])
    source = html.escape(payload.get("source_url") or "")
    section_rows = "\n".join(
        f'<div class="bar-row"><span>{html.escape(section["name"])}</span><b style="--c:{section["color"]}"><i></i></b><em>0.000</em></div>'
        for section in payload["sections"]
    )
    movement_rows = "\n".join(
        f'<div class="diff-row" data-section="{idx}"><span>{html.escape(section["name"])}</span><b>0.000</b><i>0.000</i><em>0.000</em></div>'
        for idx, section in enumerate(payload["sections"])
    )
    peak_buttons = "\n".join(
        f'<button type="button" data-second="{row["second"]}">{row["second"]}s <span>{row["energy"]:.3f}</span></button>'
        for row in payload["top_seconds"]
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} synced activation</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #17191f; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 18px; }}
    header {{ display: grid; gap: 6px; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 24px; line-height: 1.15; letter-spacing: 0; }}
    .sub {{ margin: 0; color: #5c6270; font-size: 14px; }}
    .layout {{ display: grid; grid-template-columns: minmax(280px, 430px) 1fr; gap: 16px; align-items: start; }}
    video {{ width: 100%; max-height: 78vh; background: #111; border-radius: 8px; }}
    .panel {{ background: white; border: 1px solid #dde1e8; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .timeline-wrap {{ position: relative; height: 260px; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    .readout {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 12px; }}
    .metric {{ border: 1px solid #e4e7ec; border-radius: 6px; padding: 8px; background: #fbfcfe; }}
    .metric span {{ display: block; color: #667085; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 18px; line-height: 1.2; margin-top: 2px; }}
    .bar-stack {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(120px, 220px) 1fr 56px; gap: 8px; align-items: center; font-size: 13px; }}
    .bar-row b {{ height: 14px; background: #edf0f5; border-radius: 999px; overflow: hidden; }}
    .bar-row i {{ display: block; height: 100%; width: 0%; background: var(--c); border-radius: inherit; transition: width .08s linear; }}
    .bar-row em {{ font-style: normal; font-variant-numeric: tabular-nums; color: #475467; text-align: right; }}
    .diff-title {{ margin: 14px 0 6px; font-size: 13px; color: #475467; font-weight: 650; }}
    .diff-grid {{ display: grid; gap: 4px; }}
    .diff-row {{ display: grid; grid-template-columns: minmax(120px, 220px) repeat(3, 64px); gap: 8px; align-items: center; font-size: 12px; padding: 5px 0; border-top: 1px solid #eef1f5; }}
    .diff-row:first-child {{ border-top: 0; }}
    .diff-row b, .diff-row i, .diff-row em {{ font-style: normal; font-variant-numeric: tabular-nums; text-align: right; }}
    .diff-row b {{ color: #17191f; }}
    .diff-row i {{ color: #667085; }}
    .diff-row em.up {{ color: #0f7a43; }}
    .diff-row em.down {{ color: #b42318; }}
    .peaks {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    button {{ appearance: none; border: 1px solid #cfd5df; background: #fff; border-radius: 6px; padding: 7px 9px; font: inherit; cursor: pointer; }}
    button span {{ color: #667085; margin-left: 4px; }}
    .note {{ color: #667085; font-size: 13px; margin-top: 10px; }}
    @media (max-width: 840px) {{
      main {{ padding: 12px; }}
      .layout {{ grid-template-columns: 1fr; }}
      .timeline-wrap {{ height: 220px; }}
      .readout {{ grid-template-columns: repeat(2, 1fr); }}
      .bar-row {{ grid-template-columns: 112px 1fr 48px; }}
      .diff-row {{ grid-template-columns: 112px repeat(3, 52px); }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{title}</h1>
    <p class="sub">Full-video TRIBE v2 prediction, shape {payload["shape"][0]} x {payload["shape"][1]}. Source: <a href="{source}">{source}</a></p>
  </header>
  <section class="layout">
    <div class="panel">
      <video id="video" controls playsinline src="{html.escape(video_name)}"></video>
      <p class="note">Text/lyric extraction produced {payload["text_event_count"]} text events, so this view is synced by video time rather than words.</p>
    </div>
    <div class="panel">
      <div class="timeline-wrap"><canvas id="timeline"></canvas></div>
      <div class="readout">
        <div class="metric"><span>time</span><strong id="timeMetric">0.0s</strong></div>
        <div class="metric"><span>energy</span><strong id="energyMetric">0.000</strong></div>
        <div class="metric"><span>signed</span><strong id="signedMetric">0.000</strong></div>
        <div class="metric"><span>jump</span><strong id="jumpMetric">0.000</strong></div>
      </div>
      <div class="bar-stack" id="bars">{section_rows}</div>
      <div class="diff-title">Current movement: section vs average of the other sections</div>
      <div class="diff-grid" id="movementDiff">
        <div class="diff-row"><span></span><b>section</b><i>others</i><em>diff</em></div>
        {movement_rows}
      </div>
      <div class="peaks">{peak_buttons}</div>
      <p class="note">Timeline blue = mean absolute predicted response. Orange = positive frame-to-frame jump. Bars below show current activation by rough cortical section.</p>
    </div>
  </section>
</main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById("payload").textContent);
const video = document.getElementById("video");
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const bars = [...document.querySelectorAll(".bar-row")];
const diffRows = [...document.querySelectorAll(".diff-row[data-section]")];
const timeMetric = document.getElementById("timeMetric");
const energyMetric = document.getElementById("energyMetric");
const signedMetric = document.getElementById("signedMetric");
const jumpMetric = document.getElementById("jumpMetric");

function resizeCanvas() {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(300, Math.floor(rect.width * scale));
  canvas.height = Math.max(180, Math.floor(rect.height * scale));
}}

function draw() {{
  resizeCanvas();
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const padL = 42, padR = 12, padT = 12, padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {{
    const y = padT + plotH * i / 4;
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
  }}
  ctx.stroke();

  const values = payload.energy_norm;
  ctx.strokeStyle = "#2257d7";
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((v, idx) => {{
    const x = padL + plotW * idx / Math.max(1, values.length - 1);
    const y = padT + plotH * (1 - v);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();

  ctx.fillStyle = "rgba(255,138,0,.35)";
  payload.jump.forEach((v, idx) => {{
    if (v <= 0) return;
    const x = padL + plotW * idx / Math.max(1, values.length - 1);
    const barH = Math.min(plotH, Math.max(1, v / Math.max(1e-9, payload.energy_max - payload.energy_min) * plotH));
    ctx.fillRect(x - 1, padT + plotH - barH, 2, barH);
  }});

  const current = Math.max(0, Math.min(values.length - 1, Math.round(video.currentTime || 0)));
  const x = padL + plotW * current / Math.max(1, values.length - 1);
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = "#475467";
  ctx.font = `${{Math.max(11, Math.round(w / 70))}}px system-ui`;
  ctx.fillText("0s", padL, h - 8);
  ctx.fillText(`${{values.length - 1}}s`, w - padR - 44, h - 8);
}}

function updateReadout() {{
  const idx = Math.max(0, Math.min(payload.energy.length - 1, Math.round(video.currentTime || 0)));
  timeMetric.textContent = `${{(video.currentTime || 0).toFixed(1)}}s`;
  energyMetric.textContent = payload.energy[idx].toFixed(3);
  signedMetric.textContent = payload.signed[idx].toFixed(3);
  jumpMetric.textContent = payload.jump[idx].toFixed(3);
  payload.sections.forEach((section, sidx) => {{
    const row = bars[sidx];
    const width = Math.round((section.norm[idx] || 0) * 100);
    row.querySelector("i").style.width = `${{width}}%`;
    row.querySelector("em").textContent = section.values[idx].toFixed(3);
  }});
  const moves = payload.sections.map(section => section.movement[idx] || 0);
  payload.sections.forEach((section, sidx) => {{
    const row = diffRows[sidx];
    const sectionMove = moves[sidx] || 0;
    const otherMoves = moves.filter((_, midx) => midx !== sidx);
    const others = otherMoves.reduce((sum, value) => sum + value, 0) / Math.max(1, otherMoves.length);
    const diff = sectionMove - others;
    const diffNode = row.querySelector("em");
    row.querySelector("b").textContent = sectionMove.toFixed(3);
    row.querySelector("i").textContent = others.toFixed(3);
    diffNode.textContent = `${{diff >= 0 ? "+" : ""}}${{diff.toFixed(3)}}`;
    diffNode.className = diff >= 0 ? "up" : "down";
  }});
  draw();
}}

document.querySelectorAll("button[data-second]").forEach(button => {{
  button.addEventListener("click", () => {{
    video.currentTime = Number(button.dataset.second || 0);
    video.play();
  }});
}});
video.addEventListener("timeupdate", updateReadout);
video.addEventListener("loadedmetadata", updateReadout);
window.addEventListener("resize", updateReadout);
updateReadout();
</script>
</body>
</html>
"""
    (out_dir / "activation_sync.html").write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_zip")
    parser.add_argument("video")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_src = Path(args.video)
    video_name = "singing-sampler-video.mp4"
    video_dest = out_dir / video_name
    if video_src.resolve() != video_dest.resolve():
        shutil.copy2(video_src, video_dest)

    payload = build_payload(Path(args.results_zip))
    (out_dir / "activation_sync.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_page(payload, video_name, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "html": str(out_dir / "activation_sync.html"), "shape": payload["shape"]}, indent=2))


if __name__ == "__main__":
    main()
