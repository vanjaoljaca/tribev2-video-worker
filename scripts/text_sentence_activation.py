#!/usr/bin/env python3
import argparse
import html
import io
import json
import zipfile
from pathlib import Path

import numpy as np


BANDS = [
    ("band-1", 12),
    ("band-2", 44),
    ("band-3", 90),
    ("band-4", 145),
    ("band-5", 205),
    ("band-6", 275),
]


def load_json(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def load_npy(zf, name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))


def result_dirs(zf):
    return sorted({name.rsplit("/", 1)[0] for name in zf.namelist() if name.endswith("/predictions.npy")})


def clean_text(value):
    return " ".join(str(value or "").split())


def sentence_rows(events):
    rows = []
    for row in events:
        if row.get("type") != "Sentence" or not row.get("text"):
            continue
        text = clean_text(row.get("text"))
        if not text:
            continue
        start = float(row.get("start", 0.0))
        stop = float(row.get("stop", start))
        rows.append({"text": text, "start": start, "stop": stop})
    if rows:
        return rows

    # Some clips have only word rows. Build rough sentence-like clauses from pauses.
    words = [
        row
        for row in events
        if row.get("type") == "Word" and row.get("text") and row.get("start") is not None
    ]
    words.sort(key=lambda row: float(row.get("start", 0.0)))
    chunks = []
    current = []
    last_stop = None
    for row in words:
        start = float(row.get("start", 0.0))
        stop = float(row.get("stop", start))
        if current and last_stop is not None and (start - last_stop > 0.75 or len(current) >= 18):
            chunks.append(current)
            current = []
        current.append(row)
        last_stop = stop
    if current:
        chunks.append(current)
    for chunk in chunks:
        rows.append(
            {
                "text": clean_text(" ".join(str(row.get("text", "")) for row in chunk)),
                "start": float(chunk[0].get("start", 0.0)),
                "stop": float(chunk[-1].get("stop", chunk[-1].get("start", 0.0))),
            }
        )
    return rows


def window_indices(start, stop, length):
    lo = max(0, int(np.floor(start)))
    hi = min(length, int(np.ceil(stop)) + 1)
    if hi <= lo:
        hi = min(length, lo + 1)
    return lo, hi


def local_baseline(values, start, stop, radius=6):
    center = int(round((start + stop) / 2))
    lo = max(0, center - radius)
    hi = min(len(values), center + radius + 1)
    s_lo, s_hi = window_indices(start, stop, len(values))
    mask = np.ones(hi - lo, dtype=bool)
    for idx in range(max(lo, s_lo), min(hi, s_hi)):
        mask[idx - lo] = False
    context = values[lo:hi][mask]
    if len(context) == 0:
        return float(np.median(values))
    return float(np.median(context))


def label_for_dir(zf, dirname, manifest):
    summary = load_json(zf, f"{dirname}/summary.json")
    input_name = summary.get("input")
    for item in manifest.get("summaries", []):
        if item.get("input") == input_name:
            return item.get("label") or Path(input_name or dirname).stem
    return Path(input_name or dirname).stem


def analyze_clip(zf, dirname, manifest):
    label = label_for_dir(zf, dirname, manifest)
    preds = load_npy(zf, f"{dirname}/predictions.npy")
    events = load_json(zf, f"{dirname}/events.json")
    energy = np.mean(np.abs(preds), axis=1)
    signed = np.mean(preds, axis=1)
    vertices = preds.shape[1]
    split_points = np.linspace(0, vertices, len(BANDS) + 1, dtype=int)
    band_energy = []
    for start, stop in zip(split_points[:-1], split_points[1:]):
        band_energy.append(np.mean(np.abs(preds[:, start:stop]), axis=1))
    band_energy = np.asarray(band_energy)
    rows = []
    for sent in sentence_rows(events):
        lo, hi = window_indices(sent["start"], sent["stop"], len(energy))
        if lo >= len(energy):
            continue
        sent_energy = float(np.mean(energy[lo:hi]))
        sent_signed = float(np.mean(signed[lo:hi]))
        energy_delta = sent_energy - local_baseline(energy, sent["start"], sent["stop"])
        signed_delta = sent_signed - local_baseline(signed, sent["start"], sent["stop"])
        band_values = np.mean(band_energy[:, lo:hi], axis=1)
        band_local = np.asarray(
            [
                local_baseline(band_energy[idx], sent["start"], sent["stop"])
                for idx in range(len(BANDS))
            ]
        )
        band_delta = band_values - band_local
        dominant_idx = int(np.argmax(np.abs(band_delta)))
        rows.append(
            {
                "text": sent["text"],
                "start": sent["start"],
                "stop": sent["stop"],
                "energy": sent_energy,
                "energy_delta": float(energy_delta),
                "signed": sent_signed,
                "signed_delta": float(signed_delta),
                "dominant_band": BANDS[dominant_idx][0],
                "dominant_hue": BANDS[dominant_idx][1],
                "dominant_band_delta": float(band_delta[dominant_idx]),
            }
        )
    if rows:
        energies = np.asarray([row["energy"] for row in rows])
        deltas = np.asarray([row["signed_delta"] for row in rows])
        variation = float(np.std(energies) + 0.75 * np.std(deltas) + (np.max(energies) - np.min(energies)) / 4)
    else:
        variation = 0.0
    return {"label": label, "dirname": dirname, "rows": rows, "variation_score": variation}


def choose_clip(zip_path, requested_label=None):
    with zipfile.ZipFile(zip_path) as zf:
        manifest = load_json(zf, "manifest.json")
        clips = [analyze_clip(zf, dirname, manifest) for dirname in result_dirs(zf)]
    if requested_label:
        for clip in clips:
            if requested_label in clip["label"] or requested_label in clip["dirname"]:
                return clip, clips
        raise SystemExit(f"No clip matched: {requested_label}")
    clips.sort(key=lambda clip: clip["variation_score"], reverse=True)
    return clips[0], clips


def norm(value, lo, hi):
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def render_html(clip, all_clips, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = clip["rows"]
    if not rows:
        raise SystemExit("Selected clip has no sentence rows")
    energies = np.asarray([row["energy"] for row in rows])
    signed_deltas = np.asarray([row["signed_delta"] for row in rows])
    e_lo, e_hi = float(np.min(energies)), float(np.max(energies))
    sd_abs = float(max(abs(np.min(signed_deltas)), abs(np.max(signed_deltas)), 1e-9))

    payload = {"selected": clip, "ranked_clips": all_clips}
    (out_dir / "sentence_activation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    legend = "".join(
        f'<span class="legend-chip" style="--h:{hue}">{html.escape(name)}</span>'
        for name, hue in BANDS
    )
    ranked = "\n".join(
        f"<li><code>{html.escape(item['label'])}</code> variation {item['variation_score']:.4f}</li>"
        for item in sorted(all_clips, key=lambda x: x["variation_score"], reverse=True)[:6]
    )
    sentence_html = []
    for idx, row in enumerate(rows, start=1):
        intensity = norm(row["energy"], e_lo, e_hi)
        alpha = 0.14 + 0.62 * intensity
        border_hue = 142 if row["signed_delta"] >= 0 else 355
        border_alpha = min(0.95, 0.25 + abs(row["signed_delta"]) / sd_abs * 0.7)
        direction = "up" if row["signed_delta"] >= 0 else "down"
        sentence_html.append(
            f'''<section class="sentence" style="--h:{row['dominant_hue']};--a:{alpha:.3f};--bh:{border_hue};--ba:{border_alpha:.3f}">
  <div class="meta"><span>{idx}</span><span>{row['start']:.1f}s-{row['stop']:.1f}s</span><span>{html.escape(row['dominant_band'])}</span><span>{direction} {row['signed_delta']:+.4f}</span><span>intensity {row['energy']:.4f}</span></div>
  <p>{html.escape(row['text'])}</p>
</section>'''
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentence Activation - {html.escape(clip['label'])}</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f7f2;
  color: #171713;
}}
body {{
  margin: 0;
  padding: 28px;
}}
main {{
  max-width: 980px;
  margin: 0 auto;
}}
h1 {{
  font-size: 26px;
  margin: 0 0 8px;
}}
.sub {{
  color: #5c5c52;
  margin: 0 0 20px;
  line-height: 1.45;
}}
.panel {{
  background: #ffffff;
  border: 1px solid #deded4;
  border-radius: 8px;
  padding: 16px;
  margin: 0 0 18px;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}}
.legend-chip {{
  background: hsl(var(--h) 72% 82%);
  border: 1px solid hsl(var(--h) 50% 48%);
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
}}
.sentence {{
  background: hsla(var(--h), 78%, 66%, var(--a));
  border-left: 9px solid hsla(var(--bh), 72%, 42%, var(--ba));
  border-radius: 8px;
  padding: 12px 14px;
  margin: 10px 0;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
}}
.sentence p {{
  margin: 8px 0 0;
  font-size: 18px;
  line-height: 1.45;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: #30302b;
  font-size: 12px;
  font-weight: 650;
}}
.meta span {{
  background: rgba(255,255,255,0.62);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 999px;
  padding: 3px 7px;
}}
ol {{
  margin: 8px 0 0 20px;
  padding: 0;
}}
code {{
  font-size: 12px;
}}
</style>
</head>
<body>
<main>
  <h1>Sentence Activation Text Map</h1>
  <p class="sub">Selected clip: <code>{html.escape(clip['label'])}</code>. Background strength is sentence-level activation intensity. Left border is signed movement versus local baseline: green up, red down. Hue is the dominant coarse vertex band, not an anatomical atlas label.</p>
  <div class="panel">
    <strong>Vertex-band hue legend</strong>
    <div class="legend">{legend}</div>
  </div>
  <div class="panel">
    <strong>Most varied clips by sentence score</strong>
    <ol>{ranked}</ol>
  </div>
  {''.join(sentence_html)}
</main>
</body>
</html>
"""
    (out_dir / "sentence_activation.html").write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build text-first sentence activation visualization.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--label", help="Optional substring for the clip label/dir to visualize.")
    args = parser.parse_args()
    clip, clips = choose_clip(args.zip_path, args.label)
    render_html(clip, clips, args.out_dir)
    print(json.dumps({"selected": clip["label"], "variation_score": clip["variation_score"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
