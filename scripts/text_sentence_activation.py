#!/usr/bin/env python3
import argparse
import html
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROUGH_SECTIONS = [
    ("prefrontal / executive", 12, "#2f6fbb"),
    ("motor / premotor", 34, "#c46628"),
    ("somatosensory / parietal", 142, "#2f8f5b"),
    ("temporal / auditory-language", 272, "#7654a8"),
    ("occipital / visual", 48, "#c8a021"),
    ("insula / salience-interoception", 188, "#16858a"),
    ("limbic / medial", 322, "#b23a7a"),
    ("association cortex", 218, "#667085"),
]

STOPWORDS = {
    "about",
    "actually",
    "also",
    "because",
    "really",
    "right",
    "there",
    "thing",
    "things",
    "think",
    "this",
    "that",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "you",
    "your",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "not",
    "but",
    "and",
    "the",
    "for",
    "from",
    "into",
    "then",
    "than",
    "they",
    "them",
    "it's",
    "im",
    "i",
}

TITLE_PHRASES = [
    ("booba", "kiki", "attention", "Booba Kiki Attention"),
    ("moving", "needle", "", "Moving The Needle"),
    ("discipline", "motivation", "", "Discipline Versus Motivation"),
    ("gym", "movement", "feels", "Gym Movement Feels Good"),
    ("monkey", "swear", "", "Swearing Monkey Alarm"),
    ("monkeys", "swearing", "", "Swearing Monkey Alarm"),
    ("chatgpt", "swearing", "book", "ChatGPT Swearing Book"),
    ("painting", "political", "context", "Art Political Context"),
    ("should", "political art", "", "Political Art Shoulds"),
    ("art", "political", "intent", "Political Art Intent"),
    ("political", "intent", "effect", "Political Intent Effect"),
    ("political", "effect", "intent", "Political Effect Intent"),
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


def word_rows(events):
    rows = []
    for row in events:
        if row.get("type") != "Word" or not row.get("text") or row.get("start") is None:
            continue
        text = clean_text(row.get("text"))
        if not text:
            continue
        start = float(row.get("start", 0.0))
        stop = float(row.get("stop", start))
        if stop <= start:
            stop = start + 0.18
        rows.append({"text": text, "start": start, "stop": stop})
    return sorted(rows, key=lambda item: (item["start"], item["stop"]))


def fallback_sentence_words(sentence):
    words = sentence["text"].split()
    if not words:
        return []
    duration = max(0.25, sentence["stop"] - sentence["start"])
    step = duration / len(words)
    rows = []
    for idx, word in enumerate(words):
        start = sentence["start"] + idx * step
        stop = sentence["start"] + (idx + 1) * step
        rows.append({"text": word, "start": start, "stop": stop})
    return rows


def title_case(words):
    minor = {"a", "an", "and", "as", "but", "for", "in", "of", "or", "the", "to", "vs"}
    titled = []
    for idx, word in enumerate(words):
        lower = word.lower()
        if lower == "chatgpt":
            titled.append("ChatGPT")
        elif lower == "ai":
            titled.append("AI")
        elif lower == "vs":
            titled.append("Vs")
        elif idx and lower in minor:
            titled.append(lower)
        else:
            titled.append(lower.capitalize())
    return " ".join(titled)


def transcript_title(rows, fallback):
    text = clean_text(" ".join(row["text"] for row in rows))
    lower = text.lower()
    for required_a, required_b, required_c, title in TITLE_PHRASES:
        if required_a in lower and required_b in lower and (not required_c or required_c in lower):
            return title

    first = rows[0]["text"] if rows else fallback
    cleaned = (
        first.replace("Number one.", "")
        .replace("Number one", "")
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
    )
    words = [word.strip("'\"()[]").lower() for word in cleaned.split()]
    words = [word for word in words if word and word not in STOPWORDS]
    if len(words) < 3:
        words = [word.strip("'\"()[]").lower() for word in cleaned.split() if word.strip("'\"()[]")]
    return title_case(words[:5] or [fallback])


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
    split_points = np.linspace(0, vertices, len(ROUGH_SECTIONS) + 1, dtype=int)
    section_energy = []
    for start, stop in zip(split_points[:-1], split_points[1:]):
        section_energy.append(np.mean(np.abs(preds[:, start:stop]), axis=1))
    section_energy = np.asarray(section_energy)

    def metrics_for_window(start, stop):
        lo, hi = window_indices(start, stop, len(energy))
        if lo >= len(energy):
            return None
        item_energy = float(np.mean(energy[lo:hi]))
        item_signed = float(np.mean(signed[lo:hi]))
        energy_delta = item_energy - local_baseline(energy, start, stop)
        signed_delta = item_signed - local_baseline(signed, start, stop)
        section_values = np.mean(section_energy[:, lo:hi], axis=1)
        section_local = np.asarray(
            [local_baseline(section_energy[idx], start, stop) for idx in range(len(ROUGH_SECTIONS))]
        )
        section_delta = section_values - section_local
        dominant_idx = int(np.argmax(np.abs(section_delta)))
        return {
            "energy": item_energy,
            "energy_delta": float(energy_delta),
            "signed": item_signed,
            "signed_delta": float(signed_delta),
            "dominant_section": ROUGH_SECTIONS[dominant_idx][0],
            "dominant_hue": ROUGH_SECTIONS[dominant_idx][1],
            "dominant_color": ROUGH_SECTIONS[dominant_idx][2],
            "dominant_section_delta": float(section_delta[dominant_idx]),
            "sections": [
                {
                    "name": name,
                    "hue": hue,
                    "color": color,
                    "energy": float(section_values[idx]),
                    "delta": float(section_delta[idx]),
                }
                for idx, (name, hue, color) in enumerate(ROUGH_SECTIONS)
            ],
        }

    def metrics_for_point(start, stop):
        center = max(0.0, min(float(len(energy) - 1), (start + stop) / 2))
        timeline = np.arange(len(energy), dtype=float)
        item_energy = float(np.interp(center, timeline, energy))
        item_signed = float(np.interp(center, timeline, signed))
        energy_delta = item_energy - local_baseline(energy, start, stop)
        signed_delta = item_signed - local_baseline(signed, start, stop)
        section_values = np.asarray(
            [np.interp(center, timeline, section_energy[idx]) for idx in range(len(ROUGH_SECTIONS))]
        )
        section_local = np.asarray(
            [local_baseline(section_energy[idx], start, stop) for idx in range(len(ROUGH_SECTIONS))]
        )
        section_delta = section_values - section_local
        dominant_idx = int(np.argmax(np.abs(section_delta)))
        return {
            "energy": item_energy,
            "energy_delta": float(energy_delta),
            "signed": item_signed,
            "signed_delta": float(signed_delta),
            "dominant_section": ROUGH_SECTIONS[dominant_idx][0],
            "dominant_hue": ROUGH_SECTIONS[dominant_idx][1],
            "dominant_color": ROUGH_SECTIONS[dominant_idx][2],
            "dominant_section_delta": float(section_delta[dominant_idx]),
            "sections": [
                {
                    "name": name,
                    "hue": hue,
                    "color": color,
                    "energy": float(section_values[idx]),
                    "delta": float(section_delta[idx]),
                }
                for idx, (name, hue, color) in enumerate(ROUGH_SECTIONS)
            ],
        }

    all_words = word_rows(events)
    rows = []
    for sent in sentence_rows(events):
        sent_metrics = metrics_for_window(sent["start"], sent["stop"])
        if sent_metrics is None:
            continue
        sentence_words = [
            word
            for word in all_words
            if sent["start"] - 0.05 <= (word["start"] + word["stop"]) / 2 < sent["stop"] + 0.05
        ]
        if not sentence_words:
            sentence_words = fallback_sentence_words(sent)
        analyzed_words = []
        for word in sentence_words:
            word_metrics = metrics_for_point(word["start"], word["stop"])
            if word_metrics is None:
                continue
            analyzed_words.append({**word, **word_metrics})
        rows.append(
            {
                "text": sent["text"],
                "start": sent["start"],
                "stop": sent["stop"],
                **sent_metrics,
                "words": analyzed_words,
            }
        )
    if rows:
        energies = np.asarray([row["energy"] for row in rows])
        deltas = np.asarray([row["signed_delta"] for row in rows])
        variation = float(np.std(energies) + 0.75 * np.std(deltas) + (np.max(energies) - np.min(energies)) / 4)
    else:
        variation = 0.0
    return {
        "label": label,
        "title": transcript_title(rows, label),
        "dirname": dirname,
        "rows": rows,
        "variation_score": variation,
    }


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
    if not clip["rows"]:
        raise SystemExit("Selected clip has no sentence rows")

    payload = {"selected": clip, "ranked_clips": all_clips}
    (out_dir / "sentence_activation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    legend = "".join(
        f'<span class="legend-chip" style="--c:{color}">{html.escape(name)}</span>'
        for name, _hue, color in ROUGH_SECTIONS
    )
    payload_json = json.dumps(payload).replace("</", "<\\/")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word Activation - TRIBE v2</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f6f1;
  color: #171713;
}}
body {{
  margin: 0;
  padding: 28px;
}}
main {{
  max-width: 1040px;
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
.controls {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: end;
}}
label {{
  display: grid;
  gap: 6px;
  color: #424238;
  font-size: 13px;
  font-weight: 650;
}}
select {{
  width: 100%;
  appearance: none;
  background: #fbfbf7;
  border: 1px solid #cfcfc4;
  border-radius: 6px;
  color: #151512;
  font: inherit;
  padding: 10px 12px;
}}
.score {{
  justify-self: end;
  color: #4f4f45;
  font-size: 13px;
  white-space: nowrap;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}}
.legend-chip {{
  background: color-mix(in oklab, var(--c), white 78%);
  border: 1px solid color-mix(in oklab, var(--c), black 8%);
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
}}
.sentence {{
  position: relative;
  overflow: hidden;
  background: #fffffb;
  border-left: 8px solid hsla(var(--bh), 72%, 42%, var(--ba));
  border-radius: 8px;
  padding: 0;
  margin: 12px 0;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
}}
.sentence-content {{
  position: relative;
  z-index: 1;
  padding: 14px 16px 15px;
}}
.script {{
  margin: 10px 0 0;
  font-size: 19px;
  line-height: 1.72;
  color: #11110e;
}}
.word {{
  position: relative;
  display: inline-block;
  margin: 0 0.14em 0.08em 0;
  padding: 0.01em 0.04em;
  isolation: isolate;
}}
.word-bands {{
  position: absolute;
  z-index: -1;
  left: -0.04em;
  right: -0.04em;
  top: 50%;
  height: 1.2em;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 1px;
  pointer-events: none;
}}
.word-bands i {{
  display: block;
  height: calc((1.2em - 7px) / 8);
  min-height: 1px;
  background: color-mix(in oklab, var(--c), white var(--mix));
  opacity: var(--sa);
  border-radius: 2px;
}}
.word-text {{
  text-shadow:
    0 1px 0 rgba(255,255,255,0.82),
    0 -1px 0 rgba(255,255,255,0.58),
    1px 0 0 rgba(255,255,255,0.58),
    -1px 0 0 rgba(255,255,255,0.58);
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
  background: rgba(255,255,255,0.72);
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
@media (max-width: 720px) {{
  body {{ padding: 16px; }}
  .controls {{ grid-template-columns: 1fr; }}
  .score {{ justify-self: start; }}
  .script {{ font-size: 18px; }}
}}
</style>
</head>
<body>
<main>
  <h1>Word Activation Text Map</h1>
  <p class="sub">Full script by video. Each word carries eight compressed bands, about 20% taller than the text, with brightness changing from that word's timestamp window. Left sentence border is signed movement versus local baseline: green up, red down. Rough cortical sections, not atlas-verified ROI labels.</p>
  <div class="panel controls">
    <label>Video
      <select id="videoSelect"></select>
    </label>
    <div class="score" id="score"></div>
  </div>
  <div class="panel">
    <strong>Rough section hue legend</strong>
    <div class="legend">{legend}</div>
  </div>
  <section id="script"></section>
</main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById("payload").textContent);
const clips = [...payload.ranked_clips].sort((a, b) => b.variation_score - a.variation_score);
const select = document.getElementById("videoSelect");
const score = document.getElementById("score");
const script = document.getElementById("script");

function sectionRange(clip) {{
  const values = [];
  for (const row of clip.rows) {{
    for (const word of row.words || []) {{
      for (const section of word.sections || []) values.push(section.energy);
    }}
  }}
  if (!values.length) return [0, 1];
  return [Math.min(...values), Math.max(...values)];
}}

function signedRange(clip) {{
  const values = clip.rows.map(row => Math.abs(row.signed_delta || 0));
  return Math.max(...values, 1e-9);
}}

function norm(value, lo, hi) {{
  if (hi <= lo) return 0.5;
  return Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
}}

function el(tag, className, text) {{
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}}

function wordNode(word, lo, hi) {{
  const outer = el("span", "word");
  outer.title = `${{word.start.toFixed(1)}}s-${{word.stop.toFixed(1)}}s | ${{word.dominant_section}} | signed ${{word.signed_delta.toFixed(4)}}`;
  const bands = el("span", "word-bands");
  for (const section of word.sections || []) {{
    const intensity = norm(section.energy, lo, hi);
    const mix = 84 - 62 * intensity;
    const alpha = 0.2 + 0.78 * intensity;
    const band = document.createElement("i");
    band.style.setProperty("--c", section.color);
    band.style.setProperty("--mix", `${{mix.toFixed(1)}}%`);
    band.style.setProperty("--sa", alpha.toFixed(3));
    bands.appendChild(band);
  }}
  outer.appendChild(bands);
  outer.appendChild(el("span", "word-text", word.text));
  return outer;
}}

function renderClip(index) {{
  const clip = clips[index];
  const [sectionLo, sectionHi] = sectionRange(clip);
  const signedAbs = signedRange(clip);
  score.textContent = `variation ${{clip.variation_score.toFixed(4)}} | source ${{clip.label}}`;
  script.replaceChildren();
  for (let idx = 0; idx < clip.rows.length; idx += 1) {{
    const row = clip.rows[idx];
    const direction = row.signed_delta >= 0 ? "up" : "down";
    const borderHue = row.signed_delta >= 0 ? 142 : 355;
    const borderAlpha = Math.min(0.95, 0.25 + Math.abs(row.signed_delta) / signedAbs * 0.7);
    const card = el("section", "sentence");
    card.style.setProperty("--bh", borderHue);
    card.style.setProperty("--ba", borderAlpha.toFixed(3));
    const content = el("div", "sentence-content");
    const meta = el("div", "meta");
    for (const item of [
      String(idx + 1),
      `${{row.start.toFixed(1)}}s-${{row.stop.toFixed(1)}}s`,
      row.dominant_section,
      `${{direction}} ${{row.signed_delta >= 0 ? "+" : ""}}${{row.signed_delta.toFixed(4)}}`,
      `intensity ${{row.energy.toFixed(4)}}`,
    ]) {{
      meta.appendChild(el("span", "", item));
    }}
    const text = el("p", "script");
    const words = row.words && row.words.length ? row.words : row.text.split(/\\s+/).map(token => ({{ text: token, sections: row.sections, start: row.start, stop: row.stop, dominant_section: row.dominant_section, signed_delta: row.signed_delta }}));
    for (const word of words) {{
      text.appendChild(wordNode(word, sectionLo, sectionHi));
    }}
    content.appendChild(meta);
    content.appendChild(text);
    card.appendChild(content);
    script.appendChild(card);
  }}
}}

clips.forEach((clip, index) => {{
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = `${{clip.title}} (${{clip.variation_score.toFixed(4)}})`;
  if (clip.dirname === payload.selected.dirname) option.selected = true;
  select.appendChild(option);
}});
select.addEventListener("change", event => renderClip(Number(event.target.value)));
renderClip(Number(select.value || 0));
</script>
</body>
</html>
"""
    (out_dir / "sentence_activation.html").write_text(doc, encoding="utf-8")
    render_png(clip, out_dir / "sentence_activation_preview.png")


def hsl_to_rgb(h, s=0.74, l=0.72):
    import colorsys

    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def blend_rgb(color, amount_white):
    rgb = hex_to_rgb(color)
    return tuple(int(channel * (1 - amount_white) + 255 * amount_white) for channel in rgb)


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap_text(draw, text, font, width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_png(clip, out_path):
    rows = clip["rows"]
    if not rows:
        return
    all_section_values = [section["energy"] for row in rows for section in row["sections"]]
    section_lo, section_hi = min(all_section_values), max(all_section_values)
    signed_deltas = np.asarray([row["signed_delta"] for row in rows])
    sd_abs = float(max(abs(np.min(signed_deltas)), abs(np.max(signed_deltas)), 1e-9))

    width = 1200
    margin = 48
    stripe_h = 5
    stripe_gap = 1
    card_gap = 14
    text_font = load_font(23)
    meta_font = load_font(14)
    title_font = load_font(30, bold=True)
    line_h = 32

    scratch = Image.new("RGB", (width, 200), "white")
    draw = ImageDraw.Draw(scratch)
    cards = []
    for row in rows:
        lines = wrap_text(draw, row["text"], text_font, width - margin * 2 - 42)
        height = 28 + (stripe_h + stripe_gap) * len(ROUGH_SECTIONS) + 14 + len(lines) * line_h + 18
        cards.append((row, lines, height))
    height = margin + 48 + 42 + sum(card[2] + card_gap for card in cards) + margin
    img = Image.new("RGB", (width, height), (247, 247, 242))
    draw = ImageDraw.Draw(img)
    y = margin
    draw.text((margin, y), "Sentence Activation Text Map", fill=(22, 22, 18), font=title_font)
    y += 42
    draw.text((margin, y), f"Selected: {clip['title']}  |  stripes = rough cortical sections, brightness = activation", fill=(70, 70, 62), font=meta_font)
    y += 34

    for row, lines, card_h in cards:
        x = margin
        card_w = width - margin * 2
        border_color = (23, 132, 72) if row["signed_delta"] >= 0 else (179, 45, 64)
        border_scale = min(1.0, 0.25 + abs(row["signed_delta"]) / sd_abs * 0.75)
        border_color = tuple(int(230 * (1 - border_scale) + c * border_scale) for c in border_color)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=10, fill=(255, 255, 255), outline=(218, 218, 210), width=1)
        draw.rounded_rectangle((x, y, x + 10, y + card_h), radius=8, fill=border_color)
        sy = y + 14
        for section in row["sections"]:
            intensity = norm(section["energy"], section_lo, section_hi)
            rgb = blend_rgb(section["color"], 0.76 - 0.54 * intensity)
            draw.rectangle((x + 20, sy, x + card_w - 16, sy + stripe_h), fill=rgb)
            sy += stripe_h + stripe_gap
        meta = f"{row['start']:.1f}-{row['stop']:.1f}s | {row['dominant_section']} | signed {row['signed_delta']:+.4f} | intensity {row['energy']:.4f}"
        draw.text((x + 22, sy + 6), meta, fill=(48, 48, 42), font=meta_font)
        ty = sy + 28
        for line in lines:
            draw.text((x + 22, ty), line, fill=(20, 20, 17), font=text_font)
            ty += line_h
        y += card_h + card_gap
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build text-first sentence activation visualization.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--label", help="Optional substring for the clip label/dir to visualize.")
    args = parser.parse_args()
    clip, clips = choose_clip(args.zip_path, args.label)
    render_html(clip, clips, args.out_dir)
    print(
        json.dumps(
            {
                "selected": clip["title"],
                "source_label": clip["label"],
                "variation_score": clip["variation_score"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
