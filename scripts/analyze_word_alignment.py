#!/usr/bin/env python3
import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")


def load_json(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def load_npy(zf, name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))


def normalize_word(text):
    match = WORD_RE.search(str(text or "").lower())
    return match.group(0) if match else ""


def compact_text(text, limit=180):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def local_baseline(values, second, radius=6):
    lo = max(0, second - radius)
    hi = min(len(values), second + radius + 1)
    mask = np.ones(hi - lo, dtype=bool)
    center = second - lo
    if 0 <= center < len(mask):
        mask[center] = False
    context = values[lo:hi][mask]
    if len(context) == 0:
        return float(np.median(values))
    return float(np.median(context))


def result_dirs(zf):
    dirs = set()
    for name in zf.namelist():
        if name.endswith("/predictions.npy"):
            dirs.add(name.rsplit("/", 1)[0])
    return sorted(dirs)


def label_for_dir(zf, dirname, manifest):
    summary = load_json(zf, f"{dirname}/summary.json")
    input_name = summary.get("input")
    for item in manifest.get("summaries", []):
        if item.get("input") == input_name:
            return item.get("label") or Path(input_name or dirname).stem
    return Path(input_name or dirname).stem


def analyze_zip(zip_path):
    clip_rows = []
    word_rows = []
    aggregate = defaultdict(list)

    with zipfile.ZipFile(zip_path) as zf:
        manifest = load_json(zf, "manifest.json")
        for dirname in result_dirs(zf):
            label = label_for_dir(zf, dirname, manifest)
            preds = load_npy(zf, f"{dirname}/predictions.npy")
            events = load_json(zf, f"{dirname}/events.json")
            energy = np.mean(np.abs(preds), axis=1)
            words = [
                row
                for row in events
                if row.get("type") == "Word" and normalize_word(row.get("text"))
            ]
            rows = []
            for row in words:
                second = int(round(float(row.get("start", 0.0))))
                if not 0 <= second < len(energy):
                    continue
                word = normalize_word(row.get("text"))
                baseline = local_baseline(energy, second)
                value = float(energy[second])
                delta = value - baseline
                out = {
                    "clip": label,
                    "word": word,
                    "raw_text": row.get("text"),
                    "second": second,
                    "energy": value,
                    "baseline": baseline,
                    "delta": delta,
                    "sentence": compact_text(row.get("sentence")),
                    "context": compact_text(row.get("context")),
                }
                rows.append(out)
                word_rows.append(out)
                aggregate[word].append(delta)

            rows.sort(key=lambda x: x["delta"], reverse=True)
            clip_rows.append(
                {
                    "clip": label,
                    "word_count": len(rows),
                    "top_spikes": rows[:12],
                    "top_dulls": list(reversed(rows[-12:])),
                }
            )

    aggregate_rows = []
    for word, deltas in aggregate.items():
        if len(deltas) < 2:
            continue
        arr = np.asarray(deltas, dtype=float)
        aggregate_rows.append(
            {
                "word": word,
                "count": int(len(arr)),
                "mean_delta": float(arr.mean()),
                "median_delta": float(np.median(arr)),
                "max_delta": float(arr.max()),
                "min_delta": float(arr.min()),
            }
        )

    return {
        "source_zip": str(zip_path),
        "clips": clip_rows,
        "aggregate_spike_words": sorted(
            aggregate_rows, key=lambda x: (x["mean_delta"], x["count"]), reverse=True
        )[:30],
        "aggregate_dull_words": sorted(
            aggregate_rows, key=lambda x: (x["mean_delta"], -x["count"])
        )[:30],
    }


def write_markdown(result, out_path):
    lines = [
        "# Word Alignment Analysis",
        "",
        "Scores are local associations between transcript words and mean absolute TRIBE v2 response energy.",
        "They are not causal proof: a word can align with audio, visual edits, rhythm, or nearby words.",
        "",
        "## Aggregate Spike-Associated Words",
        "",
    ]
    for row in result["aggregate_spike_words"][:20]:
        lines.append(
            f"- `{row['word']}` count `{row['count']}`, mean delta `{row['mean_delta']:.4f}`, max `{row['max_delta']:.4f}`"
        )
    lines += ["", "## Aggregate Dull-Associated Words", ""]
    for row in result["aggregate_dull_words"][:20]:
        lines.append(
            f"- `{row['word']}` count `{row['count']}`, mean delta `{row['mean_delta']:.4f}`, min `{row['min_delta']:.4f}`"
        )
    lines += ["", "## Per-Clip Extremes", ""]
    for clip in result["clips"]:
        lines += [f"### {clip['clip']}", "", "Spike-aligned words:", ""]
        for row in clip["top_spikes"][:8]:
            lines.append(
                f"- `{row['word']}` at `{row['second']}s`, delta `{row['delta']:.4f}`, energy `{row['energy']:.4f}`: {row['context'] or row['sentence']}"
            )
        lines += ["", "Dull-aligned words:", ""]
        for row in clip["top_dulls"][:8]:
            lines.append(
                f"- `{row['word']}` at `{row['second']}s`, delta `{row['delta']:.4f}`, energy `{row['energy']:.4f}`: {row['context'] or row['sentence']}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Align transcript words to TRIBE v2 response spikes/drops.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_zip(args.zip_path)
    (args.out_dir / "word_alignment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.out_dir / "WORD_ALIGNMENT.md")
    print(json.dumps({"clips": len(result["clips"]), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
