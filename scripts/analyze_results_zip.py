#!/usr/bin/env python3
import argparse
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def load_npy(zf, name):
    with zf.open(name) as f:
        return np.load(io.BytesIO(f.read()))


def nearby_words(events, second, radius=1.5, max_words=16):
    rows = [
        row
        for row in events
        if row.get("type") == "Word"
        and row.get("text")
        and abs(float(row.get("start", 0.0)) - float(second)) <= radius
    ]
    rows.sort(key=lambda row: row.get("start", 0.0))
    return " ".join(row["text"] for row in rows[:max_words])


def nearby_sentence(events, second):
    sentences = [
        row
        for row in events
        if row.get("type") == "Sentence"
        and row.get("text")
        and float(row.get("start", 0.0)) <= float(second) <= float(row.get("stop", row.get("start", 0.0)))
    ]
    if sentences:
        return sentences[0]["text"].strip()
    rows = [
        row
        for row in events
        if row.get("type") == "Sentence" and row.get("text")
    ]
    if not rows:
        return ""
    rows.sort(key=lambda row: abs(float(row.get("start", 0.0)) - float(second)))
    return rows[0]["text"].strip()


def top_indices(values, count=5, min_spacing=3):
    order = list(np.argsort(values)[::-1])
    chosen = []
    for idx in order:
        if all(abs(int(idx) - other) >= min_spacing for other in chosen):
            chosen.append(int(idx))
        if len(chosen) >= count:
            break
    return chosen


def plot_comparison(series, out_path):
    plt.figure(figsize=(11, 5.8))
    for item in series:
        y = item["abs_mean"]
        label = item["label"]
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=1.8, label=label)
    plt.title("TRIBE v2 predicted response energy")
    plt.xlabel("Second")
    plt.ylabel("Mean absolute predicted response")
    plt.grid(alpha=0.24)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_video(item, out_path):
    y = item["abs_mean"]
    diff = np.r_[0.0, np.diff(y)]
    x = np.arange(len(y))
    peaks = item["peak_seconds"]
    jumps = item["jump_seconds"]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x, y, color="#2155d9", linewidth=1.8, label="response energy")
    ax.bar(x, np.maximum(diff, 0), color="#ff8a00", alpha=0.28, label="positive jump")
    for sec in peaks[:3]:
        ax.axvline(sec, color="#d62728", linestyle="--", alpha=0.5, linewidth=1)
    for sec in jumps[:3]:
        ax.axvline(sec, color="#2ca02c", linestyle=":", alpha=0.55, linewidth=1.2)
    ax.set_title(item["label"])
    ax.set_xlabel("Second")
    ax.set_ylabel("Mean absolute predicted response")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(items, out_path):
    if not items:
        return
    max_len = max(len(item["abs_mean"]) for item in items)
    mat = np.full((len(items), max_len), np.nan)
    for row_idx, item in enumerate(items):
        values = np.asarray(item["abs_mean"], dtype=float)
        denom = values.max() - values.min()
        norm = (values - values.min()) / denom if denom else values * 0
        mat[row_idx, : len(norm)] = norm

    plt.figure(figsize=(11, max(2.4, 1.1 * len(items))))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.yticks(range(len(items)), [item["label"] for item in items], fontsize=8)
    plt.xlabel("Second")
    plt.title("Normalized TRIBE v2 energy heatmap")
    plt.colorbar(label="Normalized energy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def analyze(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source_zip": str(zip_path), "items": []}

    with zipfile.ZipFile(zip_path) as zf:
        manifest = load_json(zf, "manifest.json")
        for item_idx, manifest_item in enumerate(manifest.get("summaries", []), start=1):
            pred_name = None
            event_name = None
            summary_name = None
            label = manifest_item.get("label") or Path(manifest_item.get("input", f"item_{item_idx}")).stem
            for name in zf.namelist():
                if name.endswith("/summary.json"):
                    candidate = load_json(zf, name)
                    if candidate.get("input") == manifest_item.get("input"):
                        base = name.rsplit("/", 1)[0]
                        pred_name = f"{base}/predictions.npy"
                        event_name = f"{base}/events.json"
                        summary_name = name
                        break
            if not pred_name:
                continue

            preds = load_npy(zf, pred_name)
            events = load_json(zf, event_name)
            run_summary = load_json(zf, summary_name)
            abs_mean = np.mean(np.abs(preds), axis=1)
            signed_mean = np.mean(preds, axis=1)
            std = np.std(preds, axis=1)
            pos_jump = np.r_[0.0, np.diff(abs_mean)]
            peak_seconds = top_indices(abs_mean, count=5, min_spacing=3)
            jump_seconds = top_indices(pos_jump, count=5, min_spacing=3)

            video_summary = {
                "label": label,
                "input": manifest_item.get("input"),
                "source_url": manifest_item.get("source_url"),
                "shape": list(preds.shape),
                "modalities": run_summary.get("modalities") or manifest_item.get("modalities"),
                "timings": run_summary.get("timings") or manifest_item.get("timings"),
                "duration_seconds": int(preds.shape[0]),
                "mean_abs_energy": float(abs_mean.mean()),
                "max_abs_energy": float(abs_mean.max()),
                "energy_std": float(abs_mean.std()),
                "mean_signed": float(signed_mean.mean()),
                "peak_seconds": peak_seconds,
                "jump_seconds": jump_seconds,
                "peaks": [
                    {
                        "second": sec,
                        "energy": float(abs_mean[sec]),
                        "nearby_words": nearby_words(events, sec),
                        "sentence": nearby_sentence(events, sec),
                    }
                    for sec in peak_seconds
                ],
                "jumps": [
                    {
                        "second": sec,
                        "jump": float(pos_jump[sec]),
                        "energy": float(abs_mean[sec]),
                        "nearby_words": nearby_words(events, sec),
                        "sentence": nearby_sentence(events, sec),
                    }
                    for sec in jump_seconds
                ],
                "abs_mean": [float(x) for x in abs_mean],
            }
            plot_video(video_summary, visuals_dir / f"video_{item_idx}_energy.png")
            summary["items"].append(video_summary)

    plot_comparison(summary["items"], visuals_dir / "comparison_energy.png")
    plot_heatmap(summary["items"], visuals_dir / "energy_heatmap.png")
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_markdown(summary, out_dir):
    lines = [
        "# TRIBE v2 Video Run",
        "",
        "This report summarizes remote TRIBE v2 predictions for short-form videos.",
        "",
        "![Comparison energy](visuals/comparison_energy.png)",
        "",
        "![Energy heatmap](visuals/energy_heatmap.png)",
        "",
        "## Videos",
        "",
    ]
    for idx, item in enumerate(summary["items"], start=1):
        lines.extend(
            [
                f"### {idx}. {item['label']}",
                "",
                f"- Source: {item.get('source_url') or ''}",
                f"- Shape: `{item['shape']}`",
                f"- Mean energy: `{item['mean_abs_energy']:.4f}`",
                f"- Max energy: `{item['max_abs_energy']:.4f}`",
                f"- Energy variability: `{item['energy_std']:.4f}`",
                "",
                f"![{item['label']} energy](visuals/video_{idx}_energy.png)",
                "",
                "Top peaks:",
                "",
            ]
        )
        for peak in item["peaks"][:5]:
            text = peak["nearby_words"] or peak["sentence"]
            lines.append(f"- `{peak['second']}s` energy `{peak['energy']:.4f}`: {text}")
        lines.extend(["", "Largest upward jumps:", ""])
        for jump in item["jumps"][:5]:
            text = jump["nearby_words"] or jump["sentence"]
            lines.append(
                f"- `{jump['second']}s` jump `{jump['jump']:.4f}`, energy `{jump['energy']:.4f}`: {text}"
            )
        lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze a TRIBE v2 results zip.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    summary = analyze(args.zip_path, args.out_dir)
    write_markdown(summary, args.out_dir)
    print(json.dumps({"items": len(summary["items"]), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
