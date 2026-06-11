#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run one local TRIBE v2 prediction.")
    parser.add_argument("--input", required=True, help="Path to normalized video/audio/text stimulus.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--cache", default="./cache", help="Model cache directory.")
    args = parser.parse_args()

    # Import lazily so this script has a clean failure mode on machines where
    # TRIBE v2 has not been installed yet.
    from tribev2 import TribeModel

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache)
    kwargs = {}
    if input_path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        kwargs["video_path"] = str(input_path)
    elif input_path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}:
        kwargs["audio_path"] = str(input_path)
    elif input_path.suffix.lower() in {".txt", ".md"}:
        kwargs["text_path"] = str(input_path)
    else:
        raise SystemExit(f"Unsupported input type: {input_path.suffix}")

    events = model.get_events_dataframe(**kwargs)
    preds, segments = model.predict(events=events)

    np.save(output_dir / "predictions.npy", preds)
    events.to_json(output_dir / "events.json", orient="records", indent=2)
    summary = {
        "input": str(input_path),
        "predictions_shape": list(preds.shape),
        "segments": str(segments),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

