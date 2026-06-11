#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    stimuli_dir = root / "stimuli"
    results_dir = root / "results"
    command_template = os.environ.get("TRIBE_V2_COMMAND")

    if not command_template:
        raise SystemExit(
            "TRIBE_V2_COMMAND is required, e.g. "
            "'python /path/to/tribe_v2_infer.py --input {input} --output {output}'"
        )
    if "{input}" not in command_template or "{output}" not in command_template:
        raise SystemExit("TRIBE_V2_COMMAND must include {input} and {output} placeholders")

    videos = sorted(stimuli_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No normalized MP4 stimuli found in {stimuli_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_dir = results_dir / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        command = command_template.format(input=shlex.quote(str(video)), output=shlex.quote(str(out_dir)))
        print(f"[tribe-v2] {video.name} -> {out_dir}")
        subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

