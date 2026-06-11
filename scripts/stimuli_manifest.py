#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: stimuli_manifest.py <stimuli_dir> <manifest.tsv>")

    stimuli_dir = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    videos = sorted(stimuli_dir.glob("*.mp4"))
    with dst.open("w", encoding="utf-8") as f:
        f.write("stimulus_id\tvideo_path\taudio_path\tduration_seconds\n")
        for video in videos:
            audio = video.with_suffix(".wav")
            f.write(
                "\t".join(
                    [
                        video.stem,
                        str(video),
                        str(audio) if audio.exists() else "",
                        probe_duration(video),
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

