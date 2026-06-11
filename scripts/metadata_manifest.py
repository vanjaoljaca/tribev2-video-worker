#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def clean(value):
    return str(value if value is not None else "").replace("\t", " ").replace("\n", " ").strip()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: metadata_manifest.py <metadata.ndjson> <manifest.tsv>")

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            [
                item.get("playlist_index"),
                item.get("id"),
                item.get("upload_date"),
                item.get("duration"),
                item.get("view_count"),
                item.get("like_count"),
                item.get("comment_count"),
                item.get("webpage_url"),
                item.get("title"),
            ]
        )

    header = [
        "playlist_index",
        "id",
        "upload_date",
        "duration_seconds",
        "views",
        "likes",
        "comments",
        "url",
        "title",
    ]
    with dst.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(clean(value) for value in row) + "\n")


if __name__ == "__main__":
    main()

