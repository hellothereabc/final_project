"""
Turn a folder of phone photos into a scored test set.

Everything measured so far comes from two greenhouse datasets shot for research.
The app runs on a phone pointed at whatever is in front of it, and nothing in
sections 1-3 says what happens there. This is the smallest honest version of
that experiment: photograph real fruit, count it by hand, and run the shipped
model over it.

Boxes are not required - the thesis metric is counting-MAE, which needs counts
per stage per image and nothing else. That also keeps the labelling job small
enough to actually finish, and it puts the human judgement where the interesting
disagreement is: cross_dataset showed two annotation teams place the
half/fully boundary in different places, so the person counting is a third
annotation source and should record what they meant (`notes`).

What this script does:
  - copies photos into field_test/data/images, converting HEIC via sips
  - bakes EXIF rotation into the pixels, the same trap that cost Laboro
    counting-MAE 0.93 -> 0.82 in section 1
  - writes labels.csv with one row per image for the counts to be typed into

Usage:
    python field_test/prepare_field.py ~/Desktop/tomato_photos
    # fill in field_test/labels.csv, then:
    python field_test/eval_field.py
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).parent
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "labels.csv"

COLUMNS = ["image", "green", "half_ripened", "fully_ripened",
           "scene", "light", "distance", "occlusion", "notes"]
SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def convert(src, dst):
    """HEIC needs a detour through sips; everything else PIL opens directly."""
    if src.suffix.lower() in {".heic", ".heif"}:
        tmp = dst.with_suffix(".sips.jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(src), "--out", str(tmp)],
                       check=True, capture_output=True)
        src = tmp
    else:
        tmp = None
    with Image.open(src) as im:
        ImageOps.exif_transpose(im).convert("RGB").save(dst, quality=95)
    if tmp is not None:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="folder of photos straight off the phone")
    ap.add_argument("--prefix", default="field", help="name prefix for copies")
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    assert source.is_dir(), f"not a folder: {source}"
    photos = sorted(p for p in source.iterdir() if p.suffix.lower() in SUFFIXES)
    assert photos, f"no photos in {source}"

    IMAGES.mkdir(parents=True, exist_ok=True)
    existing = {row["image"] for row in csv.DictReader(LABELS.open())} \
        if LABELS.exists() else set()

    added = []
    for index, photo in enumerate(photos, start=1):
        name = f"{args.prefix}_{index:03d}.jpg"
        if name in existing:
            continue
        convert(photo, IMAGES / name)
        added.append(name)
        print(f"  {photo.name} -> {name}")

    write_header = not LABELS.exists()
    with LABELS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for name in added:
            writer.writerow({"image": name})

    print(f"\n{len(added)} new images in {IMAGES}")
    print(f"fill counts in {LABELS} - one row per image, blank counts are skipped")
    print("scene/light/distance/occlusion are free text and are only used to "
          "break the results down; leave them empty if you did not vary them")


if __name__ == "__main__":
    sys.exit(main())
