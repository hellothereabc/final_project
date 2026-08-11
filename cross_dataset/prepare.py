"""
Convert Laboro Tomato and tomatOD to one YOLO detection layout, 3 shared stages.

Both ship COCO and both have a silent defect:

  Laboro   ~40% of images carry EXIF orientation 6 - pixels landscape, COCO boxes
           upright, YOLO loader ignores EXIF. Same fix as ../prepare_data.py.
  tomatOD  categories in tomatOD_test.json is one record with a list for a name,
           while its annotations use ids 1/2/3. Reading categories per split puts
           the whole test set in one class without erroring, so the id->name map
           comes from train.

Writes data/cross/{laboro3,tomatod}/ plus mix.yaml, which points at both train
dirs rather than copying pixels.

Usage: python cross_dataset/prepare.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import STAGES, describe, stage_index_of  # noqa: E402

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "cross"

# As published by each dataset - a converter bug shows up as a disagreement with
# their own README. Not the counts read back out of the json, which would only
# check the converter against itself.
EXPECTED = {
    "laboro3": {"train": (643, 7781), "val": (161, 1996)},
    "tomatod": {"train": (222, 1952), "val": (55, 466)},
}

# Theirs, not ours - counted from the raw json. Listed so a real bug can't hide
# behind an "expected" mismatch.
KNOWN_DEVIATIONS = {
    # tomatOD's README advertises 2418 boxes (unripe 1592 / semi 395 / fully 431)
    # but the shipped annotations hold 2421: +1 unripe in train, +2 semi in test.
    ("tomatod", "train"): (0, +1),
    ("tomatod", "val"): (0, +2),
}

EXIF_ORIENTATION = 274


def find(*candidates):
    return next((p for p in candidates if p.exists()), None)


def laboro_source():
    """(split -> (coco json, image dir)), plus the id->name category map."""
    root = find(RAW / "laboro_tomato" / "laboro_tomato", RAW / "laboro_tomato")
    assert root, f"missing Laboro under {RAW / 'laboro_tomato'}"

    splits = {}
    for yolo_split, coco_split in (("train", "train"), ("val", "test")):
        ann = find(root / "annotations" / f"{coco_split}.json", root / f"{coco_split}.json")
        img_dir = find(root / coco_split, root / "images" / coco_split)
        assert ann and img_dir, f"cannot locate Laboro split {coco_split} under {root}"
        splits[yolo_split] = (ann, img_dir)

    cats = json.loads(splits["train"][0].read_text())["categories"]
    return splits, {c["id"]: c["name"] for c in cats}


def tomatod_source():
    root = find(RAW / "tomatod")
    assert root, f"missing tomatOD under {RAW}"
    ann_dir = find(root / "tomatOD_annotations", root)
    img_root = find(root / "tomatOD_images", root)
    assert ann_dir and img_root, f"cannot locate tomatOD payload under {root}"

    splits = {}
    for yolo_split, coco_split in (("train", "train"), ("val", "test")):
        ann = ann_dir / f"tomatOD_{coco_split}.json"
        img_dir = find(img_root / coco_split, img_root)
        assert ann.exists() and img_dir, f"cannot locate tomatOD split {coco_split}"
        splits[yolo_split] = (ann, img_dir)

    # Deliberately from train only - test's categories block is broken (see module docstring).
    cats = json.loads(splits["train"][0].read_text())["categories"]
    return splits, {c["id"]: c["name"] for c in cats}


def place_image(src, dst, coco_wh):
    """Write one image, applying EXIF rotation only if there is one.
    Returns the size actually written."""
    if dst.exists() or dst.is_symlink():
        with Image.open(dst) as im:
            return im.size

    with Image.open(src) as im:
        orientation = (im.getexif() or {}).get(EXIF_ORIENTATION, 1)
        if orientation in (1, 0, None):
            # nothing to do, symlink instead of re-encoding
            dst.symlink_to(src.resolve())
            return im.size
        fixed = ImageOps.exif_transpose(im).convert("RGB")
        fixed.save(dst, quality=95)
        return fixed.size


def convert(name, splits, cat_names, out_dir):
    print(f"\n=== {name} ===")
    print("class collapse:")
    print(describe(list(cat_names.values())))
    cat_stage = {cid: stage_index_of(cname) for cid, cname in cat_names.items()}

    totals = {}
    for split, (ann_path, img_dir) in splits.items():
        data = json.loads(ann_path.read_text())
        images = {im["id"]: im for im in data["images"]}
        by_img = {}
        for a in data["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)

        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_obj = 0
        per_stage = Counter()
        for img_id, im in images.items():
            fname = Path(im["file_name"]).name
            src = img_dir / fname
            if not src.exists():
                src = img_dir / im["file_name"]
            assert src.exists(), f"{name}/{split}: no image file for {fname}"

            w, h = place_image(src, img_out / fname, (im["width"], im["height"]))
            # boxes were drawn in the json's frame; if pixels disagree after the
            # EXIF fix the labels land in the wrong place, silently
            assert (w, h) == (im["width"], im["height"]), (
                f"{name}/{split}/{fname}: pixels {w}x{h} != COCO "
                f"{im['width']}x{im['height']} - box normalisation would be wrong"
            )

            lines = []
            for a in by_img.get(img_id, []):
                if a.get("iscrowd") or a.get("is_crowd"):  # key name differs across tomatOD splits
                    continue
                x, y, bw, bh = a["bbox"]
                idx = cat_stage[a["category_id"]]
                lines.append(
                    f"{idx} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                    f"{bw / w:.6f} {bh / h:.6f}"
                )
                per_stage[STAGES[idx]] += 1
                n_obj += 1
            (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

        totals[split] = (len(images), n_obj)
        stages = "  ".join(f"{s}={per_stage[s]}" for s in STAGES)
        print(f"{split:>5}: {len(images):>4} images, {n_obj:>5} objects   {stages}")

    write_yaml(out_dir / "data.yaml", {"train": "images/train", "val": "images/val"}, out_dir)
    return totals


def write_yaml(path, mapping, root):
    lines = [f"path: {root.resolve()}"]
    for key, value in mapping.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines += [f"  - {v}" for v in value]
        else:
            lines.append(f"{key}: {value}")
    lines += [f"nc: {len(STAGES)}", f"names: {STAGES}"]
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def check(name, totals):
    """Compare against the counts each dataset publishes about itself."""
    ok = True
    for split, got in totals.items():
        published = EXPECTED[name][split]
        drift = KNOWN_DEVIATIONS.get((name, split), (0, 0))
        allowed = (published[0] + drift[0], published[1] + drift[1])

        if got == published:
            mark = "ok"
        elif got == allowed:
            mark = f"ok (known deviation {drift[1]:+d} boxes vs README)"
        else:
            mark = f"MISMATCH - expected {allowed}"
            ok = False
        print(f"  {name}/{split}: images={got[0]} boxes={got[1]}  [{mark}]")
    return ok


def main():
    results = {}
    results["laboro3"] = convert("laboro3", *laboro_source(), OUT / "laboro3")
    results["tomatod"] = convert("tomatod", *tomatod_source(), OUT / "tomatod")

    # no leakage by construction: a frame belongs to one dataset and one split
    write_yaml(
        OUT / "mix.yaml",
        {
            "train": ["laboro3/images/train", "tomatod/images/train"],
            "val": ["laboro3/images/val", "tomatod/images/val"],
        },
        OUT,
    )

    print("\n=== counts vs published ===")
    all_ok = all(check(name, totals) for name, totals in results.items())
    print("\nall counts match published figures" if all_ok
          else "\n!! counts differ - investigate before training")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
