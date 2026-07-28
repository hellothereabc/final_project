"""
Prepare Laboro Tomato (COCO instance-seg) -> YOLO detection format.

Reads the extracted laboro_tomato/ folder, converts COCO bbox annotations to
YOLO txt labels, and lays out the images/labels tree ultralytics expects.
Also writes data.yaml. Keeps the native 6 classes (size x ripeness).

Usage: python prepare_data.py
"""
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "laboro_tomato"          # extracted dataset root
OUT = ROOT / "data" / "laboro_yolo"            # YOLO-format output
SPLITS = {"train": "train", "val": "test"}     # yolo split -> coco split name


def find_coco(split_name):
    """Locate the COCO json + image dir for a given original split."""
    ann_candidates = [
        RAW / "annotations" / f"{split_name}.json",
        RAW / f"{split_name}.json",
    ]
    ann = next((p for p in ann_candidates if p.exists()), None)
    img_candidates = [RAW / split_name, RAW / "images" / split_name]
    img_dir = next((p for p in img_candidates if p.exists()), None)
    return ann, img_dir


def main():
    assert RAW.exists(), f"missing {RAW} — extract the zip first"

    # discover category ordering from train json (consistent across splits)
    train_ann, _ = find_coco(SPLITS["train"])
    with open(train_ann) as f:
        tj = json.load(f)
    cats = sorted(tj["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    print("Classes:", names)

    for yolo_split, coco_split in SPLITS.items():
        ann_path, img_dir = find_coco(coco_split)
        assert ann_path and img_dir, f"cannot locate split {coco_split}"
        with open(ann_path) as f:
            data = json.load(f)

        images = {im["id"]: im for im in data["images"]}
        anns_by_img = {}
        for a in data["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        img_out = OUT / "images" / yolo_split
        lbl_out = OUT / "labels" / yolo_split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_obj = 0
        for img_id, im in images.items():
            w, h = im["width"], im["height"]
            fname = os.path.basename(im["file_name"])
            src = img_dir / fname
            if not src.exists():  # some file_names include subdirs
                src = img_dir / im["file_name"]
            # Bake EXIF orientation into pixels so raw pixels match the
            # annotation frame. ~40% of images (one camera) carry orientation
            # tag 6; COCO w/h are the EXIF-corrected dims, so labels normalised
            # by them only align once the rotation is applied to the pixels.
            # No-op for images without EXIF rotation.
            dst = img_out / fname
            if not dst.exists():
                img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                assert img.size == (w, h), f"{fname}: {img.size} != COCO {(w, h)}"
                img.save(dst, quality=95)

            lines = []
            for a in anns_by_img.get(img_id, []):
                x, y, bw, bh = a["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                idx = cat_id_to_idx[a["category_id"]]
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_obj += 1
            (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

        print(f"{yolo_split}: {len(images)} images, {n_obj} objects")

    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    print("Wrote", yaml)


if __name__ == "__main__":
    main()
