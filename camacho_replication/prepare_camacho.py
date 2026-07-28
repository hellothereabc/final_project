"""
Prepare the Camacho & Morocho-Cayamcela (TICEC 2023) dataset for replication.

Their repo ships `Big_tomatoes-dataset-2` in COCO-segmentation format only; the
notebook trained on a YOLO-format export (`Big_tomatoes_dataset-2-YOLO`) that was
never committed. This converts COCO -> YOLO segmentation labels, preserving their
splits and class order exactly as in their `data.yaml`.

Notes on their data (from README.roboflow.txt):
  - Roboflow already applied auto-orientation with EXIF stripping, so the EXIF
    defect fixed in ../prepare_data.py does not apply here.
  - Images are already resized to 640x640 (stretch, not letterbox).

Usage: python prepare_camacho.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "repo" / "Tomato-Dataset" / "Big_tomatoes-dataset-2"
OUT = ROOT / "data" / "camacho_yolo"

# their data.yaml order; COCO id 0 is the 'Tomatos' supercategory and carries no
# annotations, so the mapping is a straight id-1.
NAMES = ["b_fully_ripened", "b_green", "b_half_ripened"]
SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def convert(split):
    ann_path = RAW / "annotations" / f"{split}.json"
    data = json.loads(ann_path.read_text())

    cat_idx = {c["id"]: NAMES.index(c["name"])
               for c in data["categories"] if c["name"] in NAMES}

    images = {im["id"]: im for im in data["images"]}
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    img_out = OUT / "images" / split
    lbl_out = OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_obj = n_skip = 0
    for img_id, im in images.items():
        w, h = im["width"], im["height"]
        fname = im["file_name"]
        src = RAW / split / fname
        dst = img_out / fname
        if not dst.exists():
            dst.symlink_to(src.resolve())

        lines = []
        for a in by_img.get(img_id, []):
            seg = a["segmentation"]
            if not seg or not isinstance(seg, list):
                n_skip += 1
                continue
            poly = seg[0]
            if len(poly) < 6:  # need >=3 points
                n_skip += 1
                continue
            coords = []
            for i in range(0, len(poly) - 1, 2):
                coords.append(f"{poly[i] / w:.6f}")
                coords.append(f"{poly[i + 1] / h:.6f}")
            lines.append(f"{cat_idx[a['category_id']]} " + " ".join(coords))
            n_obj += 1

        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

    print(f"{split:>5}: {len(images)} images, {n_obj} objects"
          + (f", {n_skip} skipped" if n_skip else ""))


def main():
    assert RAW.exists(), f"missing {RAW} — clone their repo into camacho/repo first"
    for split in SPLITS:
        convert(split)

    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    print("Wrote", yaml)


if __name__ == "__main__":
    main()
