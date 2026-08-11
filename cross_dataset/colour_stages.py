"""
Do the two datasets mean the same thing by "half ripened"?

A mAP table can't separate a domain gap (different greenhouse, camera, lighting)
from a schema gap (annotators disagree about where green ends and half begins).
This measures the second one directly, without a model: every box reduced to its
median a* in CIELAB, then where does each dataset put its boundaries on that
axis, and how many fruit change label if one dataset's cuts are applied to the
other's fruit.

a* also moves with white balance, so every box is reported twice: raw, and
relative to the median a* of its own image (mostly foliage, so it works as a
per-image green reference). The relative one is the one to trust.

Usage: python cross_dataset/colour_stages.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import SHORT, STAGES  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "cross"
OUT = ROOT / "runs" / "colour_stages"

DATASETS = ["laboro3", "tomatod"]
SPLITS = ["train", "val"]
CENTRE = 0.5   # box edges carry leaves and stems, keep the middle


def srgb_to_lab_a(rgb):
    """a* (green-red axis) for sRGB pixels, shape (..., 3), uint8. D65.

    Written out rather than pulled from skimage - this goes in a thesis and
    "the library did it" is not a method section."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    # sRGB -> XYZ (D65); a* only needs the X and Y rows
    x = c @ np.array([0.4124564, 0.3575761, 0.1804375])
    y = c @ np.array([0.2126729, 0.7151522, 0.0721750])

    xn, yn = 0.95047, 1.0
    fx, fy = (_f(x / xn), _f(y / yn))
    return 500.0 * (fx - fy)


def _f(t):
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)


def collect(dataset):
    """One record per box: stage, a*, and its image's background a*."""
    root = DATA / dataset
    records = []

    for split in SPLITS:
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        for path in images:
            label = lbl_dir / (path.stem + ".txt")
            if not label.exists():
                continue
            rows = [ln.split() for ln in label.read_text().splitlines() if ln.strip()]
            if not rows:
                continue

            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
            h, w = arr.shape[:2]

            # median over the frame is mostly foliage, absorbs white balance
            step = max(1, min(h, w) // 200)   # subsampled, the median is stable
            background = float(np.median(srgb_to_lab_a(arr[::step, ::step])))

            for row in rows:
                stage = STAGES[int(row[0])]
                cx, cy, bw, bh = (float(v) for v in row[1:5])
                half_w, half_h = bw * w * CENTRE / 2, bh * h * CENTRE / 2
                x0 = int(max(0, cx * w - half_w))
                x1 = int(min(w, cx * w + half_w))
                y0 = int(max(0, cy * h - half_h))
                y1 = int(min(h, cy * h + half_h))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                a = float(np.median(srgb_to_lab_a(arr[y0:y1, x0:x1])))
                records.append({"stage": stage, "a": a, "a_rel": a - background})

    return records


def thresholds(records, key):
    """Where this dataset puts green|half and half|fully on the colour axis.

    1D grid search on balanced accuracy. Balanced because both datasets are
    heavily green and plain accuracy would just ignore the two ripe stages."""
    values = np.array([r[key] for r in records])
    truth = np.array([STAGES.index(r["stage"]) for r in records])
    grid = np.percentile(values, np.linspace(1, 99, 197))

    best, best_score = (None, None), -1.0
    for t1 in grid:
        for t2 in grid[grid > t1]:
            pred = np.digitize(values, [t1, t2])
            score = np.mean([
                (pred[truth == k] == k).mean()
                for k in range(3) if (truth == k).any()
            ])
            if score > best_score:
                best, best_score = (float(t1), float(t2)), float(score)
    return {"green_half": best[0], "half_fully": best[1], "balanced_accuracy": best_score}


def relabel(records, cuts, key):
    """One dataset's cuts applied to another's fruit."""
    values = np.array([r[key] for r in records])
    truth = np.array([STAGES.index(r["stage"]) for r in records])
    pred = np.digitize(values, [cuts["green_half"], cuts["half_fully"]])

    confusion = [[int(((truth == i) & (pred == j)).sum()) for j in range(3)]
                 for i in range(3)]
    return {
        "confusion": confusion,
        "changed": float((pred != truth).mean()),
        "per_stage_changed": {
            STAGES[i]: (float((pred[truth == i] != i).mean())
                        if (truth == i).any() else None)
            for i in range(3)
        },
    }


def summarise(name, records, key):
    print(f"\n{name}  ({key})")
    print(f"  {'stage':<8} {'n':>6} {'median':>8} {'p25':>8} {'p75':>8}")
    for stage in STAGES:
        vals = np.array([r[key] for r in records if r["stage"] == stage])
        if not len(vals):
            continue
        print(f"  {SHORT[stage]:<8} {len(vals):>6} {np.median(vals):>8.2f} "
              f"{np.percentile(vals, 25):>8.2f} {np.percentile(vals, 75):>8.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in DATASETS:
        print(f"reading {name} ...", flush=True)
        data[name] = collect(name)
        print(f"  {len(data[name])} boxes")

    report = {"n_boxes": {k: len(v) for k, v in data.items()}}

    for key in ("a", "a_rel"):
        label = "raw a*" if key == "a" else "a* relative to image background"
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        for name in DATASETS:
            summarise(name, data[name], key)

        cuts = {name: thresholds(data[name], key) for name in DATASETS}
        print(f"\n  boundaries each dataset implies ({label}):")
        for name in DATASETS:
            c = cuts[name]
            print(f"    {name:<9} green|half {c['green_half']:>7.2f}   "
                  f"half|fully {c['half_fully']:>7.2f}   "
                  f"(balanced acc {c['balanced_accuracy']:.3f})")

        cross = {}
        for src in DATASETS:
            for dst in DATASETS:
                if src == dst:
                    continue
                res = relabel(data[dst], cuts[src], key)
                cross[f"{src}->{dst}"] = res
                per = "  ".join(
                    f"{SHORT[s]} {v:.0%}" for s, v in res["per_stage_changed"].items()
                    if v is not None)
                print(f"\n  {dst}'s fruit judged by {src}'s boundaries: "
                      f"{res['changed']:.1%} change label")
                print(f"    by stage: {per}")

        report[key] = {"thresholds": cuts, "cross": cross,
                       "stats": {name: {stage: _stats(data[name], stage, key)
                                        for stage in STAGES} for name in DATASETS}}

    path = OUT / "colour_stages.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    plot(data)


def _stats(records, stage, key):
    vals = [r[key] for r in records if r["stage"] == stage]
    if not vals:
        return None
    return {"n": len(vals), "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)), "p75": float(np.percentile(vals, 75))}


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    colours = {"green": "#2a9d8f", "half_ripened": "#f4a261", "fully_ripened": "#e63946"}

    for col, key in enumerate(("a", "a_rel")):
        for row, name in enumerate(DATASETS):
            ax = axes[row][col]
            for stage in STAGES:
                vals = [r[key] for r in data[name] if r["stage"] == stage]
                if vals:
                    ax.hist(vals, bins=60, alpha=0.6, label=SHORT[stage],
                            color=colours[stage], density=True)
            ax.set_title(f"{name} - {'raw a*' if key == 'a' else 'a* vs image background'}",
                         fontsize=10)
            ax.set_ylabel("density")
            if row == 1:
                ax.set_xlabel("a*  (green <-- --> red)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Where each dataset puts its ripeness boundaries", fontsize=12)
    fig.tight_layout()
    path = OUT / "colour_stages.png"
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
