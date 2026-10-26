"""Two figures added after the supervisor's review of the 20 October draft.

  figures/calibration_curve.pdf   counting MAE and predicted/actual ratio
                                  against the confidence threshold, per size
                                  (runs/laboro_model_size_thresholds_nsmlx.json)
  figures/pipeline.pdf            block diagram of the deployment path
  figures/colour_axis.jpg         real Laboro fruit crops placed on the
                                  background-corrected a* axis, with the stage
                                  boundaries each dataset implies (Table III)

The boundary values are the ones printed by cross_dataset/colour_stages.py and
reported in Table III; tomatOD images are not on this machine, so only Laboro
crops are shown and the tomatOD cuts are drawn from the saved values.

Usage: python paper/make_review_figures.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tomato-paper-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "cross_dataset"))
from colour_stages import CENTRE, srgb_to_lab_a  # noqa: E402

SIZES = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO11x"]
SIZE_COL = ["#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b"]

# Background-corrected cuts, Table III (cross_dataset/colour_stages.py output).
CUTS = {"Laboro": (3.2, 31.0), "tomatOD": (-0.1, 11.7)}
STAGE_OF = {"green": "green", "half_ripened": "half-ripened", "fully_ripened": "fully ripened"}
STAGE_COL = {"green": "#31a354", "half-ripened": "#fd8d3c", "fully ripened": "#de2d26"}


def calibration_curve():
    doc = json.loads((ROOT / "runs" / "laboro_model_size_thresholds_nsmlx.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for size, colour in zip(SIZES, SIZE_COL):
        rows = sorted((r for r in doc["rows"] if r["model"] == size), key=lambda r: r["conf"])
        conf = [r["conf"] for r in rows]
        axes[0].plot(conf, [r["counting_mae"] for r in rows], color=colour, label=size, lw=1.4)
        axes[1].plot(conf, [r["pred_over_gt"] for r in rows], color=colour, lw=1.4)
    for ax in axes:
        ax.axvline(0.25, color="0.5", ls=":", lw=1)
        ax.set_xlabel("confidence threshold")
        ax.grid(alpha=0.25)
    axes[1].axhline(1.0, color="0.3", lw=0.8)
    axes[0].set_ylabel("counting MAE")
    axes[1].set_ylabel("predicted / annotated fruit")
    axes[0].set_ylim(0.8, 2.0)
    axes[1].set_ylim(0.4, 1.8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_curve.pdf")
    fig.savefig(OUT / "calibration_curve.png", dpi=200)
    plt.close(fig)


def laboro_crops():
    """(a_rel, stage, crop) for every validation box large enough to show."""
    names = ["b_fully_ripened", "b_half_ripened", "b_green",
             "l_fully_ripened", "l_half_ripened", "l_green"]
    data = ROOT / "data" / "laboro_yolo"
    out = []
    for path in sorted((data / "images" / "val").iterdir()):
        label = data / "labels" / "val" / (path.stem + ".txt")
        if not label.exists():
            continue
        with Image.open(path) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = arr.shape[:2]
        step = max(1, min(h, w) // 200)
        background = float(np.median(srgb_to_lab_a(arr[::step, ::step])))
        boxes = [[float(v) for v in ln.split()[:5]]
                 for ln in label.read_text().splitlines() if ln.strip()]
        for j, (cls, cx, cy, bw, bh) in enumerate(boxes):
            if bw * w < 220 or bh * h < 220:
                continue
            # a neighbouring fruit inside the crop would make the example misleading
            if any(_overlap((cx, cy, bw, bh), b[1:]) > 0.02
                   for k, b in enumerate(boxes) if k != j):
                continue
            stage = STAGE_OF[names[int(cls)].split("_", 1)[1]]
            hw, hh = bw * w * CENTRE / 2, bh * h * CENTRE / 2
            core = arr[int(cy * h - hh):int(cy * h + hh), int(cx * w - hw):int(cx * w + hw)]
            a_rel = float(np.median(srgb_to_lab_a(core))) - background
            x0, x1 = int(max(0, (cx - bw / 2) * w)), int(min(w, (cx + bw / 2) * w))
            y0, y1 = int(max(0, (cy - bh / 2) * h)), int(min(h, (cy + bh / 2) * h))
            crop = Image.fromarray(arr[y0:y1, x0:x1]).resize((96, 96))
            out.append((a_rel, stage, crop))
    return out


def _overlap(a, b):
    """Share of box a's area covered by box b, both (cx, cy, w, h)."""
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    return ix * iy / (a[2] * a[3])


def colour_axis():
    crops = laboro_crops()
    rng = np.random.default_rng(0)
    targets = np.linspace(-12, 50, 14)
    chosen, used = [], set()
    for t in targets:
        order = np.argsort([abs(c[0] - t) for c in crops])
        for i in order[:6]:
            if i not in used:
                used.add(i)
                chosen.append(crops[i])
                break
    rng.shuffle(chosen)
    chosen.sort(key=lambda c: c[0])

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(-16, 55)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(r"background-corrected median $a^*$ (green $-$, red $+$)")

    for k, (a, stage, crop) in enumerate(chosen):
        y = 1.0 if k % 2 == 0 else 2.15
        box = AnnotationBbox(OffsetImage(np.asarray(crop), zoom=0.42), (a, y), frameon=True,
                             bboxprops=dict(edgecolor=STAGE_COL[stage], lw=2.2))
        ax.add_artist(box)
        ax.plot([a, a], [0, y - 0.45], color=STAGE_COL[stage], lw=0.8)

    for (name, (g, f)), style, ytext in zip(CUTS.items(), ("-", "--"), (3.55, 3.15)):
        for cut, text in ((g, "green|half"), (f, "half|fully")):
            ax.axvline(cut, color="0.15", ls=style, lw=1.1)
            ax.text(cut + 0.5, ytext, f"{name}\n{text}", fontsize=6, va="top")
    handles = [plt.Line2D([], [], color=c, lw=3, label=f"Laboro label: {s}")
               for s, c in STAGE_COL.items()]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "colour_axis.jpg", dpi=220, pil_kwargs={"quality": 88})
    plt.close(fig)


def pipeline():
    """Block diagram of the deployment path (Section IV, implementation)."""
    from matplotlib.patches import FancyBboxPatch
    steps = [
        ("Training\n(laptop)", "PyTorch FP32\nYOLO11 checkpoint", "#deebf7"),
        ("Export\n(once)", "Core ML package\nFP16 + NMS", "#deebf7"),
        ("Build\n(Xcode)", "compiled model\nin the app bundle", "#deebf7"),
        ("Input\n(phone)", "photo, or camera\nframe 1280x720", "#fee6ce"),
        ("Vision", "orientation, resize\nto 640x640", "#fee6ce"),
        ("Core ML", "CPU or CPU +\nNeural Engine", "#fee6ce"),
        ("Application", "boxes above the\nthreshold, counted\nper stage", "#fee6ce"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 1.35))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (head, body, colour) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.12), 0.84, 0.76,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=colour, ec="0.3", lw=0.8))
        ax.text(i + 0.48, 0.70, head, ha="center", va="center", fontsize=6.6, weight="bold")
        ax.text(i + 0.48, 0.33, body, ha="center", va="center", fontsize=5.8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.9, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.2"))
    ax.text(1.5, 0.98, "development machine", ha="center", va="top", fontsize=6, color="#3182bd")
    ax.text(5.0, 0.98, "on the device, no network", ha="center", va="top", fontsize=6, color="#e6550d")
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "pipeline.pdf")
    fig.savefig(OUT / "pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    calibration_curve()
    colour_axis()
    pipeline()
    print("wrote", OUT / "calibration_curve.pdf", OUT / "colour_axis.jpg")
