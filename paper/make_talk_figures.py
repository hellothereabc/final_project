"""The two figures for showing the work: one per question, nothing else on them.

Both reuse the loaders of make_size_lighting_figures.py, so they read the same
saved measurements and cannot disagree with the detailed figures:

  1. model_ladder  - mAP50-95 against iPhone Neural Engine latency for n..x,
     marker area the Core ML package, against the 30 FPS frame budget.
  2. lighting      - the same validation frame at three exposures and two white
     balances, over one recall curve: stock HSV jitter against no colour jitter.

Everything that did not fit on these two lives in the detailed figures written
by make_size_lighting_figures.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tomato-paper-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lighting"))

from make_size_lighting_figures import (  # noqa: E402
    FRAME_BUDGET_MS,
    OUT,
    SIZES,
    lighting,
    mac_rows,
    map_scores,
    phone_rows,
)

PHONE_COL = "#e6550d"
STOCK_COL = "#e6550d"
PLAIN_COL = "#252525"


def fig_model_ladder() -> None:
    maps, mac, phone = map_scores(), mac_rows(), phone_rows()
    latency = [phone[s]["ne"]["median_ms"] for s in SIZES]
    score = [maps[s] for s in SIZES]
    package = [mac[s]["ne"]["package_mb"] for s in SIZES]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.axvspan(FRAME_BUDGET_MS, 40, color="#f0f0f0", zorder=0)
    ax.axvline(FRAME_BUDGET_MS, color="#525252", linestyle=":", linewidth=1.6, zorder=1)
    ax.text(FRAME_BUDGET_MS + 3.3, 0.746, "slower than 30 FPS", fontsize=11.5,
            color="#737373", rotation=90, ha="center", va="center")

    ax.plot(latency, score, color=PHONE_COL, alpha=0.35, linewidth=1.6, zorder=2)
    ax.scatter(latency, score, s=[30 + 9.5 * mb for mb in package], color=PHONE_COL,
               alpha=0.8, zorder=3)

    offsets = {"n": (14, -6), "s": (14, -6), "m": (-6, 20), "l": (16, -16), "x": (4, -30)}
    for size, x, y, mb in zip(SIZES, latency, score, package):
        ax.annotate(f"YOLO11{size}\n{mb:.0f} MB · {x:.1f} ms", (x, y), fontsize=11.5,
                    textcoords="offset points", xytext=offsets[size])

    ax.set_xlabel("Inference latency on the iPhone Neural Engine (ms)", fontsize=12.5)
    ax.set_ylabel("Accuracy (mAP50-95)", fontsize=12.5)
    ax.set_xlim(0, 40)
    ax.set_ylim(0.697, 0.801)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_title("Bigger models: accuracy stops at m, the phone bill keeps growing",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig(OUT / "talk_model_ladder.pdf", bbox_inches="tight")
    fig.savefig(OUT / "talk_model_ladder.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_lighting() -> None:
    import cv2  # local: only this figure needs OpenCV
    from PIL import Image
    from relight import relight

    rows = lighting("n")["axes"]
    recall = {("stops", r["value"]): r["by_conf"]["0.25"]["recall"] for r in rows["exposure"]}
    recall.update({("wb", r["value"]): r["by_conf"]["0.25"]["recall"] for r in rows["white_balance"]})
    plain = lighting("light_no_colour")["axes"]["exposure"]

    source = cv2.imread(str(ROOT / "data" / "laboro_yolo" / "images" / "val" / "IMG_0991.jpg"))
    crop = (661, 283, 3024, 2268)
    cells = [
        ("stops", -3.0, "−3 stops (dusk)"),
        ("stops", 0.0, "original frame"),
        ("stops", 3.0, "+3 stops (blown out)"),
        ("wb", -1.5, "cool white balance"),
        ("wb", 1.5, "warm white balance"),
    ]

    fig = plt.figure(figsize=(11.0, 7.2))
    grid = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 1.6], hspace=0.16, wspace=0.04)

    for column, (knob, value, label) in enumerate(cells):
        ax = fig.add_subplot(grid[0, column])
        frame = relight(source, **{knob: value})
        patch = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(crop)
        patch.thumbnail((720, 720))
        ax.imshow(patch)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#bdbdbd")
        ax.set_title(label, fontsize=11.5, pad=5)
        if knob == "wb":  # no curve below covers these two, so name the cost here
            ax.set_xlabel(f"recall {recall[(knob, value)]:.2f}", fontsize=11)

    ax = fig.add_subplot(grid[1, :])
    exposure = [r["value"] for r in rows["exposure"]]
    ax.plot(exposure, [r["by_conf"]["0.25"]["recall"] for r in rows["exposure"]],
            marker="o", markersize=6, linewidth=2.4, color=STOCK_COL,
            label="trained with the default HSV jitter")
    ax.plot([r["value"] for r in plain], [r["by_conf"]["0.25"]["recall"] for r in plain],
            marker="s", markersize=6, linewidth=2.4, color=PLAIN_COL, linestyle="--",
            label="trained with colour jitter switched off")

    for value in (-3.0, 0.0, 3.0):  # tie the curve back to the three frames above
        ax.axvline(value, color="#bdbdbd", linewidth=1.0, linestyle=":", zorder=0)

    ax.set_xlabel("Exposure of the test images (stops); the labels never move", fontsize=12.5)
    ax.set_ylabel("Fruit found (recall)", fontsize=12.5)
    ax.set_xlim(-3.35, 3.35)
    ax.set_ylim(0.55, 0.95)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=11.5, loc="lower left")
    ax.set_title("The default HSV jitter is already doing the work; over-exposure breaks both",
                 fontsize=13)

    fig.savefig(OUT / "talk_lighting.pdf", bbox_inches="tight")
    fig.savefig(OUT / "talk_lighting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_model_ladder()
    fig_lighting()
    print(f"wrote talk_model_ladder and talk_lighting to {OUT}")


if __name__ == "__main__":
    main()
