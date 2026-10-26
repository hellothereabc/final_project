"""Figures for the model-size ladder (section 5) and the lighting sweep.

Every number is read from a saved result file, so a figure cannot drift away
from the measurement:

  runs/laboro_yolo11*_exiffix_*_mps/results.csv   mAP50-95 per size
  runs/laboro_model_size_thresholds_nsmlx.json    counting MAE, swept threshold
  runs/runtime/mac_model_sizes.json               Mac Core ML timings
  runtime_bench/phone_iPhone18-1_ios27_*.json     iPhone timings (iOS 27.0)
  runs/lighting/<model>.json                      exposure / white-balance sweep

The one exception is YOLO11n's mAP50-95: its run lives in the reproduce repo
(`laboro_yolo11n_exiffix`), only the exported weights are here, so the value
recorded in README section 5 and in runtime_bench/model_size_results.md is
used and flagged below.

The example strip is an illustration of relight.py's transformation on one
validation frame, not an additional evaluation result.
"""
from __future__ import annotations

import csv
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "lighting"))

SIZES = ["n", "s", "m", "l", "x"]
RUN_DIRS = {
    "s": "laboro_yolo11s_exiffix_b16_mps",
    "m": "laboro_yolo11m_exiffix_b8_mps",
    "l": "laboro_yolo11l_exiffix_b8_mps",
    "x": "laboro_yolo11x_exiffix_b4_mps",
}
N_MAP50_95 = 0.706  # reproduce repo run, see module docstring
FRAME_BUDGET_MS = 1000.0 / 30.0

COL = {
    "mac_cpu": "#9ecae1",
    "mac_ne": "#3182bd",
    "phone_cpu": "#fdae6b",
    "phone_ne": "#e6550d",
}


# ---------------------------------------------------------------- loading


def map_scores() -> dict[str, float]:
    scores = {"n": N_MAP50_95}
    for size, run in RUN_DIRS.items():
        rows = list(csv.DictReader(open(ROOT / "runs" / run / "results.csv")))
        key = next(k for k in rows[0] if "mAP50-95" in k)
        scores[size] = max(float(r[key]) for r in rows)
    return scores


def counting_scores() -> dict[str, dict]:
    data = json.load(open(ROOT / "runs" / "laboro_model_size_thresholds_nsmlx.json"))
    out: dict[str, dict] = {}
    for size in SIZES:
        rows = [r for r in data["rows"] if r["model"] == f"YOLO11{size}"]
        best_all = min(rows, key=lambda r: r["counting_mae"])
        best_ripe = min(rows, key=lambda r: r["ripe_mae"])
        out[size] = {
            "mae": best_all["counting_mae"],
            "mae_conf": best_all["conf"],
            "ripe_mae": best_ripe["ripe_mae"],
            "ripe_conf": best_ripe["conf"],
        }
    return out


def mac_rows() -> dict[str, dict]:
    data = json.load(open(ROOT / "runs" / "runtime" / "mac_model_sizes.json"))
    out: dict[str, dict] = {}
    for row in data["rows"]:
        size = row["model"].replace("YOLO11", "")
        unit = "cpu" if row["units"] == "CPU_ONLY" else "ne"
        out.setdefault(size, {})[unit] = row
    return out


def phone_rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for size in SIZES:
        short = json.load(
            open(ROOT / "runtime_bench" / f"phone_iPhone18-1_ios27_yolo11{size}_laboro.json")
        )
        entry = {}
        for cfg in short["configs"]:
            entry["cpu" if cfg["computeUnits"] == "cpuOnly" else "ne"] = cfg
        sustained = ROOT / "runtime_bench" / f"phone_iPhone18-1_ios27_yolo11{size}_sustained.json"
        if size == "m":  # only measured on iOS 26.5.2, 14 September
            sustained = ROOT / "runtime_bench" / "phone_iPhone18-1_yolo11m_sustained.json"
        if sustained.exists():
            entry["sustained"] = json.load(open(sustained))["configs"][0]
        out[size] = entry
    return out


def lighting(model: str) -> dict:
    return json.load(open(ROOT / "runs" / "lighting" / f"{model}.json"))


# ---------------------------------------------------------------- figures


def fig_size_accuracy(maps, counts, mac, phone) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(len(SIZES))

    left.bar(x, [maps[s] for s in SIZES], width=0.55, color="#3182bd", label="mAP50-95")
    for i, s in enumerate(SIZES):
        left.text(i, maps[s] + 0.004, f"{maps[s]:.3f}", ha="center", fontsize=9)
    left.set(ylabel="mAP50-95", xlabel="YOLO11 size", ylim=(0.6, 0.82))
    left.set_xticks(x, SIZES)
    left.grid(axis="y", alpha=0.25)

    twin = left.twinx()
    twin.plot(x, [counts[s]["mae"] for s in SIZES], marker="o", color="#e6550d",
              linewidth=2.0, label="counting MAE (tuned)")
    twin.plot(x, [counts[s]["ripe_mae"] for s in SIZES], marker="s", color="#31a354",
              linewidth=2.0, linestyle="--", label="fully-ripe MAE (tuned)")
    twin.set(ylabel="counting MAE per image", ylim=(0.4, 1.05))
    handles = [plt.Line2D([], [], color="#3182bd", linewidth=6, label="mAP50-95")]
    handles += twin.get_legend_handles_labels()[0]
    left.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.5)
    left.set_title("Accuracy stops at m", fontsize=10.5)

    mb = [mac[s]["ne"]["package_mb"] for s in SIZES]
    load = [phone[s]["ne"]["load_ms"] / 1000.0 for s in SIZES]
    right.bar(x, mb, width=0.55, color="#756bb1")
    for i, v in enumerate(mb):
        right.text(i, v + 2, f"{v:.1f} MB", ha="center", fontsize=9)
    right.set(ylabel="Core ML FP16 package (MB)", xlabel="YOLO11 size", ylim=(0, 130))
    right.set_xticks(x, SIZES)
    right.grid(axis="y", alpha=0.25)
    rtwin = right.twinx()
    rtwin.plot(x, load, marker="o", color="#636363", linewidth=2.0)
    rtwin.set(ylabel="iPhone Neural Engine load (s)", ylim=(0, 3))
    right.set_title("Cost keeps growing after it", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT / "size_accuracy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "size_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_device_latency(mac, phone) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                      gridspec_kw={"width_ratios": [1.55, 1.0]})
    x = np.arange(len(SIZES))
    width = 0.2
    series = [
        ("Mac CPU", [mac[s]["cpu"]["median_ms"] for s in SIZES], COL["mac_cpu"]),
        ("Mac CPU+NE", [mac[s]["ne"]["median_ms"] for s in SIZES], COL["mac_ne"]),
        ("iPhone CPU", [phone[s]["cpu"]["median_ms"] for s in SIZES], COL["phone_cpu"]),
        ("iPhone CPU+NE", [phone[s]["ne"]["median_ms"] for s in SIZES], COL["phone_ne"]),
    ]
    for i, (label, values, colour) in enumerate(series):
        bars = left.bar(x + (i - 1.5) * width, values, width, label=label, color=colour)
        left.bar_label(bars, fmt="%.1f", fontsize=7, padding=1.5)

    left.axhline(FRAME_BUDGET_MS, color="#252525", linestyle=":", linewidth=1.4)
    left.text(len(SIZES) - 0.45, FRAME_BUDGET_MS * 1.06, "30 FPS budget (33.3 ms)",
              ha="right", fontsize=8.5)
    left.set(xlabel="YOLO11 size", ylabel="Median inference latency (ms, log)")
    left.set_yscale("log")
    left.set_ylim(1, 400)
    left.set_xticks(x, SIZES)
    left.grid(axis="y", alpha=0.25, which="both")
    left.legend(frameon=False, ncols=2, fontsize=8.5, loc="upper left")
    left.set_title("Mac16,8 vs iPhone 17 Pro, FP16 Core ML, same images", fontsize=10.5)

    cpu_ratio = [phone[s]["cpu"]["median_ms"] / mac[s]["cpu"]["median_ms"] for s in SIZES]
    ne_ratio = [phone[s]["ne"]["median_ms"] / mac[s]["ne"]["median_ms"] for s in SIZES]
    ne_speedup = [phone[s]["cpu"]["median_ms"] / phone[s]["ne"]["median_ms"] for s in SIZES]
    right.plot(x, cpu_ratio, marker="o", linewidth=2.0, color=COL["phone_cpu"],
               label="iPhone / Mac, CPU")
    right.plot(x, ne_ratio, marker="s", linewidth=2.0, color=COL["phone_ne"],
               label="iPhone / Mac, Neural Engine")
    right.plot(x, ne_speedup, marker="^", linewidth=2.0, color="#31a354", linestyle="--",
               label="iPhone: NE speed-up over CPU")
    right.axhline(1.0, color="#969696", linewidth=1.0)
    right.set(xlabel="YOLO11 size", ylabel="Ratio", ylim=(0, 5))
    right.set_xticks(x, SIZES)
    right.grid(alpha=0.25)
    right.legend(frameon=False, fontsize=8.5, loc="upper left")
    right.set_title("The phone gap, and what the NE buys", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT / "device_latency.pdf", bbox_inches="tight")
    fig.savefig(OUT / "device_latency.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy_vs_latency(maps, mac, phone) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    sizes_mb = [mac[s]["ne"]["package_mb"] for s in SIZES]
    area = [18 + 9.0 * mb for mb in sizes_mb]

    px = [phone[s]["ne"]["median_ms"] for s in SIZES]
    mx = [mac[s]["ne"]["median_ms"] for s in SIZES]
    y = [maps[s] for s in SIZES]

    ax.scatter(px, y, s=area, color=COL["phone_ne"], alpha=0.75, label="iPhone 17 Pro, CPU+NE")
    ax.scatter(mx, y, s=area, facecolors="none", edgecolors=COL["mac_ne"], linewidths=1.8,
               label="Mac16,8, CPU+NE")
    ax.plot(px, y, color=COL["phone_ne"], alpha=0.35, linewidth=1.2)
    ax.plot(mx, y, color=COL["mac_ne"], alpha=0.35, linewidth=1.2, linestyle="--")

    phone_offsets = {"n": (12, -6), "s": (12, -6), "m": (-14, 14), "l": (12, -14),
                     "x": (6, -20)}
    for s, xp, yp, mb in zip(SIZES, px, y, sizes_mb):
        ax.annotate(f"{s}  {mb:.0f} MB", (xp, yp), textcoords="offset points",
                    xytext=phone_offsets[s], fontsize=9)
    mac_offsets = {"n": (-2, 10), "s": (-2, 10), "m": (-6, 12), "l": (8, 10), "x": (-2, 12)}
    for s, xm, yp in zip(SIZES, mx, y):
        ax.annotate(s, (xm, yp), textcoords="offset points", xytext=mac_offsets[s],
                    fontsize=8.5, color=COL["mac_ne"])

    sustained = [(s, phone[s]["sustained"]["median_ms"]) for s in SIZES if "sustained" in phone[s]]
    ax.scatter([v for _, v in sustained], [maps[s] for s, _ in sustained], marker="x",
               color="#252525", s=45, label="iPhone sustained, 4500 frames")

    ax.axvline(FRAME_BUDGET_MS, color="#252525", linestyle=":", linewidth=1.4)
    ax.text(FRAME_BUDGET_MS - 0.6, 0.7935, "30 FPS budget", rotation=90, ha="right",
            va="top", fontsize=8.5)
    ax.set(xlabel="Neural Engine median latency (ms)", ylabel="mAP50-95",
           xlim=(0, 37), ylim=(0.695, 0.795))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.04))
    ax.set_title("Accuracy against latency; marker area is the Core ML package", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_latency.pdf", bbox_inches="tight")
    fig.savefig(OUT / "accuracy_vs_latency.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _lighting_panels(models, labels, colours, styles, title, stem) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex="col")
    data = {m: lighting(m) for m in models}

    for col, (axis, xlabel) in enumerate(
        (("exposure", "Exposure (stops)"), ("white_balance", "White balance (cool <- -> warm)"))
    ):
        for model, label, colour, style in zip(models, labels, colours, styles):
            rows = data[model]["axes"][axis]
            xs = [r["value"] for r in rows]
            axes[0][col].plot(xs, [r["by_conf"]["0.25"]["recall"] for r in rows],
                              marker="o", markersize=4, linewidth=1.9, color=colour,
                              linestyle=style, label=label)
            axes[1][col].plot(xs, [r["best_counting_mae"] for r in rows],
                              marker="o", markersize=4, linewidth=1.9, color=colour,
                              linestyle=style, label=label)
        for row in (0, 1):
            axes[row][col].grid(alpha=0.25)
            axes[row][col].axvline(0.0, color="#969696", linewidth=1.0)
        axes[1][col].set_xlabel(xlabel)

    axes[0][0].set_ylabel("Recall at conf 0.25")
    axes[1][0].set_ylabel("Counting MAE (best threshold)")
    axes[0][0].set_ylim(0.58, 0.96)
    axes[0][1].set_ylim(0.58, 0.96)
    axes[1][0].set_ylim(0.7, 2.2)
    axes[1][1].set_ylim(0.7, 2.2)
    axes[0][0].legend(frameon=False, fontsize=8.5, ncols=2, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_lighting_sizes() -> None:
    _lighting_panels(
        SIZES,
        [f"YOLO11{s}" for s in SIZES],
        ["#e6550d", "#fd8d3c", "#3182bd", "#31a354", "#756bb1"],
        ["-"] * 5,
        "Lighting sweep across the size ladder (161 val images, labels untouched)",
        "lighting_sizes",
    )


def fig_lighting_hsv() -> None:
    _lighting_panels(
        ["n", "light_hsv_v08", "light_hsv_wide", "light_no_colour"],
        ["stock HSV (n)", "hsv_v08 (brightness x2)", "hsv_wide (+ hue)", "no colour jitter"],
        ["#e6550d", "#3182bd", "#31a354", "#252525"],
        ["-", "-", "-", "--"],
        "Does wider HSV jitter at training time buy lighting robustness?",
        "lighting_hsv",
    )


def fig_lighting_examples() -> None:
    """Contact sheet: one val frame under each level, captioned with what the
    baseline n loses there. The crop holds fully ripe and green fruit side by
    side, because the stage is read from exactly the colour these knobs move."""
    import cv2  # local: only this figure needs OpenCV
    from relight import mean_luma, relight

    source = cv2.imread(str(ROOT / "data" / "laboro_yolo" / "images" / "val" / "IMG_0991.jpg"))
    crop = (661, 283, 3024, 2268)  # x1, y1, x2, y2 - red cluster plus green fruit
    rows = lighting("n")["axes"]
    recall = {
        ("stops", r["value"]): r["by_conf"]["0.25"]["recall"] for r in rows["exposure"]
    }
    recall.update({("wb", r["value"]): r["by_conf"]["0.25"]["recall"] for r in rows["white_balance"]})

    strips = [
        ("Exposure", "stops", [-3.0, -1.5, 0.0, 1.5, 3.0], lambda v: f"{v:+.1f} stops"),
        ("White balance", "wb", [-1.5, -0.66, 0.0, 0.66, 1.5],
         lambda v: f"wb {v:+.2f} " + ("(cool)" if v < 0 else "(warm)" if v > 0 else "(none)")),
    ]

    cell_w, cell_h, caption_h, header_h = 420, 353, 44, 30
    canvas = Image.new("RGB", (cell_w * 5, (cell_h + caption_h + header_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=20)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=14)

    for row, (name, knob, levels, fmt) in enumerate(strips):
        top = row * (cell_h + caption_h + header_h)
        draw.text((8, top + 6), name, fill="black", font=title_font)
        for col, value in enumerate(levels):
            frame = relight(source, **{knob: value})
            luma = mean_luma(frame)
            patch = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(crop)
            patch.thumbnail((cell_w - 8, cell_h - 8))
            x = col * cell_w + (cell_w - patch.width) // 2
            canvas.paste(patch, (x, top + header_h + (cell_h - patch.height) // 2))
            caption = fmt(value)
            detail = (f"frame luma {luma:.0f} · YOLO11n recall over the 161 val images "
                      f"{recall[(knob, value)]:.3f}")
            for line, f, dy in ((caption, font, 4), (detail, small, 23)):
                box = draw.textbbox((0, 0), line, font=f)
                draw.text((col * cell_w + (cell_w - (box[2] - box[0])) // 2,
                           top + header_h + cell_h + dy), line, fill="black", font=f)

    canvas.save(OUT / "lighting_examples.jpg", quality=92, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    maps, counts, mac, phone = map_scores(), counting_scores(), mac_rows(), phone_rows()
    fig_size_accuracy(maps, counts, mac, phone)
    fig_device_latency(mac, phone)
    fig_accuracy_vs_latency(maps, mac, phone)
    fig_lighting_sizes()
    fig_lighting_hsv()
    fig_lighting_examples()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
