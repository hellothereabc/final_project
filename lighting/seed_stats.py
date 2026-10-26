"""Seed repeats and paired tests for the colour-augmentation comparison.

Reads runs/lighting/seeds/<config>_s<seed>.json (written by sweep_seeds.sh) and
writes lighting/seed_results.md plus paper/figures/lighting_seeds.pdf.

Two sources of variation, reported separately because they answer different
questions (see the paper's statistical analysis section):

  training    every configuration is retrained at seeds 0, 1, 2 under an
              identical protocol. `stock` is repeated against itself, so its
              spread is the noise floor any difference has to clear. Reported
              as mean over seeds with the observed range; three runs do not
              justify assuming normality.

  evaluation  for a pair of configurations at one level, the per-image series
              are averaged over seeds and compared pairwise over the 161
              validation images: a 95% bootstrap interval for the paired
              difference (10,000 resamples of images) and a paired sign-flip
              permutation test (10,000 relabellings). Both are distribution
              free, which matters because per-image counting errors are small
              integers with many ties. p-values are Holm-adjusted within the
              family of levels reported for one comparison.

Usage:
    python lighting/seed_stats.py
    python lighting/seed_stats.py --resamples 50000
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = ROOT / "runs" / "lighting" / "seeds"
OUT_MD = Path(__file__).resolve().parent / "seed_results.md"
FIG_DIR = ROOT / "paper" / "figures"

CONFIGS = [
    ("stock", "stock HSV jitter (default)"),
    ("hsv_v08", "hsv\\_v08 (brightness x2)"),
    ("hsv_wide", "hsv\\_wide (brightness x2, hue x3)"),
    ("no_colour", "no colour jitter"),
]
LEVELS = [("stops", 0.0, "clean"), ("stops", -3.0, "-3 stops"), ("stops", 3.0, "+3 stops"),
          ("wb", -1.5, "wb -1.5 (cool)"), ("wb", 1.5, "wb +1.5 (warm)")]
RNG = np.random.default_rng(0)


def load() -> dict:
    """{config: {seed: {(knob, value): row}}} for every sweep on disk."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(SEEDS_DIR.glob("*_s*.json")):
        config, _, seed = path.stem.rpartition("_s")
        rows = {}
        for axis in json.load(open(path))["axes"].values():
            for row in axis:
                rows[(row["knob"], round(row["value"], 2))] = row
        out.setdefault(config, {})[int(seed)] = rows
    return out


def per_image(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image counting error (mean over stages), matched count, gt count.

    The counting error uses the best threshold of that level, which is what the
    tuned comparison reports; recall uses the fixed 0.25 threshold.
    """
    best = str(row["best_conf"])
    block = row["per_image"].get(best) or row["per_image"]["0.25"]
    err = np.mean([block["abs_err"][s] for s in sorted(block["abs_err"])], axis=0)
    fixed = row["per_image"]["0.25"]
    return err, np.array(fixed["found"], float), np.array(fixed["gt"], float)


def seed_table(data: dict) -> list[list[str]]:
    body = []
    for config, label in CONFIGS:
        seeds = data.get(config, {})
        if not seeds:
            continue
        row = [f"{label} ({len(seeds)} seeds)"]
        for knob, value, _ in LEVELS:
            recalls, maes = [], []
            for rows in seeds.values():
                entry = rows[(knob, value)]
                recalls.append(entry["by_conf"]["0.25"]["recall"])
                maes.append(entry["best_counting_mae"])
            row.append(f"{np.mean(recalls):.3f} ({min(recalls):.3f}-{max(recalls):.3f})")
            row.append(f"{np.mean(maes):.2f} ({min(maes):.2f}-{max(maes):.2f})")
        body.append(row)
    return body


def paired(a: np.ndarray, b: np.ndarray, resamples: int) -> dict:
    """Bootstrap CI and sign-flip permutation p for the paired difference a-b."""
    diff = a - b
    n = len(diff)
    idx = RNG.integers(0, n, size=(resamples, n))
    boot = diff[idx].mean(axis=1)
    flips = RNG.choice([-1.0, 1.0], size=(resamples, n))
    null = (diff * flips).mean(axis=1)
    observed = diff.mean()
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (resamples + 1)
    return {"diff": float(observed),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "p": float(p)}


def recall_ci(found_a, found_b, gt, resamples: int) -> dict:
    n = len(gt)
    idx = RNG.integers(0, n, size=(resamples, n))
    ratio = lambda f, i: f[i].sum(axis=1) / gt[i].sum(axis=1)
    boot = ratio(found_a, idx) - ratio(found_b, idx)
    observed = found_a.sum() / gt.sum() - found_b.sum() / gt.sum()
    return {"diff": float(observed),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m, adjusted, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def compare(data: dict, resamples: int) -> dict:
    """Every configuration against stock, seed-averaged, level by level."""
    results = {}
    for config, _ in CONFIGS:
        if config == "stock" or config not in data or "stock" not in data:
            continue
        shared = sorted(set(data[config]) & set(data["stock"]))
        if not shared:
            continue
        rows, ps = [], []
        for knob, value, label in LEVELS:
            err_a, found_a, gt = (np.mean(x, axis=0) for x in zip(
                *(per_image(data[config][s][(knob, value)]) for s in shared)))
            err_b, found_b, gt_b = (np.mean(x, axis=0) for x in zip(
                *(per_image(data["stock"][s][(knob, value)]) for s in shared)))
            assert np.array_equal(gt, gt_b), "validation ground truth differs between runs"
            mae = paired(err_a, err_b, resamples)
            rec = recall_ci(found_a, found_b, gt, resamples)
            rows.append({"level": label, "mae": mae, "recall": rec})
            ps.append(mae["p"])
        for row, adjusted in zip(rows, holm(ps)):
            row["mae"]["p_holm"] = adjusted
        results[config] = {"seeds": shared, "rows": rows}
    return results


def figure(data: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tomato-paper-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"stock": "#e6550d", "hsv_v08": "#3182bd",
               "hsv_wide": "#31a354", "no_colour": "#252525"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for axis_idx, (knob, xlabel) in enumerate(
            (("stops", "Exposure (stops)"), ("wb", "White balance (cool to warm)"))):
        ax = axes[axis_idx]
        for config, label in CONFIGS:
            seeds = data.get(config, {})
            if not seeds:
                continue
            levels = sorted({v for k, v in next(iter(seeds.values())) if k == knob})
            curves = np.array([[rows[(knob, v)]["by_conf"]["0.25"]["recall"] for v in levels]
                               for rows in seeds.values()])
            mean = curves.mean(axis=0)
            ax.plot(levels, mean, marker="o", markersize=4, linewidth=2.0,
                    color=colours[config], linestyle="--" if config == "no_colour" else "-",
                    label=f"{label.replace(chr(92), '')} ({len(seeds)})")
            if len(curves) > 1:
                ax.fill_between(levels, curves.min(axis=0), curves.max(axis=0),
                                color=colours[config], alpha=0.18, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        ax.axvline(0.0, color="#969696", linewidth=1.0)
    axes[0].set_ylabel("Recall at conf 0.25")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Seed repeats: mean over seeds, band is the observed range", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "lighting_seeds.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "lighting_seeds.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def table(header, body) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
                     + ["| " + " | ".join(r) + " |" for r in body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit(f"no sweeps in {SEEDS_DIR}; run lighting/sweep_seeds.sh first")
    counts = {c: sorted(s) for c, s in data.items()}
    print("sweeps found:", counts)

    parts = [
        "# Colour augmentation: seed repeats and paired tests",
        "",
        "Generated by `python lighting/seed_stats.py` from `runs/lighting/seeds/*.json`.",
        "Do not edit by hand.",
        "",
        f"Seeds available per configuration: "
        + ", ".join(f"`{c}` {s}" for c, s in sorted(counts.items())) + ".",
        "",
        "## Training variation: mean over seeds (observed range)",
        "",
        "Recall is at conf 0.25; counting MAE is at the best threshold of that level.",
        "",
    ]
    header = ["Configuration"]
    for _, _, label in LEVELS:
        header += [f"{label} recall", f"{label} MAE"]
    parts += [table(header, seed_table(data)), ""]

    results = compare(data, args.resamples)
    if results:
        parts += [
            "## Evaluation variation: each configuration against stock",
            "",
            f"Seed-averaged per-image series, paired over the 161 validation images;",
            f"{args.resamples:,} bootstrap resamples and {args.resamples:,} sign-flip",
            "permutations. A negative MAE difference favours the configuration over stock.",
            "p is Holm-adjusted across the five levels of each comparison.",
            "",
        ]
        for config, result in results.items():
            parts += [f"### {config} vs stock (seeds {result['seeds']})", ""]
            body = []
            for row in result["rows"]:
                mae, rec = row["mae"], row["recall"]
                body.append([
                    row["level"],
                    f"{mae['diff']:+.3f}",
                    f"[{mae['lo']:+.3f}, {mae['hi']:+.3f}]",
                    f"{mae['p_holm']:.3f}",
                    f"{rec['diff']:+.3f}",
                    f"[{rec['lo']:+.3f}, {rec['hi']:+.3f}]",
                ])
            parts += [table(["Level", "MAE diff", "95% CI", "p (Holm)",
                             "recall diff", "95% CI"], body), ""]
    else:
        parts += ["## Evaluation variation", "",
                  "Not computed: the stock configuration and at least one other must share",
                  "a seed. Train the remaining seeds first.", ""]

    OUT_MD.write_text("\n".join(parts) + "\n")
    figure(data)
    print(f"wrote {OUT_MD} and {FIG_DIR / 'lighting_seeds.pdf'}")


if __name__ == "__main__":
    main()
