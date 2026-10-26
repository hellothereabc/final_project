"""Assemble a dated folder to show the supervisor.

Two figures carry the meeting - `01_models.png` and `02_lighting.png`, written
by make_talk_figures.py - and `summary.md` holds the tables and the readings.
The detailed figures go to `extra/`, to open only if a question needs them.

The tables in summary.md are sliced out of the generated files
(runtime_bench/device_tables.md, lighting/results.md) rather than retyped, so
regenerating those regenerates the hand-out. Nothing here is a source: delete
the folder and run this again.

    python paper/make_meeting_folder.py [DD.MM.YYYY]
    python paper/make_meeting_folder.py 07.10.2026 --module lighting

With --module the folder instead carries one extract of the paper, built by
build.py, together with the generated tables and figures that extract rests on.
That is the unit to show when the point is one experiment rather than progress
across the whole project.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"

HEADLINE = [("talk_model_ladder", "01_models"), ("talk_lighting", "02_lighting")]
EXTRA = [
    "size_accuracy.png",
    "accuracy_vs_latency.png",
    "device_latency.png",
    "lighting_examples.jpg",
    "lighting_sizes.png",
    "lighting_hsv.png",
]


def section(path: Path, heading: str) -> str:
    """The body under a '## heading' of a generated markdown file."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == f"## {heading}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1:end]).strip()


def summary(date: str) -> str:
    devices = ROOT / "runtime_bench" / "device_tables.md"
    light = ROOT / "lighting" / "results.md"
    return f"""# Tomato ripeness - progress for {date}

Two questions from the 19 September meeting, both answered and measured:
*"try bigger models until it stops working"* and *"data augmentation using
HSV/HSL to simulate light conditions"*.

Two figures, one per question: **`01_models.png`** and **`02_lighting.png`**.
Everything else is in `extra/`, for questions.

Everything below is the six-class Laboro Tomato protocol, 643 train / 161 val
images, 60 epochs, image size 640, seed 0, **one seed per model**: treat
differences under ~0.02 recall or ~0.01 mAP as noise. Latency is inference-only
Core ML (FP16, embedded NMS) on the same three images - no camera capture or
rendering. Mac is a Mac16,8; the phone is an iPhone 17 Pro on iOS 27.0.

## 1. How far up the size ladder is it worth going?  -  figure 01

YOLO11n, s, m, l, x trained under one protocol, exported to Core ML, swept over
confidence, and benchmarked on the Mac and on the phone.

{section(devices, "Accuracy and cost")}

- **Accuracy stops at m.** 0.706 -> 0.747 -> 0.782, then l is 0.002 below and x
  0.008 below m: inside one-seed noise.
- **The fully-ripe count never improves.** n's 0.590 is the best on the ladder;
  every larger model lands between 0.602 and 0.658. The class this work is
  about gains nothing from capacity on this split.
- **Cost keeps growing after the accuracy stops.** m -> x is 2.8x the package,
  2.0x the Neural Engine latency and 1.7x the load time, for no mAP.
- **Where it stops working**, two different answers: accuracy stops improving at
  **m**; real time on the phone stops at **x**, which has no margin left -
  33.8 FPS over a sustained 4,500-frame run, p90 31.8 ms against a 33.3 ms
  budget, and the largest thermal drift on the ladder (+8.8%).

On the phone's CPU rather than its Neural Engine the ladder breaks three sizes
earlier: s is the last size above 30 FPS (38.3), m is at 20.4, x at 8.9. The
laptop is no substitute for measuring on the phone, but it is close: the phone
trails the Mac by a flat ~1.5x on CPU, while on the Neural Engine the gap
closes as the model grows - 3.85x at n down to 1.21x at x - so a small model
looks far worse on the phone than the laptop suggests. Both devices see the
same fruit: every matched box has IoU 0.99-1.00 and the same stage label.
Full tables in `extra/`, `device_tables.md`.

## 2. Does HSV augmentation buy robustness to lighting?  -  figure 02

The test perturbation is deliberately *not* HSV jitter, so no model is scored on
its own augmentation: exposure in stops and a red/blue white-balance shift,
both applied in linear light, labels untouched.

Each cell is recall at conf 0.25 / counting MAE at the best threshold.

{section(light, "Headline: clean frame against the four extremes")}

- **The stock HSV jitter is what buys lighting robustness.** Turning it off is a
  wash on clean frames - it even has the best clean mAP - and loses about 10
  recall points at every extreme.
- **Widening it beyond stock buys nothing.** hsv_v08 sits on the baseline
  everywhere; hsv_wide gains ~0.03 at the white-balance extremes and pays 0.02
  clean mAP.
- **Under-exposure is nearly free down to -3 stops; over-exposure breaks
  everyone**, big models included: clipping removes the very saturation the
  ripeness stage is read from - the +3 stops frame in figure 02 shows it.
- A **cool** cast costs more than a warm one; a **warm** cast costs counting
  more than detection, because it pushes green fruit toward "riper".

Answer to the question: the augmentation asked for is already in the default
recipe and is doing its job. It should be kept, not strengthened.

## Caveats to state out loud

- One seed per model; no confidence intervals.
- Laboro Tomato only, six classes. The cross-dataset (tomatOD) models are not
  part of this comparison.
- The phone moved from iOS 26.5.2 to 27.0 between the two measurement days:
  YOLO11n's Neural Engine latency went 4.05 -> 6.4-6.8 ms (repeated, not noise)
  while m was unchanged. The YOLO11m sustained run is still from 26.5.2.
- Power consumption is not reported: the Instruments runs on 14 September ended
  unreliably, so no defensible number exists.

## If a question needs more (`extra/`)

| File | What it shows |
|---|---|
| `size_accuracy.png` | mAP and tuned counting MAE per size, next to package size and phone load time |
| `accuracy_vs_latency.png` | Figure 01 with the Mac alongside the phone and the sustained runs marked |
| `device_latency.png` | Mac CPU / Mac NE / iPhone CPU / iPhone NE per size, log axis, 33.3 ms budget, ratios |
| `lighting_examples.jpg` | All five exposure and five white-balance levels of the sample frame |
| `lighting_sizes.png` | Recall and counting MAE along both lighting axes for n-x |
| `lighting_hsv.png` | The same for all three HSV training variants |

Sources: every number is generated from saved result files by
`runtime_bench/make_device_tables.py`, `lighting/make_tables.py`,
`paper/make_talk_figures.py` and `paper/make_size_lighting_figures.py`; the full
tables are `runtime_bench/device_tables.md` and `lighting/results.md`, the
written record is `runtime_bench/model_size_results.md`.
"""


MODULE_EXTRAS = {
    "lighting": [
        ("lighting/seed_results.md", "seed_results.md"),
        ("paper/figures/lighting_seeds.png", "seeds_figure.png"),
        ("paper/figures/lighting_examples.jpg", "perturbation_examples.jpg"),
        ("lighting/results.md", "full_sweep_tables.md"),
    ],
    "ladder": [
        ("runtime_bench/device_tables.md", "device_tables.md"),
        ("paper/figures/accuracy_vs_latency.png", "accuracy_vs_latency.png"),
        ("paper/figures/device_latency.png", "device_latency.png"),
    ],
}


def module_folder(date: str, module: str) -> Path:
    """One extract of the paper plus the generated evidence behind it."""
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "paper" / "build.py"), module],
                   check=True, cwd=ROOT / "paper")
    out = FIG / date
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "paper" / "build" / f"{module}.pdf",
                 out / f"00_{module}_extract.pdf")
    for source, name in MODULE_EXTRAS.get(module, []):
        shutil.copy2(ROOT / source, out / name)
    (out / "README.md").write_text(f"""# {module} extract - {date.replace('.', ' ')}

`00_{module}_extract.pdf` is the module of the paper covering this experiment:
its protocol and its results, with the section numbers it carries in the full
paper. It is an extract, not the paper; the rest is in progress.

The other files are the generated evidence the extract rests on, in case a
number is questioned:

""" + "\n".join(f"- `{name}`" for _, name in MODULE_EXTRAS.get(module, []))
        + """

All of them are regenerated from saved result files by the scripts in the
repository; none is transcribed by hand.
""")
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else "22.09.2026"
    if "--module" in sys.argv:
        module = sys.argv[sys.argv.index("--module") + 1]
        print(f"wrote {module_folder(date, module)}")
        return
    out = FIG / date
    extra = out / "extra"
    out.mkdir(parents=True, exist_ok=True)
    extra.mkdir(exist_ok=True)

    for stem, name in HEADLINE:
        for suffix in (".png", ".pdf"):
            shutil.copy2(FIG / f"{stem}{suffix}", out / f"{name}{suffix}")
    for name in EXTRA:
        shutil.copy2(FIG / name, extra / name)

    (out / "summary.md").write_text(summary(date.replace(".", " ")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
