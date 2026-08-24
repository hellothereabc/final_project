"""Build occlusion.ipynb from the JSON the sweeps leave in runs/occlusion."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))          # noqa: E731
code = lambda s: c.append(nbf.v4.new_code_cell(s))            # noqa: E731

md("""# Occlusion

Sections 1-3 measured the detector on fruit an annotator could see whole. A phone
in a greenhouse is pointed at fruit behind leaves, behind other fruit, half out
of frame. This section asks what that costs, and - the part the earlier sections
set up - whether it costs the same thing everywhere.

Three measurements:

1. **Synthetic sweep.** Hide a known fraction of every annotated fruit and watch
   detection, staging and counting separately.
2. **Colour-directed occlusion.** A half-ripened fruit is red on one side and
   green on the other. Hide the red half, hide the green half: same fruit, same
   fraction, opposite colour evidence.
3. **Natural occlusion.** No pasting at all - fruit that other fruit already
   overlap in the original frames.

Model and data are the ones from section 3: `runs/cross_laboro3`, Laboro's own
161-image validation split, conf 0.25.""")

code("""import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = Path("../runs/occlusion")
STAGES = ["green", "half_ripened", "fully_ripened"]
SHORT = {"green": "green", "half_ripened": "half", "fully_ripened": "fully"}

strict = json.loads((RUNS / "laboro3_laboro3_leaf_iou50.json").read_text())
loose = json.loads((RUNS / "laboro3_laboro3_leaf_iou30.json").read_text())
print(f"{strict['n_images']} images, {strict['levels'][0]['n_gt']} annotated fruit, "
      f"conf {strict['conf']}, occluder {strict['occluder']}")""")

md("""## How the occluder is built

The strip runs in from one side of the box and covers a set fraction of its area,
so "40% hidden" means the same thing for a large fruit and a small one. The
patch itself is cropped from a fruit-free part of the same frame, which carries
that frame's lighting and white balance with it - a flat rectangle of any colour
would be a novel object rather than a leaf. `--occluder black` keeps the flat
version as a control.""")

code("""from io import BytesIO

from IPython.display import Image, display
from PIL import Image as PILImage

for name in ["clean.jpg", "random_leaf_40.jpg", "hide_red_leaf_40.jpg"]:
    path = RUNS / "examples" / name
    if not path.exists():
        continue
    thumb = PILImage.open(path)
    thumb.thumbnail((420, 420))
    buffer = BytesIO()
    thumb.save(buffer, format="JPEG", quality=80)
    print(name)
    display(Image(data=buffer.getvalue()))""")

md("""## 1. What breaks first

Ground truth stays the whole fruit, occluded or not, so recall answers "did it
find the fruit" rather than "did it find the visible part". Stage accuracy is
computed only over fruit that were found, which keeps the two failures apart.""")

code("""def table(document, mode):
    rows = [r for r in document["levels"] if r["mode"] in ("none", mode)]
    print(f"{'hidden':>7}{'recall':>9}{'green':>8}{'half':>7}{'fully':>7}"
          f"{'stage-acc':>11}{'conf':>7}{'MAE':>7}")
    for row in rows:
        by_stage = "".join(f"{row['recall_by_stage'][s]:>8.2f}" if s == "green"
                           else f"{row['recall_by_stage'][s]:>7.2f}" for s in STAGES)
        print(f"{row['fraction'] * 100:>6.0f}%{row['recall']:>9.3f}{by_stage}"
              f"{row['stage_accuracy']:>11.3f}{row['mean_conf']:>7.3f}"
              f"{row['counting_mae']:>7.2f}")

table(strict, "random")""")

md("""Detection and staging come apart. Recall falls by an order of magnitude
across the sweep while stage accuracy barely moves - the model stops *finding*
fruit long before it starts misreading ripeness. Counting-MAE, the metric this
thesis is about, is driven almost entirely by the first of those, which is the
same point section 3 made about domain shift: a single error number cannot say
which mechanism produced it.

The per-stage columns rank the way the colour argument predicts: `fully` survives
occlusion best and `green` worst, on a dataset where green fruit are also the
smallest and the most numerous.

The zero-occlusion row is a check rather than a result: counting-MAE 1.25 is the
laboro3/Laboro cell of section 3's matrix, reproduced by a different code path.""")

md("""### The cliff at 50% is bookkeeping, not perception

Recall drops off a shelf between 40% and 50% hidden. That is where the geometry
of the metric turns over, not the model: ground truth is the whole fruit, a
detector that boxes only the visible part scores IoU about `1 - fraction`
against it, and at 50% that lands exactly on the matching threshold. Re-running
the identical sweep at IoU 0.3 separates "did not find it" from "found it,
boxed what was left".""")

code("""print(f"{'hidden':>7}{'recall @0.5':>13}{'recall @0.3':>13}{'MAE @0.5':>10}{'MAE @0.3':>10}")
for row in [r for r in loose["levels"] if r["mode"] in ("none", "random")]:
    twin = next(r for r in strict["levels"]
                if r["mode"] == row["mode"] and r["fraction"] == row["fraction"])
    print(f"{row['fraction'] * 100:>6.0f}%{twin['recall']:>13.3f}{row['recall']:>13.3f}"
          f"{twin['counting_mae']:>10.2f}{row['counting_mae']:>10.2f}")""")

md("""Counting-MAE is identical in the two runs, as it must be - counts do not
depend on how detections are matched to ground truth - which makes the pair a
clean decomposition. Both curves belong in the write-up: the strict one is the
honest answer to "is the fruit localised", the loose one to "is it seen at all".""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for document, label, style in ((strict, "IoU 0.5", "-"), (loose, "IoU 0.3", "--")):
    rows = sorted([r for r in document["levels"] if r["mode"] in ("none", "random")],
                  key=lambda r: r["fraction"])
    axes[0].plot([r["fraction"] * 100 for r in rows], [r["recall"] for r in rows],
                 style, marker="o", label=label)
axes[0].set(xlabel="% of each fruit hidden", ylabel="recall", title="detection")
axes[0].legend(); axes[0].grid(alpha=0.3)

for mode in ("random", "hide_red", "hide_green"):
    rows = sorted([r for r in strict["levels"] if r["mode"] in ("none", mode)],
                  key=lambda r: r["fraction"])
    axes[1].plot([r["fraction"] * 100 for r in rows],
                 [r["counting_mae"] for r in rows], marker="o", label=mode)
axes[1].set(xlabel="% of each fruit hidden", ylabel="counting-MAE per image",
            title="counting")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RUNS / "curves.png", dpi=120)
plt.show()""")

md("""## 2. Hiding the red half against hiding the green half

Same fruit, same fraction, opposite evidence. `hide_red` covers the redder half
of each box, `hide_green` the greener one, chosen by median CIELAB a\\* - the axis
`cross_dataset/colour_stages.py` measured the stage boundaries on.

If the model reads ripeness off the visible colour, the two should push the stage
call in opposite directions; if it reads shape, size or context instead, they
should not differ.""")

code("""for mode in ("hide_red", "hide_green"):
    print(f"--- {mode} ---")
    table(strict, mode)
    print()""")

code("""from math import comb

RANK = {stage: index for index, stage in enumerate(STAGES)}

flip = json.loads((RUNS / "colour_flip_laboro3_laboro3_40.json").read_text())
print(f"paired, {int(flip['fraction'] * 100)}% hidden; stage scale green 0, half 1, fully 2")
print(f"{'GT stage':<10}{'n':>6}{'red hidden':>13}{'green hidden':>15}{'shift':>9}"
      f"{'disagree':>11}")
for stage, row in flip["by_stage"].items():
    print(f"{SHORT[stage]:<10}{row['n']:>6}{row['mean_hide_red']:>13.3f}"
          f"{row['mean_hide_green']:>15.3f}{row['shift']:>+9.3f}"
          f"{row['disagreement']:>11.1%}")

# McNemar over discordant pairs: fruit whose call moved up the ripeness scale when
# the green half was hidden, against those that moved down.
print()
for stage in STAGES:
    up = down = 0
    for key, count in flip["flips"].get(stage, {}).items():
        red_call, green_call = key.split("->")
        if RANK[green_call] > RANK[red_call]:
            up += count
        elif RANK[green_call] < RANK[red_call]:
            down += count
    total = up + down
    if not total:
        continue
    tail = sum(comb(total, k) for k in range(max(up, down), total + 1)) / 2 ** total
    print(f"{SHORT[stage]:<8} riper when green hidden: {up:>3}   "
          f"riper when red hidden: {down:>3}   two-sided p = {min(1.0, 2 * tail):.3f}")""")

md("""The aggregate sweep above showed almost nothing between the two modes. Paired,
it separates: **which half is hidden decides the stage of a half-ripened fruit
and almost nothing else.** 17% of half-ripened fruit change their call between
the two conditions, against 2% of green and 5% of fully ripened ones - the
classes that look the same from either side. The direction is the one the colour
boundaries predict: among the half-ripened fruit that moved, close to three times
as many read riper with the green side covered as the other way round.

The mean shift is small (+0.08 of a stage) because most fruit do not move at all,
and that is the honest way to report it: the effect is concentrated in the class
that has two colours to choose between, which is also the class section 3 found
the two datasets disagreeing about.

Caveats: only fruit detected in *both* passes are compared, which selects the
easier ones; the two conditions remove different pixels beyond colour, including
specular highlights; and splitting a box by median a\\* is a coarse stand-in for
"the red side".""")

md("""## 3. Occlusion that was already there

Fruit overlapping other fruit, measured on the untouched frames: for every
annotated box, the share of its area covered by other annotated boxes. It is a
lower bound on what is hidden - leaves and stems carry no labels, and box overlap
does not know which fruit is in front - and it is real rather than pasted.""")

code("""natural = json.loads((RUNS / "natural_laboro3_laboro3.json").read_text())
print(f"{'covered by other fruit':<24}{'n':>7}{'recall':>9}{'stage-acc':>11}{'conf':>8}")
for row in natural["rows"]:
    print(f"{row['bin']:<24}{row['n']:>7}{row['recall']:>9.3f}"
          f"{row['stage_accuracy']:>11.3f}{row['mean_conf']:>8.3f}")""")

md("""Naturally overlapped fruit are barely harder than isolated ones: recall
holds within a point or two out to 50% box coverage and only falls past that.
The synthetic sweep, at the same nominal fraction, has already lost a third of
its detections. Two readings, and they are not exclusive:

- **The proxy overstates occlusion.** Fruit are round and boxes are rectangles,
  so two boxes can overlap by a third while neither fruit hides any of the other.
  Whatever "30-50% covered" means here, it is not "a third of the fruit is
  invisible".
- **The occluder is out of distribution.** Fruit behind fruit is in the training
  set; a pasted rectangle with a straight edge is not. Some of the synthetic
  damage may be the model reacting to the patch rather than to the missing
  pixels - which is what the `black` control is for, and what a fruit-shaped or
  leaf-shaped occluder would test.

Either way the gap is the finding, and it lands on the same nerve as section 3:
a protocol that hides a fixed fraction with a pasted patch is not measuring
robustness to occlusion, it is measuring reaction to an occluder. The number it
produces is not wrong, but it is not the field number either.""")

md("""## 4. Two controls

**Does the occluder matter?** `--occluder black` pastes a flat black rectangle
instead of a crop of the frame's own background. Same geometry, same fraction,
different appearance.

**Does the checkpoint matter?** The same sweep with `runs/cross_mix`, the model
that actually ships in the app - trained on both datasets rather than on Laboro
alone.""")

code("""black = json.loads((RUNS / "laboro3_laboro3_black_iou50.json").read_text())
mix = json.loads((RUNS / "mix_laboro3_leaf_iou50.json").read_text())


def curve(document):
    return {r["fraction"]: r for r in document["levels"] if r["mode"] in ("none", "random")}

leaf_c, black_c, mix_c = curve(strict), curve(black), curve(mix)
print(f"{'hidden':>7}{'leaf':>9}{'black':>9}{'mix/leaf':>11}   "
      f"{'MAE leaf':>9}{'MAE black':>10}{'MAE mix':>9}")
for fraction in sorted(set(leaf_c) & set(black_c) & set(mix_c)):
    print(f"{fraction * 100:>6.0f}%{leaf_c[fraction]['recall']:>9.3f}"
          f"{black_c[fraction]['recall']:>9.3f}{mix_c[fraction]['recall']:>11.3f}   "
          f"{leaf_c[fraction]['counting_mae']:>9.2f}"
          f"{black_c[fraction]['counting_mae']:>10.2f}"
          f"{mix_c[fraction]['counting_mae']:>9.2f}")""")

md("""The black patch is consistently harder than the background crop at the same
fraction. Nothing about the fruit differs between the two runs - only what covers
it - so that gap is the model reacting to the occluder rather than to the missing
pixels. It is a lower bound on the effect, too: a background crop is still a
pasted rectangle with straight edges.

Which makes the reading of section 3 more concrete. A synthetic-occlusion number
is an occluder-dependent quantity, and the choice of occluder is rarely reported.
Between the two here it is worth about ten points of recall at 30-40% hidden -
more than the difference between several of the models in section 3's matrix.""")

md("""## Open

- Two checkpoints, one dataset. Neither sweep has been run on tomatOD, whose
  fruit are smaller in frame, so none of this is yet a statement about the task
  rather than about Laboro.
- The occluder is a rectangle. Leaves are not rectangles, and a detector that has
  learned "tomatoes are round" may be losing the fruit to the straight edge as
  much as to the missing pixels. The `black` control separates colour from
  texture but not shape from either.
- Every fruit in a frame is occluded at once, which also changes the scene: fewer
  intact exemplars, more clutter. Occluding one fruit per pass would isolate the
  per-fruit effect at N times the compute.
- Box overlap is a poor proxy for occlusion on round fruit and does not know
  which fruit is in front. Laboro ships instance masks; recomputing coverage
  from masks would replace the proxy with a measurement and make the
  synthetic-vs-natural gap quantitative rather than suggestive.""")

nb["cells"] = c
nbf.write(nb, "occlusion.ipynb")
print("wrote occlusion.ipynb")
