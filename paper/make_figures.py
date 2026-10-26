"""Create paper figures from committed evidence.

The curve uses the values reported in README section 4. The example montage
applies the same rectangular, same-image background-patch intervention to a
representative red tomato from an app sample. It is an illustration of the
transformation rather than an additional evaluation result.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tomato-paper-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"


def occlusion_curve() -> None:
    hidden = [0, 20, 40, 60, 70]
    recall = [0.884, 0.778, 0.571, 0.160, 0.057]
    stage_accuracy = [0.944, 0.940, 0.920, 0.900, 0.860]

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.plot(hidden, recall, marker="o", linewidth=2.1, label="Detection recall")
    ax.plot(
        hidden,
        stage_accuracy,
        marker="s",
        linewidth=2.1,
        label="Stage accuracy (detected fruit)",
    )
    ax.set(xlabel="Occluded fraction of each annotated box (%)", ylabel="Score")
    ax.set_xlim(-2, 72)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(hidden)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "occlusion_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "occlusion_curve.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def occlusion_examples() -> None:
    source = Image.open(
        ROOT / "ios" / "TomatoRipeness" / "Resources" / "samples" / "sample2.jpg"
    ).convert("RGB")

    # Representative foreground red tomato in sample2.jpg. The crop at upper
    # right contains foliage and no fruit, matching the same-image background
    # patch used by occlusion/occlude.py. Coordinates are only for this visual.
    fruit_box = (274, 327, 422, 488)
    foliage = source.crop((480, 50, 608, 178))
    display_crop = (65, 225, 465, 575)

    images = []
    for fraction in (0.0, 0.2, 0.4, 0.6):
        view = source.copy()
        if fraction:
            x1, y1, x2, y2 = fruit_box
            strip_height = round((y2 - y1) * fraction)
            fill = foliage.resize((x2 - x1, strip_height), Image.Resampling.BILINEAR)
            view.paste(fill, (x1, y1))
        images.append(view.crop(display_crop))

    labels = ("Original", "20% occluded", "40% occluded", "60% occluded")
    cell_width, cell_height, caption_height = 330, 300, 34
    canvas = Image.new("RGB", (cell_width * 4, cell_height + caption_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)

    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail((cell_width - 10, cell_height - 10))
        x = index * cell_width + (cell_width - image.width) // 2
        y = (cell_height - image.height) // 2
        canvas.paste(image, (x, y))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (index * cell_width + (cell_width - text_width) // 2, cell_height + 6),
            label,
            fill="black",
            font=font,
        )

    canvas.save(OUT / "occlusion_examples.jpg", quality=92, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    occlusion_curve()
    occlusion_examples()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
