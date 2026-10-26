"""
Photometric perturbations that stand in for a change of lighting.

Two axes, each a single scalar, each applied in linear light rather than on the
8-bit values so that the operation means what a camera would do:

  exposure       gain of 2**stops on every channel. Negative stops darken the
                 way a shaded row or an evening frame is darker; positive stops
                 blow out the highlights, and the clipping is the point - an
                 over-exposed red tomato loses the very saturation the stage
                 label is read from.
  white balance  opposite gains on red and blue, green fixed. Positive is warm
                 (tungsten, low sun), negative is cool (shade, overcast).
                 Ripeness sits on the green->red axis (cross_dataset measured it
                 as CIELAB a*), so a warm cast pushes every fruit toward "riper"
                 and a cool cast toward "greener" before the model sees it.

These are deliberately *not* the HSV jitter ultralytics trains with. The
training-time augmentation is a knob on the pipeline; the test-time
perturbation has to be independent of it, otherwise a model that was trained
under HSV jitter is being tested on its own augmentation and the result says
only that it fitted it.

Labels are untouched: nothing here moves a pixel, only recolours it.
"""
import numpy as np

_SRGB = np.linspace(0.0, 1.0, 256)


def _decode(v):
    """sRGB -> linear, IEC 61966-2-1."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _encode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def _lut(gain):
    """256-entry lookup for `x -> encode(decode(x) * gain)`."""
    return np.round(_encode(_decode(_SRGB) * gain) * 255.0).astype(np.uint8)


def relight(image, stops=0.0, wb=0.0, wb_strength=0.35):
    """Apply exposure and white-balance change to a BGR uint8 image.

    stops        exposure in photographic stops, gain 2**stops
    wb           white-balance shift in [-1, 1]; +1 is the warmest, -1 the coolest
    wb_strength  red/blue gain at |wb| = 1: gains are 1 +- wb_strength * wb
    """
    if stops == 0.0 and wb == 0.0:
        return image.copy()
    exposure = 2.0 ** stops
    gains = {  # OpenCV channel order
        0: exposure * (1.0 - wb_strength * wb),   # blue
        1: exposure,                              # green
        2: exposure * (1.0 + wb_strength * wb),   # red
    }
    out = np.empty_like(image)
    for channel, gain in gains.items():
        out[..., channel] = _lut(gain)[image[..., channel]]
    return out


# The sweep the write-up uses. Exposure is symmetric around the original frame;
# white balance runs from a clearly blue shade to a clearly orange evening. The
# stock YOLO11n held recall within 0.02 out to +-1.5 stops and +-0.66 wb on the
# first pass, so the ends were pushed out until the curve actually breaks.
EXPOSURE_STOPS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
WB_SHIFTS = [-1.5, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.5]


def mean_luma(image):
    """Rec. 601 luma of a BGR image, 0-255, for reporting how dark a level got."""
    b, g, r = [image[..., i].astype(np.float32) for i in range(3)]
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())
