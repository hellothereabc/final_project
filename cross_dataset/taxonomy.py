"""
Shared 3-stage label space: Laboro's 6 classes and tomatOD's 3 collapse here.

Mapped by class name rather than index, since indices differ between datasets.
Rule order matters: "semi-ripe" and "unripe" both contain "ripe", so those get
checked first. Same order as String.ripenessStage in the iOS app's
BoxOverlay.swift, and the names below survive that rule too, so a model trained
here drops into the app unchanged.
"""

# ordered by ripening, so confusions land next to the diagonal
STAGES = ["green", "half_ripened", "fully_ripened"]
STAGE_IDX = {name: i for i, name in enumerate(STAGES)}

# Short forms for tables and plots.
SHORT = {"green": "green", "half_ripened": "half", "fully_ripened": "fully"}


class UnknownClassError(ValueError):
    """Class name no rule matches. Raised rather than bucketed by default -
    a silent fallback would quietly corrupt every count downstream."""


def stage_of(class_name):
    """Map a dataset's class name onto a shared ripeness stage."""
    name = class_name.lower()
    if "half" in name or "semi" in name:
        return "half_ripened"
    if "green" in name or "unripe" in name:
        return "green"
    if "full" in name or "ripe" in name:
        return "fully_ripened"
    raise UnknownClassError(f"{class_name!r} matches no ripeness rule, add it here")


def stage_index_of(class_name):
    return STAGE_IDX[stage_of(class_name)]


def describe(class_names):
    """Mapping table for prepare.py's log."""
    width = max(len(n) for n in class_names)
    return "\n".join(f"  {n:<{width}} -> {stage_of(n)}" for n in sorted(class_names))
