"""
Load ultralytics 8.0.x checkpoints under a modern ultralytics.

Camacho's released `YOLOv8-seg-trained.pt` was pickled by ultralytics 8.0.39,
whose modules lived under `ultralytics.yolo.*`. That namespace was removed in
later releases, so unpickling raises ModuleNotFoundError. This installs an
import hook aliasing `ultralytics.yolo.X` -> `ultralytics.X` (with a few
explicit redirects where the layout also changed), which is enough to
deserialise the model graph.

Import this module before torch.load / YOLO(...).
"""
import importlib
import sys
import types

# 8.0.x path -> modern path, for the cases where it is not a plain prefix swap.
EXPLICIT = {
    "ultralytics.yolo.utils": "ultralytics.utils",
    "ultralytics.yolo.utils.loss": "ultralytics.utils.loss",
    "ultralytics.yolo.utils.ops": "ultralytics.utils.ops",
    "ultralytics.yolo.utils.tal": "ultralytics.utils.tal",
    "ultralytics.yolo.utils.metrics": "ultralytics.utils.metrics",
    "ultralytics.yolo.utils.checks": "ultralytics.utils.checks",
    "ultralytics.yolo.utils.torch_utils": "ultralytics.utils.torch_utils",
    "ultralytics.yolo.engine.model": "ultralytics.engine.model",
    "ultralytics.yolo.engine.results": "ultralytics.engine.results",
    "ultralytics.yolo.data": "ultralytics.data",
    "ultralytics.yolo.v8": "ultralytics.models.yolo",
}


class _LegacyFinder:
    def find_module(self, fullname, path=None):  # py2-style API torch still probes
        return self if fullname.startswith("ultralytics.yolo") else None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        target = EXPLICIT.get(fullname)
        if target is None and fullname.startswith("ultralytics.yolo."):
            target = "ultralytics." + fullname[len("ultralytics.yolo."):]
        mod = None
        if target:
            try:
                mod = importlib.import_module(target)
            except ImportError:
                mod = None
        if mod is None:
            # container package with no modern counterpart (e.g. ultralytics.yolo)
            mod = types.ModuleType(fullname)
            mod.__path__ = []
        sys.modules[fullname] = mod
        return mod

    # PEP 451 API
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("ultralytics.yolo"):
            return None
        from importlib.machinery import ModuleSpec
        return ModuleSpec(fullname, _Loader(self))


class _Loader:
    def __init__(self, finder):
        self.finder = finder

    def create_module(self, spec):
        return self.finder.load_module(spec.name)

    def exec_module(self, module):
        pass


def install():
    import ultralytics  # noqa: F401  (ensure the real package is imported first)
    if not any(isinstance(f, _LegacyFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())


install()
