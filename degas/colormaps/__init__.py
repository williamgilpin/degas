# Expose every colormap module in this folder as <name> and <name>_r,
# e.g. degas.colormaps.batlow and degas.colormaps.batlow_r
import importlib
import pkgutil

import matplotlib

__all__ = []


def _register(cmap, name):
    # skip names already registered, e.g. by cmcrameri
    if name not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap, name=name)


def _load_colormaps():
    for _module_info in pkgutil.iter_modules(__path__):
        name = _module_info.name
        if name.startswith("_"):
            continue
        module = importlib.import_module(f".{name}", __name__)
        forward = None
        reversed_ = None
        for attr, value in vars(module).items():
            if attr.startswith("cmap_"):
                if attr.endswith("_r"):
                    reversed_ = value
                else:
                    forward = value
        if forward is None:
            continue
        globals()[name] = forward
        __all__.append(name)
        _register(forward, name)
        if reversed_ is not None:
            globals()[f"{name}_r"] = reversed_
            __all__.append(f"{name}_r")
            _register(reversed_, f"{name}_r")


_load_colormaps()
