from importlib.resources import files

from .degas import *
from .colormaps import *
from .colormaps.vernalis import cmap_vn
from .colormaps.vernalis import cmap_vn_r

# Path to the styles directory (Python 3.9+)
data_path = str(files(__package__) / "styles")
