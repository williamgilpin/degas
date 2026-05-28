from importlib.resources import files

from .degas import *
from .vernalis import cmap_vn
from .vernalis import cmap_vn as vernalis

# Path to the styles directory (Python 3.9+)
data_path = str(files(__package__) / "styles")
