from importlib.resources import files

from .degas import *

# Path to the styles directory (Python 3.9+)
data_path = str(files(__package__) / "styles")
