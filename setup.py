from setuptools import setup

# read the contents of the README file so that PyPI can use it as the long description
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name = 'degas',
      packages=['degas'],
      install_requires = ["numpy", "matplotlib"],
      # packages=find_packages(),
      package_dir={'degas': 'degas'},
      package_data={'degas': ['styles/*']},
      long_description=long_description,
      long_description_content_type='text/markdown'
)