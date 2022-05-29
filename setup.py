from setuptools import setup, find_packages

setup(name = 'degas',
      packages=['degas'],
      # packages=find_packages(),
      package_dir={'degas': 'degas'},
      package_data={'degas': ['styles/*']},
      )