## degas

A minimal collection of plotting utilities for matplotlib.

## Installation and requirements

Dependencies
+ matplotlib
+ numpy

Install via pip

	pip install git+git://github.com/williamgilpin/dysts



## Example usage

```python3
import numpy as np
from degas.degas import plot_err
   
x = np.linspace(0, 1, 100)
y = x**3
err = .05*np.random.random(len(x))

plot_err(y, err, x=x)
```
