## degas

A collection of plotting utilities and helper functions for matplotlib.

## Installation and requirements

Dependencies
+ matplotlib
+ numpy

Install via pip

	pip install git+https://github.com/williamgilpin/degas



## Example usage

```python3
import numpy as np
from degas.degas import plot_err
   
x = np.linspace(0, 1, 100)
y = x**3
err = .05 * np.random.random(len(x))

plot_err(y, err, x=x)
```

## Styles

`degas` ships with a set of matplotlib style sheets in the `degas/styles` folder:

All styles use `vernalis` as the default colormap for images, and cycle through the degas
color schemes for line plots:

+ `default`: matplotlib's built-in defaults, plus the degas high-contrast color cycle
+ `degas_figure`: publication-style figures with Open Sans fonts, gray axes, and the degas high-contrast color cycle
+ `pastel_rainbow`: similar to `degas_figure`, with the 12-color degas pastel rainbow color cycle

Activate a style with `set_style`, which applies it to all subsequent plots:

```python3
import degas as dg

dg.set_style("degas_figure")
```

The styles are ordinary matplotlib style sheets, so they can also be used directly with
`plt.style.use`. The path to the styles folder is available as `dg.data_path`:

```python3
import os
import matplotlib.pyplot as plt
import degas as dg

plt.style.use(os.path.join(dg.data_path, "degas_figure.mplstyle"))

# or apply a style to a single figure only
with plt.style.context(os.path.join(dg.data_path, "pastel_rainbow.mplstyle")):
    plt.plot([0, 1], [0, 1])
```

Note: `degas_figure` and `pastel_rainbow` use the [Open Sans](https://fonts.google.com/specimen/Open+Sans)
font, which needs to be installed separately; matplotlib falls back to its default font if it
is not available.

## Colormaps

All colormaps in the `degas/colormaps` folder are available as top-level attributes, along
with reversed versions with an `_r` suffix. Importing `degas` also registers them with
matplotlib, so they can be passed by name as `cmap="vernalis"`. These include
[vernalis](https://gist.github.com/mwaskom/9f4c519e185637a4d0a7dd7bd900207e) as well as the
[Scientific colour maps](https://www.fabiocrameri.ch/colourmaps/) (batlow, roma, vik, etc.):

```python3
import numpy as np
import matplotlib.pyplot as plt
import degas as dg

data = np.random.random((32, 32))
plt.imshow(data, cmap=dg.vernalis)
plt.imshow(data, cmap=dg.batlow_r)
```
