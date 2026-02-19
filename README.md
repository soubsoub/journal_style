# journal_style

Matplotlib style utilities for producing publication-ready figures formatted to the specifications of major scientific journals.

## Supported journals

| Key        | Journals                                      |
|------------|-----------------------------------------------|
| `nature`   | Nature, Nature Methods, Nature Communications |
| `elsevier` | Cell, Lancet, most Elsevier journals          |
| `acs`      | ACS Nano, JACS, Analytical Chemistry          |
| `ieee`     | IEEE Transactions, IEEE Access                |
| `plos`     | PLOS ONE, PLOS Biology                        |

## Quick start in Google Colab

Add this cell at the top of your notebook:

```python
# Download the module directly from GitHub
import urllib.request
url = "https://raw.githubusercontent.com/YOUR_USERNAME/journal_style/main/journal_style.py"
urllib.request.urlretrieve(url, "journal_style.py")
```

Then import and use:

```python
import matplotlib.pyplot as plt
import numpy as np
from journal_style import journal_style

x = np.linspace(0, 2 * np.pi, 100)

with journal_style("nature", mode="color") as palette:
    fig, ax = plt.subplots(figsize=palette.figsize_single)
    for i, (col, ls, mk) in enumerate(palette.cycle(3)):
        ax.plot(x, np.sin(x + i * 0.5), color=col, linestyle=ls, marker=mk, markevery=20, label=f"Series {i+1}")
    ax.set_xlabel("x (rad)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()
```

## Installation (local)

```bash
pip install matplotlib
# Optional: pip install SciencePlots
```

Copy `journal_style.py` into your project directory or notebook folder.

## API summary

```python
from journal_style import set_color_style, set_grayscale_style, reset_style, journal_style

# Direct call
palette = set_color_style(journal="nature")

# Context manager (recommended for notebooks - restores rcParams on exit)
with journal_style("nature", mode="color") as palette:
    fig, ax = plt.subplots(figsize=palette.figsize_single)
    ax.plot(x, y, color=palette.colors[0])

# Grayscale (for print / BW journals)
with journal_style("ieee", mode="grayscale") as palette:
    fig, ax = plt.subplots(figsize=palette.figsize_double)

# Reset to matplotlib defaults
reset_style()
```

### Palette attributes

| Attribute        | Description                                      |
|------------------|--------------------------------------------------|
| `colors`         | List of hex color strings                        |
| `linestyles`     | `['-', '--', ':', '-.']`                         |
| `markers`        | `['o', 's', '^', 'D']`                           |
| `hatches`        | `['/', '\\', 'x', '.']`                          |
| `fill_alpha`     | Default alpha for shaded areas                   |
| `figsize_single` | `(width, height)` for single-column figures      |
| `figsize_double` | `(width, height)` for double-column figures      |
| `journal`        | Journal preset name                              |
| `mode`           | `'color'` or `'grayscale'`                       |

### palette.cycle(n)

Convenience generator yielding `(color, linestyle, marker)` tuples for multi-series plots:

```python
for i, (col, ls, mk) in enumerate(palette.cycle(4)):
    ax.plot(x, y[i], color=col, linestyle=ls, marker=mk)
```

## Dependencies

- **Required:** `matplotlib`
- **Optional:** `SciencePlots` (`pip install SciencePlots`)
