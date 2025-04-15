import matplotlib
import numpy as np

PALETTE = [
    (1, 0.5, 0),
    (0.5, 0.5, 0.85),
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 0.9),
    (0, 1, 1),
    (0.4, 0.4, 0.4),
    (0.5, 0.85, 0.5),
    (0.5, 0.15, 0.5),
    (0.15, 0.5, 0.5),
    (0.5, 0.5, 0.15),
    (0.9, 0.9, 0),
    (1, 0, 1),
    (0, 0.5, 1),
    (0.85, 0.5, 0.5),
    (0.5, 1, 0),
    (0.5, 0, 1),
    (1, 0, 0.5),
    (0, 0.9, 0.6),
    (0.3, 0.6, 0),
    (0, 0.3, 0.6),
    (0.6, 0.3, 0),
    (0.3, 0, 0.6),
    (0, 0.6, 0.3),
    (0.6, 0, 0.3),
]

DEFAULT_VIRIDIS = matplotlib.cm.get_cmap("viridis")
DEFAULT_VIRIDIS.set_under("white")
EPS = 0.99e-6

DEFAULT_BONE = matplotlib.colors.ListedColormap(
    matplotlib.colormaps["bone_r"](np.linspace(0.05, 0.4, 256))
)
DEFAULT_BONE.set_under("white")
