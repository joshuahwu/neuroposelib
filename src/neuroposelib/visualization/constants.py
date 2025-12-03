import matplotlib
import numpy as np
import matplotlib.colors as mcolors

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

_PLANE = {"x": 0, "y": 1, "z": 2}


CUSTOM_CMAPS = {}

cmap_colors = [
    (0.0, "#083006"),  # very dark green
    (0.4, "#55aa55"), # medium green
    (0.5, "white"),    # center = white
    (0.6, "#aa55aa"), # medium magenta
    (1.0, "#300630"),  # very dark magenta
]
darker_piyg = mcolors.LinearSegmentedColormap.from_list("darker_PiYG", cmap_colors)

cmap_colors = [
    (0.0, "#00008B"),  # dark blue
    (0.28, "#6495ED"),  # lighter blue
    (0.302, "white"),   # fade into white
    (0.698, "white"),   # extend white
    (0.72, "#F08080"),  # light red
    (1.0, "#8B0000"),  # dark red
]

wide_white_seismic = mcolors.LinearSegmentedColormap.from_list("wide_white_seismic", cmap_colors)

CUSTOM_CMAPS["darker_piyg"] = darker_piyg
CUSTOM_CMAPS["default_viridis"] = DEFAULT_VIRIDIS
CUSTOM_CMAPS["default_bone"] = DEFAULT_BONE
CUSTOM_CMAPS["wide_white_seismic"] = wide_white_seismic