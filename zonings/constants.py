import numpy as np

# Types

NDArray = np.ndarray
Number = float | int
Box = tuple[tuple[int, int], tuple[int, int]]

# String Formats
YIELD_FILE_PATH_FORMAT = "data/yield/{slug}_clipped.tif"
PROTEIN_FILE_PATH_FORMAT = "data/protein/Protein_P_{slug}.tif"

# Numerics
MAP_PIXEL_TOL_KM = 0.001  # 1 metre
KM2_TO_HA = 100
