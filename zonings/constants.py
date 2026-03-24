from zonings.models import PriceInfo

# String Formats
YIELD_FILE_PATH_FORMAT = "data/yield/{slug}_clipped.tif"
PROTEIN_FILE_PATH_FORMAT = "data/protein/Protein_P_{slug}.tif"

# Numerics
MAP_PIXEL_TOL_KM = 0.001  # 1 metre
KM2_TO_HA = 100

DEFAULT_PRICING = PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360])

YIELD_ERROR_TONNES_PER_HA = 0.3
GPC_ERROR = 0.0056
