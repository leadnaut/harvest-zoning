from zonings.models import Field, ZoningConfig, Zone
from zonings.scoring import Blender
from multiprocessing import Pool


def make_zones(field: Field, config: ZoningConfig) -> list[Zone]:
    print("Creating Boxes")
    boxes = [
        (x1, y1, x2, y2)
        for x1 in range(field.width)
        for y1 in range(field.height)
        for x2 in range(x1 + config.minimum_width, field.width)
        for y2 in range(y1 + config.minimum_height, field.height)
        if (
            x2 - x1 >= config.minimum_width
            and y2 - y1 >= config.minimum_height
            and (
                config.minimum_pixels is None
                or field.pixels_in_box(x1, y1, x2, y2) >= config.minimum_pixels
            )
        )
    ]
    print("Starting Scoring")
    blender = Blender(field)
    with Pool(8) as pool:
        zones = pool.map(blender, boxes)

    return zones
    
