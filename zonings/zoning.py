from multiprocessing import Pool

from zonings.models import Field, Zone, ZoningConfig
from zonings.scoring import Blender


def make_zones(field: Field, config: ZoningConfig) -> list[Zone]:
    print("Creating Boxes")
    boxes = [
        (x1, y1, x2, y2)
        for x1 in range(field.width)
        for y1 in range(field.height)
        for x2 in range(x1 + config.minimum_width-1, field.width)
        for y2 in range(y1 + config.minimum_height-1, field.height)
        if (
            x2 - x1 >= config.minimum_width
            and y2 - y1 >= config.minimum_height
            and (
                config.minimum_pixels is None
                or field.pixels_in_box(x1, y1, x2, y2) >= config.minimum_pixels
            )
        )
    ]
    nboxes = len(boxes)
    print(f"Starting Scoring {nboxes} Boxes")
    blender = Blender(field, config.pricing)
    with Pool(8) as pool:
        zpool = pool.imap_unordered(blender, boxes, chunksize=nboxes//8)
        for i, _ in enumerate(zpool):
            if i % 100 != 0:
                continue
            print(f"done {i/nboxes * 100 :.2f}%", end="\r")
        print('')
        zones = list(zpool)

    return zones
