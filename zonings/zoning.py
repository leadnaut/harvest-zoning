from multiprocessing import Pool

from zonings.constants import Box
from zonings.models import Field, PriceInfo, Zone, ZoningConfig


class Blender:
    def __init__(self, field: Field, price_info: PriceInfo) -> None:
        self.field = field
        self.price_info = price_info

    def __call__(self, box: Box) -> Zone:
        total_yield = self.field.yield_box_sums[box]
        total_protein = self.field.protein_box_sums[box]
        return Zone(
            box,
            self.price_info.calculate_price(
                total_protein / total_yield, total_yield
            )
            if total_yield > 0.0001
            else 0,
        )


def make_zones(field: Field, config: ZoningConfig) -> list[Zone]:
    print("Creating Boxes")
    boxes = [
        ((x1, y1), (x2, y2))
        for x1 in range(field.width)
        for y1 in range(field.height)
        for x2 in range(x1 + config.minimum_width - 1, field.width)
        for y2 in range(y1 + config.minimum_height - 1, field.height)
        if (
            x2 - x1 >= config.minimum_width
            and y2 - y1 >= config.minimum_height
            and (
                config.minimum_pixels is None
                or field.pixels_in_box(((x1, y1), (x2, y2)))
                >= config.minimum_pixels
            )
        )
    ]
    nboxes = len(boxes)
    print(f"Starting Scoring {nboxes} Boxes")
    blender = Blender(field, config.pricing)
    zones = []
    with Pool(8) as pool:
        zpool = pool.imap_unordered(blender, boxes, chunksize=nboxes // 8)
        for i, z in enumerate(zpool):
            zones.append(z)
            if i % 100 != 0:
                continue
            print(f"Progress: {i / nboxes * 100:.2f}%", end="\r")
        print("\nDone!")

    return zones
