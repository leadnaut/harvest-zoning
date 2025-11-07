from typing import overload

from zonings.models import (
    Box,
    Field,
    PriceInfo,
    SField,
    SZone,
    Zone,
    ZoningConfig,
)


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


class SBlender:
    def __init__(self, field: SField, price_info: PriceInfo) -> None:
        self.field = field
        self.price_info = price_info

    def __call__(self, box: Box) -> SZone:
        return SZone(
            box,
            [
                self.price_info.price_box_in_sfield(box, self.field, s)
                for s in range(self.field.num_scenarios)
            ],
        )


@overload
def make_zones(field: Field, config: ZoningConfig) -> list[Zone]: ...


@overload
def make_zones(field: SField, config: ZoningConfig) -> list[SZone]: ...


def make_zones(
    field: Field | SField, config: ZoningConfig
) -> list[Zone] | list[SZone]:
    print("Creating Boxes")
    boxes = [
        Box(x1, y1, x2, y2)
        for x1 in range(field.width)
        for y1 in range(field.height)
        for x2 in range(x1, field.width)
        for y2 in range(y1, field.height)
        if (
            x2 - x1 + 1 >= config.minimum_width
            and y2 - y1 + 1 >= config.minimum_height
            and (
                config.minimum_pixels is None
                or field.field_box_sums[Box(x1, y1, x2, y2)]
                >= config.minimum_pixels
            )
        )
    ]
    nboxes = len(boxes)
    print(f"Starting Scoring {nboxes} Boxes")
    blender: Blender | SBlender
    if isinstance(field, Field):
        blender = Blender(field, config.pricing)
    else:
        blender = SBlender(field, config.pricing)
    zones: list[Zone] | list[SZone] = []
    for i, b in enumerate(boxes):
        zones.append(blender(b))  # type: ignore
        if i % 1000 == 0:
            print(f"{i / nboxes * 100:.2f}%", end="\r")

    return zones
