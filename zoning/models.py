from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np

from zoning.utils import subsequence_sums


@dataclass(frozen=True)
class Field:
    height: int
    width: int
    field_map: list[list[int]]
    yield_maps: list[list[float]]
    gpc_maps: list[list[float]]

    @cached_property
    def field_row_sum(self) -> dict[tuple[int, int, int], int]:
        """
        field_row_sum[y,x1,x2] returns the number of field pixels on row y between
        x1 (inclusive) and x2 (inclusive).
        """

        lookup: dict[tuple[int, int, int], int] = {}
        for y in range(self.height):
            sums = subsequence_sums(self.field_map[y])
            lookup.update(((y, *k), sums[k]) for k in sums)
        return lookup

    def pixels_in_box(self, x1, y1, x2, y2) -> int:
        return sum(self.field_row_sum[y, x1, x2] for y in range(y1, y2))


@dataclass(frozen=True)
class Zone:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ZoningConfig:
    minimum_width: int
    minimum_height: int
    minimum_pixels: Optional[float]


def make_zones(field: Field, config: ZoningConfig) -> list[Zone]:
    boxes = [
        Zone(x1, y1, x2, y2)
        for x1 in range(field.width)
        for y1 in range(field.height)
        for x2 in range(x1 + 1, field.width)
        for y2 in range(y1 + 1, field.height)
        if (
            x2 - x1 >= config.minimum_width
            and y2 - y1 >= config.minimum_height
            and (
                config.minimum_pixels is None
                or field.pixels_in_box(x1, y1, x2, y2) >= config.minimum_pixels
            )
        )
    ]
    return boxes


if __name__ == "__main__":
    f = Field(100, 100, [[1 for _ in range(100)] for _ in range(100)], [], [])
    f.field_row_sum
