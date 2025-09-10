from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np

from zonings.utils import subsequence_sums


@dataclass(frozen=True)
class Field:
    height: int
    width: int
    field_map: list[list[int]]
    yield_map: list[list[float]]
    gpc_map: list[list[float]]

    def __post_init__(self) -> None:
        print("Initializing Field")
        self.field_row_sum
        self.field_col_sum
        print("Finished Initializing Field")

    @cached_property
    def field_row_sum(self) -> dict[tuple[int, int, int], int]:
        """
        field_row_sum[y,x1,x2] returns the number of field pixels on row y between
        x1 (inclusive) and x2 (inclusive).
        """
        lookup = {}
        for y in range(self.height):
            sums = subsequence_sums(self.field_map[y])
            lookup.update(((y, *k), sums[k]) for k in sums)
        return lookup

    @cached_property
    def field_col_sum(self) -> dict[tuple[int, int, int], int]:
        field_transpose = np.asarray(self.field_map).transpose().tolist()
        lookup = {}
        for x in range(self.width):
            sums = subsequence_sums(field_transpose[x])
            lookup.update(((x, *k), sums[k]) for k in sums)
        return lookup

    def pixels_in_box(self, x1, y1, x2, y2) -> int:
        return sum(self.field_row_sum[y, x1, x2] for y in range(y1, y2))


@dataclass(frozen=True)
class Zone:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass(frozen=True)
class PriceInfo:
    protein_minimums: list[float]
    price_per_tonnes: list[float]

    def calculate_price(self, gpc: float, yield_tonnes: float) -> float:
        for (protein, price) in zip(
            self.protein_minimums[::-1], self.price_per_tonnes[::-1]
        ):
            if gpc > protein:
                return price * yield_tonnes
        return 0


@dataclass(frozen=True)
class ZoningConfig:
    minimum_width: int
    minimum_height: int
    pricing: PriceInfo
    minimum_pixels: Optional[float] = None
