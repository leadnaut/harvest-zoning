from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np

from zonings.constants import Box
from zonings.utils import calculate_box_sums


@dataclass
class Field:
    field_id: str
    height: int
    width: int
    pixel_area: float
    field_map: list[list[int]]
    yield_map: list[list[float]]
    gpc_map: list[list[float]]
    coordinates: Optional[tuple[float, float]] = None

    field_box_sums: dict[Box, int] = field(init=False)
    yield_box_sums: dict[Box, float] = field(init=False)
    protein_box_sums: dict[Box, float] = field(init=False)

    def __post_init__(self):
        print(f"intializing field {self.field_id}")
        self.field_box_sums = calculate_box_sums(self.field_map)
        self.yield_box_sums = calculate_box_sums(self.yield_map)
        self.protein_box_sums = calculate_box_sums(
            np.multiply(self.yield_map, self.gpc_map).tolist()
        )
        print(f"finished intializing field {self.field_id}")

    @cached_property
    def _protein_total_map(self) -> list[list[float]]:
        return (np.asarray(self.yield_map) * np.asarray(self.gpc_map)).tolist()

    def pixels_in_box(self, box: Box) -> int:
        return self.field_box_sums[box]


@dataclass(frozen=True)
class Zone:
    box: Box
    score: float


@dataclass(frozen=True)
class PriceInfo:
    protein_minimums: list[float]
    price_per_tonnes: list[float]

    def calculate_price(self, gpc: float, yield_tonnes: float) -> float:
        for protein, price in zip(
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
