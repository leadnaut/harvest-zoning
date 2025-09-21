from dataclasses import dataclass, field
from functools import cached_property
from typing import Generator, Optional

import numpy as np

from zonings.utils import Box, calculate_box_sums


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
        print(f"Intializing field {self.field_id}")
        self.field_box_sums = calculate_box_sums(self.field_map)
        self.yield_box_sums = calculate_box_sums(self.yield_map)
        self.protein_box_sums = calculate_box_sums(
            np.multiply(self.yield_map, self.gpc_map).tolist()
        )
        print(f"Finished intializing field {self.field_id}")

    @cached_property
    def _protein_total_map(self) -> list[list[float]]:
        return (np.asarray(self.yield_map) * np.asarray(self.gpc_map)).tolist()

    def pixels_in_box(self, box: Box) -> int:
        return self.field_box_sums[box]

    def bounding_box(self) -> Box:
        return Box(0, 0, self.width - 1, self.height - 1)


@dataclass(frozen=True)
class Zone:
    box: Box
    score: float

    def iter_contents(self) -> Generator[tuple[int, int], None, None]:
        for x in range(self.box.x1, self.box.x2 + 1):
            for y in range(self.box.y1, self.box.y2 + 1):
                yield (x, y)

    def __str__(self) -> str:
        return f"Zone((x1, y1)={(self.box.x1, self.box.y2)}, (x2,y2)={(self.box.x2, self.box.y2)}, score={round(self.score, 2)})"


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


@dataclass(frozen=True)
class MipConfig:
    max_cg_iterations: Optional[int] = None
    max_variables_added_per_cg_iteration: int = 500


@dataclass(frozen=True)
class SolveInfo:
    total_solve_seconds: float
    column_generation_seconds: float
    column_generation_iterations: int
    total_variables: int


@dataclass(frozen=True)
class Solution:
    zones: list[Zone]
    revenue: float
    solve_info: Optional[SolveInfo] = None
