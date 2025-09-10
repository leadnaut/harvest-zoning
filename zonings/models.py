from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np

from zonings.utils import subsequence_sums, calculate_axis_sums



class Field:
    def __init__(self, field_id: str, height: int, width: int, field_map: list[list[int]], yield_map: list[list[float]], gpc_map: list[list[float]]) -> None:
        self.field_id = field_id
        self.height = height
        self.width = width
        self.field_map = field_map
        self.yield_map = yield_map
        self.gpc_map = gpc_map
        
        print(f"intializing field {field_id}")
        self.field_row_sums = calculate_axis_sums(self.field_map)
        self.field_col_sums = calculate_axis_sums(self.field_map, rows=False)
        self.yield_row_sums = calculate_axis_sums(self.yield_map)
        self.yield_col_sums = calculate_axis_sums(self.yield_map, rows=False)
        self.protein_row_sums = calculate_axis_sums(np.asarray(self.yield_map) * np.asarray(self.gpc_map).tolist())
        print(f"finished intializing field {field_id}")

    @cached_property
    def _protein_total_map(self) -> list[list[float]]:
        return (np.asarray(self.yield_map) * np.asarray(self.gpc_map)).tolist()

    def pixels_in_box(self, x1, y1, x2, y2) -> int:
        return sum(self.field_row_sums[y, x1, x2] for y in range(y1, y2+1))


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
