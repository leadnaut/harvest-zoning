from dataclasses import dataclass, field
from typing import Generator, Generic, Optional, TypeVar

import numpy as np

from zonings.utils import Box, calculate_box_sums

T = TypeVar("T")
ListGrid = list[list[T]]


@dataclass
class Field:
    field_id: str
    height: int
    width: int
    pixel_area: float
    field_map: ListGrid[int]
    yield_map: ListGrid[float]
    gpc_map: ListGrid[float]
    coordinates: Optional[tuple[float, float]] = None

    field_box_sums: dict[Box, int] = field(init=False)
    yield_box_sums: dict[Box, float] = field(init=False)
    protein_box_sums: dict[Box, float] = field(init=False)

    def __post_init__(self) -> None:
        print(f"Intializing field {self.field_id}")
        self.field_box_sums = calculate_box_sums(self.field_map)
        self.yield_box_sums = calculate_box_sums(self.yield_map)
        self.protein_box_sums = calculate_box_sums(
            np.multiply(self.yield_map, self.gpc_map).tolist()
        )
        print(f"Finished intializing field {self.field_id}")

    def pixels_in_box(self, box: Box) -> int:
        return self.field_box_sums[box]

    def bounding_box(self) -> Box:
        return Box(0, 0, self.width - 1, self.height - 1)


@dataclass
class SField:
    field_id: str
    height: int
    width: int
    pixel_area: float
    num_scenarios: int
    field_map: ListGrid[int]
    yield_maps: list[ListGrid[float]]
    gpc_maps: list[ListGrid[float]]
    coordinates: Optional[tuple[float, float]] = None

    field_box_sums: dict[Box, int] = field(init=False)
    yield_box_sums: list[dict[Box, float]] = field(
        default_factory=list, init=False
    )
    protein_box_sums: list[dict[Box, float]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        print(f"Initializing stochastic field {self.field_id}")
        if not (
            self.num_scenarios == len(self.yield_maps) == len(self.gpc_maps)
        ):
            raise ValueError("Number of maps does not match scenarios")
        self.field_box_sums = calculate_box_sums(self.field_map)
        for s in range(self.num_scenarios):
            self.yield_box_sums.append(calculate_box_sums(self.yield_maps[s]))
            self.protein_box_sums.append(
                calculate_box_sums(
                    np.multiply(self.yield_maps[s], self.gpc_maps[s]).tolist()
                )
            )
        print(f"Finished initalizing stochastic field {self.field_id}")


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
class SZone:
    box: Box
    scores: list[float]

    def iter_contents(self) -> Generator[tuple[int, int], None, None]:
        for x in range(self.box.x1, self.box.x2 + 1):
            for y in range(self.box.y1, self.box.y2 + 1):
                yield (x, y)


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


Z = TypeVar("Z", bound=Zone | SZone)


@dataclass(frozen=True)
class Solution(Generic[Z]):
    zones: list[Z]
    revenue: float
    solve_info: Optional[SolveInfo] = None
