from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Generator, Generic, Optional, TypeVar

import numpy as np

from zonings.utils import subsequence_sums

T = TypeVar("T")
ListGrid = list[list[T]]


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @cached_property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @cached_property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @cached_property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2 + 1) / 2, (self.y1 + self.y2 + 1) / 2

    def split(self, x: Optional[int] = None, y: Optional[int] = None) -> tuple["Box", "Box"]:
        if x is None and y is None:
            raise ValueError("have to split somewhere")
        if x is not None and self.x1 <= x < self.x2:
            return (
                Box(self.x1, self.y1, x, self.y2),
                Box(x + 1, self.y1, self.x2, self.y2),
            )
        if y is not None and self.y1 <= y < self.y2:
            return (
                Box(self.x1, self.y1, self.x2, y),
                Box(self.x1, y + 1, self.x2, self.y2),
            )
        raise ValueError("Cut out of bounds")

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Box):
            raise NotImplementedError()
        return (self.x1, self.y1, self.x2, self.y2) < (
            other.x1,
            other.y1,
            other.x2,
            other.y2,
        )


NumberT = TypeVar("NumberT", bound=float | int)


@dataclass(frozen=True)
class BoxDataLookup(Generic[NumberT]):
    """data should contain a value per possible box such that the value for box
    x1,y1,x2,y2 is at `data[y1 * max_y + y2][x1 * max_x + x2]`. data can contain
    junk data for invalid boxes (i.e. if x2 < x1)."""

    data: np.ndarray
    max_x: int
    max_y: int

    def __getitem__(self, b: Box) -> NumberT:
        return self.data[b.y1 * self.max_y + b.y2][b.x1 * self.max_x + b.x2]

    @classmethod
    def from_grid(cls, grid: ListGrid[NumberT]) -> "BoxDataLookup[NumberT]":
        row_sums = [subsequence_sums(row) for row in grid]

        return BoxDataLookup(subsequence_sums(row_sums), max_x=len(grid[0]), max_y=len(grid))


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

    def __add__(self, other: Any) -> "Point":
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, int | float):
            return Point(self.x + other, self.y + other)
        raise NotImplementedError

    __radd__ = __add__

    def __mul__(self, other: Any) -> "Point":
        if isinstance(other, int | float):
            return Point(self.x * other, self.y * other)
        raise NotImplementedError

    __rmul__ = __mul__

    def __sub__(self, other: Any) -> "Point":
        return self + (-other)

    __rsub__ = __sub__


@dataclass(frozen=True)
class Quadrilateral:
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

    def point_on_top_border(self, t: float) -> Point:
        return self.top_left + t * (self.top_right - self.top_left)

    def point_on_left_border(self, t: float) -> Point:
        return self.top_left + t * (self.bottom_left - self.top_left)

    def point_on_bottom_border(self, t: float) -> Point:
        return self.bottom_left + t * (self.bottom_right - self.bottom_left)

    def point_on_right_border(self, t: float) -> Point:
        return self.top_right + t * (self.bottom_right - self.top_right)

    def inscribed_box(self) -> Box:
        return Box(
            x1=round(max(self.top_left.x, self.bottom_left.x)),
            y1=round(max(self.top_left.y, self.top_right.y)),
            x2=round(min(self.top_right.x, self.bottom_right.x)),
            y2=round(min(self.bottom_left.y, self.bottom_right.y)),
        )


@dataclass(frozen=True)
class PriceInfo:
    protein_minimums: list[float]
    price_per_tonnes: list[float]

    @cached_property
    def _reversed_lookup(self) -> list[tuple[float, float]]:
        return list(zip(reversed(self.protein_minimums), reversed(self.price_per_tonnes)))

    def calculate_price(self, gpc: float, yield_tonnes: float) -> float:
        if gpc < 0:
            raise ValueError("negative gpc")
        for protein, price in self._reversed_lookup:
            if gpc > protein:
                return price * yield_tonnes
        return 0


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
    skip_init: bool = False

    field_box_sums: BoxDataLookup[int] = field(init=False)
    yield_box_sums: BoxDataLookup[float] = field(init=False)
    protein_box_sums: BoxDataLookup[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.skip_init:
            return
        print(f"Intializing field {self.field_id}")
        self.field_box_sums = BoxDataLookup.from_grid(self.field_map)
        self.yield_box_sums = BoxDataLookup.from_grid(self.yield_map)
        self.protein_box_sums = BoxDataLookup.from_grid(
            np.multiply(self.yield_map, self.gpc_map).tolist()
        )
        print(f"Finished intializing field {self.field_id}")

    def pixels_in_box(self, box: Box) -> int:
        return self.field_box_sums[box]

    def bounding_box(self) -> Box:
        return Box(0, 0, self.width - 1, self.height - 1)

    def to_dict(self) -> dict[str, Any]:
        return (
            {
                k: v
                for (k, v) in asdict(self).items()
                if k in {"field_id", "height", "width", "pixel_area"}
            }
            | {
                "gpc": self.protein_box_sums[self.bounding_box()]
                / self.yield_box_sums[self.bounding_box()]
            }
            | (
                {}
                if self.coordinates is None
                else {"lat": self.coordinates[0], "lon": self.coordinates[1]}
            )
        )

    def get_box_price(self, box: Box, pricer: PriceInfo) -> float:
        box_yield = self.yield_box_sums[box]
        if box_yield < 0.001:
            return 0
        box_gpc = self.protein_box_sums[box] / box_yield
        return pricer.calculate_price(box_gpc, box_yield)


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

    field_box_sums: BoxDataLookup[int] = field(init=False)
    yield_box_sums: list[BoxDataLookup[float]] = field(default_factory=list, init=False)
    protein_box_sums: list[BoxDataLookup[float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        print(f"Initializing stochastic field {self.field_id}")
        if not (self.num_scenarios == len(self.yield_maps) == len(self.gpc_maps)):
            raise ValueError("Number of maps does not match scenarios")
        self.field_box_sums = BoxDataLookup.from_grid(self.field_map)
        for s in range(self.num_scenarios):
            self.yield_box_sums.append(BoxDataLookup.from_grid(self.yield_maps[s]))
            self.protein_box_sums.append(
                BoxDataLookup.from_grid(np.multiply(self.yield_maps[s], self.gpc_maps[s]).tolist())
            )
        print(f"Finished initalizing stochastic field {self.field_id}")

    def bounding_box(self) -> Box:
        return Box(0, 0, self.width - 1, self.height - 1)

    def to_dict(self) -> dict[str, Any]:
        return (
            {
                k: v
                for (k, v) in self.__dict__.items()
                if k in {"field_id", "height", "width", "pixel_area"}
            }
            | {
                f"gpc_{s}": self.protein_box_sums[s][self.bounding_box()]
                / self.yield_box_sums[s][self.bounding_box()]
                for s in range(self.num_scenarios)
            }
            | (
                {}
                if self.coordinates is None
                else {"lat": self.coordinates[0], "lon": self.coordinates[1]}
            )
        )

    def __hash__(self) -> int:
        return hash(self.field_id)

    def get_box_prices(self, box: Box, pricer: PriceInfo) -> list[float]:
        prices = []
        for s in range(self.num_scenarios):
            box_yield = self.yield_box_sums[s][box]
            if box_yield < 0.001:
                prices.append(0.0)
            else:
                box_gpc = self.protein_box_sums[s][box] / box_yield
                prices.append(pricer.calculate_price(box_gpc, box_yield))

        return prices


@dataclass
class ScenarioMap:
    yield_map: ListGrid[float]
    gpc_map: ListGrid[float]

    yield_box_sums: BoxDataLookup[float] = field(init=False)
    protein_box_sums: BoxDataLookup[float] = field(init=False)

    def __post_init__(self) -> None:
        self.yield_box_sums = BoxDataLookup.from_grid(self.yield_map)
        self.protein_box_sums = BoxDataLookup.from_grid(self.gpc_map)

    def get_box_price(self, box: Box, pricer: PriceInfo) -> float:
        box_yield = self.yield_box_sums[box]
        if box_yield < 0.001:
            return 0
        box_gpc = self.protein_box_sums[box] / box_yield
        return pricer.calculate_price(box_gpc, box_yield)


@dataclass(frozen=True)
class Zone:
    box: Box
    score: float

    def iter_contents(self) -> Generator[tuple[int, int], None, None]:
        for x in range(self.box.x1, self.box.x2 + 1):
            for y in range(self.box.y1, self.box.y2 + 1):
                yield (x, y)

    @cached_property
    def turns(self) -> int:
        return min(self.box.height, self.box.width)

    def __str__(self) -> str:
        return f"Zone((x1, y1)={(self.box.x1, self.box.y2)}, (x2,y2)={(self.box.x2, self.box.y2)}, score={round(self.score, 2)})"


@dataclass(frozen=True)
class DeterministicSolution:
    zones: list[Zone]
    revenue: float


@dataclass(frozen=True)
class SZone:
    box: Box
    scores: list[float]

    def iter_contents(self) -> Generator[tuple[int, int], None, None]:
        for x in range(self.box.x1, self.box.x2 + 1):
            for y in range(self.box.y1, self.box.y2 + 1):
                yield (x, y)

    def __hash__(self) -> int:
        return hash(self.box)


@dataclass(frozen=True)
class StochasticSolution:
    zones: list[SZone]
    revenue: list[float]


@dataclass(frozen=True)
class ZoningConfig:
    minimum_width: int
    minimum_height: int
    pricing: PriceInfo
    minimum_pixels: Optional[float] = None


@dataclass(frozen=True)
class CGSolverConfig:
    max_cg_iterations: Optional[int] = None
    max_variables_added_per_cg_iteration: int = 500


@dataclass(frozen=True)
class CGSolveInfo:
    total_solve_seconds: float
    column_generation_seconds: float
    column_generation_iterations: int
    total_variables: int


@dataclass(frozen=True)
class DPSolveInfo:
    total_solve_seconds: float
    lookup_size: int
    lookup_hits: int
