from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from zonings.types import NDArray, Number

NumberT = TypeVar("NumberT", bound=Number)
SummableT = TypeVar("SummableT", bound=Number | NDArray)


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return self.x2 - self.x1 + 1

    def height(self) -> int:
        return self.y2 - self.y1 + 1

    def split(
        self, x: Optional[int] = None, y: Optional[int] = None
    ) -> tuple["Box", "Box"]:
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


def no_numpy(value) -> Any:
    return getattr(value, "tolist", lambda: value)()


@dataclass(frozen=True)
class BoxDataLookup(Generic[NumberT]):
    """data should contain a value per possible box such that the value for box
    x1,y1,x2,y2 is at `data[y1 * max_y + y2][x1 * max_x + x2]`. data can contain
    junk data for invalid boxes (i.e. if x2 < x1)."""

    data: NDArray
    max_x: int
    max_y: int

    def __getitem__(self, b: Box) -> NumberT:
        return self.data[b.y1 * self.max_y + b.y2][b.x1 * self.max_x + b.x2]


def subsequence_sums(
    seq: list[SummableT],
) -> NDArray:
    """calculates the sums of all possible subsequences of an array and returns
    a lookup array where `lookup[i * len(seq) + j]` is the sum of elements
    between indices i and j inclusive"""
    seq_len = len(seq)
    sums = [
        np.asarray(
            [sum(seq[x] for x in range(x2 + 1)) for x2 in range(seq_len)]
        )
    ]
    prev_x1 = seq[0]
    for x1 in range(1, seq_len):
        sums.append(sums[-1] - prev_x1)
        prev_x1 = seq[x1]

    return np.concatenate(sums)


def calculate_box_sums(grid: list[list[NumberT]]) -> BoxDataLookup[NumberT]:
    row_sums = [subsequence_sums(row) for row in grid]

    return BoxDataLookup(
        subsequence_sums(row_sums), max_x=len(grid[0]), max_y=len(grid)
    )
