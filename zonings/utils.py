from typing import Any, TypeVar

import numpy as np

from zonings.constants import Box, NDArray, Number

NumberT = TypeVar("NumberT", bound=Number)
SummableT = TypeVar("SummableT", bound=Number | NDArray)


def no_numpy(value) -> Any:
    return getattr(value, "tolist", lambda: value)()


def subsequence_sums(
    sequence: list[SummableT],
) -> dict[tuple[int, int], SummableT]:
    """calculates the sums of all possible subsequences of an array and returns a
    lookup dictionary where lookup[i, j] is the sum of elements between indices
    i and j inclusive"""
    seq_len = len(sequence)
    sums: list[NDArray] = [
        np.asarray(
            [sum(sequence[x] for x in range(x2 + 1)) for x2 in range(seq_len)]
        )
    ]
    prev_x1 = sequence[0]
    for x1 in range(1, seq_len):
        sums.append(sums[-1] - prev_x1)
        prev_x1 = sequence[x1]

    return {
        (x1, x2): no_numpy(sums[x1][x2])
        for x1 in range(seq_len)
        for x2 in range(x1, seq_len)
    }


def calculate_axis_sums(
    grid: list[list[NumberT]], rows=True
) -> dict[tuple[int, int, int], NumberT]:
    if not rows:
        grid = [
            [grid[j][i] for i in range(len(grid[0]))] for j in range(len(grid))
        ]

    lookup: dict[tuple[int, int, int], NumberT] = {}
    for i in range(len(grid)):
        sums = subsequence_sums(grid[i])
        lookup.update(((i, *k), sums[k]) for k in sums)  # type: ignore

    return lookup


def calculate_box_sums(grid: list[list[NumberT]]) -> dict[Box, NumberT]:
    (height, width) = (len(grid), len(grid[0]))
    row_sums = calculate_axis_sums(grid)

    row_sum_arrays = [
        np.asarray(
            list(
                row_sums.get((y, x1, x2), 0)
                for x1 in range(width)
                for x2 in range(width)
            )
        )
        for y in range(height)
    ]

    y_axis_lookup = subsequence_sums(row_sum_arrays)

    lookup = {
        ((x1, y1), (x2, y2)): y_axis_lookup[y1, y2][x1 * width + x2]
        for x1 in range(width)
        for y1 in range(height)
        for x2 in range(x1, width)
        for y2 in range(y1, height)
    }

    return lookup
