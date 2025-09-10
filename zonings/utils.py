from typing import TypeVar

import numpy as np

Summable = TypeVar("Summable", float, int)


def subsequence_sums(sequence: list[Summable]) -> dict[tuple[int, int], Summable]:
    """ calculates the sums of all possible subsequences of an array and returns a
    lookup dictionary where lookup[i, j] is the sum of elements between indices
    i and j inclusive"""
    seq_len = len(sequence)
    sums: list[np.ndarray] = [
        np.asarray([sum(sequence[x] for x in range(x2 + 1)) for x2 in range(seq_len)])
    ]
    prev_x1 = sequence[0]
    for x1 in range(1, seq_len):
        sums.append(sums[-1] - prev_x1)
        prev_x1 = sequence[x1]

    return {
        (x1, x2): sums[x1][x2].item()
        for x1 in range(seq_len)
        for x2 in range(x1, seq_len)
    }


def calculate_axis_sums(grid: list[list[Summable]], rows=True) -> dict[tuple[int, int, int], Summable]:
    if not rows:
        grid = np.asarray(grid).transpose().tolist()
    
    lookup = {}
    for i in range(len(grid)):
        sums = subsequence_sums(grid[i])
        lookup.update(((i, *k), sums[k]) for k in sums)
    return lookup
