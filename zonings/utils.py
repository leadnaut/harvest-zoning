from typing import Any, TypeVar, cast

import numpy as np


def no_numpy(value) -> Any:
    return getattr(value, "tolist", lambda: value)()


SummableT = TypeVar("SummableT", bound=float | int | np.ndarray)
T = TypeVar("T", bound=float | int)
ListGrid = list[list[T]]


def subsequence_sums(
    seq: list[SummableT],
) -> np.ndarray:
    """calculates the sums of all possible subsequences of an array and returns
    a lookup array where `lookup[i * len(seq) + j]` is the sum of elements
    between indices i and j inclusive"""
    seq_len = len(seq)
    sums = [np.asarray([sum(seq[x] for x in range(x2 + 1)) for x2 in range(seq_len)])]
    prev_x1 = seq[0]
    for x1 in range(1, seq_len):
        sums.append(sums[-1] - prev_x1)
        prev_x1 = seq[x1]

    return np.concatenate(sums)


def cvar(alpha: float, seq: list[float | int]) -> float:
    number = int(alpha * len(seq))
    return sum(sorted(seq)[:number]) / number


def sum_list_grid(list_grid: ListGrid[T]) -> T:
    return cast(T, sum(sum(row) for row in list_grid))
