from math import ceil

import numpy as np
import rasterio  # type: ignore
from geopy.distance import geodesic  # type: ignore
from rasterio.errors import RasterioIOError  # type: ignore

from zonings.constants import (
    KM2_TO_HA,
    MAP_PIXEL_TOL_KM,
    PROTEIN_FILE_PATH_FORMAT,
    YIELD_FILE_PATH_FORMAT,
    NDArray,
)
from zonings.models import Field
from zonings.utils import no_numpy


def _average_pixels(values: NDArray, mask: NDArray, merge_size: int):
    """
    averages merge_size x merge_size squares in the values array. the mask array
    controls what pixels are counted in the averaging
    """
    merged_shape = tuple(ceil(s / merge_size) for s in values.shape)
    pad_tuple = tuple(
        (0, merged_shape[i] * merge_size - values.shape[i])
        for i in range(len(merged_shape))
    )
    padded_values = np.pad(values, pad_tuple, mode="constant")
    padded_bounds = np.pad(mask, pad_tuple, mode="constant")
    sums = sum(
        (
            padded_values[i::merge_size, j::merge_size]
            for i in range(merge_size)
            for j in range(merge_size)
        ),
        start=np.zeros(merged_shape),
    )
    counts = sum(
        (
            padded_bounds[i::merge_size, j::merge_size]
            for i in range(merge_size)
            for j in range(merge_size)
        ),
        start=np.zeros(merged_shape),
    )

    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def load_field(slug: str, merge_size: int) -> Field:
    yield_file_path = YIELD_FILE_PATH_FORMAT.format(slug=slug)
    protein_file_path = PROTEIN_FILE_PATH_FORMAT.format(slug=slug)

    try:
        with rasterio.open(yield_file_path) as yield_data:
            y_pixel_length = (
                geodesic(
                    (yield_data.bounds.top, yield_data.bounds.left),
                    (yield_data.bounds.top, yield_data.bounds.right),
                ).km
                / yield_data.width
            )
            yield_array: NDArray = yield_data.read(1)
            yield_array *= y_pixel_length**2 * KM2_TO_HA
            field_map = yield_array > 0.001
    except (FileNotFoundError, RasterioIOError):
        print(f"Couldn't find yield file (looked for {yield_file_path})")
        quit()

    try:
        with rasterio.open(protein_file_path) as protein_data:
            p_pixel_length = (
                geodesic(
                    (protein_data.bounds.top, protein_data.bounds.left),
                    (protein_data.bounds.top, protein_data.bounds.right),
                ).km
                / protein_data.width
            )
            if abs(y_pixel_length - p_pixel_length) > MAP_PIXEL_TOL_KM:
                print(
                    f"Difference between yield and protein map pixel lengths is {abs(y_pixel_length - p_pixel_length)}"
                )
            protein_array: NDArray = protein_data.read(1)
    except (FileNotFoundError, RasterioIOError):
        print(f"Couldn't find protein file (looked for {protein_file_path})")
        quit()

    merged_yield = _average_pixels(yield_array, field_map, merge_size)
    merged_protein = _average_pixels(protein_array, field_map, merge_size)
    merged_pixel_size = y_pixel_length * merge_size

    return Field(
        field_id=slug,
        height=merged_yield.shape[0],
        width=merged_yield.shape[1],
        pixel_area=merged_pixel_size**2,
        field_map=no_numpy(merged_yield > 0.001),
        yield_map=no_numpy(merged_yield),
        gpc_map=no_numpy(merged_protein / 100),
        coordinates=(
            (yield_data.bounds.top + yield_data.bounds.bottom) / 2,
            (yield_data.bounds.left + yield_data.bounds.right) / 2,
        ),
    )
