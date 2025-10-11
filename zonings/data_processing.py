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
)
from zonings.models import Field, SField
from zonings.types import NDArray
from zonings.utils import no_numpy


def pnormalise_arrays(*arrays: NDArray, pad_value: float = 0) -> list[NDArray]:
    """
    pad a series of arrays so they are all the same shape. arrays must have the
    same dimension. the returned array's shape will be the dimension-wise
    maximum of the input arrays.
    """
    n_arrays = len(arrays)
    if any(
        len(arrays[0].shape) != len(arrays[i].shape) for i in range(n_arrays)
    ):
        raise ValueError(
            "Attempted to normalise arrays of different dimensions"
        )
    n_dims = len(arrays[0].shape)

    result = [a for a in arrays]
    for axis in range(len(arrays[0].shape)):
        target_size = max(a.shape[axis] for a in arrays)

        for i in range(n_arrays):
            a = result[i]
            if a.shape[axis] < target_size:
                result[i] = np.concat(
                    [
                        a,
                        np.full(
                            shape=[
                                target_size - a.shape[i]
                                if i == axis
                                else a.shape[i]
                                for i in range(n_dims)
                            ],
                            fill_value=pad_value,
                        ),
                    ],
                    axis=axis,
                )

    return result


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


def load_field(slug: str, merge_size: int, skip_init: bool = False) -> Field:
    yield_file_path = YIELD_FILE_PATH_FORMAT.format(slug=slug)
    protein_file_path = PROTEIN_FILE_PATH_FORMAT.format(slug=slug)

    try:
        with rasterio.open(yield_file_path) as yield_data:
            y_pixel_length: float = (
                geodesic(
                    (yield_data.bounds.top, yield_data.bounds.left),
                    (yield_data.bounds.top, yield_data.bounds.right),
                ).km
                / yield_data.width
            )
            yield_array: NDArray = yield_data.read(1)
            yield_array *= y_pixel_length**2 * KM2_TO_HA
    except (FileNotFoundError, RasterioIOError):
        raise FileNotFoundError(
            f"Couldn't find yield file (looked for {yield_file_path})"
        )

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
        raise FileNotFoundError(
            f"Couldn't find protein file (looked for {protein_file_path})"
        )

    if yield_array.shape != protein_array.shape:
        print(
            f"Yield and protein data for {slug} have different shapes. Normalising"
        )
        yield_array, protein_array = pnormalise_arrays(
            yield_array, protein_array
        )

    field_map = yield_array > 0.0001

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
        skip_init=skip_init,
        coordinates=(
            (yield_data.bounds.top + yield_data.bounds.bottom) / 2,
            (yield_data.bounds.left + yield_data.bounds.right) / 2,
        ),
    )


def field_to_sfield(
    field: Field, yield_error: float, gpc_error: float, num_scenarios: int
) -> SField:
    yield_maps = []
    gpc_maps = []
    field_mask = np.asarray(field.field_map)
    for _ in range(num_scenarios):
        y_map = np.add(
            field.yield_map,
            np.random.normal(0, yield_error, (field.height, field.width))
            * field_mask,
        )
        g_map = np.add(
            field.gpc_map,
            np.random.normal(0, gpc_error, (field.height, field.width))
            * field_mask,
        )
        yield_maps.append(no_numpy(y_map))
        gpc_maps.append(no_numpy(g_map))

    return SField(
        field_id=field.field_id,
        height=field.height,
        width=field.width,
        pixel_area=field.pixel_area,
        num_scenarios=num_scenarios,
        field_map=field.field_map,
        yield_maps=yield_maps,
        gpc_maps=gpc_maps,
        coordinates=field.coordinates,
    )


def load_sfield(
    field_slug: str,
    merge_size: int,
    yield_error: float,
    gpc_error: float,
    num_scenarios: int,
) -> SField:
    field = load_field(field_slug, merge_size, True)
    return field_to_sfield(field, yield_error, gpc_error, num_scenarios)
