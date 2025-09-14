import rasterio
from geopy.distance import geodesic  # type: ignore

from zonings.constants import (
    MAP_PIXEL_TOL_KM,
    PROTEIN_FILE_PATH_FORMAT,
    YIELD_FILE_PATH_FORMAT,
    NDArray,
)
from zonings.models import Field


def _merge_pixels(map: NDArray, bound_map: NDArray, edge_size: int):
    pass


def load_field(slug: str) -> Field:
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
            field_map = yield_array > 0.0001
    except FileNotFoundError:
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
            protein_array: NDArray = protein_data.read(1).tolist()
    except FileNotFoundError:
        print(f"Couldn't find protein file (looked for {protein_file_path})")
        quit()

    return Field(
        field_id=slug,
        height=yield_data.height,
        width=yield_data.width,
        field_map=field_map.tolist(),
        yield_map=yield_array.tolist(),
        gpc_map=protein_array.tolist(),
        coordinates=(
            (yield_data.bounds.top + yield_data.bounds.bottom) / 2,
            (yield_data.bounds.left + yield_data.bounds.right) / 2,
        ),
    )
