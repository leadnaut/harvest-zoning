import rasterio
from geopy.distance import geodesic # type: ignore
from matplotlib.pyplot import imshow, show

from zonings.models import Field


def load_field(slug: str) -> Field:
    with rasterio.open(f"data/yield/{slug}_clipped.tif") as yield_data:
        pixel_length = (
            geodesic(
                (yield_data.bounds.top, yield_data.bounds.left),
                (yield_data.bounds.top, yield_data.bounds.right),
            ).km
            / yield_data.width
        )
        pixel_area = pixel_length * pixel_length
        print(yield_data.tags())
        yield_map = yield_data.read(1).tolist()
        field_map = map(lambda row: list(map(lambda i: i > 0, row)), yield_map)

    return Field(
        field_id=slug,
        height=yield_data.height,
        width=yield_data.width,
        field_map=list(field_map),
        yield_map=yield_map,
        gpc_map=yield_map,  # TODO: FIX
    )
