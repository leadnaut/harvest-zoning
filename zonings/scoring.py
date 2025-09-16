from zonings.constants import Box
from zonings.models import Field, PriceInfo, Zone


class Blender:
    def __init__(self, field: Field, price_info: PriceInfo) -> None:
        self.field = field
        self.price_info = price_info

    def __call__(self, box: Box) -> Zone:
        total_yield = self.field.yield_box_sums[box]
        total_protein = self.field.protein_box_sums[box]
        return Zone(
            box,
            self.price_info.calculate_price(
                total_protein / total_yield, total_yield
            ),
        )
