from zonings.models import Field, PriceInfo, Zone


class Blender:
    def __init__(self, field: Field, price_info: PriceInfo) -> None:
        self.field = field
        self.price_info = price_info

    def __call__(self, box: tuple[int, int, int, int]) -> Zone:
        x1, y1, x2, y2 = box
        total_yield = sum(
            self.field.yield_row_sums[y, x1, x2] for y in range(y1, y2 + 1)
        )
        total_protein = sum(
            self.field.protein_row_sums[y, x1, x2] for y in range(y1, y2 + 1)
        )
        return Zone(
            *box,
            self.price_info.calculate_price(
                total_protein / total_yield, total_yield
            ),
        )
