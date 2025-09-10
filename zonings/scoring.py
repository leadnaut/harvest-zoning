from zonings.models import Field, PriceInfo, Zone


class Blender:
    def __init__(self, field: Field, price_info: PriceInfo) -> None:
        self.field = field

    def __call__(self, box: tuple[int, int, int, int]) -> Zone:
        score = 0.0
        return Zone(*box, score)
