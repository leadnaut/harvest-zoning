from random import random

from zonings.models import Field, PriceInfo, ZoningConfig
from zonings.zoning import make_zones
from zonings.data_processing import load_field

if __name__ == "__main__":
    load_field("cy2022_3")
    # f = Field(
    #     "tester",
    #     100,
    #     100,
    #     [[1] * 100] * 100,
    #     [[random() * 10 for i in range(100)] for j in range(100)],
    #     [[random() for i in range(100)] for j in range(100)],
    # )
    # zones = make_zones(f, ZoningConfig(20, 20, PriceInfo([], [])))
