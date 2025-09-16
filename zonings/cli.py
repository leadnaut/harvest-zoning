import click

from zonings.data_processing import load_field
from zonings.models import PriceInfo, ZoningConfig
from zonings.utils import calculate_box_sums
from zonings.zoning import make_zones


@click.group
def cli():
    pass


@cli.command()
@click.argument("field_slug")
def read_field(field_slug):
    f = load_field(field_slug, 2)


@cli.command()
@click.argument("field")
def zone_field(field):
    f = load_field(field, 2)
    zs = make_zones(
        f,
        ZoningConfig(
            3, 3, PriceInfo([0, 10.5, 11.5, 13, 14], [200, 325, 355, 360])
        ),
    )


@cli.command()
def debug():
    my_map = [[i * 5 + j for j in range(5)] for i in range(3)]

    box_sums = calculate_box_sums(my_map)
    print(box_sums[(3, 0), (4, 2)])
