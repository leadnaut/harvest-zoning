import click

from zonings.data_processing import load_field
from zonings.models import PriceInfo, SolverConfig, ZoningConfig
from zonings.solver import ZoneSolver
from zonings.utils import calculate_box_sums
from zonings.zoning import make_zones


@click.group
def cli():
    pass


@cli.command()
@click.argument("field_slug")
def read_field(field_slug: str):
    f = load_field(field_slug, 2)


@cli.command()
@click.argument("field_slug")
def zone_field(field_slug: str):
    f = load_field(field_slug, 2)
    whole_box = ((0, 0), (f.width - 1, f.height - 1))
    print(f.protein_box_sums[whole_box] / f.yield_box_sums[whole_box])
    zs = make_zones(
        f,
        ZoningConfig(
            3,
            3,
            PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]),
        ),
    )
    print(len(zs))
    mip = ZoneSolver(field_slug, zs, f, SolverConfig(max_zones=4))
    sol = mip.solve()
    print(sol.zones)


@cli.command()
def debug():
    my_map = [[i * 5 + j for j in range(5)] for i in range(3)]

    box_sums = calculate_box_sums(my_map)
    print(box_sums[(3, 0), (4, 2)])
