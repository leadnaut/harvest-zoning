import click

from zonings.data_processing import load_field
from zonings.utils import calculate_box_sums


@click.group
def cli():
    pass


@cli.command()
@click.argument("field_slug")
def read_field(field_slug):
    f = load_field(field_slug, 2)
    print(f.gpc_map)


@cli.command()
def debug():
    my_map = [[i * 5 + j for j in range(5)] for i in range(3)]

    box_sums = calculate_box_sums(my_map)
    print(box_sums[(3, 0), (4, 2)])
