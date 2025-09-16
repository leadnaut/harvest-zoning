import io
from pathlib import Path

import click

from zonings.data_processing import load_field
from zonings.pipelines import zone_field_pipeline
from zonings.utils import calculate_box_sums


@click.group
def cli():
    pass


@cli.command()
@click.argument("field_slug")
def read_field(field_slug: str):
    f = load_field(field_slug, 2)


@cli.command()
@click.argument("field_slug")
@click.argument("output", type=click.Path(writable=True, path_type=Path))
def zone_field(field_slug: str, output: Path):
    zone_field_pipeline(field_slug, output)


@cli.command()
@click.argument("field_list", type=click.File())
@click.argument(
    "output_dir", type=click.Path(exists=True, writable=True, path_type=Path)
)
def zone_batch(field_list: io.TextIOWrapper, output_dir: Path):
    for slug in field_list:
        zone_field_pipeline(slug.strip(), output_dir)


@cli.command()
def debug():
    my_map = [[i * 5 + j for j in range(5)] for i in range(3)]

    box_sums = calculate_box_sums(my_map)
    print(box_sums[(3, 0), (4, 2)])
