import click

from zonings.data_processing import load_field
from zonings.visualisations import view_map


@click.group
def cli():
    pass


@cli.command()
@click.argument("field_slug")
def read_field(field_slug):
    f = load_field(field_slug)
    print(f.gpc_map)
