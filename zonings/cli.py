import io
from pathlib import Path

import click

from zonings.data_processing import load_field
from zonings.models import PriceInfo, ZoningConfig
from zonings.pipelines import dynamic_pipeline, mip_pipeline
from zonings.solvers import DynamicSolver


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
def mip_solve_field(field_slug: str, output: Path):
    mip_pipeline(field_slug, output)


@cli.command()
@click.argument("field_slug")
@click.argument("output", type=click.Path(writable=True, path_type=Path))
def dynamic_solve_field(field_slug: str, output: Path):
    dynamic_pipeline(field_slug, output)


@cli.command()
@click.argument("solve_type", type=click.Choice(["mip", "dynamic"]))
@click.argument("field_list", type=click.File())
@click.argument("output_dir", type=click.Path(writable=True, path_type=Path))
def solve_batch(
    solve_type: str, field_list: io.TextIOWrapper, output_dir: Path
):
    for slug in field_list:
        if slug.startswith("#"):  # commented out
            continue
        if solve_type == "mip":
            mip_pipeline(slug.strip(), output_dir)
        elif solve_type == "dynamic":
            dynamic_pipeline(slug.strip(), output_dir)


@cli.command()
def debug():
    field = load_field("cy2022_30", 2)
    solver = DynamicSolver(
        field,
        max_zones=4,
        config=ZoningConfig(
            3,
            3,
            PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]),
        ),
    )

    print(solver.solve())
