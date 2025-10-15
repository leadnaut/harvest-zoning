import csv
import io
from pathlib import Path

import click

from zonings.pipelines import (
    dynamic_pipeline,
    mip_pipeline,
    stochastic_dynamic_pipeline,
    stochastic_mip_pipeline,
)


@click.group
def cli():
    """Zones fields using different algorithms"""
    pass


@cli.command()
@click.argument("solve_type", type=click.Choice(["mip", "dp", "smip", "sdp"]))
@click.argument("field_slug")
@click.argument("output_dir", type=click.Path(writable=True, path_type=Path))
@click.option(
    "--alpha",
    "-a",
    type=float,
    default=0.2,
    help="alpha-cvar level. does nothing with non-stochastic solvers",
)
@click.option(
    "--num_zones",
    "-n",
    type=int,
    default=4,
    help="maximum number of zones in the solution",
)
def solve(
    solve_type: str,
    field_slug: str,
    output_dir: Path,
    alpha: float,
    num_zones: int,
):
    """zones a single field"""
    match solve_type:
        case "mip":
            mip_pipeline(field_slug, output_dir, nzones=num_zones)
        case "dp":
            dynamic_pipeline(field_slug, output_dir, nzones=num_zones)
        case "smip":
            stochastic_mip_pipeline(
                field_slug, output_dir, alpha=alpha, nzones=num_zones
            )
        case "sdp":
            stochastic_dynamic_pipeline(
                field_slug, output_dir, alpha=alpha, nzones=num_zones
            )
        case _:
            raise NotImplementedError("Unreachable")


@cli.command()
@click.argument("solve_type", type=click.Choice(["mip", "dp", "smip"]))
@click.argument("field_list", type=click.File())
@click.argument("output_dir", type=click.Path(writable=True, path_type=Path))
def solve_batch(
    solve_type: str, field_list: io.TextIOWrapper, output_dir: Path
):
    """zones a lit of fields"""
    for slug in field_list:
        if slug.startswith("#"):  # commented out
            continue
        if solve_type == "mip":
            mip_pipeline(slug.strip(), output_dir)
        elif solve_type == "dp":
            dynamic_pipeline(slug.strip(), output_dir)
        elif solve_type == "smip":
            stochastic_mip_pipeline(slug.strip(), output_dir)


@cli.command(hidden=True)
@click.argument("field_slug")
@click.argument("alpha", type=float)
def debug(field_slug: str, alpha: float):
    with open(f"data/outs_smip/{field_slug}_kpis.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        sol_scores = [
            float(v) for k, v in row.items() if k.startswith("solution")
        ]
        base_scores = [float(v) for k, v in row.items() if k.startswith("base")]

        cvar_scenarios = int(alpha * len(sol_scores))
        print(
            "solution cvar:",
            sum(sorted(sol_scores)[:cvar_scenarios]) / cvar_scenarios,
        )
        print(
            "baseline_cvar:",
            sum(sorted(base_scores)[:cvar_scenarios]) / cvar_scenarios,
        )


# @cli.command()
# @click.argument("field_list", type=click.File("r"))
# @click.argument("output_file", type=click.File("w"))
# def grid_search(field_list: io.TextIOWrapper, output_file: io.TextIOWrapper):
#     pricing = PriceInfo(
#         [0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]
#     )
#     fields: list[Field] = []
#     for slug in field_list:
#         if slug.startswith("#"):
#             continue
#         field = load_field(slug.strip(), 2)
#         if (
#             field.protein_box_sums[field.bounding_box()]
#             / field.yield_box_sums[field.bounding_box()]
#             > pricing.protein_minimums[-1]
#         ):
#             print(f"Skipping {slug.strip()} because of average protein content")
#             continue
#         fields.append(field)
#     print([f.field_id for f in fields])
#     output_file.write(
#         "field,average_gpc,min_zone_dim,max_zones,solution,benefit,solve_time,timed_out,zones_used\n"
#     )
#     line_format = "{id},{gpc:.4f},{min_dim},{max_zones},{sol:.2f},{benefit:.2f},{solve_time:.4f},{timed_out},{zones_used}\n"
#     for min_dimension, max_zones in product(range(1, 20, 2), range(2, 7)):
#         for field in fields:
#             solver = DynamicSolver(
#                 field,
#                 max_zones,
#                 ZoningConfig(min_dimension, min_dimension, pricing),
#                 timeout=600,
#             )
#             solution = solver.solve()

#             gpc = (
#                 field.protein_box_sums[field.bounding_box()]
#                 / field.yield_box_sums[field.bounding_box()]
#             )
#             benefit = solution.revenue - pricing.calculate_price(
#                 gpc, field.yield_box_sums[field.bounding_box()]
#             )
#             assert solution.solve_info
#             output_file.write(
#                 line_format.format(
#                     id=field.field_id,
#                     gpc=gpc,
#                     min_dim=min_dimension,
#                     max_zones=max_zones,
#                     sol=solution.revenue,
#                     benefit=benefit,
#                     solve_time=solution.solve_info.total_solve_seconds,
#                     timed_out=solution.solve_info.total_solve_seconds > 600,
#                     zones_used=len(solution.zones),
#                 )
#             )
#             output_file.flush()
