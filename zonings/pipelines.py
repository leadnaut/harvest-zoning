import csv
import os
from math import floor, log10
from pathlib import Path

from zonings.data_processing import load_field
from zonings.models import (
    Field,
    MipConfig,
    PriceInfo,
    Solution,
    ZoningConfig,
)
from zonings.solvers import CGMipSolver, DynamicSolver
from zonings.zoning import make_zones


def _guess_good_solve_parameters(n_zones: int) -> MipConfig:
    return MipConfig(
        max_variables_added_per_cg_iteration=10 ** (floor(log10(n_zones)) - 3)
        * 5
    )


def write_outputs(
    field: Field,
    solution: Solution,
    pricing: PriceInfo,
    output_dir: Path,
    field_slug: str,
) -> None:
    whole_field = field.bounding_box()
    whole_yield = field.yield_box_sums[whole_field]
    whole_protein = field.protein_box_sums[whole_field]
    kpis = {
        "field_width": field.width,
        "field_height": field.height,
        "field_average_gpc": round(whole_protein / whole_yield, 4),
        "optimal_revenue": round(solution.revenue, 2),
        "zones_used": len(solution.zones),
        "base_revenue": round(
            pricing.calculate_price(whole_protein / whole_yield, whole_yield), 2
        ),
    }
    if solution.solve_info:
        kpis.update(
            {
                "solve_time": round(solution.solve_info.total_solve_seconds, 2),
                "cg_time": round(
                    solution.solve_info.column_generation_seconds, 2
                ),
                "cg_iters": solution.solve_info.column_generation_iterations,
                "total_variables": solution.solve_info.total_variables,
            }
        )
    if field.coordinates:
        kpis.update(
            {
                "field_lat": field.coordinates[0],
                "field_lon": field.coordinates[1],
            }
        )

    with open(output_dir / f"{field_slug}_kpis.csv", "w") as file:
        writer = csv.DictWriter(file, kpis.keys())
        writer.writeheader()
        writer.writerow(kpis)

    with open(output_dir / f"{field_slug}_zones.csv", "w") as file:
        writer = csv.DictWriter(file, ["x1", "y1", "x2", "y2", "score"])
        writer.writeheader()
        for z in solution.zones:
            writer.writerow(
                {
                    "x1": z.box.x1,
                    "y1": z.box.y1,
                    "x2": z.box.x2,
                    "y2": z.box.y2,
                    "score": z.score,
                }
            )


def mip_pipeline(field_slug: str, output_dir: Path) -> None:
    if not output_dir.exists():
        os.mkdir(output_dir)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        field = load_field(field_slug, 2)
    except FileNotFoundError as e:
        print(e.args)
        return None
    pricing = PriceInfo(
        [0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]
    )
    zones = make_zones(
        field,
        ZoningConfig(
            3, 3, pricing, minimum_pixels=int(field.width * field.height * 0.1)
        ),
    )

    mip = CGMipSolver(zones, 4, field, _guess_good_solve_parameters(len(zones)))
    sol = mip.solve()

    # write outputs:
    write_outputs(field, sol, pricing, output_dir, field_slug)


def dynamic_pipeline(field_slug: str, output_dir: Path) -> None:
    if not output_dir.exists():
        os.mkdir(output_dir)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        field = load_field(field_slug, 2)
    except FileNotFoundError as e:
        print(e.args)
        return
    pricing = PriceInfo(
        [0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]
    )

    solver = DynamicSolver(
        field,
        5,
        ZoningConfig(3, 3, pricing, int(field.width * field.height * 0.1)),
        600,
    )
    sol = solver.solve()

    write_outputs(field, sol, pricing, output_dir, field_slug)
