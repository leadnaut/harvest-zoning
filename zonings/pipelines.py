import csv
import os
from pathlib import Path

from zonings.data_processing import load_field
from zonings.models import PriceInfo, SolverConfig, ZoningConfig
from zonings.solver import ZoneSolver
from zonings.zoning import make_zones


def zone_field_pipeline(field_slug: str, output_dir: Path) -> None:
    if not output_dir.exists():
        os.mkdir(output_dir)

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
            3,
            3,
            pricing,
        ),
    )

    mip = ZoneSolver(field_slug, zones, field, SolverConfig(4))
    sol = mip.solve()

    # write outputs:
    whole_field = ((0, 0), (field.width - 1, field.height - 1))
    whole_yield = field.yield_box_sums[whole_field]
    whole_protein = field.protein_box_sums[whole_field]
    kpis = {
        "field_width": field.width,
        "field_height": field.height,
        "field_average_gpc": round(whole_protein / whole_yield, 4),
        "optimal_revenue": round(sol.revenue, 2),
        "zones_used": len(sol.zones),
        "base_revenue": round(
            pricing.calculate_price(whole_protein / whole_yield, whole_yield), 2
        ),
    }
    if sol.solve_info:
        kpis.update(
            {
                "solve_time": round(sol.solve_info.total_solve_seconds, 2),
                "cg_time": round(sol.solve_info.column_generation_seconds, 2),
                "cg_iters": sol.solve_info.column_generation_iterations,
                "total_variables": sol.solve_info.total_variables,
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
        for z in sol.zones:
            writer.writerow(
                {
                    "x1": z.box[0][0],
                    "y1": z.box[0][1],
                    "x2": z.box[1][0],
                    "y2": z.box[1][1],
                    "score": z.score,
                }
            )
