import csv
import os
from dataclasses import asdict
from math import floor, log10
from pathlib import Path
from typing import Any

import numpy as np

from zonings.data_processing import field_to_sfield, load_field
from zonings.models import (
    CGSolveInfo,
    Field,
    MipConfig,
    PriceInfo,
    SField,
    Solution,
    SZone,
    Zone,
    ZoningConfig,
)
from zonings.solvers import CGMipSolver, CVarDynamicSolver, DynamicSolver
from zonings.zoning import make_zones


def _guess_good_solve_parameters(n_zones: int) -> MipConfig:
    return MipConfig(
        max_variables_added_per_cg_iteration=10 ** (floor(log10(n_zones)) - 3)
        * 5
    )


def output_results(
    field: Field | SField,
    solution: Solution[Zone] | Solution[SZone],
    pricing: PriceInfo,
    output_dir: Path,
    field_slug: str,
    solve_info: CGSolveInfo | None = None,
) -> None:
    kpis = field.to_dict()

    # base revenue/s
    if isinstance(solution.revenue, list) and isinstance(field, SField):
        kpis |= {
            f"solution_{s}": solution.revenue[s]
            for s in range(field.num_scenarios)
        }
        kpis |= {
            f"base_{s}": pricing.price_box_in_sfield(
                field.bounding_box(), field, s
            )
            for s in range(field.num_scenarios)
        }

    elif isinstance(solution.revenue, float) and isinstance(field, Field):
        kpis |= {"solution": solution.revenue}
        kpis |= {
            "base": pricing.price_box_in_field(field.bounding_box(), field)
        }
    else:
        raise TypeError()

    if solve_info is not None:
        kpis |= asdict(solve_info)

    with open(output_dir / f"{field_slug}_kpis.csv", "w") as file:
        writer = csv.DictWriter(file, kpis.keys())
        writer.writeheader()
        writer.writerow(kpis)

    with open(output_dir / f"{field_slug}_zones.csv", "w") as file:
        zones_info: list[dict[str, Any]] = []
        for z in solution.zones:
            if isinstance(z, Zone):
                zones_info.append(asdict(z.box) | {"score": z.score})
            elif isinstance(z, SZone):
                zones_info.append(
                    asdict(z.box)
                    | {f"score_{s}": z.scores[s] for s in range(len(z.scores))}
                )
        writer = csv.DictWriter(file, zones_info[0].keys())
        writer.writeheader()
        writer.writerows(zones_info)


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
    sol, info = mip.solve()

    # write outputs:
    output_results(field, sol, pricing, output_dir, field_slug, solve_info=info)


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

    output_results(field, sol, pricing, output_dir, field_slug)


def sdynamic_pipeline(field_slug: str, output_dir: Path) -> None:
    np.random.seed(2025)
    try:
        field = field_to_sfield(load_field(field_slug, 2), 0.56, 0.4, 50)
    except FileNotFoundError as e:
        print(e.args)
        return

    pricing = PriceInfo(
        [0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]
    )

    solver = CVarDynamicSolver(field, 4, ZoningConfig(3, 3, pricing))
    sol = solver.solve(0.2)

    print(sol.zones)
