import csv
from dataclasses import asdict
from math import floor, log10
from pathlib import Path
from typing import Any

import numpy as np

from zonings.constants import DEFAULT_PRICING
from zonings.data_processing import load_field, load_sfield
from zonings.models import (
    CGSolveInfo,
    CGSolverConfig,
    DeterministicSolution,
    DPSolveInfo,
    Field,
    PriceInfo,
    SField,
    StochasticSolution,
    SZone,
    Zone,
    ZoningConfig,
)
from zonings.solvers import (
    DeterministicMIPSolver,
    DynamicSolver,
    StochasticCGMIPSolver,
    StochasticDynamicSolver,
)
from zonings.zoning import make_zones


def _guess_good_solve_parameters(n_zones: int) -> CGSolverConfig:
    return CGSolverConfig(
        max_variables_added_per_cg_iteration=10 ** (floor(log10(n_zones)) - 3) * 5
    )


def output_results(
    field: Field | SField,
    solution: DeterministicSolution | StochasticSolution,
    pricing: PriceInfo,
    output_dir: Path,
    field_slug: str,
    solve_info: CGSolveInfo | DPSolveInfo | None = None,
) -> None:
    kpis = field.to_dict()

    # base revenue/s
    if isinstance(solution, StochasticSolution) and isinstance(field, SField):
        kpis |= {f"solution_{s}": solution.revenue[s] for s in range(field.num_scenarios)}
        kpis |= {
            f"base_{s}": score
            for s, score in enumerate(field.get_box_prices(field.bounding_box(), pricing))
        }

    elif isinstance(solution, DeterministicSolution) and isinstance(field, Field):
        kpis |= {"solution": solution.revenue}
        kpis |= {"base": field.get_box_price(field.bounding_box(), pricing)}
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
                    asdict(z.box) | {f"score_{s}": z.scores[s] for s in range(len(z.scores))}
                )
        writer = csv.DictWriter(file, zones_info[0].keys())
        writer.writeheader()
        writer.writerows(zones_info)


def mip_pipeline(field_slug: str, output_dir: Path, nzones: int = 4) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        field = load_field(field_slug, 2)
    except FileNotFoundError as e:
        print(e.args)
        return None
    pricing = PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360])
    zones = make_zones(
        field,
        ZoningConfig(3, 3, pricing, minimum_pixels=int(field.width * field.height * 0.1)),
    )

    mip = DeterministicMIPSolver(
        zones, nzones, field.width, field.height, _guess_good_solve_parameters(len(zones))
    )
    sol, info = mip.solve()

    # write outputs:
    output_results(field, sol, pricing, output_dir, field_slug, solve_info=info)


def stochastic_mip_pipeline(
    field_slug,
    output_dir: Path,
    nzones: int = 4,
    alpha: float = 0.2,
    cvar_weight: float = 0.5,
) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        np.random.seed(2025)
        field = load_sfield(field_slug=field_slug, merge_size=2, num_scenarios=50)
        if field is None:
            return
    except FileNotFoundError as e:
        print(e.args)
        return None
    zones = make_zones(field, ZoningConfig(3, 3, DEFAULT_PRICING))

    solver = StochasticCGMIPSolver(
        zones,
        nzones,
        alpha,
        1 - cvar_weight,
        field,
        _guess_good_solve_parameters(len(zones)),
    )

    sol, info = solver.solve()

    output_results(field, sol, DEFAULT_PRICING, output_dir, field_slug, info)


def dynamic_pipeline(field_slug: str, output_dir: Path, nzones: int = 4) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        field = load_field(field_slug, 2)
    except FileNotFoundError as e:
        print(e.args)
        return

    solver = DynamicSolver(
        field,
        nzones,
        ZoningConfig(3, 3, DEFAULT_PRICING),
        600,
    )
    sol, info = solver.solve()

    output_results(field, sol, DEFAULT_PRICING, output_dir, field_slug, info)


def stochastic_dynamic_pipeline(
    field_slug: str,
    output_dir: Path,
    nzones: int = 4,
    alpha: float = 0.2,
    cvar_weight: float = 0.5,
) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if (output_dir / f"{field_slug}_kpis.csv").exists():
        return

    try:
        np.random.seed(2025)
        field = load_sfield(field_slug=field_slug, merge_size=2, num_scenarios=100)
        if field is None:
            return
    except FileNotFoundError as e:
        print(e.args)
        return

    solver = StochasticDynamicSolver(
        field=field,
        max_zones=nzones,
        cvar_alpha=alpha,
        expectation_weight=1 - cvar_weight,
        config=ZoningConfig(3, 3, DEFAULT_PRICING),
        timeout=1200,
    )
    sol, info = solver.solve()

    output_results(field, sol, DEFAULT_PRICING, output_dir, field_slug, info)
