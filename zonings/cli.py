import csv
import io
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import sleep, time

import click
import numpy as np

from zonings.constants import DEFAULT_PRICING
from zonings.data_processing import load_field, load_sfield
from zonings.models import CGSolverConfig, Field, PriceInfo, ZoningConfig
from zonings.pipelines import (
    dynamic_pipeline,
    mip_pipeline,
    stochastic_dynamic_pipeline,
    stochastic_mip_pipeline,
)
from zonings.solvers import (
    DeterministicMIPSolver,
    DynamicSolver,
    StochasticCGMIPSolver,
    StochasticDynamicSolver,
    StochasticMipSolver,
    TurnAwareMIPSolver,
)
from zonings.visualisations import view_sfield_solution
from zonings.zoning import make_zones


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
@click.argument("solve_type", type=click.Choice(["mip", "dp", "smip", "sdp"]))
@click.argument("field_list", type=click.File())
@click.argument("output_dir", type=click.Path(writable=True, path_type=Path))
@click.option(
    "--alpha",
    "-a",
    type=float,
    default=0.2,
    help="alpha-cvar level. does nothing with non-stochastic solvers (default 0.2)",
)
@click.option(
    "--cvar-weight",
    "-w",
    type=float,
    default=0.5,
    help="weight given to cvar when optimising stochastic fields",
)
@click.option(
    "--num_zones",
    "-n",
    type=int,
    default=4,
    help="maximum number of zones in the solution (default 4)",
)
def solve_batch(
    solve_type: str,
    field_list: io.TextIOWrapper,
    output_dir: Path,
    alpha: float,
    cvar_weight: float,
    num_zones: int,
):
    """zones a list of fields"""
    for slug in field_list:
        if slug.startswith("#"):  # commented out
            continue
        if solve_type == "mip":
            mip_pipeline(slug.strip(), output_dir, nzones=num_zones)
        elif solve_type == "dp":
            dynamic_pipeline(slug.strip(), output_dir, nzones=num_zones)
        elif solve_type == "smip":
            stochastic_mip_pipeline(
                slug.strip(),
                output_dir,
                nzones=num_zones,
                alpha=alpha,
                cvar_weight=cvar_weight,
            )
        elif solve_type == "sdp":
            stochastic_dynamic_pipeline(
                slug.strip(),
                output_dir,
                nzones=num_zones,
                alpha=alpha,
                cvar_weight=cvar_weight,
            )


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


@cli.command(hidden=True)
def debug2():
    np.random.seed(2025)
    f = load_sfield(field_slug="cy2022_118", merge_size=2, num_scenarios=100)
    assert f is not None
    solver = StochasticDynamicSolver(
        field=f,
        max_zones=4,
        cvar_alpha=0.1,
        expectation_weight=0,
        config=ZoningConfig(4, 4, DEFAULT_PRICING),
    )

    sol, info = solver.solve()
    view_sfield_solution(f, sol)


@cli.command(hidden=True)
def debug3():
    out = open("data/grid_fields.txt", "w")
    with open("data/fields.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            field = load_field(line.strip(), 2)
            avg_gpc = (
                field.protein_box_sums[field.bounding_box()]
                / field.yield_box_sums[field.bounding_box()]
            )
            if avg_gpc < 0.14:
                out.write(line)
    out.close()


@cli.command()
@click.argument("field_list", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def grid_search(field_list: io.TextIOWrapper, output_file: io.TextIOWrapper):
    pricing = PriceInfo(
        [0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360]
    )
    fields: list[Field] = []
    for slug in field_list:
        if slug.startswith("#"):
            continue
        field = load_field(slug.strip(), 2)
        if (
            field.protein_box_sums[field.bounding_box()]
            / field.yield_box_sums[field.bounding_box()]
            >= pricing.protein_minimums[-1]
        ):
            print(f"Skipping {slug.strip()} because of average protein content")
            continue
        fields.append(field)
    print([f.field_id for f in fields])
    output_file.write(
        "field,average_gpc,min_zone_dim,max_zones,solution,benefit,solve_time,timed_out,zones_used\n"
    )
    line_format = "{id},{gpc:.4f},{min_dim},{max_zones},{sol:.2f},{benefit:.2f},{solve_time:.4f},{timed_out},{zones_used}\n"
    for min_dimension, max_zones in product(range(1, 20, 2), range(2, 7)):
        for field in fields:
            solver = DynamicSolver(
                field,
                max_zones,
                ZoningConfig(min_dimension, min_dimension, pricing),
                timeout=600,
            )
            solution, info = solver.solve()

            gpc = (
                field.protein_box_sums[field.bounding_box()]
                / field.yield_box_sums[field.bounding_box()]
            )
            benefit = solution.revenue - pricing.calculate_price(
                gpc, field.yield_box_sums[field.bounding_box()]
            )
            output_file.write(
                line_format.format(
                    id=field.field_id,
                    gpc=gpc,
                    min_dim=min_dimension,
                    max_zones=max_zones,
                    sol=solution.revenue,
                    benefit=benefit,
                    solve_time=info.total_solve_seconds,
                    timed_out=info.total_solve_seconds > 600,
                    zones_used=len(solution.zones),
                )
            )
            output_file.flush()


@cli.command(hidden=True)
def speed_test():
    fields = []

    with open("data/speed_fields.txt", "r") as file:
        for line in file:
            fields.append(load_field(line.strip(), 2))

    mip_times = defaultdict(list)
    dp_times = defaultdict(list)
    for n in range(1, 7):
        print(f"######## {n} zones")
        sleep(3)
        for f in fields:
            tic = time()
            zones = make_zones(f, ZoningConfig(3, 3, DEFAULT_PRICING))
            mip_sol, mip_info = DeterministicMIPSolver(
                zones, n, f, CGSolverConfig()
            ).solve()
            toc = time()
            mip_times[n].append(toc - tic)

            dp_sol, dp_info = DynamicSolver(
                f, n, ZoningConfig(3, 3, DEFAULT_PRICING), None
            ).solve()
            dp_times[n].append(dp_info.total_solve_seconds)

    for n in range(1, 7):
        print(
            f"{n},{sum(mip_times[n]) / len(mip_times[n])},{sum(dp_times[n]) / len(dp_times[n])}"
        )


@cli.command(hidden=True)
def quality_test():
    fields = []

    with open("data/quality_fields.txt", "r") as file:
        for line in file:
            np.random.seed(2025)
            fields.append(
                load_sfield(
                    field_slug=line.strip(),
                    merge_size=2,
                    num_scenarios=100,
                )
            )

    mip_results = []
    mip_times = []
    cg_mip_results = []
    cg_mip_times = []
    dp_results = []
    dp_times = []

    for f in fields:
        tic = time()
        mip_sol = StochasticMipSolver(
            make_zones(f, ZoningConfig(3, 3, DEFAULT_PRICING)), 4, 0.2, 0, f
        ).solve()
        toc = time()

        mip_times.append(toc - tic)
        mip_results.append(float(sum(sorted(mip_sol.revenue)[:20]) / 20))

        tic = time()
        cg_mip_sol, mip_info = StochasticCGMIPSolver(
            make_zones(f, ZoningConfig(3, 3, DEFAULT_PRICING)),
            4,
            0.2,
            0,
            f,
            CGSolverConfig(),
        ).solve()
        toc = time()
        cg_mip_times.append(toc - tic)
        cg_mip_results.append(float(sum(sorted(cg_mip_sol.revenue)[:20]) / 20))

        dp_sol, dp_info = StochasticDynamicSolver(
            field=f,
            max_zones=4,
            cvar_alpha=0.2,
            expectation_weight=0,
            config=ZoningConfig(3, 3, DEFAULT_PRICING),
        ).solve()

        dp_times.append(dp_info.total_solve_seconds)

        dp_results.append(float(sum(sorted(dp_sol.revenue)[:20]) / 20))

        if dp_results[-1] - mip_results[-1] > 0.0001:
            print("mip", cg_mip_results[-1])
            print("dp", dp_results[-1])

            view_sfield_solution(f, mip_sol)
            view_sfield_solution(f, dp_sol)

    print("MIP")
    print(mip_results)
    print(mip_times)
    print("CG MIP")
    print(cg_mip_results)
    print(cg_mip_times)
    print("Dynamic")
    print(dp_results)
    print(dp_times)

    print(
        "CG Gap",
        np.mean(
            list(
                (m - c) / m * 100 for (m, c) in zip(mip_results, cg_mip_results)
            )
        ),
    )
    print(
        "DP Gap",
        np.mean(
            list((m - d) / m * 100 for (m, d) in zip(mip_results, dp_results))
        ),
    )

@cli.command
@click.argument("field_slug")
@click.argument("output_file", type=click.Path(writable=True, dir_okay=False, path_type=Path))
@click.option(
    "--num-zones",
    "-n",
    type=int,
    default=4
)
def find_turn_pareto_frontier(field_slug: str, output_file: Path,num_zones: int) -> None:
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

    gpc = (
        field.protein_box_sums[field.bounding_box()]
        / field.yield_box_sums[field.bounding_box()]
    )

    base_revenue = pricing.calculate_price(
        gpc, field.yield_box_sums[field.bounding_box()]
    )

    field_turns = min(field.bounding_box().width, field.bounding_box().height)

    with open(output_file, "w+") as output:
        solver = TurnAwareMIPSolver(zones, num_zones, float("inf"), field, CGSolverConfig())
        sol, _ = solver.solve()
        turns = sum(z.turns for z in sol.zones)
        output.write(f"revenue,revenue_ratio,turns,turn_ratio,zones\n")
        output.write(f"{sol.revenue},{sol.revenue / base_revenue},{turns},{turns/field_turns},{len(sol.zones)}\n")
        output.flush()
        while len(sol.zones) > 1:
            solver = TurnAwareMIPSolver(zones, num_zones, turns-1, field, CGSolverConfig())
            sol, _ = solver.solve()
            turns = sum(z.turns for z in sol.zones)
            output.write(f"{sol.revenue},{sol.revenue / base_revenue},{turns},{(turns/field_turns)},{len(sol.zones)}\n")
            output.flush()

    print("field turns", field_turns)
    print("field gpc", gpc)
