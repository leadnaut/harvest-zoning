import csv
import io
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import sleep, time

import click
import numpy as np
from matplotlib import pyplot as plt

from zonings.constants import DEFAULT_PRICING, GPC_ERROR, KM2_TO_HA, YIELD_ERROR_TONNES_PER_HA
from zonings.data_processing import field_to_sfield, load_field, load_sfield
from zonings.models import (
    CGSolverConfig,
    Field,
    PriceInfo,
    ScenarioMap,
    ZoningConfig,
)
from zonings.pipelines import (
    _guess_good_solve_parameters,
    dynamic_pipeline,
    mip_pipeline,
    stochastic_dynamic_pipeline,
    stochastic_mip_pipeline,
)
from zonings.solution_testing import score_solution_on_scenarios
from zonings.solvers import (
    DeterministicMIPSolver,
    DynamicSolver,
    StochasticCGMIPSolver,
    StochasticDynamicSolver,
    StochasticMipSolver,
    TurnAwareMIPSolver,
)
from zonings.utils import cvar, sum_list_grid
from zonings.visualisations import view_field_solution, view_sfield_solution
from zonings.zoning import flatten_szones, make_zones


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
            stochastic_mip_pipeline(field_slug, output_dir, alpha=alpha, nzones=num_zones)
        case "sdp":
            stochastic_dynamic_pipeline(field_slug, output_dir, alpha=alpha, nzones=num_zones)
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
        sol_scores = [float(v) for k, v in row.items() if k.startswith("solution")]
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
    f = load_field(slug="cy2022_29", merge_size=2)
    zones = make_zones(f, ZoningConfig(4, 4, pricing=DEFAULT_PRICING))
    solver = TurnAwareMIPSolver(
        field=f,
        zones=zones,
        max_zones=4,
        max_turns=16,
        config=CGSolverConfig(),
    )

    sol, info = solver.solve()
    ax = view_field_solution(f, sol)
    plt.show()


@cli.command(hidden=True)
def debug3():
    out = open("data/high_benefit_fields.txt", "w")
    with open("data/fields.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            field = load_field(line.strip(), 2)
            avg_gpc = (
                field.protein_box_sums[field.bounding_box()]
                / field.yield_box_sums[field.bounding_box()]
            )
            if 0.1275 < avg_gpc < 0.13:
                out.write(line)
    out.close()


@cli.command()
@click.argument("field_list", type=click.File("r"))
@click.argument("output_path", type=click.Path(dir_okay=False, writable=True, path_type=Path))
def grid_search(field_list: io.TextIOWrapper, output_path: Path):
    pricing = PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360])
    fields: list[Field] = []

    pre_calced = set()
    if output_path.exists():
        with open(output_path, "r") as output_file:
            pre_calced_count = 0
            line_count = 0
            for line in csv.DictReader(output_file):
                line_count += 1
                if float(line["benefit"]) >= 0 and line["timed_out"] == "False":
                    pre_calced.add(
                        (str(line["field"]), int(line["min_zone_dim"]), int(line["max_zones"]))
                    )
                    pre_calced_count += 1
        print(f"{pre_calced_count} / {line_count} pre-calced")
        output_file = open(output_path, "a")
    else:
        output_file = open(output_path, "x")
        output_file.write(
            "field,average_gpc,min_zone_dim,max_zones,solution,benefit,solve_time,timed_out,zones_used\n"
        )

    for slug in field_list:
        if slug.startswith("#"):
            continue
        field = load_field(slug.strip(), 2)
        fields.append(field)
    print([f.field_id for f in fields])

    line_format = "{id},{gpc:.4f},{min_dim},{max_zones},{sol:.2f},{benefit:.2f},{solve_time:.4f},{timed_out},{zones_used}\n"
    timeout = 1200
    for min_dimension, max_zones, field in product(range(1, 20, 3), range(2, 7), fields):
        if (field.field_id, min_dimension, max_zones) in pre_calced:
            continue
        solver = DynamicSolver(
            field, max_zones, ZoningConfig(min_dimension, min_dimension, pricing), timeout=timeout
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
                timed_out=info.total_solve_seconds > timeout,
                zones_used=len(solution.zones),
            )
        )
        output_file.flush()
    output_file.close()


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
            mip_sol, mip_info = DeterministicMIPSolver(zones, n, f, CGSolverConfig()).solve()
            toc = time()
            mip_times[n].append(toc - tic)

            dp_sol, dp_info = DynamicSolver(f, n, ZoningConfig(3, 3, DEFAULT_PRICING), None).solve()
            dp_times[n].append(dp_info.total_solve_seconds)

    for n in range(1, 7):
        print(f"{n},{sum(mip_times[n]) / len(mip_times[n])},{sum(dp_times[n]) / len(dp_times[n])}")


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
        np.mean(list((m - c) / m * 100 for (m, c) in zip(mip_results, cg_mip_results))),
    )
    print(
        "DP Gap",
        np.mean(list((m - d) / m * 100 for (m, d) in zip(mip_results, dp_results))),
    )


@cli.command
@click.argument("field-list", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "output-dir",
    type=click.Path(exists=True, writable=True, file_okay=False, path_type=Path),
)
@click.option("--num-zones", "-n", type=int, default=4)
def find_turn_pareto_frontier(field_list: Path, output_dir: Path, num_zones: int) -> None:
    fields: list[Field] = []
    with open(field_list, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields.append(
                load_field(
                    slug=line.strip(),
                    merge_size=2,
                )
            )

    pricing = PriceInfo([0, 0.105, 0.115, 0.13, 0.14], [200, 325, 330, 355, 360])
    for field in fields:
        fpath = output_dir / f"{field.field_id}.csv"
        if fpath.exists():
            continue

        zones = make_zones(
            field,
            ZoningConfig(3, 3, pricing, minimum_pixels=int(field.width * field.height * 0.1)),
        )

        gpc = (
            field.protein_box_sums[field.bounding_box()]
            / field.yield_box_sums[field.bounding_box()]
        )

        base_revenue = pricing.calculate_price(gpc, field.yield_box_sums[field.bounding_box()])

        field_turns = min(field.bounding_box().width, field.bounding_box().height)

        with open(fpath, "w+") as output:
            output.write("revenue,revenue_ratio,turns,turn_ratio,zones\n")
            max_turns = None
            while True:
                solver = TurnAwareMIPSolver(
                    zones, num_zones, max_turns, field, _guess_good_solve_parameters(len(zones))
                )
                sol, _ = solver.solve()
                ax = view_field_solution(field, sol)
                ax.set_title(f"Field {field.field_id}, Max Turns = {max_turns}")
                plt.savefig(output_dir / f"{field.field_id}_{max_turns}.pdf")
                plt.close()

                turns = sum(z.turns for z in sol.zones)

                output.write(
                    f"{sol.revenue},{sol.revenue / base_revenue},{turns},{turns / field_turns},{len(sol.zones)}\n"
                )
                output.flush()
                max_turns = turns - 1
                if sol.revenue < base_revenue:
                    break

        print("field turns", field_turns)
        print("field gpc", gpc)


@cli.command
@click.argument("field-list", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def flatten_zone_test(field_list: Path, output: Path):
    N_SCENARIOS = 200
    ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with open(field_list, "r") as field_file:
        slugs = [line.strip() for line in field_file if not line.startswith("#")]

    if output.exists():
        pre_exists = True
        output_file = open(output, "r")
        pre_calced = {
            (str(line["slug"]), float(line["alpha"])) for line in csv.DictReader(output_file)
        }
        output_file.close()
    else:
        pre_exists = False
        pre_calced = set()

    if pre_exists:
        output_file = open(output, "a")
    else:
        output_file = open(output, "x")

    writer = csv.DictWriter(
        output_file,
        [
            "slug",
            "alpha",
            "optimal_objective",
            "flattened_objective",
            "optimal_time",
            "flattened_time",
        ],
    )
    if not pre_exists:
        writer.writeheader()

    for slug in slugs:
        if all((slug, alpha) in pre_calced for alpha in ALPHAS):
            continue
        field = load_field(slug, merge_size=2)
        sfield = field_to_sfield(field, YIELD_ERROR_TONNES_PER_HA, GPC_ERROR, N_SCENARIOS)
        scenarios = [
            ScenarioMap(yields, gpcs) for yields, gpcs in zip(sfield.yield_maps, sfield.gpc_maps)
        ]
        szones = make_zones(
            sfield, ZoningConfig(minimum_width=3, minimum_height=3, pricing=DEFAULT_PRICING)
        )
        for alpha in ALPHAS:
            if (slug, alpha) in pre_calced:
                continue

            flat_zones = flatten_szones(
                szones, lambda scores: 0.5 * cvar(alpha, scores) + 0.5 * sum(scores) / len(scores)
            )

            sto_sol, sto_info = StochasticMipSolver(
                szones,
                max_zones=4,
                alpha=alpha,
                expectation_weight=0.5,
                field=sfield,
            ).solve()

            det_sol, det_info = DeterministicMIPSolver(
                flat_zones,
                max_zones=4,
                field=field,
                config=_guess_good_solve_parameters(len(flat_zones)),
            ).solve()

            det_revenues = score_solution_on_scenarios(det_sol, scenarios)

            writer.writerow(
                {
                    "slug": slug,
                    "alpha": alpha,
                    "optimal_objective": cvar(alpha, sto_sol.revenue),
                    "flattened_objective": cvar(alpha, det_revenues),
                    "optimal_time": sto_info.total_solve_seconds,
                    "flattened_time": det_info.total_solve_seconds,
                }
            )
            output_file.flush()
    output_file.close()


@cli.command()
@click.argument("field_list", type=click.Path(path_type=Path))
def field_stats(field_list: Path) -> None:
    with open(field_list, "r") as field_file:
        slugs = [line.strip() for line in field_file if not line.startswith("#")]

    field_area_pixelss = []
    field_area_has = []
    gpcs = []
    for slug in slugs:
        field = load_field(slug, merge_size=2, skip_init=True)
        short_slug = slug.removeprefix("cy2022_")

        width_pix, height_pix = field.width, field.height
        width_km, height_km = map(lambda x: x * field.pixel_size_km, (width_pix, height_pix))

        field_area_pixels = sum_list_grid(field.field_map)
        field_area_ha = field_area_pixels * field.pixel_area * KM2_TO_HA
        if field_area_ha > 300:
            continue

        field_area_has.append(field_area_ha)
        field_area_pixelss.append(field_area_pixels)

        total_yield = sum_list_grid(field.yield_map)
        average_gpc_percent = sum_list_grid(field.gpc_map) / field_area_pixels * 100
        gpcs.append(average_gpc_percent)

        print(
            f"{short_slug} & ${width_pix}\\times{height_pix}$ & ${width_km:.2f}\\km\\times{height_km:.2f}\\km$ & {field_area_pixels} & {field_area_ha:.2f}ha & {total_yield:.2f}t & {average_gpc_percent:.2f}\\%\\\\"
        )

    areas_and_slugs = list(zip(field_area_has, slugs))
    data_s_s = set(s for a, s in areas_and_slugs if a < 50)
    data_s_m = set(s for a, s in areas_and_slugs if 50 <= a < 125)
    data_s_l = set(s for a, s in areas_and_slugs if 125 <= a)

    gpcs_and_slugs = list(zip(gpcs, slugs))
    data_q_l = set(s for p, s in gpcs_and_slugs if p < 13)
    data_q_m = set(s for p, s in gpcs_and_slugs if 13 <= p < 14)
    data_q_h = set(s for p, s in gpcs_and_slugs if 14 < p)

    # print("size S", *sorted(data_s_s),sep="\n")
    # print("size M", *sorted(data_s_m),sep="\n")
    # print("size L", *sorted(data_s_l),sep="\n")

    # print("gpc L", *sorted(data_q_l),sep="\n")
    # print("gpc M", *sorted(data_q_m),sep="\n")
    # print("gpc H", *sorted(data_q_h),sep="\n")

    # for (s, sset), (q, qset) in product(
    #     [("s", data_s_s), ("m", data_s_m), ("l", data_s_l)],
    #     [("l", data_q_l), ("m", data_q_m), ("h", data_q_h)]
    # ):
    #     print(f"size {s} qual {q}: {len(sset & qset)}")

    # plt.hist(gpcs, bins=20)
    # plt.title("Test Data Field Average GPC")
    # plt.xlabel("Average GPC (%)")
    # plt.ylabel("Count")
    # plt.savefig("plots/field_gpcs.png", dpi=200)
