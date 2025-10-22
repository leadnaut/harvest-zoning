from typing import overload

import seaborn as sns
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from zonings.models import Field, SField, Solution, SZone, Zone


@overload
def plotting_zone_info(zone: Zone, field: Field) -> str: ...
@overload
def plotting_zone_info(zone: SZone, field: SField, alpha: float) -> str: ...
def plotting_zone_info(
    zone: Zone | SZone, field: Field | SField, alpha: float | None = None
) -> str:
    if type(zone) is Zone and type(field) is Field:
        gpc = field.protein_box_sums[zone.box] / field.yield_box_sums[zone.box]
        return f"GPC = {gpc * 100:.2f}%"

    if type(zone) is SZone and type(field) is SField and alpha is not None:
        gpcs = sorted(
            field.protein_box_sums[s][zone.box]
            / field.yield_box_sums[s][zone.box]
            for s in range(field.num_scenarios)
        )
        worst = gpcs[0] * 100
        best = gpcs[-1] * 100
        average = sum(gpcs) / len(gpcs) * 100

        return (
            f"Worst: {worst: .2f}%\nAverage: {average:.2f}%\nBest: {best:.2f}%"
        )

    return "test"


def view_field_solution(field: Field, solution: Solution[Zone]):
    ax = sns.heatmap(field.gpc_map, vmin=0.1)
    for z in solution.zones:
        plot_zone_on_axes(ax, z, field)

    plt.show()
    return ax


def view_sfield_solution(
    field: SField, solution: Solution[SZone], alpha: float
):
    maximum_gpc = max(map(max, map(max, field.gpc_maps)))

    fig = plt.figure()

    def view_scenario(s: int):
        plt.clf()
        ax = sns.heatmap(field.gpc_maps[s], vmin=0.1, vmax=maximum_gpc)
        for z in solution.zones:
            plot_zone_on_axes(ax, z, field, alpha)

    view_scenario(0)

    anim = FuncAnimation(
        fig,
        view_scenario,  # type: ignore
        init_func=lambda: view_scenario(0),  # type: ignore
        frames=field.num_scenarios,
    )
    plt.show()


@overload
def plot_zone_on_axes(ax: Axes, z: Zone, field: Field) -> Axes: ...
@overload
def plot_zone_on_axes(
    ax: Axes, z: SZone, field: SField, alpha: float
) -> Axes: ...
def plot_zone_on_axes(
    ax: Axes, z: Zone | SZone, field: Field | SField, alpha: float | None = None
) -> Axes:
    ax.add_patch(
        Rectangle(
            xy=(z.box.x1, z.box.y1),
            width=z.box.width,
            height=z.box.height,
            linewidth=4,
            edgecolor=(30 / 256, 214 / 256, 45 / 256),
            facecolor="none",
        )
    )
    txt = ax.text(
        *z.box.centre,
        plotting_zone_info(z, field, alpha),  # type: ignore
        color="w",
        fontsize="large",
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )
    txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="k")])

    return ax
