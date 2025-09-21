from dataclasses import dataclass
from heapq import heappush, heapreplace
from time import time
from typing import Any

import gurobipy as gp

from zonings.models import (
    Field,
    MipConfig,
    Solution,
    SolveInfo,
    Zone,
    ZoningConfig,
)
from zonings.utils import Box, calculate_box_sums


@dataclass(frozen=True)
class CGQueueNode:
    reduced_cost: float
    zone: Zone

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CGQueueNode):
            raise NotImplementedError
        return self.reduced_cost < other.reduced_cost


class CGMipSolver:
    def __init__(
        self,
        solver_id: str,
        zones: list[Zone],
        max_zones: int,
        field: Field,
        config: MipConfig,
    ) -> None:
        self.all_zones = set(zones)
        self.model_zones: set[Zone] = set()
        self.field = field
        self.config = config

        self.model = gp.Model(solver_id)
        self.X: dict[Zone, gp.Var] = {}

        # Objective
        self.model.setObjective(gp.LinExpr(0), gp.GRB.MAXIMIZE)

        # Maximum zones constraints
        self.limit_constraint = self.model.addConstr(gp.LinExpr(0) <= max_zones)
        # No overlapping constraints
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }

    def choose_starting_zones(self) -> list[Zone]:
        return []

    def add_vars(self, zones: list[Zone], vtype=gp.GRB.CONTINUOUS):
        self.X.update((z, self.model.addVar(vtype=vtype)) for z in zones)
        self.model_zones.update(zones)

        for z in zones:
            self.X[z].Obj = z.score
            self.model.chgCoeff(self.limit_constraint, self.X[z], 1)
            for x, y in z.iter_contents():
                self.model.chgCoeff(
                    self.overlap_constraints[x, y], self.X[z], 1
                )

    def find_entering_variables(self) -> tuple[list[Zone], int] | None:
        # get dual variables
        limit_dual = self.limit_constraint.Pi
        cover_constraint_dual_grid = [
            [self.overlap_constraints[x, y].Pi for x in range(self.field.width)]
            for y in range(self.field.height)
        ]
        cover_dual_box_sums = calculate_box_sums(cover_constraint_dual_grid)

        # calculate reduced costs of all the zones
        best_zones: list[CGQueueNode] = []
        positive_rc_zones = 0
        for z in self.all_zones:
            if z in self.model_zones:
                continue
            reduced_cost = z.score - cover_dual_box_sums[z.box] - limit_dual
            if reduced_cost > 0.001:
                positive_rc_zones += 1
                if (
                    len(best_zones)
                    < self.config.max_variables_added_per_cg_iteration
                ):
                    heappush(best_zones, CGQueueNode(reduced_cost, z))
                elif (
                    reduced_cost > best_zones[0].reduced_cost
                ):  # better than the smallest rc in queue
                    heapreplace(best_zones, CGQueueNode(reduced_cost, z))

        if len(best_zones) == 0:
            return None
        return [i.zone for i in best_zones], positive_rc_zones

    def solve(self) -> Solution:
        print("Beginning column generation")
        solve_start_t = time()
        cg_iterations = 0
        while (
            not self.config.max_cg_iterations
            or cg_iterations < self.config.max_cg_iterations
        ):
            self.model.setParam("OutputFlag", 0)
            self.model.optimize()
            cg_iterations += 1
            if entering_variables := self.find_entering_variables():
                print(
                    f"{cg_iterations}: Added {len(entering_variables[0])}/{entering_variables[1]}"
                )
                self.add_vars(entering_variables[0])
                continue
            break
        cg_end_t = time()
        print(f"Column generation done. {len(self.X)} total variables")
        for k in self.X:
            self.X[k].setAttr("vtype", gp.GRB.BINARY)

        self.model.setParam("OutputFlag", 1)
        self.model.optimize()
        solve_end_t = time()

        return Solution(
            [z for z in self.X if self.X[z].X > 0.01],
            self.model.ObjVal,
            SolveInfo(
                total_solve_seconds=solve_end_t - solve_start_t,
                column_generation_seconds=cg_end_t - solve_start_t,
                column_generation_iterations=cg_iterations,
                total_variables=len(self.X),
            ),
        )


class DynamicSolver:
    def __init__(
        self, field: Field, max_zones: int, config: ZoningConfig
    ) -> None:
        self.field = field
        self.max_zones = max_zones
        self.config = config
        self.lookup: dict[tuple[Box, int], tuple[float, list[Box]]] = {}
        self.cache_hits = 0

    def _score_box(self, box: Box) -> float:
        box_yield = self.field.yield_box_sums[box]
        if box_yield < 0.00001:
            return 0
        box_gpc = self.field.protein_box_sums[box] / box_yield
        return self.config.pricing.calculate_price(box_gpc, box_yield)

    def _combine_solution(
        self, s1: tuple[float, list[Box]], s2: tuple[float, list[Box]]
    ) -> tuple[float, list[Box]]:
        return (s1[0] + s2[0], s1[1] + s2[1])

    def zone_box(self, box: Box, n_zones: int) -> tuple[float, list[Box]]:
        result: tuple[float, list[Box]]
        if (box, n_zones) in self.lookup:
            self.cache_hits += 1
            return self.lookup[box, n_zones]

        if n_zones == 1:
            result = (self._score_box(box), [box])

        elif (
            box.height() < self.config.minimum_height
            or box.width() < self.config.minimum_width
        ):
            result = (0.0, [])

        else:
            split_values = [(self._score_box(box), [box])]  # do nothing option
            horizontal_splits = [box.split(x=x) for x in range(box.x1, box.x2)]
            vertical_splits = [box.split(y=y) for y in range(box.y1, box.y2)]
            split_values.extend(
                self._combine_solution(
                    self.zone_box(b1, n1), self.zone_box(b2, n_zones - 1 - n1)
                )
                for b1, b2 in horizontal_splits + vertical_splits
                for n1 in range(0, n_zones)
            )
            result = max(
                split_values, key=lambda tup: (round(tup[0], 2), -len(tup[1]))
            )

        self.lookup[box, n_zones] = result
        return result

    def solve(self) -> Solution:
        print("Starting dynamic programming solve")
        tic = time()
        self.cache_hits = 0
        self.lookup = {}
        val, boxes = self.zone_box(self.field.bounding_box(), self.max_zones)
        toc = time()
        print(
            f"Solve done! Calculated {len(self.lookup)} nodes with {self.cache_hits} cache hits"
        )
        return Solution(
            [Zone(b, self._score_box(b)) for b in boxes],
            val,
            SolveInfo(
                toc - tic,
                0,
                0,
                0,
            ),
        )
