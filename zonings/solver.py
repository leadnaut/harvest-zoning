from dataclasses import dataclass
from heapq import heappush, heapreplace
from typing import Any

import gurobipy as gp

from zonings.models import Field, Solution, SolverConfig, Zone
from zonings.utils import calculate_box_sums


@dataclass(frozen=True)
class CGQueueNode:
    reduced_cost: float
    zone: Zone

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CGQueueNode):
            raise NotImplementedError
        return self.reduced_cost < other.reduced_cost


class ZoneSolver:
    def __init__(
        self,
        solver_id: str,
        zones: list[Zone],
        field: Field,
        config: SolverConfig,
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
        self.limit_constraint = self.model.addConstr(
            gp.LinExpr(0) <= config.max_zones
        )
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

    def find_entering_variables(self) -> list[Zone] | None:
        # get dual variables
        limit_dual = self.limit_constraint.Pi
        cover_constraint_dual_grid = [
            [self.overlap_constraints[x, y].Pi for x in range(self.field.width)]
            for y in range(self.field.height)
        ]
        cover_dual_box_sums = calculate_box_sums(cover_constraint_dual_grid)

        # calculate reduced costs of all the zones
        best_zones: list[CGQueueNode] = []
        rcs = []
        for z in self.all_zones:
            if z in self.model_zones:
                continue
            reduced_cost = z.score - cover_dual_box_sums[z.box] - limit_dual
            rcs.append(reduced_cost)
            if reduced_cost > 0.001:
                if (
                    len(best_zones)
                    < self.config.max_variables_added_per_cg_iteration
                ):
                    heappush(best_zones, CGQueueNode(reduced_cost, z))
                elif (
                    reduced_cost > best_zones[0].reduced_cost
                ):  # better than the smallest rc in queue
                    heapreplace(best_zones, CGQueueNode(reduced_cost, z))

        print(sum(rcs) / len(rcs))
        if len(best_zones) == 0:
            return None

        return [i.zone for i in best_zones]

    def get_solution(self) -> Solution:
        return Solution(
            revenue=self.model.ObjVal,
            zones=[z for z in self.X if self.X[z].X > 0.001],
        )

    def solve(self) -> Solution:
        print("Beginning column generation")
        iteration = 0
        while (
            not self.config.max_cg_iterations
            or iteration < self.config.max_cg_iterations
        ):
            self.model.setParam("OutputFlag", 0)
            self.model.optimize()
            iteration += 1
            if entering_variables := self.find_entering_variables():
                print(f"{iteration}: Added {len(entering_variables)}")
                self.add_vars(entering_variables)
                continue
            break

        print(f"Column generation done. {len(self.X)} total variables")
        for k in self.X:
            self.X[k].setAttr("vtype", gp.GRB.BINARY)

        self.model.setParam("OutputFlag", 1)
        self.model.optimize()

        return self.get_solution()
