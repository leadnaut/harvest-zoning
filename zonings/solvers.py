from dataclasses import dataclass
from heapq import heappush, heapreplace
from time import time
from typing import Any, Generic, Optional, TypeVar

import gurobipy as gp

from zonings.models import (
    Field,
    MipConfig,
    SField,
    Solution,
    SolveInfo,
    SZone,
    Zone,
    ZoningConfig,
)
from zonings.utils import Box, calculate_box_sums

T = TypeVar("T")


@dataclass(frozen=True)
class CGQueueNode(Generic[T]):
    reduced_cost: float
    variable: T

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CGQueueNode):
            raise NotImplementedError
        return self.reduced_cost < other.reduced_cost


class CGMipSolver:
    def __init__(
        self,
        zones: list[Zone],
        max_zones: int,
        field: Field,
        config: MipConfig,
    ) -> None:
        self.all_zones = set(zones)
        self.model_zones: set[Zone] = set()
        self.field = field
        self.config = config

        self.model = gp.Model()
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

    def add_vars(self, zones: list[Zone], vtype: str = gp.GRB.CONTINUOUS):
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
        best_zones: list[CGQueueNode[Zone]] = []
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
        return [i.variable for i in best_zones], positive_rc_zones

    def solve(self) -> Solution[Zone]:
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


class StochasticCGMipSolver:
    def __init__(
        self,
        zones: list[SZone],
        max_zones: int,
        alpha: float,
        lam: float,
        field: SField,
        config: MipConfig,
    ) -> None:
        self.all_zones = set(zones)
        self.model_zones: set[SZone] = set()
        self.field = field
        self.config = config
        self.num_scenarios = len(self.field.yield_maps)
        self.model = gp.Model()
        # Variables
        self.X: dict[SZone, gp.Var] = {}
        self.Beta = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.BetaM = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.Var = self.model.addVar()
        self.CVar = self.model.addVar()
        # Objective:
        self.model.setObjective(
            lam * gp.quicksum(self.Beta[s] for s in range(self.num_scenarios))
            + (1 - lam) * self.CVar,
            gp.GRB.MAXIMIZE,
        )
        # Constraints
        self.limit_constraint = self.model.addConstr(gp.LinExpr(0) <= max_zones)
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }
        self.return_constraints = {
            s: self.model.addConstr(self.Beta[s] == 0)
            for s in range(self.num_scenarios)
        }
        self.var_constraints = {
            s: self.model.addConstr(self.Beta[s] + self.BetaM[s] >= self.Var)
            for s in range(self.num_scenarios)
        }
        self.cvar_constraint = self.model.addConstr(
            self.CVar
            == self.Var
            - (1 / (alpha * self.num_scenarios))
            * gp.quicksum(self.BetaM[s] for s in range(self.num_scenarios))
        )

    def choose_starting_zones(self) -> list[SZone]:
        return []

    def add_vars(
        self, zones: list[SZone], vtype: str = gp.GRB.CONTINUOUS
    ) -> None:
        self.X.update((z, self.model.addVar(vtype=vtype)) for z in zones)
        self.model_zones.update(zones)

        for z in zones:
            self.model.chgCoeff(self.limit_constraint, self.X[z], 1)
            for s in range(self.num_scenarios):
                self.model.chgCoeff(
                    self.return_constraints[s], self.X[z], -z.scores[s]
                )
            for x, y in z.iter_contents():
                self.model.chgCoeff(
                    self.overlap_constraints[x, y], self.X[z], 1
                )

    def find_entering_variables(self) -> tuple[list[SZone], int] | None:
        # get dual variables
        limit_dual = self.limit_constraint.Pi
        cover_constraint_dual_grid = [
            [self.overlap_constraints[x, y].Pi for x in range(self.field.width)]
            for y in range(self.field.height)
        ]
        cover_dual_box_sums = calculate_box_sums(cover_constraint_dual_grid)
        return_duals = [
            self.return_constraints[s].Pi for s in range(self.num_scenarios)
        ]

        best_zones: list[CGQueueNode[SZone]] = []
        positive_rc_zones = 0
        for z in self.all_zones:
            if z in self.model_zones:
                continue
            reduced_cost = (
                -cover_dual_box_sums[z.box]
                - limit_dual
                + sum(
                    z.scores[s] * return_duals[s]
                    for s in range(self.num_scenarios)
                )
            )
            if reduced_cost > 0.001:
                positive_rc_zones += 1
                if (
                    len(best_zones)
                    < self.config.max_variables_added_per_cg_iteration
                ):
                    heappush(best_zones, CGQueueNode(reduced_cost, z))
                elif reduced_cost > best_zones[0].reduced_cost:
                    heapreplace(best_zones, CGQueueNode(reduced_cost, z))

        if len(best_zones) == 0:
            return None
        return [i.variable for i in best_zones], positive_rc_zones

    def solve(self) -> Solution[SZone]:
        print("Beginning column generation")
        solve_start_t = time()
        cg_iterations = 0
        while (
            not self.config.max_cg_iterations
            or cg_iterations < self.config.max_cg_iterations
        ):
            self.model.setParam("OutputFlag", 0)
            self.model.optimize()
            if self.model.Status == gp.GRB.INF_OR_UNBD:
                print("infeasible cg mip")
                break
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
        self,
        field: Field,
        max_zones: int,
        config: ZoningConfig,
        timeout: Optional[float] = None,
    ) -> None:
        self.field = field
        self.max_zones = max_zones
        self.config = config
        self.lookup: dict[tuple[Box, int], tuple[float, list[Box]]] = {}
        self.cache_hits = 0
        self.solve_start = 0.0
        self.timeout = timeout

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
        if (
            self.timeout is not None
            and time() - self.solve_start > self.timeout
        ):
            return (0, [])

        if (box, n_zones) in self.lookup:
            self.cache_hits += 1
            return self.lookup[box, n_zones]

        if (
            box.height() < self.config.minimum_height
            or box.width() < self.config.minimum_width
            or (
                self.config.minimum_pixels
                and self.field.field_box_sums[box] < self.config.minimum_pixels
            )
        ):
            result = (0.0, [])

        elif n_zones == 1:
            result = (self._score_box(box), [box])

        else:
            split_values = [(self._score_box(box), [box])]  # do nothing option
            horizontal_splits = [box.split(x=x) for x in range(box.x1, box.x2)]
            vertical_splits = [box.split(y=y) for y in range(box.y1, box.y2)]
            split_values.extend(
                self._combine_solution(
                    self.zone_box(b1, n1), self.zone_box(b2, n_zones - n1)
                )
                for b1, b2 in horizontal_splits + vertical_splits
                for n1 in range(1, n_zones)
            )
            result = max(
                split_values, key=lambda tup: (round(tup[0], 2), -len(tup[1]))
            )

        self.lookup[box, n_zones] = result
        return result

    def solve(self) -> Solution[Zone]:
        print(f"Starting dynamic programming solve for {self.field.field_id}")
        tic = time()
        if self.timeout is not None:
            self.solve_start = tic
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


class CVarDynamicSolver:
    def __init__(
        self,
        field: SField,
        max_zones: int,
        config: ZoningConfig,
        time_out: Optional[float] = None,
    ) -> None:
        self.field = field
        self.max_zones = max_zones
        self.config = config
        self.time_out = time_out

        self.lookup: dict[tuple[Box, int], tuple[float, list[Box]]] = {}
        self.solve_start = 0.0
        self.cache_hits = 0
        self.cvar_scenarios: set[int] = set()

    def _score_box(self, box: Box) -> float:
        cvar_scenario_score = 0.0
        pseudo_var = 0.0
        for s in self.cvar_scenarios:
            score = self.config.pricing.price_box_in_sfield(box, self.field, s)
            cvar_scenario_score += score
            pseudo_var = max(pseudo_var, score)

        positive_revenue = 0.0
        for s in range(self.field.num_scenarios):
            positive_revenue += max(
                pseudo_var
                - self.config.pricing.price_box_in_sfield(box, self.field, s),
                0,
            )
        positive_revenue /= self.field.num_scenarios
        return pseudo_var - (
            positive_revenue
            / (1 - len(self.cvar_scenarios) / self.field.num_scenarios)
        )

    def _combine_solution(
        self, s1: tuple[float, list[Box]], s2: tuple[float, list[Box]]
    ) -> tuple[float, list[Box]]:
        return (s1[0] + s2[0], s1[1] + s2[1])

    def zone_box(self, box: Box, n_zones: int) -> tuple[float, list[Box]]:
        result: tuple[float, list[Box]]
        if (
            self.time_out is not None
            and time() - self.solve_start > self.time_out
        ):
            return (0, [])

        if (box, n_zones) in self.lookup:
            self.cache_hits += 1
            return self.lookup[box, n_zones]

        if (
            box.height() < self.config.minimum_height
            or box.width() < self.config.minimum_width
            or (
                self.config.minimum_pixels
                and self.field.field_box_sums[box] < self.config.minimum_pixels
            )
        ):
            result = (0.0, [])
        elif n_zones == 1:
            result = (self._score_box(box), [box])
        else:
            split_values = [(self._score_box(box), [box])]  # do nothing option
            horizontal_splits = [box.split(x=x) for x in range(box.x1, box.x2)]
            vertical_splits = [box.split(y=y) for y in range(box.y1, box.y2)]
            split_values.extend(
                self._combine_solution(
                    self.zone_box(b1, n1), self.zone_box(b2, n_zones - n1)
                )
                for b1, b2 in horizontal_splits + vertical_splits
                for n1 in range(1, n_zones)
            )
            result = max(
                split_values, key=lambda tup: (round(tup[0], 2), -len(tup[1]))
            )

        self.lookup[box, n_zones] = result
        return result

    def solve(self, cvar_alpha: float) -> Solution:
        n_cvar_scenarios = int(self.field.num_scenarios * cvar_alpha)
        self.cvar_scenarios = set(
            range(n_cvar_scenarios)
        )  # take the first n as a guess

        historical_cvar_scenarios = [set(range(n_cvar_scenarios))]
        while True:
            print(self.cvar_scenarios)
            proposed_solution = self.zone_box(
                self.field.bounding_box(), self.max_zones
            )
            scenario_scores: list[tuple[float, int]] = []
            for s in range(self.field.num_scenarios):
                scenario_scores.append(
                    (
                        sum(
                            self.config.pricing.price_box_in_sfield(
                                b, self.field, s
                            )
                            for b in proposed_solution[1]
                        ),
                        s,
                    )
                )

            worst_alpha_scores = sorted(scenario_scores)[:n_cvar_scenarios]

            print(
                sum(score for (score, _) in worst_alpha_scores)
                / n_cvar_scenarios
            )
            print(proposed_solution[1])

            new_cvar_scenarios = set(s for (_, s) in worst_alpha_scores)
            if new_cvar_scenarios in historical_cvar_scenarios:
                break

            self.cvar_scenarios = new_cvar_scenarios
            historical_cvar_scenarios.append(new_cvar_scenarios)
            self.lookup = {}

        return Solution([SZone(b, []) for b in proposed_solution[1]], 0, None)
