from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from heapq import heappush, heapreplace
from time import time
from typing import Any, Callable, Hashable, Optional

import gurobipy as gp

from zonings.models import (
    Box,
    BoxDataLookup,
    CGSolveInfo,
    CGSolverConfig,
    DeterministicSolution,
    DPSolveInfo,
    Field,
    SField,
    StochasticSolution,
    SZone,
    Zone,
    ZoningConfig,
)
from zonings.utils import cvar


@dataclass(frozen=True)
class CGQueueNode[T]:
    reduced_cost: float
    variable: T

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CGQueueNode):
            raise NotImplementedError
        return self.reduced_cost < other.reduced_cost


class Sense(IntEnum):
    MAXIMISE = 1
    MINIMISE = -1


class CGSolver[VariableType: Hashable, SolutionType](ABC):
    def __init__(
        self,
        cg_variables: list[VariableType],
        config: CGSolverConfig,
        sense: Sense,
    ) -> None:
        self.all_cg_variables = set(cg_variables)
        self.sense = sense
        self.model = gp.Model()
        self.cg_X: dict[VariableType, gp.Var] = {}
        self.config = config

    @abstractmethod
    def _get_starting_variables(self) -> list[VariableType]: ...

    def _add_variables_to_model(self, variables: list[VariableType]) -> None:
        self.cg_X.update((v, self.model.addVar()) for v in variables)

        for v in variables:
            self._add_variable_to_objective_and_constraints(v)

    @abstractmethod
    def _add_variable_to_objective_and_constraints(self, v: VariableType) -> None: ...

    @abstractmethod
    def _update_lp_sol_based_attributes(self) -> None:
        """this runs before reduced costs for the current iteration are calculated
        providing a chance for attrs needed during rc calculation to be updated
        """
        ...

    @abstractmethod
    def _calculate_reduced_cost(self, variable: VariableType) -> float: ...

    def _find_entering_variables(self) -> tuple[list[VariableType], int] | None:
        best_variables: list[CGQueueNode[VariableType]] = []
        good_rc_variables = 0

        self._update_lp_sol_based_attributes()

        for v in self.all_cg_variables:
            if v in self.cg_X:
                continue
            rc = self._calculate_reduced_cost(v)
            if rc * self.sense > 0.001:
                good_rc_variables += 1
                if len(best_variables) < self.config.max_variables_added_per_cg_iteration:
                    heappush(best_variables, CGQueueNode(rc, v))
                elif abs(rc) > abs(best_variables[0].reduced_cost):
                    heapreplace(best_variables, CGQueueNode(rc, v))

        if len(best_variables) == 0:
            return None
        return [i.variable for i in best_variables], good_rc_variables

    @abstractmethod
    def _extract_solution(self) -> SolutionType: ...

    def solve(self) -> tuple[SolutionType, CGSolveInfo]:
        print("Beginning Column Generation")
        solve_start_t = time()
        cg_iterations = 0
        self.model.setParam("OutputFlag", 0)

        # add initial variables
        initial_variables = self._get_starting_variables()
        self._add_variables_to_model(initial_variables)
        total_variables = len(initial_variables)
        while not self.config.max_cg_iterations or cg_iterations < self.config.max_cg_iterations:
            self.model.optimize()
            cg_iterations += 1

            if entering_vars := self._find_entering_variables():
                print(
                    f"{cg_iterations}: Added {len(entering_vars[0])} ({entering_vars[1]} variables with good RCs)"
                )
                self._add_variables_to_model(entering_vars[0])
                total_variables += len(entering_vars[0])
                continue
            break
        cg_end_t = time()
        print(f"Column generation done! {total_variables} total variables.")

        for v in self.cg_X:
            self.cg_X[v].setAttr("vtype", gp.GRB.BINARY)
        self.model.setParam("OutputFlag", 1)
        print("Starting MIP Solve")
        self.model.optimize()
        solve_end_t = time()

        return (
            self._extract_solution(),
            CGSolveInfo(
                solve_end_t - solve_start_t,
                cg_end_t - solve_start_t,
                cg_iterations,
                total_variables,
            ),
        )


class DeterministicMIPSolver(CGSolver[Zone, DeterministicSolution]):
    def __init__(
        self,
        zones: list[Zone],
        max_zones: int,
        field: Field,
        config: CGSolverConfig,
    ) -> None:
        super().__init__(zones, config, Sense.MAXIMISE)

        self.field_width = field.width
        self.field_height = field.height

        self.model.setObjective(gp.LinExpr(0), gp.GRB.MAXIMIZE)

        self.limit_constraint = self.model.addConstr(gp.LinExpr(0) <= max_zones)
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }

        # RC Calculation Helpers
        self.cover_dual_box_sums: BoxDataLookup[float]

    def _get_starting_variables(self) -> list[Zone]:
        return []

    def _add_variable_to_objective_and_constraints(self, v: Zone) -> None:
        self.cg_X[v].Obj = v.score
        self.model.chgCoeff(self.limit_constraint, self.cg_X[v], 1)
        for x, y in v.iter_contents():
            self.model.chgCoeff(self.overlap_constraints[x, y], self.cg_X[v], 1)

    def _update_lp_sol_based_attributes(self) -> None:
        self.cover_dual_box_sums = BoxDataLookup.from_grid(
            [
                [self.overlap_constraints[x, y].Pi for x in range(self.field_width)]
                for y in range(self.field_height)
            ]
        )

    def _calculate_reduced_cost(self, variable: Zone) -> float:
        return variable.score - self.cover_dual_box_sums[variable.box] - self.limit_constraint.Pi

    def _extract_solution(self) -> DeterministicSolution:
        return DeterministicSolution(
            [z for z in self.cg_X if round(self.cg_X[z].X) == 1],
            self.model.ObjVal,
        )


class TurnAwareMIPSolver(CGSolver[Zone, DeterministicSolution]):
    def __init__(
        self,
        zones: list[Zone],
        max_zones: int,
        max_turns: float | None,
        field: Field,
        config: CGSolverConfig,
    ) -> None:
        super().__init__(zones, config, Sense.MAXIMISE)

        self.field = field
        self.model.setObjective(gp.LinExpr(0), gp.GRB.MAXIMIZE)
        self.zone_limit_constraint = self.model.addConstr(gp.LinExpr(0) <= max_zones)

        self.turn_limit_constraint = self.model.addConstr(
            gp.LinExpr(0) <= (max_turns or gp.GRB.INFINITY)
        )
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }
        self.cover_dual_box_sums: BoxDataLookup[float]

    def _get_starting_variables(self) -> list[Zone]:
        return []

    def _add_variable_to_objective_and_constraints(self, v: Zone) -> None:
        self.cg_X[v].Obj = v.score
        self.model.chgCoeff(self.zone_limit_constraint, self.cg_X[v], 1)
        self.model.chgCoeff(self.turn_limit_constraint, self.cg_X[v], v.turns)

        for x, y in v.iter_contents():
            self.model.chgCoeff(self.overlap_constraints[x, y], self.cg_X[v], 1)

    def _update_lp_sol_based_attributes(self) -> None:
        self.cover_dual_box_sums = BoxDataLookup.from_grid(
            [
                [self.overlap_constraints[x, y].Pi for x in range(self.field.width)]
                for y in range(self.field.height)
            ]
        )

    def _calculate_reduced_cost(self, variable: Zone) -> float:
        return (
            variable.score
            - self.cover_dual_box_sums[variable.box]
            - self.zone_limit_constraint.Pi
            - (self.turn_limit_constraint.Pi * variable.turns)
        )

    def _extract_solution(self) -> DeterministicSolution:
        return DeterministicSolution(
            [z for z in self.cg_X if round(self.cg_X[z].X) == 1],
            self.model.ObjVal,
        )


class StochasticMipSolver:
    def __init__(
        self,
        zones: list[SZone],
        max_zones: int,
        alpha: float,
        expectation_weight: float,
        field: SField,
    ) -> None:
        self.alpha = alpha
        self.zones = zones
        self.model = gp.Model()

        self.num_scenarios = field.num_scenarios
        self.X = {z: self.model.addVar(vtype=gp.GRB.BINARY) for z in self.zones}
        self.Beta = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.BetaM = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.Var = self.model.addVar()
        self.CVar = self.model.addVar()

        self.model.ModelSense = gp.GRB.MAXIMIZE
        self.model.setObjectiveN(
            expectation_weight * gp.quicksum(self.Beta[s] for s in range(self.num_scenarios))
            + (1 - expectation_weight) * self.CVar,
            index=0,
            priority=2,
            name="main_obj",
        )
        # self.model.setObjectiveN(
        #     -1 * gp.quicksum(var for var in self.X.values()), index=1, priority=1, name="nzones"
        # )

        self.limit_constraint = self.model.addConstr(
            gp.quicksum(self.X[z] for z in self.X) <= max_zones
        )
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }
        for z in self.zones:
            for x, y in z.iter_contents():
                self.model.chgCoeff(self.overlap_constraints[x, y], self.X[z], 1)
        self.return_constraints = {
            s: self.model.addConstr(
                self.Beta[s] - gp.quicksum(self.X[z] * z.scores[s] for z in self.X) == 0
            )
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

    def solve(self) -> tuple[StochasticSolution, CGSolveInfo]:
        self.model.setParam("PreSolve", 0)
        tic = time()
        self.model.optimize()
        toc = time()
        print("Objective Value", self.model.ObjVal)
        print("CVaR variable", self.CVar.X)
        zones = [z for z in self.X if self.X[z].X > 0.01]
        scores = [sum(z.scores[s] for z in zones) for s in range(self.num_scenarios)]
        calculated_cvar = cvar(self.alpha, scores)
        print("Calculated CVaR", calculated_cvar)
        # assert abs(calculated_cvar - self.CVar.X) < 1e-5
        return StochasticSolution(
            zones,
            scores,
        ), CGSolveInfo(toc - tic, 0, 0, 0)


class StochasticCGMIPSolver(CGSolver[SZone, StochasticSolution]):
    def __init__(
        self,
        zones: list[SZone],
        max_zones: int,
        alpha: float,
        expectation_weight: float,
        field: SField,
        config: CGSolverConfig,
    ) -> None:
        super().__init__(zones, config, Sense.MAXIMISE)

        self.field = field
        self.num_scenarios = self.field.num_scenarios

        # Non-CG Variables
        self.Beta = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.BetaM = {s: self.model.addVar() for s in range(self.num_scenarios)}
        self.Var = self.model.addVar()
        self.CVar = self.model.addVar()
        # Objective
        self.model.setObjective(
            expectation_weight
            * gp.quicksum(self.Beta[s] for s in range(self.num_scenarios))
            / self.num_scenarios
            + (1 - expectation_weight) * self.CVar,
            gp.GRB.MAXIMIZE,
        )
        self.limit_constraint = self.model.addConstr(gp.LinExpr(0) <= max_zones)
        self.overlap_constraints = {
            (x, y): self.model.addConstr(gp.LinExpr(0) <= 1)
            for x in range(field.width)
            for y in range(field.height)
        }
        self.return_constraints = {
            s: self.model.addConstr(self.Beta[s] == 0) for s in range(self.num_scenarios)
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

        # RC Calculation Helpers
        self.cover_dual_box_sums: BoxDataLookup[float]

    def _get_starting_variables(self) -> list[SZone]:
        return [next(z for z in self.all_cg_variables if z.box == self.field.bounding_box())]

    def _add_variable_to_objective_and_constraints(self, v: SZone) -> None:
        self.model.chgCoeff(self.limit_constraint, self.cg_X[v], 1)
        for s in range(self.num_scenarios):
            self.model.chgCoeff(self.return_constraints[s], self.cg_X[v], -v.scores[s])
        for x, y in v.iter_contents():
            self.model.chgCoeff(self.overlap_constraints[x, y], self.cg_X[v], 1)

    def _update_lp_sol_based_attributes(self) -> None:
        self.cover_dual_box_sums = BoxDataLookup.from_grid(
            [
                [self.overlap_constraints[x, y].Pi for x in range(self.field.width)]
                for y in range(self.field.height)
            ]
        )

    def _calculate_reduced_cost(self, variable: SZone) -> float:
        return (
            -self.cover_dual_box_sums[variable.box]
            - self.limit_constraint.Pi
            + sum(
                variable.scores[s] * self.return_constraints[s].Pi
                for s in range(self.num_scenarios)
            )
        )

    def _extract_solution(self) -> StochasticSolution:
        return StochasticSolution(
            [z for z in self.cg_X if round(self.cg_X[z].X) == 1],
            [self.Beta[s].X for s in range(self.num_scenarios)],
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
        return self.field.get_box_price(box, self.config.pricing)

    def _combine_solution(
        self, s1: tuple[float, list[Box]], s2: tuple[float, list[Box]]
    ) -> tuple[float, list[Box]]:
        return (s1[0] + s2[0], s1[1] + s2[1])

    def zone_box(self, box: Box, n_zones: int) -> tuple[float, list[Box]]:
        result: tuple[float, list[Box]]
        if self.timeout is not None and time() - self.solve_start > self.timeout:
            return (0, [])

        if (box, n_zones) in self.lookup:
            self.cache_hits += 1
            return self.lookup[box, n_zones]

        if (
            box.height < self.config.minimum_height
            or box.width < self.config.minimum_width
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
                self._combine_solution(self.zone_box(b1, n1), self.zone_box(b2, n_zones - n1))
                for b1, b2 in horizontal_splits + vertical_splits
                for n1 in range(1, n_zones)
            )
            result = max(split_values, key=lambda tup: (round(tup[0], 2), -len(tup[1])))

        self.lookup[box, n_zones] = result
        return result

    def solve(self) -> tuple[DeterministicSolution, DPSolveInfo]:
        print(f"Starting dynamic programming solve for {self.field.field_id}")
        tic = time()
        if self.timeout is not None:
            self.solve_start = tic
        self.cache_hits = 0
        self.lookup = {}
        val, boxes = self.zone_box(self.field.bounding_box(), self.max_zones)
        toc = time()
        print(f"Solve done! Calculated {len(self.lookup)} nodes with {self.cache_hits} cache hits")
        return DeterministicSolution(
            [Zone(b, self._score_box(b)) for b in boxes],
            val,
        ), DPSolveInfo(toc - tic, len(self.lookup), self.cache_hits)


@dataclass(slots=True)
class SDPPartialSol:
    objective: float
    boxes: list[Box]
    scores: list[float]


class StochasticDynamicSolver:
    def __init__(
        self,
        field: SField,
        max_zones: int,
        cvar_alpha: float,
        expectation_weight: float,
        config: ZoningConfig,
        timeout: Optional[float] = None,
    ) -> None:
        self.field = field
        self.max_zones = max_zones
        self.config = config
        self.timeout = timeout

        self.lookup: dict[tuple[Box, int], SDPPartialSol] = {}
        self.solve_start = 0.0
        self.cache_hits = 0
        self.total_scenarios = self.field.num_scenarios
        self.cvar_scenarios = int(cvar_alpha * self.total_scenarios)

        self.objective: Callable[[list[float]], float] = lambda scores: (
            expectation_weight * sum(scores) / self.total_scenarios
            + (1 - expectation_weight)
            * sum(sorted(scores)[: self.cvar_scenarios])
            / self.cvar_scenarios
        )

    def _base_case(self, box: Box) -> SDPPartialSol:
        # apply objective at every step
        scores = self.field.get_box_prices(box, self.config.pricing)
        return SDPPartialSol(self.objective(scores), [box], scores)

    def _combine_and_score_sub_solutions(
        self, s1: SDPPartialSol, s2: SDPPartialSol
    ) -> SDPPartialSol:
        scores = [s1.scores[s] + s2.scores[s] for s in range(self.total_scenarios)]
        return SDPPartialSol(self.objective(scores), s1.boxes + s2.boxes, scores)

    def zone_box(self, box: Box, n_zones: int) -> SDPPartialSol:
        """returns (objective value, boxes, revenue in each scenario)"""
        result: SDPPartialSol
        if self.timeout is not None and time() - self.solve_start > self.timeout:
            return SDPPartialSol(0, [], [0 for _ in range(self.total_scenarios)])

        if (box, n_zones) in self.lookup:
            self.cache_hits += 1
            return self.lookup[box, n_zones]

        if (
            box.height < self.config.minimum_height
            or box.width < self.config.minimum_width
            or (
                self.config.minimum_pixels
                and self.field.field_box_sums[box] < self.config.minimum_pixels
            )
        ):
            result = SDPPartialSol(0, [], [0 for _ in range(self.total_scenarios)])

        elif n_zones == 1:
            result = self._base_case(box)

        else:
            split_solutions = [self._base_case(box)]  # do nothing option
            horizontal_splits = [box.split(x=x) for x in range(box.x1, box.x2)]
            vertical_splits = [box.split(y=y) for y in range(box.y1, box.y2)]
            split_solutions.extend(
                self._combine_and_score_sub_solutions(
                    self.zone_box(b1, n1), self.zone_box(b2, n_zones - n1)
                )
                for b1, b2 in horizontal_splits + vertical_splits
                for n1 in range(1, n_zones)
            )

            result = max(
                split_solutions,
                key=lambda sol: (sol.objective, len(sol.boxes)),
            )

        self.lookup[box, n_zones] = result
        return result

    def solve(self) -> tuple[StochasticSolution, DPSolveInfo]:
        print(f"Starting stochastic dynamic programming solve for {self.field.field_id}")
        tic = time()
        if self.timeout is not None:
            self.solve_start = tic
        self.cache_hits = 0
        self.lookup = {}
        sol = self.zone_box(self.field.bounding_box(), self.max_zones)
        toc = time()
        print(f"Solve done! Calculated {len(self.lookup)} nodes with {self.cache_hits} cache hits")
        print(f"Objective: {sol.objective}")
        zones = [
            SZone(
                b,
                self.field.get_box_prices(b, self.config.pricing),
            )
            for b in sol.boxes
        ]
        scores = [sum(z.scores[s] for z in zones) for s in range(self.total_scenarios)]
        if (calculated := self.objective(scores)) != sol.objective:
            print(f"Calculated: {calculated}\nReturned: {sol.objective}")
        return StochasticSolution(
            zones,
            scores,
        ), DPSolveInfo(toc - tic, len(self.lookup), self.cache_hits)
