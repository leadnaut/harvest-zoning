from zonings.constants import DEFAULT_PRICING
from zonings.models import DeterministicSolution, PriceInfo, ScenarioMap, StochasticSolution


def score_solution_on_scenarios(
    solution: DeterministicSolution | StochasticSolution,
    scenarios: list[ScenarioMap],
    pricing: PriceInfo = DEFAULT_PRICING,
) -> list[float]:
    boxes = [zone.box for zone in solution.zones]

    revenues: list[float] = []
    for scenario in scenarios:
        revenues.append(sum(scenario.get_box_price(box, pricing) for box in boxes))

    return revenues
