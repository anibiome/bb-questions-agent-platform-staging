import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineState:
    mean: float
    var: float
    n: int


def update_ewma_baseline(prev: BaselineState, value: float, alpha: float = 0.2) -> BaselineState:
    """
    EWMA update for within-person baseline.

    - mean_t = alpha*x + (1-alpha)*mean_{t-1}
    - var_t  = alpha*(x-mean_t)^2 + (1-alpha)*var_{t-1}
    """
    a = float(alpha)
    a = max(0.01, min(0.8, a))
    mean_new = a * float(value) + (1.0 - a) * float(prev.mean)
    var_new = a * (float(value) - mean_new) ** 2 + (1.0 - a) * float(prev.var)
    return BaselineState(mean=float(mean_new), var=float(max(0.0, var_new)), n=int(prev.n) + 1)


def baseline_std(state: BaselineState) -> float:
    return float(math.sqrt(max(0.0, state.var)))

