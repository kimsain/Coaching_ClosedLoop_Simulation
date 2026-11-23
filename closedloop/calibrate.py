"""Fit the user-model intercept so the current policy reproduces the target completion rate."""

from . import with_base_logit
from .simulate import run_pair


def current_completion(cfg, seeds):
    results = [run_pair(cfg, s, policies=("current",))[1]["current"] for s in seeds]
    return sum(r["completed"].sum() for r in results) / sum(r["assigned"].sum() for r in results)


def calibrate(cfg):
    """Bisection on base_logit. Completion under the current policy rises monotonically with it."""
    c = cfg["calibration"]
    seeds = cfg["simulation"]["calibration_seeds"]
    target = c["target_completion"]
    lo, hi = c["search_range"]
    for _ in range(c["max_iter"]):
        mid = (lo + hi) / 2
        rate = current_completion(with_base_logit(cfg, mid), seeds)
        if abs(rate - target) * 100 < c["tolerance_pp"] / 10:
            break
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    return {"base_logit": mid, "rate": rate, "target": target, "gap_pp": (rate - target) * 100}
