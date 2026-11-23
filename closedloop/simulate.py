"""Daily closed loop: policy decides, user responds, state and policy memory update."""

import numpy as np

from .policies import make_policy
from .users import make_draws, make_population


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(cfg, users, draws, policy_name):
    um = cfg["user_model"]
    domains, n_slots = cfg["domains"], len(cfg["slots"])
    n, d = users["h0"].shape
    days = cfg["simulation"]["weeks"] * 7

    eligible = np.ones((n, d), bool)
    eligible[:, domains.index("medication")] = users["has_med"]
    policy = make_policy(cfg, policy_name, eligible, users["stated"], n_slots)

    offset = np.array([um["domain_offset"][k] for k in domains])
    slot_match = np.arange(n_slots)[None, :] == users["pref"][:, None]
    m, h = users["m0"].copy(), users["h0"].copy()
    fatigue, active_yesterday = np.zeros(n), np.zeros(n, bool)

    out = {
        "assigned": np.zeros((n, d)), "completed": np.zeros((n, d)), "difficulty": np.zeros((n, d)),
        "sends": np.zeros(n), "opens": np.zeros(n), "max_daily_sends": np.zeros(n, int),
        "week_assigned": np.zeros(days // 7), "week_completed": np.zeros(days // 7),
    }

    for t in range(days):
        assigned, difficulty = policy.routines()
        sends = policy.notifications()

        p_open = sigmoid(
            um["open_base"] + um["open_motivation"] * m[:, None] - um["open_fatigue"] * fatigue[:, None]
            + um["open_slot_match"] * slot_match + um["open_active_yesterday"] * active_yesterday[:, None]
        )
        opens = sends & (draws["open"][t] < p_open)

        time_today = np.clip(users["tc"] + draws["time"][t], 0, 1)
        logit = (
            um["base_logit"] + offset[None, :] + um["motivation"] * m[:, None] + um["habit"] * h
            - um["burden"] * difficulty * (1 + time_today[:, None])
            - um["fatigue"] * fatigue[:, None] + um["opened_boost"] * opens.any(1)[:, None]
        )
        completed = assigned & (draws["complete"][t] < sigmoid(logit))

        # User state update: today's outcomes shape tomorrow's probabilities.
        n_assigned, n_done = assigned.sum(1), completed.sum(1)
        done_frac = np.divide(n_done, n_assigned, out=np.zeros(n), where=n_assigned > 0)
        m = users["m0"] + um["motivation_reversion"] * (m - users["m0"]) + um["motivation_lr"] * (done_frac - 0.5)
        h = np.where(completed, h + um["habit_gain"] * (1 - h), np.where(assigned, h * (1 - um["habit_decay"]), h))
        fatigue = fatigue * um["fatigue_decay"] + um["fatigue_gain"] * users["fs"] * sends.sum(1)
        active_yesterday = n_done > 0

        policy.observe(sends, opens, assigned, completed)

        out["assigned"] += assigned
        out["completed"] += completed
        out["difficulty"] += difficulty * assigned
        out["sends"] += sends.sum(1)
        out["opens"] += opens.sum(1)
        out["max_daily_sends"] = np.maximum(out["max_daily_sends"], sends.sum(1))
        out["week_assigned"][t // 7] += n_assigned.sum()
        out["week_completed"][t // 7] += n_done.sum()

    out["days"] = days
    return out


def run_pair(cfg, seed, policies=("current", "new")):
    """Same users and same random draws for every policy, so only the policy differs."""
    rng = np.random.default_rng(seed)
    users = make_population(rng, cfg)
    draws = make_draws(rng, cfg, len(users["m0"]))
    return users, {name: simulate(cfg, users, draws, name) for name in policies}
