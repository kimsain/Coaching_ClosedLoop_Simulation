"""Summary numbers computed from one simulate() result."""

import numpy as np


def user_rates(res):
    return res["completed"].sum(1) / np.maximum(res["assigned"].sum(1), 1)


def summarize(res, domains):
    n, days = len(res["sends"]), res["days"]
    assigned, completed = res["assigned"], res["completed"]
    out = {
        "completion": completed.sum() / assigned.sum(),
        "user_mean_completion": user_rates(res).mean(),
        "notifs_per_user_day": res["sends"].sum() / (n * days),
        "open_rate": res["opens"].sum() / max(res["sends"].sum(), 1),
        "opens_per_user_day": res["opens"].sum() / (n * days),
        "routines_per_user_day": assigned.sum() / (n * days),
        "completed_per_user_day": completed.sum() / (n * days),
        "mean_difficulty": res["difficulty"].sum() / assigned.sum(),
    }
    for j, name in enumerate(domains):
        out[f"completion_{name}"] = completed[:, j].sum() / assigned[:, j].sum()
    return out


def weekly_completion(res):
    return res["week_completed"] / res["week_assigned"]


def paired(cur, new):
    """Within-user comparison and a split of the overall gap into composition and within-user parts."""
    diff = user_rates(new) - user_rates(cur)
    w_cur = cur["assigned"].sum(1)
    rate_cur = cur["completed"].sum() / cur["assigned"].sum()
    rate_new = new["completed"].sum() / new["assigned"].sum()
    # New-policy per-user rates, weighted by how many routines each user had under the current policy.
    rate_new_cur_weights = (user_rates(new) * w_cur).sum() / w_cur.sum()
    return {
        "mean_diff": diff.mean(),
        "median_diff": float(np.median(diff)),
        "n_improved": int((diff > 0).sum()),
        "n_worse": int((diff < 0).sum()),
        "n_users": len(diff),
        "gap_total": rate_new - rate_cur,
        "gap_within_user": rate_new_cur_weights - rate_cur,
        "gap_composition": rate_new - rate_new_cur_weights,
        "diffs": diff,
    }
