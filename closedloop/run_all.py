"""One command: calibrate, run all policies over seeds, write reports/RESULTS.md and figures.

    python -m closedloop.run_all
"""

import json

import numpy as np
import pandas as pd

from . import ROOT, load_config, with_base_logit
from .calibrate import calibrate, current_completion
from .metrics import paired, summarize, user_rates, weekly_completion
from .policies import ABLATIONS
from .report import write_figures, write_results
from .simulate import simulate, run_pair
from .users import make_draws, make_persona_runs

POLICY_ORDER = ["current", "new", *ABLATIONS]
MOTIVATION_BINS = [-np.inf, -1, 0, 1, np.inf]
MOTIVATION_LABELS = ["-1 미만", "-1 이상 0 미만", "0 이상 1 미만", "1 이상"]


def run_population(cfg):
    rows, weekly, pairs, per_user = [], [], [], []
    for seed in cfg["simulation"]["eval_seeds"]:
        users, res = run_pair(cfg, seed, policies=POLICY_ORDER)
        cur, new = res["current"], res["new"]
        per_user.append(pd.DataFrame({
            "motivation": users["m0"], "current": user_rates(cur), "new": user_rates(new),
            "new_difficulty": new["difficulty"].sum(1) / new["assigned"].sum(1),
        }))
        for name, r in res.items():
            rows.append({"seed": seed, "policy": name, **summarize(r, cfg["domains"])})
            weekly += [{"seed": seed, "policy": name, "week": w + 1, "completion": v}
                       for w, v in enumerate(weekly_completion(r))]
        pairs.append({"seed": seed, **paired(cur, new)})
    return pd.DataFrame(rows), pd.DataFrame(weekly), pairs, motivation_groups(pd.concat(per_user))


def motivation_groups(df):
    """Pooled over seeds: who gains and who loses, by latent baseline motivation."""
    df = df.assign(group=pd.cut(df.motivation, MOTIVATION_BINS, right=False, labels=MOTIVATION_LABELS), diff=df.new - df.current)
    return df.groupby("group", observed=True).agg(
        users=("diff", "size"), current=("current", "mean"), new=("new", "mean"),
        worse=("diff", lambda d: int((d < 0).sum())), new_difficulty=("new_difficulty", "mean"),
    ).reset_index()


def run_personas(cfg):
    sim, d = cfg["simulation"], len(cfg["domains"])
    rows = []
    for i, (name, persona) in enumerate(cfg["personas"].items()):
        users = make_persona_runs(persona, sim["persona_runs"], d)
        draws = make_draws(np.random.default_rng(sim["persona_seed"] + i), cfg, sim["persona_runs"])
        for policy in ("current", "new"):
            r = simulate(cfg, users, draws, policy)
            rate = user_rates(r)
            rows.append({
                "persona": name, "policy": policy,
                "completion_median": np.median(rate), "completion_min": rate.min(), "completion_max": rate.max(),
                "notifs_per_day": np.median(r["sends"] / r["days"]),
                "opens_per_day": np.median(r["opens"] / r["days"]),
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    cal = calibrate(cfg)
    cfg = with_base_logit(cfg, cal["base_logit"])
    cal["eval_rate"] = current_completion(cfg, cfg["simulation"]["eval_seeds"])

    summary, weekly, pairs, groups = run_population(cfg)
    personas = run_personas(cfg)

    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    write_figures(summary, weekly, pairs, cfg, out / "figures")
    write_results(cal, summary, pairs, groups, personas, cfg, out / "RESULTS.md")
    metrics = {
        "calibration": {k: float(v) for k, v in cal.items()},
        "summary": summary.to_dict(orient="records"),
        "paired": [{k: v for k, v in p.items() if k != "diffs"} for p in pairs],
        "personas": personas.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"calibration gap {cal['gap_pp']:+.2f}pp; wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
