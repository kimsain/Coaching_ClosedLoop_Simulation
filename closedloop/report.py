"""Figures and the generated reports/RESULTS.md."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"current": "#eb6834", "new": "#2a78d6"}
EN = {"current": "Current", "new": "New"}
KO = {"current": "현행", "new": "새 방식", "new_notify_only": "새 알림 + 현행 루틴",
      "new_routine_only": "새 루틴 + 현행 알림"}
ROWS = [  # (metric, label, is_rate)
    ("completion", "루틴 완료율(%)", True),
    ("user_mean_completion", "사용자 평균 완료율(%)", True),
    ("notifs_per_user_day", "사용자 1명 하루 알림 수", False),
    ("open_rate", "알림 1건당 열람률(%)", True),
    ("opens_per_user_day", "사용자 1명 하루 열람 수", False),
    ("routines_per_user_day", "사용자 1명 하루 배정 루틴 수", False),
    ("completed_per_user_day", "사용자 1명 하루 완료 루틴 수", False),
    ("mean_difficulty", "배정 루틴 평균 난이도(1~3)", False),
]

plt.rcParams.update({
    "font.family": ["Pretendard", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def _fmt(values, is_rate):
    k = 100 if is_rate else 1
    d = 1 if is_rate else 2
    return f"{values.mean() * k:.{d}f} ({values.min() * k:.{d}f}, {values.max() * k:.{d}f})"


def write_figures(summary, weekly, pairs, cfg, out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for p in ("current", "new"):
        g = weekly[weekly.policy == p].groupby("week").completion
        mean, lo, hi = g.mean() * 100, g.min() * 100, g.max() * 100
        ax.fill_between(mean.index, lo, hi, color=COLORS[p], alpha=0.15, linewidth=0)
        ax.plot(mean.index, mean, color=COLORS[p], linewidth=2, marker="o", markersize=4)
        ax.annotate(EN[p], (mean.index[-1], mean.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color="#0b0b0b")
    ax.set(xlabel="Week", ylabel="Routine completion (%)", title="Weekly completion, mean and seed range")
    ax.set_xticks(range(1, cfg["simulation"]["weeks"] + 1))
    fig.savefig(out / "weekly_completion.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x, w = np.arange(len(cfg["domains"])), 0.38
    for i, p in enumerate(("current", "new")):
        vals = [summary[summary.policy == p][f"completion_{d}"].mean() * 100 for d in cfg["domains"]]
        bars = ax.bar(x + (i - 0.5) * w, vals, w - 0.02, color=COLORS[p], label=EN[p])
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, color="#52514e")
    ax.set_xticks(x, cfg["domains"])
    ax.set(ylabel="Completion (%)", title="Completion by routine domain (mean of seeds)")
    ax.legend(frameon=False)
    fig.savefig(out / "domain_completion.png")
    plt.close(fig)

    panels = [("notifs_per_user_day", "Notifications / user / day", 1),
              ("open_rate", "Open rate per notification (%)", 100),
              ("opens_per_user_day", "Opens / user / day", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (col, title, k) in zip(axes, panels):
        vals = [summary[summary.policy == p][col].mean() * k for p in ("current", "new")]
        bars = ax.bar([EN["current"], EN["new"]], vals, color=[COLORS["current"], COLORS["new"]], width=0.6)
        ax.bar_label(bars, fmt="%.2f" if k == 1 else "%.1f", padding=2, fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "notifications.png")
    plt.close(fig)

    diffs = np.concatenate([p["diffs"] for p in pairs]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(diffs, bins=40, color=COLORS["new"], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set(xlabel="New minus current, same user (pp)", ylabel="Users",
           title=f"Within-user difference, {len(diffs)} user-runs pooled over seeds")
    fig.savefig(out / "paired_difference.png")
    plt.close(fig)


def write_results(cal, summary, pairs, groups, personas, cfg, path):
    sim = cfg["simulation"]
    seeds = ", ".join(map(str, sim["eval_seeds"]))
    by = {p: g for p, g in summary.groupby("policy")}
    L = [
        "# 시뮬레이션 결과",
        "",
        "이 문서는 `python -m closedloop.run_all`이 만듭니다. 모든 숫자는 합성 데이터로 돌린 시뮬레이션 결과이고, "
        "실제 사용자에게서 측정한 효과가 아닙니다.",
        "",
        f"평가 시드는 {seeds}이고 시드마다 가상 사용자 {sim['n_users']}명을 {sim['weeks']}주 동안 돌립니다. "
        "표의 값은 시드 평균이고 괄호 안은 시드별 최솟값과 최댓값입니다.",
        "",
        "## 1. 보정",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 목표 완료율(합성 관측값) | {cal['target'] * 100:.1f}% |",
        f"| 보정 시드({', '.join(map(str, sim['calibration_seeds']))})의 현행 완료율 | {cal['rate'] * 100:.2f}% |",
        f"| 보정 차이 | {cal['gap_pp']:+.2f}%p (허용 범위 ±{cfg['calibration']['tolerance_pp']:.1f}%p) |",
        f"| 평가 시드의 현행 완료율 | {cal['eval_rate'] * 100:.2f}% (목표와 {(cal['eval_rate'] - cal['target']) * 100:+.2f}%p) |",
        f"| 보정한 사용자 모형 절편(base_logit) | {cal['base_logit']:.4f} |",
        "",
        "## 2. 집단 결과",
        "",
        "| 지표 | 현행 | 새 방식 |",
        "|---|---|---|",
    ]
    for col, label, is_rate in ROWS:
        L.append(f"| {label} | {_fmt(by['current'][col], is_rate)} | {_fmt(by['new'][col], is_rate)} |")

    L += ["", "## 3. 영역별 완료율(%)", "", "| 영역 | 현행 | 새 방식 |", "|---|---|---|"]
    for d in cfg["domains"]:
        L.append(f"| {d} | {_fmt(by['current'][f'completion_{d}'], True)} | {_fmt(by['new'][f'completion_{d}'], True)} |")

    L += ["", "## 4. 같은 사용자끼리 비교", "",
          "두 방식은 같은 사용자와 같은 난수를 씁니다. 차이는 새 방식에서 현행을 뺀 값입니다. "
          "전체 차이는 사용자 안 차이와 구성 효과의 합입니다. 구성 효과는 사용자별 배정 루틴 수가 달라져 생기는 몫입니다.",
          "", "| 시드 | 사용자 평균 차이(%p) | 중앙값(%p) | 좋아진 사용자 | 나빠진 사용자 | 전체 차이(%p) | 사용자 안 차이(%p) | 구성 효과(%p) |",
          "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p['seed']} | {p['mean_diff'] * 100:+.1f} | {p['median_diff'] * 100:+.1f} | "
                 f"{p['n_improved']}/{p['n_users']} | {p['n_worse']}/{p['n_users']} | {p['gap_total'] * 100:+.1f} | "
                 f"{p['gap_within_user'] * 100:+.1f} | {p['gap_composition'] * 100:+.2f} |")

    L += ["", "기초 의욕(motivation, 잠재 변수) 구간별로 모든 평가 시드를 합친 결과입니다.", "",
          "| 기초 의욕 구간 | 사용자 수 | 현행 완료율(%) | 새 방식 완료율(%) | 나빠진 사용자 | 새 방식 평균 난이도 |",
          "|---|---|---|---|---|---|"]
    for _, g in groups.iterrows():
        L.append(f"| {g.group} | {g.users} | {g.current * 100:.1f} | {g.new * 100:.1f} | {g.worse} | {g.new_difficulty:.2f} |")

    L += ["", "## 5. 차이를 나눠 보는 실험", "",
          "새 방식의 알림 규칙과 루틴 규칙을 하나씩만 현행에 붙여 돌린 결과입니다.", "",
          "| 방식 | 루틴 완료율(%) | 하루 알림 수 | 하루 열람 수 | 평균 난이도 |", "|---|---|---|---|---|"]
    for name, label in KO.items():
        g = by[name]
        L.append(f"| {label} | {_fmt(g['completion'], True)} | {_fmt(g['notifs_per_user_day'], False)} | "
                 f"{_fmt(g['opens_per_user_day'], False)} | {_fmt(g['mean_difficulty'], False)} |")

    L += ["", f"## 6. 대표 페르소나({sim['persona_runs']}회 반복)", "",
          "완료율은 반복 실행의 중앙값이고 괄호 안은 최솟값과 최댓값입니다.", "",
          "| 페르소나 | 방식 | 완료율(%) | 하루 알림 수 | 하루 열람 수 |", "|---|---|---|---|---|"]
    for _, r in personas.iterrows():
        L.append(f"| {r.persona} | {KO[r.policy]} | {r.completion_median * 100:.1f} "
                 f"({r.completion_min * 100:.1f}, {r.completion_max * 100:.1f}) | "
                 f"{r.notifs_per_day:.2f} | {r.opens_per_day:.2f} |")

    L += ["", "## 그림", ""] + [f"![{f}](figures/{f}.png)" for f in
                                 ("weekly_completion", "domain_completion", "notifications", "paired_difference")]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
