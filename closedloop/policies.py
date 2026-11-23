"""The two policies. Each one only sees observed outcomes (sends, opens, completions)."""

import numpy as np


class CurrentPolicy:
    """Same routines at the same difficulty and fixed pushes to every user, every day."""

    def __init__(self, params, eligible, stated, n_slots):
        self.eligible = eligible
        self.difficulty = np.full(eligible.shape, params["difficulty"])
        self.sends = np.zeros((eligible.shape[0], n_slots), bool)
        self.sends[:, params["push_slots"]] = True

    def routines(self):
        return self.eligible, self.difficulty

    def notifications(self):
        return self.sends.copy()

    def observe(self, sends, opens, assigned, completed):
        pass


class NewPolicy:
    """Adaptive: learned timing, stop rules after non-response, feedback on difficulty."""

    def __init__(self, params, eligible, stated, n_slots):
        n, d = eligible.shape
        self.p = params
        self.eligible = eligible
        self.difficulty = np.full((n, d), params["start_difficulty"])
        self.done_streak = np.zeros((n, d), int)
        self.miss_streak = np.zeros((n, d), int)
        self.done_count = np.zeros((n, d), int)
        self.idle_days = np.zeros(n, int)
        self.unopened = np.zeros(n, int)
        self.pause_left = np.zeros(n, int)
        self.slot_sends = np.zeros((n, n_slots))
        self.slot_opens = np.zeros((n, n_slots))
        self.prior = (np.arange(n_slots)[None, :] == stated[:, None]).astype(float)

    def routines(self):
        n = self.eligible.shape[0]
        dormant = self.idle_days >= self.p["dormant_after_days"]
        # A dormant user keeps only the routine they completed most often, at the easiest level.
        best = np.where(self.eligible, self.done_count, -1).argmax(1)
        single = np.zeros_like(self.eligible)
        single[np.arange(n), best] = True
        assigned = np.where(dormant[:, None], single, self.eligible)
        difficulty = np.where(dormant[:, None], 1, self.difficulty)
        return assigned, difficulty

    def notifications(self):
        n, n_slots = self.slot_sends.shape
        # Optimistic open-rate estimate per slot; the stated slot starts slightly ahead.
        est = (self.slot_opens + 1 + self.prior) / (self.slot_sends + 2 + self.prior)
        primary = est.argmax(1)
        active = self.pause_left == 0
        sends = np.zeros((n, n_slots), bool)
        sends[np.arange(n), primary] = active
        if self.p["max_per_day"] >= 2:
            lapsing = (self.idle_days >= 1) & (self.idle_days <= self.p["followup_window_days"])
            sends[np.arange(n), (primary + 1) % n_slots] |= active & lapsing
        return sends

    def observe(self, sends, opens, assigned, completed):
        p = self.p
        self.slot_sends += sends
        self.slot_opens += opens
        for s in range(sends.shape[1]):
            sent = sends[:, s]
            self.unopened = np.where(sent & opens[:, s], 0, self.unopened + sent)

        self.pause_left = np.maximum(self.pause_left - 1, 0)
        stop = self.unopened >= p["stop_after_unopened"]
        self.pause_left = np.where(stop, p["pause_days"], self.pause_left)
        # After a pause one probe is sent; if it is ignored the pause starts again.
        self.unopened = np.where(stop, p["stop_after_unopened"] - 1, self.unopened)

        self.idle_days = np.where(completed.any(1), 0, self.idle_days + 1)
        self.done_count += completed

        missed = assigned & ~completed
        self.done_streak = np.where(completed, self.done_streak + 1, np.where(missed, 0, self.done_streak))
        self.miss_streak = np.where(missed, self.miss_streak + 1, np.where(completed, 0, self.miss_streak))
        up = self.done_streak >= p["level_up_after"]
        down = self.miss_streak >= p["level_down_after"]
        self.difficulty = np.clip(self.difficulty + up - down, 1, p["max_difficulty"])
        self.done_streak[up] = 0
        self.miss_streak[down] = 0


class MixedPolicy:
    """Routines from one policy, notifications from another. Used only to split the effect."""

    def __init__(self, routine_part, notify_part):
        self.routine_part, self.notify_part = routine_part, notify_part

    def routines(self):
        return self.routine_part.routines()

    def notifications(self):
        return self.notify_part.notifications()

    def observe(self, *outcomes):
        self.routine_part.observe(*outcomes)
        self.notify_part.observe(*outcomes)


POLICIES = {"current": CurrentPolicy, "new": NewPolicy}
# name: (policy that sets routines, policy that sets notifications)
ABLATIONS = {"new_notify_only": ("current", "new"), "new_routine_only": ("new", "current")}


def make_policy(cfg, name, eligible, stated, n_slots):
    def build(key):
        return POLICIES[key](cfg["policies"][key], eligible, stated, n_slots)

    if name in POLICIES:
        return build(name)
    routine_from, notify_from = ABLATIONS[name]
    return MixedPolicy(build(routine_from), build(notify_from))


def notification_cap(cfg, name):
    name = ABLATIONS.get(name, (None, name))[1]
    p = cfg["policies"][name]
    return len(p["push_slots"]) if name == "current" else p["max_per_day"]
