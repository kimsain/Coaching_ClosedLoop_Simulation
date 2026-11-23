"""Virtual users: latent traits that the policies never see directly."""

import numpy as np


def make_population(rng, cfg):
    """Draw n_users with latent motivation, habit, time constraint and fatigue sensitivity."""
    p = cfg["population"]
    n = cfg["simulation"]["n_users"]
    n_domains, n_slots = len(cfg["domains"]), len(cfg["slots"])

    pref = rng.integers(0, n_slots, n)
    # Stated slot equals the real best slot with some probability, otherwise another slot.
    other = (pref + rng.integers(1, n_slots, n)) % n_slots
    stated = np.where(rng.random(n) < p["stated_slot_accuracy"], pref, other)

    a, b = p["time_constraint_beta"]
    lo, hi = p["fatigue_sensitivity"]
    return {
        "m0": rng.normal(p["motivation_mean"], p["motivation_sd"], n),
        "tc": rng.beta(a, b, n),
        "fs": rng.uniform(lo, hi, n),
        "pref": pref,
        "stated": stated,
        "has_med": rng.random(n) < p["medication_share"],
        "h0": rng.uniform(0, p["habit_init_max"], (n, n_domains)),
    }


def make_persona_runs(persona, n_runs, n_domains):
    """Replicate one named persona n_runs times; each copy gets its own random draws."""
    n = n_runs
    return {
        "m0": np.full(n, persona["motivation"], float),
        "tc": np.full(n, persona["time_constraint"], float),
        "fs": np.full(n, persona["fatigue_sensitivity"], float),
        "pref": np.full(n, persona["pref_slot"]),
        "stated": np.full(n, persona["stated_slot"]),
        "has_med": np.full(n, persona["medication"], bool),
        "h0": np.full((n, n_domains), persona["habit"], float),
    }


def make_draws(rng, cfg, n):
    """Pre-drawn random numbers shared by both policies (common random numbers)."""
    days = cfg["simulation"]["weeks"] * 7
    return {
        "open": rng.random((days, n, len(cfg["slots"]))),
        "complete": rng.random((days, n, len(cfg["domains"]))),
        "time": rng.normal(0, cfg["population"]["time_noise_sd"], (days, n)),
    }
