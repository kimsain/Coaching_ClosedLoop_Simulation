"""Closed-loop simulation of two notification policies for a coaching app."""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "policies.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def with_base_logit(cfg, base_logit):
    """Return a copy of cfg with the calibrated user-model intercept."""
    out = copy.deepcopy(cfg)
    out["user_model"]["base_logit"] = float(base_logit)
    return out
