import copy

import numpy as np
import pytest

from closedloop import load_config, with_base_logit
from closedloop.calibrate import calibrate
from closedloop.policies import notification_cap
from closedloop.run_all import POLICY_ORDER
from closedloop.simulate import run_pair


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def calibrated(cfg):
    cal = calibrate(cfg)
    return cal, with_base_logit(cfg, cal["base_logit"])


def test_same_seed_same_result(cfg):
    _, a = run_pair(cfg, 7)
    _, b = run_pair(cfg, 7)
    _, c = run_pair(cfg, 8)
    for name in a:
        for key in ("assigned", "completed", "sends", "opens"):
            np.testing.assert_array_equal(a[name][key], b[name][key])
    assert not np.array_equal(a["current"]["completed"], c["current"]["completed"])


def test_calibration_within_tolerance(cfg, calibrated):
    cal, _ = calibrated
    assert abs(cal["gap_pp"]) <= cfg["calibration"]["tolerance_pp"]


def test_notification_caps(calibrated):
    _, cfg = calibrated
    _, res = run_pair(cfg, 11, policies=POLICY_ORDER)
    for name, r in res.items():
        assert r["max_daily_sends"].max() <= notification_cap(cfg, name), name
    # The current policy pushes every user every day, inactive or not.
    assert (res["current"]["sends"] == notification_cap(cfg, "current") * res["current"]["days"]).all()


def test_stop_rule_cuts_notifications_to_non_openers(calibrated):
    _, cfg = calibrated
    cfg = copy.deepcopy(cfg)
    cfg["user_model"]["open_base"] = -50.0  # nobody ever opens
    _, res = run_pair(cfg, 11)
    days = res["new"]["days"]
    assert res["new"]["opens"].sum() == 0
    # With a 4-send stop rule and 7-day pauses, a non-opener gets well under one notification a day.
    assert res["new"]["sends"].max() / days < 0.4
    assert res["current"]["sends"].min() / days == 2
