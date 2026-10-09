"""
Pool-mode tests (dryrun)

Run:
    pip3 install pytest
    pytest -q
"""
import queue
import random
import tempfile
import time

import numpy as np
import controller_all_in_one_poolmode as ctrl


def _make_cfg(tmpdir: str, **overrides):
    d = ctrl.config_template()
    d.update({
        "hal": "dryrun",
        "run_base_dir": tmpdir,
        "save_npz": False,
        "npz_every_n": 9999,
        "pca_address": 0x40,
        "left_thruster_ch": 0,
        "right_thruster_ch": 1,
        "servo_ch": 2,

        "pool_mode": True,
        "pool_search_timeout_s": 10.0,
        "pool_search_rotate_s": 0.2,
        "pool_search_pause_s": 0.1,
        "pool_search_forward_s": 0.2,
        "pool_search_cycles_per_forward": 1,

        # make detection easier
        "blob_temp_threshold_c": 28.0,
        "forward_cmd": 0.0,
        "verify_votes": 1,
        "verify_accept_count": 1,
        "verify_timeout_s": 2.0,
    })
    d.update(overrides)
    return ctrl.parse_config(d)


def test_starts_in_search_pattern_when_pool_mode_enabled():
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_cfg(td)
        rec = ctrl.RunRecorder(cfg.run_base_dir, enable_npz=False, npz_every_n=9999)
        hal = ctrl.make_hal(cfg)
        latest = ctrl.LatestValue()
        verify_in = queue.Queue()
        verify_out = queue.Queue()
        stop_event = ctrl.threading.Event()

        controller = ctrl.Controller(cfg, rec, latest, hal.thrusters, hal.servo, verify_in, verify_out, stop_event)
        assert controller.ctx.state == ctrl.State.SEARCH_PATTERN


def test_search_pattern_transitions_to_coarse_align_when_target_detected():
    random.seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_cfg(td)
        rec = ctrl.RunRecorder(cfg.run_base_dir, enable_npz=False, npz_every_n=9999)
        hal = ctrl.make_hal(cfg)
        latest = ctrl.LatestValue()
        verify_in = queue.Queue()
        verify_out = queue.Queue()
        stop_event = ctrl.threading.Event()

        controller = ctrl.Controller(cfg, rec, latest, hal.thrusters, hal.servo, verify_in, verify_out, stop_event)

        # Feed frames and step until it sees a stable target and transitions
        for _ in range(200):
            latest.set(hal.thermal.read())
            controller.step()
            if controller.ctx.state == ctrl.State.COARSE_ALIGN:
                break
            time.sleep(0.001)

        assert controller.ctx.state == ctrl.State.COARSE_ALIGN
