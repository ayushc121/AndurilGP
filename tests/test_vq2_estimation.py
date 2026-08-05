"""VQ2 state estimation, vision fusion, and guidance behaviour."""

import math
import time

import numpy as np
import pytest

from vq2 import controller as vq2
from vq2 import estimator as vq2_est


# Attitude estimation

def test_ahrs_seeds_at_launch_pitch():
    """The drone starts on an angled block; assuming level starts us wrong."""
    ahrs = vq2.GyroAHRS(initial_pitch_deg=vq2.LAUNCH_PITCH_DEG)
    _, pitch, _ = ahrs.euler_deg()
    assert pitch == pytest.approx(vq2.LAUNCH_PITCH_DEG, abs=1e-6)


def test_ahrs_holds_still_with_no_rotation():
    """Zero gyro input must not drift the attitude estimate."""
    ahrs = vq2.GyroAHRS(initial_pitch_deg=-10.0)
    for _ in range(500):
        ahrs.update(0.0, 0.0, 0.0, 1 / 120)
    _, pitch, _ = ahrs.euler_deg()
    assert pitch == pytest.approx(-10.0, abs=1e-6)


def test_ahrs_integrates_a_known_yaw_rate():
    """One second at 30 deg/s about the yaw axis is 30 degrees of yaw."""
    ahrs = vq2.GyroAHRS()
    rate = math.radians(30.0)
    for _ in range(1200):
        ahrs.update(0.0, 0.0, rate, 1 / 1200)
    _, _, yaw = ahrs.euler_deg()
    assert yaw == pytest.approx(30.0, abs=0.1)


def test_ahrs_keeps_quaternion_normalised():
    """Integration must renormalise, or the rotation silently gains scale."""
    ahrs = vq2.GyroAHRS()
    for i in range(2000):
        ahrs.update(0.5 * math.sin(i / 50), 0.3, -0.2, 1 / 120)
    assert np.linalg.norm(ahrs.quaternion) == pytest.approx(1.0, abs=1e-9)


# Fixtures for the controller under test

@pytest.fixture
def ctrl(sim_conn, shared_data):
    controller = vq2.Controller(sim_conn, shared_data, 0)
    controller.phase = vq2.Phase.FLYING
    return controller


def detection(fwd=20.0, right=0.0, down=0.0, frame_id=1, pnp=False):
    estimate = {'body_x_m': fwd, 'body_y_m': right, 'body_z_m': down,
                'frame_id': frame_id, 'pnp_ok': pnp}
    if pnp:
        estimate['pnp_rvec'] = [0.0, 0.0, 0.0]
    return estimate


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# Gate observation

def test_no_detection_is_not_valid(ctrl):
    assert not ctrl._observe_gate(None, IDENTITY_QUAT).valid


def test_gate_behind_camera_is_rejected(ctrl):
    """Negative forward distance is geometrically impossible, not just unlikely."""
    assert not ctrl._observe_gate(detection(fwd=-2.0), IDENTITY_QUAT).valid


def test_nan_pose_is_rejected(ctrl):
    """A degenerate detection must not reach the steering maths."""
    assert not ctrl._observe_gate(detection(fwd=float('nan')), IDENTITY_QUAT).valid


def test_bearing_points_toward_an_offset_gate(ctrl):
    """A gate to the right must give a positive bearing."""
    view = ctrl._observe_gate(detection(fwd=20.0, right=5.0), IDENTITY_QUAT)
    assert view.bearing_deg == pytest.approx(math.degrees(math.atan2(5, 20)), abs=1e-6)


def test_bearing_is_clamped(ctrl):
    """An extreme bearing must saturate rather than command a violent bank."""
    view = ctrl._observe_gate(detection(fwd=1.0, right=50.0), IDENTITY_QUAT)
    assert view.bearing_deg == pytest.approx(vq2.BEARING_CLAMP_DEG)


def test_blend_fades_roll_out_near_the_gate(ctrl):
    """Far out, full roll authority; at the gate plane, none."""
    far = ctrl._observe_gate(detection(fwd=30.0), IDENTITY_QUAT)
    near = ctrl._observe_gate(detection(fwd=0.5, frame_id=2), IDENTITY_QUAT)
    assert far.blend == 1.0
    assert near.blend < 0.1


def test_elevation_error_freezes_when_the_gate_is_lost(ctrl):
    """Losing the gate must not snap thrust to hover mid-correction."""
    ctrl._observe_gate(detection(fwd=20.0, down=4.0), IDENTITY_QUAT)
    held = ctrl._elev_err
    assert held != 0.0
    ctrl._observe_gate(None, IDENTITY_QUAT)
    assert ctrl._elev_err == held


def test_close_range_geometry_is_not_trusted_for_elevation(ctrl):
    """Inside MIN_RANGE_FOR_ELEV_M the gate fills the frame; hold, don't update."""
    ctrl._observe_gate(detection(fwd=20.0, down=4.0), IDENTITY_QUAT)
    held = ctrl._elev_err
    ctrl._observe_gate(detection(fwd=1.0, down=99.0, frame_id=2), IDENTITY_QUAT)
    assert ctrl._elev_err == held


def test_rate_is_zero_across_a_detection_gap(ctrl):
    """Distant frames mean a re-acquisition; differencing them invents a rate."""
    ctrl._observe_gate(detection(fwd=20.0, right=0.0, frame_id=1), IDENTITY_QUAT)
    view = ctrl._observe_gate(detection(fwd=20.0, right=8.0, frame_id=99), IDENTITY_QUAT)
    assert view.bearing_rate == 0.0


def test_rate_is_computed_across_adjacent_frames(ctrl):
    """Consecutive frames are the case the derivative is meant for."""
    ctrl._observe_gate(detection(fwd=20.0, right=0.0, frame_id=1), IDENTITY_QUAT)
    view = ctrl._observe_gate(detection(fwd=20.0, right=4.0, frame_id=2), IDENTITY_QUAT)
    assert view.bearing_rate > 0.0


# Velocity fusion

def test_fusion_pulls_velocity_toward_vision(ctrl):
    """Vision has to move the estimate, or it cannot bound IMU drift."""
    vision_vel = {'vy_body_mps': 10.0, 'vz_body_mps': 0.0}
    v_right, _, source = ctrl._fuse_velocity(vision_vel, 1, 0.0, 0.0)
    assert source == 'fused'
    assert 0.0 < v_right < 10.0


def test_each_camera_frame_is_fused_once(ctrl):
    """Re-fusing the same frame would over-weight one measurement."""
    vision_vel = {'vy_body_mps': 10.0, 'vz_body_mps': 0.0}
    ctrl._fuse_velocity(vision_vel, 1, 0.0, 0.0)
    v_right, _, source = ctrl._fuse_velocity(vision_vel, 1, 0.0, 0.0)
    assert source == 'imu'
    assert v_right == 0.0


def test_fusion_only_touches_the_two_corrected_axes(ctrl):
    """Fusion returns lateral and vertical only — forward stays on the IMU."""
    result = ctrl._fuse_velocity({'vy_body_mps': 10.0, 'vz_body_mps': 10.0},
                                 1, 0.0, 0.0)
    assert len(result) == 3          # (v_right, v_down, source)
    assert isinstance(result[2], str)


# Damping

def test_damping_is_zero_without_a_gate(ctrl):
    """Damping against a target we cannot see is worse than no damping."""
    view = ctrl._observe_gate(None, IDENTITY_QUAT)
    lateral, vertical, source = ctrl._damping_terms(view, 5.0, 5.0)
    assert (lateral, vertical, source) == (0.0, 0.0, 'none')


def test_damping_clamps_an_absurd_bearing_rate(ctrl):
    """One bad detection gives a triple-digit derivative; the clamp bounds it."""
    view = ctrl._observe_gate(detection(fwd=25.0, frame_id=5), IDENTITY_QUAT)
    view.bearing_rate = 5000.0
    lateral, _, _ = ctrl._damping_terms(view, 0.0, 0.0)
    limit = math.radians(vq2.BEARING_RATE_CLAMP_DEG_S) * 25.0
    assert abs(lateral) <= limit + 1e-9


def test_damping_falls_back_to_imu_between_frames(ctrl):
    """Between camera frames the IMU carries the derivative for ~33 ms."""
    view = ctrl._observe_gate(detection(fwd=20.0, frame_id=7), IDENTITY_QUAT)
    ctrl._damping_terms(view, 1.0, 1.0)
    _, _, source = ctrl._damping_terms(view, 2.5, 2.0)
    assert source == 'imu'


# Command output

def test_thrust_is_tilt_compensated(ctrl, sim_conn):
    """Banking must not cost altitude: thrust rises to cover the lift lost."""
    ctrl._att_deg = (0.0, 0.0, 0.0)
    ctrl._fly()
    level = sim_conn.mav.attitude_targets[-1]['thrust']

    ctrl._att_deg = (30.0, 0.0, 0.0)
    ctrl._fly()
    banked = sim_conn.mav.attitude_targets[-1]['thrust']
    assert banked > level


def test_thrust_stays_in_range(ctrl, sim_conn):
    """A saturated command must clip, not wrap or go negative."""
    ctrl._elev_err = -1e6
    ctrl._elev_err_at = time.monotonic()
    ctrl._fly()
    thrust = sim_conn.mav.attitude_targets[-1]['thrust']
    assert 0.0 <= thrust <= 1.0


def test_gate_below_reduces_thrust(ctrl, sim_conn):
    """Positive elevation error means the gate is below us — descend toward it."""
    ctrl._att_deg = (0.0, 0.0, 0.0)
    ctrl._elev_err = 0.0
    ctrl._elev_err_at = time.monotonic()
    ctrl._fly()
    neutral = sim_conn.mav.attitude_targets[-1]['thrust']

    ctrl._elev_err = 5.0
    ctrl._elev_err_at = time.monotonic()
    ctrl._fly()
    assert sim_conn.mav.attitude_targets[-1]['thrust'] < neutral


def test_elevation_hold_expires(ctrl):
    """A held elevation error must stop applying once it is too old."""
    ctrl._elev_err = 4.32
    ctrl._elev_err_at = time.monotonic()
    assert ctrl._held_elev_err() == pytest.approx(4.32)

    ctrl._elev_err_at = time.monotonic() - (vq2.ELEV_HOLD_MAX_S + 0.1)
    assert ctrl._held_elev_err() == 0.0


def test_elevation_hold_survives_a_short_blackout(ctrl):
    """Inside the window the value is still applied — that is the point of it."""
    ctrl._elev_err = 2.0
    ctrl._elev_err_at = time.monotonic() - (vq2.ELEV_HOLD_MAX_S * 0.5)
    assert ctrl._held_elev_err() == pytest.approx(2.0)


def test_no_elevation_measurement_means_no_command(ctrl):
    """Before any gate is seen there is nothing to hold, so thrust sees zero."""
    assert ctrl._elev_err_at is None
    assert ctrl._held_elev_err() == 0.0


def test_commands_use_absolute_attitude_mask(ctrl, sim_conn):
    """Type mask 7 tells the sim to use the quaternion and ignore body rates."""
    ctrl._fly()
    assert sim_conn.mav.attitude_targets[-1]['type_mask'] == 7


def test_reset_clears_vision_state_but_keeps_estimator_running(ctrl):
    """A sim reset must not leave a previous run's gate memory behind."""
    ctrl._elev_err = 12.0
    ctrl._tilt_ema = 4.0
    ctrl._reset_flight_state()
    assert ctrl._elev_err == 0.0
    assert ctrl._tilt_ema is None
    assert ctrl.phase is vq2.Phase.WAIT_FOR_DATA


# launch yaw and the acceleration cap

def test_yaw_starts_at_the_pad_heading(ctrl):
    """Yaw is absolute, so it has to begin at the heading the drone is on."""
    assert ctrl._yaw_cmd == pytest.approx(vq2.LAUNCH_YAW_CMD_DEG)


def test_yaw_holds_when_there_is_no_gate(ctrl):
    """No detection means no reason to move the setpoint."""
    view = ctrl._observe_gate(None, IDENTITY_QUAT)
    assert ctrl._step_yaw(view) == pytest.approx(vq2.LAUNCH_YAW_CMD_DEG)


def test_yaw_step_is_rate_limited(ctrl):
    """One noisy bearing must not snap the nose across the course."""
    view = ctrl._observe_gate(detection(fwd=1.0, right=50.0), IDENTITY_QUAT)
    before = ctrl._yaw_cmd
    after = ctrl._step_yaw(view)
    assert abs(after - before) <= vq2.YAW_RATE_LIMIT_DEG_S / vq2.CONTROL_HZ + 1e-9


def test_yaw_walks_toward_an_offset_gate(ctrl):
    """Repeated steps should accumulate in one direction."""
    start = ctrl._yaw_cmd
    for i in range(20):
        view = ctrl._observe_gate(detection(fwd=20.0, right=5.0, frame_id=i + 1),
                                  IDENTITY_QUAT)
        ctrl._step_yaw(view)
    assert ctrl._yaw_cmd != pytest.approx(start)


def test_yaw_setpoint_stays_wrapped(ctrl):
    """The setpoint must stay in [-180, 180) however far it is walked."""
    for i in range(400):
        view = ctrl._observe_gate(detection(fwd=2.0, right=40.0, frame_id=i + 1),
                                  IDENTITY_QUAT)
        ctrl._step_yaw(view)
        assert -180.0 <= ctrl._yaw_cmd < 180.0


def test_accel_cap_does_not_freeze_a_free_fall():
    """
    At zero commanded thrust the drone is falling and must read as falling.

    Capping total acceleration instead of the thrust part returns zero below
    hover, which scaled a real +9.81 down to nothing and inverted the sign the
    thrust D term reads.
    """
    level = (1.0, 0.0, 0.0, 0.0)
    a_ned, _, _ = vq2.body_accel_to_ned(level, 0.0, 0.0, 0.0, cmd_thrust=0.0)
    assert a_ned[2] == pytest.approx(vq2.G, abs=1e-9)


def test_accel_cap_bounds_an_implausible_reading():
    """A wild accelerometer sample is clipped to what the thrust could produce."""
    level = (1.0, 0.0, 0.0, 0.0)
    a_ned, max_accel, saturated = vq2.body_accel_to_ned(
        level, 500.0, 0.0, 0.0, cmd_thrust=vq2.HOVER_THRUST)
    assert saturated
    assert a_ned[0] == pytest.approx(max_accel, rel=1e-9)


def test_accel_cap_passes_a_plausible_reading_through():
    """Inside the bound nothing is altered."""
    level = (1.0, 0.0, 0.0, 0.0)
    a_ned, _, saturated = vq2.body_accel_to_ned(
        level, 1.0, 0.0, 0.0, cmd_thrust=vq2.HOVER_THRUST)
    assert not saturated
    assert a_ned[0] == pytest.approx(1.0)


# gate-relative VIO

@pytest.fixture
def vio(shared_data):
    from vq2.estimator import GateVIO
    return GateVIO(shared_data)


def _imu(ts_us, ax=0.0, ay=0.0, az=-9.81):
    return {'time_usec': ts_us, 'xacc': ax, 'yacc': ay, 'zacc': az,
            'xgyro': 0.0, 'ygyro': 0.0, 'zgyro': 0.0}


def _pnp(fwd=15.0, right=0.0, down=0.0, frame_id=1):
    return {'body_x_m': fwd, 'body_y_m': right, 'body_z_m': down,
            'frame_id': frame_id, 'pnp_ok': True, 'bw': 60, 'bh': 60}


def test_vio_starts_stale(vio):
    """Nothing has anchored it yet, so nothing may steer on it."""
    est = vio.get_estimate()
    assert est.stale
    assert not est.valid


def test_first_fix_anchors_and_clears_stale(vio):
    vio._vision_step(_pnp())
    est = vio.get_estimate()
    assert est.valid
    assert not est.stale


def test_gate_body_points_back_at_the_gate(vio):
    """The estimate's job is a vector TO the gate, in the body frame."""
    vio._vision_step(_pnp(fwd=15.0))
    est = vio.get_estimate()
    assert est.gate_body[0] == pytest.approx(15.0, abs=0.5)


def test_range_filter_rejects_an_impossible_detection(vio):
    """Beyond the range window a fix is the far-lock artefact, not a gate."""
    vio._vision_step(_pnp(fwd=500.0))
    assert not vio.get_estimate().valid


def test_aspect_filter_rejects_a_clipped_box(vio):
    """A gate opening stays roughly square; an extreme ratio means a clipped contour."""
    bad = _pnp()
    bad['bw'], bad['bh'] = 200, 20
    vio._vision_step(bad)
    assert not vio.get_estimate().valid


def test_a_dark_run_forces_a_handoff(vio):
    """Sustained blackout means we flew through — the anchor is the old gate."""
    vio._vision_step(_pnp())
    assert not vio.get_estimate().stale
    for _ in range(vq2_est.REANCHOR_DARK_FRAMES):
        vio._vision_step(None)
    assert vio.get_estimate().stale


def test_reanchor_keeps_velocity(vio):
    """
    A handoff must not throw the velocity estimate away.

    World velocity is gate-independent, so it carries across a handoff; the
    whole point of the filter is that it stays clean through one.
    """
    vio._vision_step(_pnp(frame_id=1))
    with vio._lock:
        vio.ekf.x[3:6] = np.array([3.0, 0.0, 0.0])
    for _ in range(vq2_est.REANCHOR_DARK_FRAMES):
        vio._vision_step(None)
    vio._vision_step(_pnp(fwd=12.0, frame_id=2))
    assert vio.get_estimate().vel[0] == pytest.approx(3.0, abs=1e-6)
    assert vio.n_reanchor == 1


def test_repeated_frame_is_not_fused_twice(vio):
    """The 400 Hz loop sees each 30 Hz frame more than once."""
    vio._vision_step(_pnp(frame_id=7))
    before = vio.n_accepted
    vio._vision_step(_pnp(frame_id=7))
    assert vio.n_accepted == before


def test_estimate_reads_nothing_before_it_is_anchored(vio):
    """Predicting alone must not produce a velocity anyone could steer on."""
    vio._predict(_imu(1_000_000), cmd_thrust=0.3)
    vio._predict(_imu(1_100_000, ax=2.0), cmd_thrust=0.3)
    est = vio.get_estimate()
    assert not est.valid
    assert est.vel[0] == 0.0


def test_predict_moves_the_estimate_once_anchored(vio):
    """After a fix, a real acceleration has to show up as velocity."""
    vio._vision_step(_pnp())
    vio._predict(_imu(1_000_000), cmd_thrust=0.3)
    vio._predict(_imu(1_100_000, ax=2.0), cmd_thrust=0.3)
    assert abs(vio.get_estimate().vel[0]) > 0.0


def test_range_noise_grows_with_range():
    """Far fixes are weak on the range axis and must be weighted that way."""
    assert vq2_est.sigma_range_for(40.0) > vq2_est.sigma_range_for(10.0)
    assert vq2_est.sigma_range_for(5.0) == vq2_est.EKF_SIGMA_RANGE_FLOOR


def test_yaw_consumes_each_camera_frame_once(ctrl):
    """A 30 Hz frame spans two 60 Hz ticks; stepping twice doubles the slew."""
    view = ctrl._observe_gate(detection(fwd=20.0, right=5.0, frame_id=7),
                              IDENTITY_QUAT)
    first = ctrl._step_yaw(view)
    assert ctrl._step_yaw(view) == pytest.approx(first)
