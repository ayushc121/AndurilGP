"""VQ2 state estimation, vision fusion, and guidance behaviour."""

import math

import numpy as np
import pytest

from vq2 import controller as vq2


# --------------------------------------------------------------------------
# Attitude estimation
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Fixtures for the controller under test
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Gate observation
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Velocity fusion
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Damping
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Command output
# --------------------------------------------------------------------------

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
    ctrl._fly()
    thrust = sim_conn.mav.attitude_targets[-1]['thrust']
    assert 0.0 <= thrust <= 1.0


def test_gate_below_reduces_thrust(ctrl, sim_conn):
    """Positive elevation error means the gate is below us — descend toward it."""
    ctrl._att_deg = (0.0, 0.0, 0.0)
    ctrl._elev_err = 0.0
    ctrl._fly()
    neutral = sim_conn.mav.attitude_targets[-1]['thrust']

    ctrl._elev_err = 5.0
    ctrl._fly()
    assert sim_conn.mav.attitude_targets[-1]['thrust'] < neutral


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
