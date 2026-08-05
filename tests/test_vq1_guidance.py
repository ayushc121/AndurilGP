"""VQ1 guidance and attitude conventions."""

import math

import pytest

from vq1 import controller as vq1


def test_euler_quaternion_round_trip():
    """Angles must survive the conversion the wire format forces on them."""
    roll, pitch, yaw = 12.0, -7.5, 100.0
    qw, qx, qy, qz = vq1.euler_to_quat(math.radians(roll), math.radians(pitch),
                                       math.radians(yaw))
    out = vq1.quat_to_euler(qw, qx, qy, qz)
    assert out == pytest.approx((roll, pitch, yaw), abs=1e-6)


def test_minimum_closing_speed_on_course_axis():
    """Without a floor the drone stalls short of the gate plane and never advances."""
    v_north, _ = vq1.Controller._velocity_setpoint(0.001, 0.0)
    assert v_north >= vq1.V_MIN_CLOSE


def test_no_floor_on_cross_course_axis():
    """Laterally, settling to zero is the goal, so there is no floor."""
    _, v_east = vq1.Controller._velocity_setpoint(0.0, 0.0)
    assert v_east == 0.0


def test_velocity_setpoint_is_clamped():
    """A distant gate must not command an unbounded velocity."""
    v_north, v_east = vq1.Controller._velocity_setpoint(500.0, 500.0)
    assert v_north <= vq1.V_MAX + vq1.V_MIN_CLOSE
    assert v_east == pytest.approx(vq1.V_MAX)


def test_tilt_factor_compensates_bank():
    """A 30 deg bank costs ~13% of vertical thrust; the factor gives it back."""
    roll = math.radians(30.0)
    qx = math.sin(roll / 2)
    assert vq1.tilt_factor(qx, 0.0) == pytest.approx(math.cos(roll), abs=1e-9)
    # 30 deg of bank costs ~13% of vertical thrust.
    assert 1.0 / vq1.tilt_factor(qx, 0.0) == pytest.approx(1.155, abs=1e-3)


def test_tilt_factor_never_divides_by_zero():
    """Past vertical the factor must clamp, not explode the thrust command."""
    assert vq1.tilt_factor(0.99, 0.0) == vq1.MIN_TILT_FACTOR


def test_body_velocity_rotates_into_world():
    """Facing east, forward body velocity is east velocity in NED. Invert this
    and every steering correction inverts with it."""
    yaw = math.radians(90.0)
    qw, qx, qy, qz = vq1.euler_to_quat(0.0, 0.0, yaw)
    odo = {'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz, 'vx': 5.0, 'vy': 0.0, 'vz': 0.0}
    north, east, down = vq1.body_to_world_velocity(odo)
    assert north == pytest.approx(0.0, abs=1e-9)
    assert east == pytest.approx(5.0, abs=1e-9)
    assert down == pytest.approx(0.0, abs=1e-9)


def test_hold_commands_hover_when_stationary(sim_conn, shared_data):
    """With no gate to chase — pre-start or course complete — the drone holds."""
    ctrl = vq1.Controller(sim_conn, shared_data, 0)
    ctrl._fly({'x': 0, 'y': 0, 'z': -5, 'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0,
               'vx': 0, 'vy': 0, 'vz': 0}, None, None)
    sent = sim_conn.mav.attitude_targets[-1]
    assert sent['thrust'] == pytest.approx(vq1.HOVER_THRUST)
    assert sent['type_mask'] == 7


def test_hold_arrests_a_descent(sim_conn, shared_data):
    """Hover thrust cancels gravity, not momentum; the damping term stops a fall."""
    ctrl = vq1.Controller(sim_conn, shared_data, 0)
    ctrl._fly({'x': 0, 'y': 0, 'z': -5, 'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0,
               'vx': 0, 'vy': 0, 'vz': 3.0}, None, None)    # +vz is downward
    assert sim_conn.mav.attitude_targets[-1]['thrust'] > vq1.HOVER_THRUST


def test_thrust_rises_when_below_target_gate(sim_conn, shared_data):
    """Gate above the drone (more negative Z) must command more than hover."""
    ctrl = vq1.Controller(sim_conn, shared_data, 0)
    odo = {'x': 0, 'y': 0, 'z': 0.0, 'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0,
           'vx': 0, 'vy': 0, 'vz': 0}
    gates = [{'pos_x': 20.0, 'pos_y': 0.0, 'pos_z': -10.0}]
    ctrl._fly(odo, {'active_gate_index': 0}, gates)
    assert sim_conn.mav.attitude_targets[-1]['thrust'] > vq1.HOVER_THRUST


# Vision fallback — the path the detector was developed on

def _centred_bbox(range_m):
    """A gate detection dead ahead at a given range."""
    from vq1 import vision as vis
    width = vis.FX * vq1.GATE_WIDTH_M / range_m
    return {'bx': vis.CX - width / 2, 'by': vis.CY - width / 2,
            'bw': width, 'bh': width}


LEVEL = {'x': 0.0, 'y': 0.0, 'z': -5.0}


def test_back_projection_recovers_range():
    """Bbox width against the known gate must give the right forward distance."""
    north, _, _ = vq1.Controller._vision_gate_position(
        _centred_bbox(15.0), LEVEL, 0.0, 0.0, 0.0)
    # The camera is tilted up, so a centred gate is cos(20 deg) of the range away.
    assert north == pytest.approx(15.0 * math.cos(math.radians(vq1.CAM_TILT_DEG)), rel=1e-6)


def test_back_projection_follows_yaw():
    """Facing east, a gate dead ahead must project east, not north."""
    north, east, _ = vq1.Controller._vision_gate_position(
        _centred_bbox(15.0), LEVEL, 0.0, 0.0, 90.0)
    assert north == pytest.approx(0.0, abs=1e-9)
    assert east > 10.0


def test_back_projection_places_gate_above_when_centred():
    """The camera points 20 deg up, so a centred gate is higher than the drone."""
    _, _, down = vq1.Controller._vision_gate_position(
        _centred_bbox(15.0), LEVEL, 0.0, 0.0, 0.0)
    assert down < LEVEL['z']    # NED: more negative is higher


def test_back_projection_rejects_no_detection():
    assert vq1.Controller._vision_gate_position(None, LEVEL, 0.0, 0.0, 0.0) is None


def test_back_projection_rejects_degenerate_bbox():
    """A zero-width bbox would divide by zero in the range estimate."""
    assert vq1.Controller._vision_gate_position({'bw': 0}, LEVEL, 0.0, 0.0, 0.0) is None


def test_telemetry_overrides_vision(sim_conn, shared_data):
    """Both targets are computed every tick; telemetry wins whenever it exists."""
    ctrl = vq1.Controller(sim_conn, shared_data, 0)
    shared_data['vision_gate_estimate'] = _centred_bbox(15.0)
    odo = {**LEVEL, 'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0, 'vx': 0, 'vy': 0, 'vz': 0}

    gates = [{'pos_x': 40.0, 'pos_y': 0.0, 'pos_z': -5.0}]
    ctrl._fly(odo, {'active_gate_index': 0}, gates)
    with_telemetry = sim_conn.mav.attitude_targets[-1]['thrust']

    ctrl._fly(odo, None, None)          # same frame, telemetry gone
    with_vision = sim_conn.mav.attitude_targets[-1]['thrust']

    # Vision puts the gate above the drone, telemetry puts it level: the two
    # paths must produce different altitude commands, proving both are live.
    assert with_vision != with_telemetry


def test_vision_steers_when_telemetry_is_absent(sim_conn, shared_data):
    """With no gate telemetry, the detection is what the drone flies toward."""
    ctrl = vq1.Controller(sim_conn, shared_data, 0)
    shared_data['vision_gate_estimate'] = _centred_bbox(15.0)
    odo = {**LEVEL, 'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0, 'vx': 0, 'vy': 0, 'vz': 0}
    ctrl._fly(odo, None, None)
    # A gate 14 m ahead demands forward pitch, not the level hold.
    assert sim_conn.mav.attitude_targets[-1]['quaternion'] != vq1.euler_to_quat(0, 0, vq1.COURSE_YAW_RAD)
