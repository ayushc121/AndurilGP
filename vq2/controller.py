"""
Vision-only gate controller (Virtual Qualifier 2).

VQ2 blocks odometry, attitude and gate positions, leaving a 30 Hz camera and a
120 Hz IMU. All state comes from the gate-relative estimator in estimator.py —
this file owns one and steers on it. Guidance is reactive, with no map and no
planned trajectory — see README for why that capped out at three gates.

Frame is NED (X north, Y east, Z down). More negative Z is higher.
"""

import math
import threading
import time
from enum import Enum, auto

import numpy as np
from pymavlink import mavutil

from .ahrs import HOVER_THRUST, LAUNCH_YAW_CMD_DEG, euler_to_quat
from .estimator import GateVIO

# Loop timing

CONTROL_HZ    = 60    # command rate: 2:1 with the 30 Hz camera
CAMERA_FPS    = 30.0
DEBUG_EVERY_N = 20

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25
# Guidance

VIS_VEL_EMA_ALPHA = 0.35   # smooth vision velocity first
OF_ALPHA          = 0.60   # IMU share of the blend

# one bad detection makes a huge finite difference, so cap them
BEARING_RATE_CLAMP_DEG_S = 60.0
ELEV_RATE_CLAMP_M_S      = 5.0

BEARING_CLAMP_DEG = 25.0   # feeds the roll loop
YAW_CLAMP_DEG     = 12.0   # tighter on purpose
YAW_RATE_LIMIT_DEG_S = 45.0

YAW_BEARING_SIGN = -1.0    # +1 was flown, gave positive feedback. don't flip
YAW_BEARING_MIN_M = 1.0    # below this the bearing is noise, hold instead
PERP_BLEND_DIST_M = 6.0    # inside this, roll authority fades out
MIN_RANGE_FOR_ELEV_M = 3.0 # closer than this the gate fills the frame

# hold the last elevation error this long after losing the gate. unbounded,
# a stale error is a permanent climb/descent — one run rode +4.32 m into the ground
ELEV_HOLD_MAX_S = 1.5

DESIRED_PITCH_DEG = -3.0   # constant nose-down: the only source of forward speed
K_BEARING         = 4.5    # deg of bank per deg of bearing error
K_LAT_D           = 9.0    # deg of bank per m/s of lateral closure
MAX_BANK_DEG      = 25.0

K_P_THRUST = 0.014         # thrust per metre of gate elevation error
K_D_THRUST = 0.0175        # thrust per m/s of vertical closure

ROLL_CMD_SIGN = -1.0       # sim roll axis is opposite to NED


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()


class GateView:
    """One frame's worth of gate geometry, in the drone's body frame."""

    __slots__ = ('valid', 'source', 'frame_id', 'fwd', 'right', 'down',
                 'bearing_deg', 'bearing_rate', 'elev_rate', 'blend')

    def __init__(self):
        self.valid = False
        self.source = 'BLIND'
        self.frame_id = None
        self.fwd = self.right = self.down = float('nan')
        self.bearing_deg = 0.0
        self.bearing_rate = 0.0
        self.elev_rate = 0.0
        self.blend = 0.0


# Controller

class Controller:
    """Vision-only flight controller.

    Owns a GateVIO and steers on its output: attitude, velocity and the gate
    vector all come from the filter, so there is one estimator in the process
    rather than a controller and a filter integrating the same gyro apart from
    each other.
    """

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        self._state_lock = threading.Lock()
        self._was_armed = False
        self._disarm_at = None

        self.vio = GateVIO(data)
        self._vio_started = False

        self._reset_flight_state()

    def _reset_flight_state(self):
        """Clear per-run state. The estimator thread survives, but is reseeded
        here rather than relying on the sim's IMU clock to rewind — a stale
        AHRS yaw would snap the nose to the wrong absolute heading at release.
        """
        self.vio.reset()
        lock = self.data.get('lock')
        if lock is not None:
            with lock:
                self.data['cmd_thrust'] = HOVER_THRUST
        with self._state_lock:
            self.phase = Phase.WAIT_FOR_DATA
            self._last_arm_attempt = 0.0
            self._tick = 0
            self._wait_start_sim_ms = None

        # Vision state. Within a run `_elev_err` survives a detection gap on
        # purpose — see `_observe_gate`.
        self._elev_err = 0.0
        self._elev_err_at = None       # when _elev_err was last measured
        self._yaw_cmd = LAUNCH_YAW_CMD_DEG   # absolute heading setpoint
        self._yaw_frame = None
        self._vis_vy_ema = 0.0
        self._vis_vz_ema = 0.0
        self._prev_bearing = None
        self._prev_bearing_frame = None
        self._prev_elev = None
        self._prev_elev_frame = None
        self._last_fused_frame = None
        self._last_damping_frame = None
        self._vy_at_vision = 0.0
        self._vd_at_vision = 0.0
        print('Controller state reset.', flush=True)

    # public

    def arm(self):
        self._send_arm()

    # estimation

    def _start_estimator(self):
        if self._vio_started:
            return
        self.vio.start()
        self._vio_started = True
        print('Gate-relative estimator started.', flush=True)

    def get_thread_for_join(self):
        """Shutdown hook — core.setup joins this on the way out."""
        return self.vio.get_thread_for_join()

    # control

    def update(self):
        """One control tick. Dispatches on flight phase, then sleeps."""
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            return self._idle()

        with lock:
            imu = self.data.get('imu')
            race_status = self.data.get('race_status')
            armed = self.data.get('armed', False)

        if self._handle_disarm(armed, lock):
            return self._idle()

        if self.phase is Phase.WAIT_FOR_DATA:
            self._wait_for_data(armed, imu)
        elif self.phase is Phase.WAIT_FOR_START:
            self._wait_for_start(race_status)
        elif self.phase is Phase.FLYING:
            self._fly()

        self._idle()

    def _idle(self):
        time.sleep(1.0 / CONTROL_HZ)

    def _handle_disarm(self, armed, lock):
        """Detect a sim reset and hold off re-arming. True = skip this tick."""
        if self._was_armed and not armed:
            if self._disarm_at is None:
                print('Disarm detected — waiting before re-arm.', flush=True)
                self._disarm_at = time.time()
                with lock:
                    self.data['imu'] = None
                    self.data['race_status'] = None
                self._reset_flight_state()
            self._was_armed = armed
            return True

        if not armed and self._disarm_at is not None:
            if time.time() - self._disarm_at < POST_DISARM_WAIT:
                self._was_armed = armed
                return True
            print('Post-disarm wait done. Ready to re-arm.', flush=True)
            self._disarm_at = None
            self._last_arm_attempt = 0.0

        self._was_armed = armed
        return False

    def _wait_for_data(self, armed, imu):
        if not armed:
            now = time.time()
            if now - self._last_arm_attempt >= ARM_RETRY_S:
                print('Sending arm command...', flush=True)
                self._send_arm()
                self._last_arm_attempt = now
            return
        if imu is not None:
            self._start_estimator()
            print('Armed and IMU ready. Waiting for race start.', flush=True)
            self.phase = Phase.WAIT_FOR_START
        else:
            self._send_attitude_target(0.0, 0.0, 0.0, 0.0)

    def _wait_for_start(self, race_status):
        """Hold on the pad at zero thrust until the countdown elapses."""
        self._send_attitude_target(0.0, 0.0, 0.0, 0.0)
        if race_status is None:
            return

        sim_ms = race_status['sim_boot_time_ms']
        start_ms = race_status['race_start_boot_time_ms']
        if self._wait_start_sim_ms is None:
            self._wait_start_sim_ms = sim_ms

        is_fresh = start_ms > 0 and start_ms >= self._wait_start_sim_ms
        if is_fresh and sim_ms >= start_ms:
            print('Countdown complete — flying.', flush=True)
            self.phase = Phase.FLYING

    # guidance

    def _observe_gate(self, vision, quat, est=None):
        """Body-frame gate geometry, from the filter if it has a live anchor.

        Two rungs. EKF is the filtered estimate and survives a dropped frame;
        VIS is the raw detection, used while the anchor is stale. Neither means
        blind, and blind must not steer — coasting on a drifting estimate
        measured 2.74 m/s of velocity error against 0.56 while tracked.
        """
        view = GateView()

        gate_body = None
        if est is not None and est.valid and not est.stale:
            gate_body = est.gate_body
            view.source = 'EKF'
        elif vision is not None:
            gate_body = (vision.get('body_x_m', float('nan')),
                         vision.get('body_y_m', float('nan')),
                         vision.get('body_z_m', float('nan')))
            view.source = 'VIS'

        if gate_body is None:
            self._prev_bearing = self._prev_bearing_frame = None
            self._prev_elev = self._prev_elev_frame = None
            return view

        view.frame_id = vision.get('frame_id') if vision is not None else None
        view.fwd, view.right, view.down = (float(v) for v in gate_body)
        if any(math.isnan(v) for v in (view.fwd, view.right, view.down)) or view.fwd <= 0.1:
            self._prev_bearing = self._prev_bearing_frame = None
            self._prev_elev = self._prev_elev_frame = None
            view.source = 'BLIND'
            return view

        view.valid = True
        view.bearing_deg = float(np.clip(math.degrees(math.atan2(view.right, view.fwd)),
                                         -BEARING_CLAMP_DEG, BEARING_CLAMP_DEG))
        view.blend = float(np.clip(view.fwd / PERP_BLEND_DIST_M, 0.0, 1.0))
        view.bearing_rate = self._frame_rate(
            view.bearing_deg, view.frame_id, '_prev_bearing', '_prev_bearing_frame')

        # rotate out through attitude so a bank isn't read as altitude error
        if view.fwd > MIN_RANGE_FOR_ELEV_M:
            qw, qx, qy, qz = quat
            elev = (2 * (qx * qz - qw * qy) * view.fwd
                    + 2 * (qy * qz + qw * qx) * view.right
                    + (1 - 2 * (qx * qx + qy * qy)) * view.down)
            view.elev_rate = self._frame_rate(
                elev, view.frame_id, '_prev_elev', '_prev_elev_frame')
            self._elev_err = elev
            self._elev_err_at = time.monotonic()
        # closer than that the gate fills the frame — hold the last value

        return view

    def _frame_rate(self, value, frame_id, value_attr, frame_attr):
        """Finite difference per second, over gaps of 1-3 frames. A wider gap means
        the gate was lost and re-acquired, so the difference is noise."""
        prev_v = getattr(self, value_attr)
        prev_f = getattr(self, frame_attr)
        rate = 0.0
        if frame_id is not None:
            if prev_f is not None and 0 < frame_id - prev_f <= 3:
                rate = (value - prev_v) / ((frame_id - prev_f) / CAMERA_FPS)
            setattr(self, value_attr, value)
            setattr(self, frame_attr, frame_id)
        return rate

    def _fuse_velocity(self, vision_vel, frame_id, v_right, v_down):
        """Correct lateral/vertical velocity with vision, once per camera frame.
        Forward stays on the IMU — apparent gate growth is a weak range signal."""
        if vision_vel is None or frame_id is None or frame_id == self._last_fused_frame:
            return v_right, v_down, 'imu'

        self._vis_vy_ema = (VIS_VEL_EMA_ALPHA * vision_vel['vy_body_mps']
                            + (1 - VIS_VEL_EMA_ALPHA) * self._vis_vy_ema)
        self._vis_vz_ema = (VIS_VEL_EMA_ALPHA * vision_vel['vz_body_mps']
                            + (1 - VIS_VEL_EMA_ALPHA) * self._vis_vz_ema)
        self._last_fused_frame = frame_id
        return (OF_ALPHA * v_right + (1 - OF_ALPHA) * self._vis_vy_ema,
                OF_ALPHA * v_down + (1 - OF_ALPHA) * self._vis_vz_ema,
                'fused')

    def _damping_terms(self, view, v_right, v_down_ned):
        """Closure rates for the D terms. Vision on a fresh frame, IMU delta between
        frames, zero with no gate in view."""
        fresh = view.valid and view.frame_id is not None and view.frame_id != self._last_damping_frame

        if fresh:
            rate = float(np.clip(view.bearing_rate,
                                 -BEARING_RATE_CLAMP_DEG_S, BEARING_RATE_CLAMP_DEG_S))
            lateral = -math.radians(rate) * max(view.fwd, 0.5)
            vertical = -float(np.clip(view.elev_rate,
                                      -ELEV_RATE_CLAMP_M_S, ELEV_RATE_CLAMP_M_S))
        elif view.valid:
            lateral = v_right - self._vy_at_vision
            vertical = v_down_ned - self._vd_at_vision
            return lateral, vertical, 'imu'
        else:
            lateral = vertical = 0.0
            self._last_damping_frame = None

        self._vy_at_vision = v_right
        self._vd_at_vision = v_down_ned
        if fresh:
            self._last_damping_frame = view.frame_id
        return lateral, vertical, 'vision' if fresh else 'none'

    def _step_yaw(self, vision):
        """Walk the absolute heading setpoint toward the gate, rate-limited.

        Deliberately the raw camera bearing, not the filter's `gate_body`: that
        one is rotated by the AHRS, so steering on it closes the yaw loop over
        the filter's own yaw error. This is the only attitude-free bearing in
        the tree. With no gate the setpoint is held, not steered.
        """
        if vision is None:
            return self._yaw_cmd
        bx, by = vision.get('body_x_m'), vision.get('body_y_m')
        if bx is None or by is None or not (math.isfinite(bx) and math.isfinite(by)):
            return self._yaw_cmd
        if math.hypot(bx, by) < YAW_BEARING_MIN_M:
            return self._yaw_cmd

        # one 30 Hz frame covers two 60 Hz ticks, and the step is relative to
        # the last command, so consuming it twice slews at double the limit
        frame_id = vision.get('frame_id')
        if frame_id is not None and frame_id == self._yaw_frame:
            return self._yaw_cmd
        self._yaw_frame = frame_id

        bearing = math.degrees(math.atan2(by, bx))
        step_limit = YAW_RATE_LIMIT_DEG_S / CONTROL_HZ
        correction = YAW_BEARING_SIGN * float(np.clip(
            bearing, -YAW_CLAMP_DEG, YAW_CLAMP_DEG))
        self._yaw_cmd += float(np.clip(correction, -step_limit, step_limit))
        self._yaw_cmd = (self._yaw_cmd + 180.0) % 360.0 - 180.0
        return self._yaw_cmd

    def _held_elev_err(self):
        """Elevation error, or zero once the held value is too old to trust."""
        # holding through a short blackout keeps the climb going across a gate;
        # holding forever is a stale command with nothing to end it
        if self._elev_err_at is None:
            return 0.0
        if time.monotonic() - self._elev_err_at > ELEV_HOLD_MAX_S:
            return 0.0
        return self._elev_err

    def _fly(self):
        """Run one guidance/control pass and send the resulting setpoint."""
        # attitude comes off the AHRS whether or not the filter has anchored,
        # so an unanchored estimate still flies — on the VIS rung, as before.
        est = self.vio.get_estimate()
        roll_deg, pitch_deg, yaw_deg = est.rpy_deg
        v_down_ned = float(est.vel[2])
        v_fwd, v_right, v_down = (float(x) for x in est.vel_body)
        quat = est.quat

        vision = self.data.get('vision_gate_estimate')
        vision_vel = self.data.get('vision_velocity')

        view = self._observe_gate(vision, quat, est)
        v_right, v_down, vel_source = self._fuse_velocity(
            vision_vel, view.frame_id, v_right, v_down)
        d_lateral, d_vertical, d_source = self._damping_terms(view, v_right, v_down_ned)

        # roll steers, damped by lateral closure. blend hands authority to yaw
        # near the gate plane so we square up instead of still turning
        p_lat = K_BEARING * view.bearing_deg * view.blend
        d_lat = K_LAT_D * d_lateral * view.blend
        roll_target = float(np.clip(p_lat - d_lat, -MAX_BANK_DEG, MAX_BANK_DEG))

        yaw_target = self._step_yaw(vision)

        # PD on gate elevation, divided by lift lost to tilt.
        # positive elev_err = gate is below us
        elev_err = self._held_elev_err()
        tilt_loss = max(0.01, math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)))
        thrust = (HOVER_THRUST - elev_err * K_P_THRUST
                  + d_vertical * K_D_THRUST) / tilt_loss
        thrust = float(np.clip(thrust, 0.0, 1.0))

        if self._tick % DEBUG_EVERY_N == 0:
            print(f'[NAV/{view.source}] '
                  f'vel[{vel_source}]=({v_fwd:+5.1f}f {v_right:+5.1f}r {v_down:+5.1f}d) '
                  f'att=({roll_deg:+5.1f}r {pitch_deg:+5.1f}p {yaw_deg:+6.1f}y) '
                  f'gate=({view.fwd:+5.1f}f {view.right:+5.1f}r {view.down:+5.1f}d) '
                  f'elev={elev_err:+5.2f} age={est.age_s:.2f} '
                  f'D[{d_source}]=({d_lateral:+5.2f} {d_vertical:+5.2f}) '
                  f'cmd=(r={roll_target:+5.1f} y={yaw_target:+5.1f} T={thrust:.3f})',
                  flush=True)

        # absolute setpoints, not errors — the sim closes its own attitude loop
        self._send_attitude_target(ROLL_CMD_SIGN * roll_target,
                                   DESIRED_PITCH_DEG, yaw_target, thrust)

    # mavlink

    def _send_attitude_target(self, roll_deg, pitch_deg, yaw_deg, thrust):
        """Type mask 7 = use the quaternion, ignore the body-rate fields."""
        lock = self.data.get('lock')
        if lock is not None and self.phase is Phase.FLYING:
            # the estimator's acceleration cap reads this back. only while
            # flying: on the pad the command is 0, and a zero cap scales the
            # measured specific force away, leaving a free fall we aren't in
            with lock:
                self.data['cmd_thrust'] = thrust
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            7,
            euler_to_quat(math.radians(roll_deg), math.radians(pitch_deg),
                          math.radians(yaw_deg)),
            0, 0, 0,
            thrust)

    def _send_arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
