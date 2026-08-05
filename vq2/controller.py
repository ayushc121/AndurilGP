"""
Vision-only gate controller (Virtual Qualifier 2).

VQ2 blocks odometry, attitude and gate positions, leaving a 30 Hz camera and a
120 Hz IMU. Attitude and velocity come from the IMU; bearing, elevation and
gate normal come from the camera. Guidance is reactive, with no map and no
planned trajectory — see README for why that capped out at three gates.

Frame is NED (X north, Y east, Z down). More negative Z is higher.
"""

import math
import threading
import time
from collections import deque
from enum import Enum, auto

import cv2
import numpy as np
from pymavlink import mavutil

# --------------------------------------------------------------------------
# Loop timing
# --------------------------------------------------------------------------

CONTROL_HZ         = 60    # command rate: 2:1 with the 30 Hz camera
ESTIMATION_POLL_HZ = 400   # oversamples the 120 Hz IMU so no sample is missed
CAMERA_FPS         = 30.0
DEBUG_EVERY_N      = 20

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25

# --------------------------------------------------------------------------
# Platform constants
# --------------------------------------------------------------------------

G                = 9.81
HOVER_THRUST     = 0.264   # measured hover trim (see sysid/)
LAUNCH_PITCH_DEG = -17.8   # the drone starts on an angled launch block

# Moving average over the accelerometer: ~42 ms window at 120 Hz. Kills the
# contact and motor spikes that would otherwise integrate into velocity error.
ACC_SMOOTH_N = 5

# --------------------------------------------------------------------------
# Vision / IMU velocity fusion
# --------------------------------------------------------------------------

VIS_VEL_EMA_ALPHA = 0.35   # smoothing on raw vision velocity, applied first
OF_ALPHA          = 0.60   # IMU's share of the blend; vision gets the rest

# Bound the damage one bad detection can do to a finite difference.
BEARING_RATE_CLAMP_DEG_S = 60.0
ELEV_RATE_CLAMP_M_S      = 5.0

# --------------------------------------------------------------------------
# Guidance gains
# --------------------------------------------------------------------------

BEARING_CLAMP_DEG = 25.0   # bearing feeding the roll loop
YAW_CLAMP_DEG     = 12.0   # tighter: yaw authority is limited on purpose
PERP_BLEND_DIST_M = 6.0    # inside this range, yaw shifts from bearing to normal
TILT_EMA_ALPHA    = 0.25   # PnP gate-normal is ~±10 deg noisy; this gets it to ~±4
MIN_RANGE_FOR_ELEV_M = 3.0 # below this the gate fills the frame; geometry is junk

DESIRED_PITCH_DEG = -3.0   # constant nose-down: the only source of forward speed
K_BEARING         = 4.5    # deg of bank per deg of bearing error
K_LAT_D           = 9.0    # deg of bank per m/s of lateral closure
MAX_BANK_DEG      = 25.0

K_P_THRUST = 0.014         # thrust per metre of gate elevation error
K_D_THRUST = 0.0175        # thrust per m/s of vertical closure

# The sim's roll and yaw axes run opposite to NED; pitch agrees.
ROLL_CMD_SIGN = -1.0
YAW_CMD_SIGN  = -1.0


# --------------------------------------------------------------------------
# Attitude estimation
# --------------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler angles (radians, NED) to quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


class GyroAHRS:
    """
    Attitude from gyro integration alone, seeded at the known launch pitch.

    No accelerometer correction on purpose: under thrust the apparent gravity
    vector is gravity plus linear acceleration, so a complementary filter would
    be wrong for most of a race.
    """

    def __init__(self, initial_pitch_deg=0.0):
        self.q = np.array(euler_to_quat(0.0, math.radians(initial_pitch_deg), 0.0))

    def update(self, gx, gy, gz, dt):
        """Integrate one gyro sample (rad/s, sign-corrected). Returns Euler degrees."""
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        q = np.array([
            qw + h * (-qx * gx - qy * gy - qz * gz),
            qx + h * (qw * gx + qy * gz - qz * gy),
            qy + h * (qw * gy - qx * gz + qz * gx),
            qz + h * (qw * gz + qx * gy - qy * gx),
        ])
        self.q = q / np.linalg.norm(q)
        return self.euler_deg()

    def euler_deg(self):
        qw, qx, qy, qz = self.q
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @property
    def quaternion(self):
        return self.q.copy()


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()


class GateView:
    """One frame's worth of gate geometry, in the drone's body frame."""

    __slots__ = ('valid', 'frame_id', 'fwd', 'right', 'down',
                 'bearing_deg', 'bearing_rate', 'elev_rate', 'tilt_deg', 'blend')

    def __init__(self):
        self.valid = False
        self.frame_id = None
        self.fwd = self.right = self.down = float('nan')
        self.bearing_deg = 0.0
        self.bearing_rate = 0.0
        self.elev_rate = 0.0
        self.tilt_deg = float('nan')
        self.blend = 0.0


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------

class Controller:
    """
    Vision-only flight controller. An estimation thread integrates IMU at
    400 Hz; the 60 Hz control loop snapshots that state and turns the latest
    detection into an attitude setpoint.
    """

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms

        self._state_lock = threading.Lock()
        self._est_running = False
        self._est_thread = None
        self._was_armed = False
        self._disarm_at = None

        self.ahrs = GyroAHRS(initial_pitch_deg=LAUNCH_PITCH_DEG)
        self.vel_ned = np.zeros(3)    # [north, east, down] m/s
        self.vel_body = np.zeros(3)   # [forward, right, down] m/s
        self.pos_ned = np.zeros(3)    # metres, relative to the arm point
        self._att_deg = (0.0, LAUNCH_PITCH_DEG, 0.0)
        self._last_imu_ts_us = None
        self._acc_buf = deque(maxlen=ACC_SMOOTH_N)

        self._reset_flight_state()

    def _reset_flight_state(self):
        """Clear per-run state. The estimation thread survives a sim reset."""
        with self._state_lock:
            self.phase = Phase.WAIT_FOR_DATA
            self._last_arm_attempt = 0.0
            self._tick = 0
            self._wait_start_sim_ms = None
            self._last_imu_ts_us = None
            self.ahrs = GyroAHRS(initial_pitch_deg=LAUNCH_PITCH_DEG)
            self.vel_ned[:] = 0.0
            self.vel_body[:] = 0.0
            self.pos_ned[:] = 0.0
            self._att_deg = (0.0, LAUNCH_PITCH_DEG, 0.0)
            self._acc_buf.clear()

        # Vision state. Within a run `_elev_err` survives a detection gap on
        # purpose — see `_observe_gate`.
        self._elev_err = 0.0
        self._tilt_ema = None
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

    # ---------------------------------------------------------------- public

    def arm(self):
        self._send_arm()

    # ------------------------------------------------------------ estimation

    def _start_estimation_thread(self):
        if self._est_thread is not None and self._est_thread.is_alive():
            return
        self._est_running = True
        self._est_thread = threading.Thread(target=self._estimation_loop,
                                            daemon=True, name='imu-estimation')
        self._est_thread.start()
        print('IMU estimation thread started.', flush=True)

    def _estimation_loop(self):
        interval = 1.0 / ESTIMATION_POLL_HZ
        while self._est_running:
            data_lock = self.data.get('lock')
            if data_lock:
                with data_lock:
                    imu = self.data.get('imu')
                if imu is not None:
                    self._process_imu(imu)
            time.sleep(interval)

    def _process_imu(self, imu):
        """Integrate one IMU sample. Ignores repeats of the last timestamp."""
        ts_us = imu['time_usec']
        with self._state_lock:
            if self._last_imu_ts_us is None:
                self._last_imu_ts_us = ts_us
                return
            if ts_us == self._last_imu_ts_us:
                return
            dt = max(0.0005, min(0.1, (ts_us - self._last_imu_ts_us) * 1e-6))
            self._last_imu_ts_us = ts_us

            # This simulator reports gyro rates inverted relative to NED.
            self._att_deg = self.ahrs.update(-imu['xgyro'], -imu['ygyro'],
                                             -imu['zgyro'], dt)

            self._acc_buf.append((imu['xacc'], imu['yacc'], imu['zacc']))
            n = len(self._acc_buf)
            acc = [sum(s[i] for s in self._acc_buf) / n for i in range(3)]
            self._integrate_kinematics(acc, dt)

    def _integrate_kinematics(self, acc, dt):
        """
        Strapdown integration: specific force -> NED velocity -> position.

        Caller holds _state_lock. Position has no correction term and drifts
        without bound; only velocity is corrected, and only by vision.
        """
        qw, qx, qy, qz = self.ahrs.quaternion
        ax, ay, az = acc

        a_n = (1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az
        a_e = 2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az
        a_d = 2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az + G

        self.vel_ned += np.array([a_n, a_e, a_d]) * dt
        self.pos_ned += self.vel_ned * dt

        vn, ve, vd = self.vel_ned
        self.vel_body[0] = (1 - 2 * (qy * qy + qz * qz)) * vn + 2 * (qx * qy + qw * qz) * ve + 2 * (qx * qz - qw * qy) * vd
        self.vel_body[1] = 2 * (qx * qy - qw * qz) * vn + (1 - 2 * (qx * qx + qz * qz)) * ve + 2 * (qy * qz + qw * qx) * vd
        self.vel_body[2] = 2 * (qx * qz + qw * qy) * vn + 2 * (qy * qz - qw * qx) * ve + (1 - 2 * (qx * qx + qy * qy)) * vd

    # --------------------------------------------------------------- control

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
            self._start_estimation_thread()
            print('Armed and IMU ready. Waiting for race start.', flush=True)
            self.phase = Phase.WAIT_FOR_START
        else:
            self._send_attitude_target(0.0, 0.0, 0.0, 0.0)

    def _wait_for_start(self, race_status):
        """
        Hold on the pad at zero thrust until the countdown elapses.

        `race_start_boot_time_ms` persists across runs, so it is only trusted
        once it is at or after the clock reading taken when this phase began.
        """
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

    # -------------------------------------------------------------- guidance

    def _observe_gate(self, vision, quat):
        """Turn the latest detection into body-frame gate geometry."""
        view = GateView()
        if vision is None:
            self._tilt_ema = None
            self._prev_bearing = self._prev_bearing_frame = None
            self._prev_elev = self._prev_elev_frame = None
            return view

        view.frame_id = vision.get('frame_id')
        view.fwd = vision.get('body_x_m', float('nan'))
        view.right = vision.get('body_y_m', float('nan'))
        view.down = vision.get('body_z_m', float('nan'))
        if any(math.isnan(v) for v in (view.fwd, view.right, view.down)) or view.fwd <= 0.1:
            self._tilt_ema = None
            self._prev_bearing = self._prev_bearing_frame = None
            self._prev_elev = self._prev_elev_frame = None
            return view

        view.valid = True
        view.bearing_deg = float(np.clip(math.degrees(math.atan2(view.right, view.fwd)),
                                         -BEARING_CLAMP_DEG, BEARING_CLAMP_DEG))
        view.blend = float(np.clip(view.fwd / PERP_BLEND_DIST_M, 0.0, 1.0))
        view.bearing_rate = self._frame_rate(
            view.bearing_deg, view.frame_id, '_prev_bearing', '_prev_bearing_frame')

        # Rotate the gate vector out through attitude, so a banked drone
        # doesn't read its own tilt as altitude error.
        if view.fwd > MIN_RANGE_FOR_ELEV_M:
            qw, qx, qy, qz = quat
            elev = (2 * (qx * qz - qw * qy) * view.fwd
                    + 2 * (qy * qz + qw * qx) * view.right
                    + (1 - 2 * (qx * qx + qy * qy)) * view.down)
            view.elev_rate = self._frame_rate(
                elev, view.frame_id, '_prev_elev', '_prev_elev_frame')
            self._elev_err = elev
        # Closer than that, the gate fills the frame: hold the last value.

        view.tilt_deg = self._gate_tilt(vision)
        return view

    def _frame_rate(self, value, frame_id, value_attr, frame_attr):
        """
        Finite difference per second, over gaps of 1-3 frames only. A wider gap
        means the gate was lost and re-acquired, so the difference is noise.
        """
        prev_v = getattr(self, value_attr)
        prev_f = getattr(self, frame_attr)
        rate = 0.0
        if frame_id is not None:
            if prev_f is not None and 0 < frame_id - prev_f <= 3:
                rate = (value - prev_v) / ((frame_id - prev_f) / CAMERA_FPS)
            setattr(self, value_attr, value)
            setattr(self, frame_attr, frame_id)
        return rate

    def _gate_tilt(self, vision):
        """
        Gate face normal in the body frame, used to square up on approach.
        EMA because raw PnP normals are ~±10 deg noisy; resets on dropout.
        """
        if not (vision.get('pnp_ok') and vision.get('pnp_rvec') is not None):
            self._tilt_ema = None
            return float('nan')

        rotation, _ = cv2.Rodrigues(np.array(vision['pnp_rvec'], dtype=np.float64))
        normal = rotation @ np.array([0.0, 0.0, 1.0])
        if normal[2] > 0:      # IPPE returns a sign-ambiguous normal
            normal = -normal

        # Camera is pitched 20 deg up from the body frame.
        ct, st = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
        fwd = ct * (-normal[2]) + st * (-normal[1])
        right = -normal[0]
        raw = float(np.clip(math.degrees(math.atan2(right, fwd)), -30.0, 30.0))

        self._tilt_ema = raw if self._tilt_ema is None else (
            TILT_EMA_ALPHA * raw + (1.0 - TILT_EMA_ALPHA) * self._tilt_ema)
        return self._tilt_ema

    def _fuse_velocity(self, vision_vel, frame_id, v_right, v_down):
        """
        Correct lateral/vertical velocity with vision, once per camera frame.
        Forward stays on the IMU — apparent gate growth is a weak range signal.
        """
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
        """
        Closure rates for the D terms: vision on a fresh frame, IMU delta
        between frames, zero with no gate in view.
        """
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

    def _fly(self):
        """Run one guidance/control pass and send the resulting setpoint."""
        with self._state_lock:
            roll_deg, pitch_deg, yaw_deg = self._att_deg
            v_down_ned = float(self.vel_ned[2])
            v_fwd, v_right, v_down = (float(x) for x in self.vel_body)
            quat = self.ahrs.quaternion

        vision = self.data.get('vision_gate_estimate')
        vision_vel = self.data.get('vision_velocity')

        view = self._observe_gate(vision, quat)
        v_right, v_down, vel_source = self._fuse_velocity(
            vision_vel, view.frame_id, v_right, v_down)
        d_lateral, d_vertical, d_source = self._damping_terms(view, v_right, v_down_ned)

        # Roll steers, damped by lateral closure. `blend` hands authority from
        # roll to yaw as the gate plane approaches, so the drone stops steering
        # and starts squaring up on the face normal.
        p_lat = K_BEARING * view.bearing_deg * view.blend
        d_lat = K_LAT_D * d_lateral * view.blend
        roll_target = float(np.clip(p_lat - d_lat, -MAX_BANK_DEG, MAX_BANK_DEG))

        yaw_bearing = float(np.clip(view.bearing_deg, -YAW_CLAMP_DEG, YAW_CLAMP_DEG))
        if view.valid and not math.isnan(view.tilt_deg):
            yaw_target = view.blend * yaw_bearing + (1.0 - view.blend) * view.tilt_deg
        elif view.valid:
            yaw_target = yaw_bearing
        else:
            yaw_target = 0.0

        # PD on gate elevation, divided by the lift lost to tilt.
        # Positive elev_err means the gate is below us.
        tilt_loss = max(0.01, math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg)))
        thrust = (HOVER_THRUST - self._elev_err * K_P_THRUST
                  + d_vertical * K_D_THRUST) / tilt_loss
        thrust = float(np.clip(thrust, 0.0, 1.0))

        if self._tick % DEBUG_EVERY_N == 0:
            print(f'[NAV] vel[{vel_source}]=({v_fwd:+5.1f}f {v_right:+5.1f}r {v_down:+5.1f}d) '
                  f'att=({roll_deg:+5.1f}r {pitch_deg:+5.1f}p {yaw_deg:+6.1f}y) '
                  f'gate=({view.fwd:+5.1f}f {view.right:+5.1f}r {view.down:+5.1f}d) '
                  f'elev={self._elev_err:+5.2f} '
                  f'D[{d_source}]=({d_lateral:+5.2f} {d_vertical:+5.2f}) '
                  f'cmd=(r={roll_target:+5.1f} y={yaw_target:+5.1f} T={thrust:.3f})',
                  flush=True)

        # Roll and pitch are unity-gain P on attitude error; yaw is the raw
        # relative bearing. The sim takes all three as absolute setpoints —
        # see the README's "known issues".
        self._send_attitude_target(
            ROLL_CMD_SIGN * (roll_target - roll_deg),
            DESIRED_PITCH_DEG - pitch_deg,
            YAW_CMD_SIGN * yaw_target,
            thrust)

    # ---------------------------------------------------------------- mavlink

    def _send_attitude_target(self, roll_deg, pitch_deg, yaw_deg, thrust):
        """Type mask 7 = use the quaternion, ignore the body-rate fields."""
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
