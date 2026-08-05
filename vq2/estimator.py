"""
Gate-relative visual-inertial estimator.

The IMU predicts, the camera corrects. State is the drone's position and
velocity relative to the gate it is currently anchored to, so no map is needed
and nothing here reads telemetry — the same inputs VQ2 actually has.

Runs as a passive observer: it never commands the aircraft. On VQ1 that means
the telemetry controller can fly while this estimates blind, which is how the
numbers in the README were measured against real odometry.
"""

import math
import threading
import time

import numpy as np

from .controller import (GyroAHRS, HOVER_THRUST, LAUNCH_PITCH_DEG,
                         body_accel_to_ned, quat_to_R_wb)
from .gate_ekf import GateEKF, build_R_anisotropic

POLL_HZ = 400              # IMU predict rate, catches every ~120 Hz sample
VIS_HZ = 35                # vision tick, camera is ~30 Hz

EKF_SIGMA_A = 1.5          # accel process noise, m/s^2
EKF_SIGMA_A_SAT = 8.0      # inflated when the thrust cap clips a_ned
EKF_SIGMA_LAT = 0.3        # PnP lateral/vertical noise, the tight axis
EKF_SIGMA_RANGE_FLOOR = 1.0
EKF_VISION_LATENCY_S = 1.0 / 30

REANCHOR_DARK_FRAMES = 5   # dark vision ticks before we assume a passthrough
REANCHOR_REJECTS = 6       # consecutive chi2 rejects before forcing recovery

VALID_RANGE_MIN_M = 1.0
VALID_RANGE_MAX_M = 75.0
VALID_ASPECT_MAX = 2.0     # max(w/h, h/w), rejects clipped or degenerate boxes

CAM_TILT_DEG = 20.0
_CT = math.cos(math.radians(CAM_TILT_DEG))
_ST = math.sin(math.radians(CAM_TILT_DEG))

# body <- camera. camera is x right, y down, z forward; body is fwd, right, down
R_BODY_CAM = np.array([
    [0.0, _ST, _CT],
    [1.0, 0.0, 0.0],
    [0.0, _CT, -_ST],
])


def sigma_range_for(range_m):
    """Range noise grows with the square of range. Fitted, not guessed."""
    return max(EKF_SIGMA_RANGE_FLOOR, (range_m * range_m) / 500.0)


class VIOEstimate:
    """Snapshot for a consumer. Check `stale` before steering on anything here."""

    __slots__ = ('p_rel', 'vel', 'stale', 'age_s', 'valid',
                 'gate_body', 'vel_body', 'quat', 'rpy_deg')

    def __init__(self, p_rel, vel, stale, age_s, valid,
                 gate_body, vel_body, quat, rpy_deg):
        self.p_rel = p_rel          # drone relative to the anchor gate, world NED
        self.vel = vel              # world NED
        self.stale = stale          # anchor cannot be trusted — coast, don't steer
        self.age_s = age_s          # since the last accepted fix, not since predict
        self.valid = valid
        self.gate_body = gate_body  # vector TO the gate, body frame
        self.vel_body = vel_body
        self.quat = quat
        self.rpy_deg = rpy_deg


class GateVIO:
    """Passive estimator thread. Construct, `start()`, read `get_estimate()`."""

    def __init__(self, data):
        self.data = data
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._reset_state()

    def _reset_state(self):
        with self._lock:
            self.ahrs = GyroAHRS(initial_pitch_deg=LAUNCH_PITCH_DEG)
            self.ekf = GateEKF(sigma_a=EKF_SIGMA_A)
            # pure dead-reckoning on the same input, never corrected. keeps the
            # "beats raw strapdown" comparison decidable from one run
            self._sd_pos = np.zeros(3)
            self._sd_vel = np.zeros(3)
        self._last_imu_ts_us = None
        self._last_frame_id = None
        self._dark_frames = 0
        self._reject_run = 0
        self._need_anchor = True
        self._last_vis_t = 0.0
        self._last_accept_t = None
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_reanchor = 0

    def start(self):
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='gate-vio')
        self._thread.start()

    def get_thread_for_join(self):
        self._running = False
        return self._thread

    def get_estimate(self):
        """Thread-safe snapshot. Always check `.stale` before steering on it."""
        with self._lock:
            valid = self.ekf.initialised
            p = self.ekf.position if valid else np.zeros(3)
            v = self.ekf.velocity if valid else np.zeros(3)
            stale = bool(self._need_anchor or not valid)
            quat = self.ahrs.quaternion
            rpy = self.ahrs.euler_deg()
        age = (time.time() - self._last_accept_t) if self._last_accept_t else float('inf')

        R_bw = quat_to_R_wb(quat).T
        return VIOEstimate(p_rel=p, vel=v, stale=stale, age_s=age, valid=valid,
                           gate_body=R_bw @ (-p), vel_body=R_bw @ v,
                           quat=quat, rpy_deg=rpy)

    def _loop(self):
        interval = 1.0 / POLL_HZ
        vis_interval = 1.0 / VIS_HZ
        while self._running:
            lock = self.data.get('lock')
            if lock is None:
                time.sleep(interval)
                continue
            with lock:
                imu = self.data.get('imu')
                vision = self.data.get('vision_gate_estimate')
                cmd_thrust = self.data.get('cmd_thrust', HOVER_THRUST)

            try:
                if imu is not None:
                    self._predict(imu, cmd_thrust)
                now = time.time()
                if now - self._last_vis_t >= vis_interval:
                    self._last_vis_t = now
                    self._vision_step(vision)
            except Exception as exc:                  # noqa: BLE001
                print(f'[vio] fault (continuing): {exc!r}', flush=True)

            time.sleep(interval)

    def _predict(self, imu, cmd_thrust):
        ts_us = imu['time_usec']
        if self._last_imu_ts_us is None:
            self._last_imu_ts_us = ts_us
            return
        if ts_us <= self._last_imu_ts_us:
            # sim reset makes the timestamp go backwards
            if ts_us < self._last_imu_ts_us:
                self._reset_state()
                self._last_imu_ts_us = ts_us
            return

        dt = max(0.0005, min(0.1, (ts_us - self._last_imu_ts_us) * 1e-6))
        self._last_imu_ts_us = ts_us

        # this sim reports gyro rates inverted relative to NED
        with self._lock:
            self.ahrs.update(-imu['xgyro'], -imu['ygyro'], -imu['zgyro'], dt)
            quat = self.ahrs.quaternion

        a_ned, _, saturated = body_accel_to_ned(
            quat, imu['xacc'], imu['yacc'], imu['zacc'], cmd_thrust)

        with self._lock:
            # a clipped a_ned is a guess, so widen the process noise to match
            self.ekf.sigma_a = EKF_SIGMA_A_SAT if saturated else EKF_SIGMA_A
            self.ekf.predict(a_ned, dt)
            self._sd_pos += self._sd_vel * dt + 0.5 * dt * dt * a_ned
            self._sd_vel += a_ned * dt

    @staticmethod
    def _passes_validity(vision, gate_body):
        """Reject a detection before it can reach the filter or become an anchor."""
        rng = float(np.linalg.norm(gate_body))
        if not VALID_RANGE_MIN_M <= rng <= VALID_RANGE_MAX_M:
            return False
        bw, bh = vision.get('bw'), vision.get('bh')
        if bw and bh:
            # a gate opening stays roughly square at any sane angle
            if max(bw / bh, bh / bw) > VALID_ASPECT_MAX:
                return False
        return True

    def _vision_step(self, vision):
        """One vision tick: fuse a fix, or count toward a handoff."""
        if vision is None:
            self._dark_frames += 1
            if self._dark_frames >= REANCHOR_DARK_FRAMES:
                self._need_anchor = True        # passthrough, hand off to the next gate
            return

        self._dark_frames = 0
        frame_id = vision.get('frame_id')
        if frame_id is not None and frame_id == self._last_frame_id:
            return
        if not vision.get('pnp_ok'):
            return

        gate_body = np.array([vision.get('body_x_m', float('nan')),
                              vision.get('body_y_m', float('nan')),
                              vision.get('body_z_m', float('nan'))], dtype=float)
        if not np.all(np.isfinite(gate_body)) or gate_body[0] <= 0.1:
            return
        if not self._passes_validity(vision, gate_body):
            return

        self._last_frame_id = frame_id
        with self._lock:
            quat = self.ahrs.quaternion
        R_wb = quat_to_R_wb(quat)
        z = -(R_wb @ gate_body)                 # drone position relative to the gate
        if not np.all(np.isfinite(z)):
            return

        R = build_R_anisotropic(R_wb @ R_BODY_CAM, EKF_SIGMA_LAT,
                                sigma_range_for(float(np.linalg.norm(gate_body))))
        self._fuse(z, R)

    def _fuse(self, z, R):
        with self._lock:
            ekf = self.ekf
            if not ekf.initialised:
                # seed velocity from what predict has been integrating since t=0,
                # which beats starting at zero
                ekf.init_position(z, vel_ned=ekf.velocity)
                self._sd_pos = z.copy()
                self._sd_vel = ekf.velocity.copy()
                self._accept(anchor=True)
                return

            if self._need_anchor:
                # first valid fix, not two that agree — agreement against our
                # own drifted prediction is circular, and once went blind for good
                ekf.reanchor(z, R)
                self._sd_pos = z.copy()
                self.n_reanchor += 1
                self._accept(anchor=True)
                return

            tau = EKF_VISION_LATENCY_S
            z_fwd = z + ekf.velocity * tau          # first-order latency projection
            R_eff = R + (tau * tau) * ekf.P[3:6, 3:6]
            accepted, _ = ekf.update(z_fwd, R_eff)
            if accepted:
                self._accept(anchor=False)
            else:
                self.n_rejected += 1
                self._reject_run += 1
                if self._reject_run >= REANCHOR_REJECTS:
                    self._need_anchor = True

    def _accept(self, anchor):
        """Caller holds the lock."""
        self._need_anchor = False
        self._reject_run = 0
        self.n_accepted += 1
        self._last_accept_t = time.time()

    @property
    def strapdown_baseline(self):
        """Uncorrected dead-reckoning on the same input, for comparison."""
        with self._lock:
            return self._sd_pos.copy(), self._sd_vel.copy()
