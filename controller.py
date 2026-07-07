import time
import math
import threading
import numpy as np
import cv2
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CONTROL_HZ         = 60     # command rate — 2:1 with 30 Hz camera and 120 Hz physics
ESTIMATION_POLL_HZ = 400    # estimation thread poll rate — catches every 120 Hz IMU sample

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25
DEBUG_EVERY_N    = 20       # print interval (~3x per s at 60 Hz — reduce once tuned)
HOVER_THRUST     = 0.264
MAVLINK_CMD_SIM_RESET = 31000

# Known launch attitude — drone always starts on an angled block.
LAUNCH_PITCH_DEG = -17.8    # NED: negative = nose down

G = 9.81   # m/s²

THRUST_ACCEL_COEFF = 9.81 / HOVER_THRUST**2   # m/s² per thrust^2


# -----------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler angles (radians, NED) → quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    return [cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy]


# -----------------------------------------------------------------------
# Gyro attitude integrator
# -----------------------------------------------------------------------

class GyroAHRS:
    """
    Pure gyroscope attitude integrator.

    No accelerometer correction — the gyro is assumed accurate enough that
    complementary correction would only introduce errors (accelerometer is
    corrupted by thrust during manoeuvres, which is most of a race).

    Attitude is seeded at construction from the known launch geometry.
    The quaternion then evolves exclusively from integrated gyro rates.

    NED sign conventions (ZYX Euler):
        positive roll  = right wing down
        positive pitch = nose up
        positive yaw   = clockwise from above
    """

    def __init__(self, initial_pitch_deg=0.0, initial_roll_deg=0.0):
        self.q = np.array(euler_to_quat(
            math.radians(initial_roll_deg),
            math.radians(initial_pitch_deg),
            0.0,
        ))
        self._initialized = True

    def update(self, gx, gy, gz, dt):
        """
        Integrate one gyro sample into the quaternion.

        Args:
            gx, gy, gz : sign-corrected body-frame angular rates (rad/s)
            dt         : elapsed time since last call (s)

        Returns:
            (roll_deg, pitch_deg, yaw_deg)
        """
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        qw_n = qw + h*(-qx*gx - qy*gy - qz*gz)
        qx_n = qx + h*( qw*gx + qy*gz - qz*gy)
        qy_n = qy + h*( qw*gy - qx*gz + qz*gx)
        qz_n = qz + h*( qw*gz + qx*gy - qy*gx)
        n = math.sqrt(qw_n**2 + qx_n**2 + qy_n**2 + qz_n**2)
        self.q = np.array([qw_n, qx_n, qy_n, qz_n]) / n
        return self._euler_deg()

    def _euler_deg(self):
        """ZYX Euler angles in degrees from current quaternion."""
        qw, qx, qy, qz = self.q
        roll  = math.atan2(2.0*(qw*qx + qy*qz),
                           1.0 - 2.0*(qx*qx + qy*qy))
        sp    = max(-1.0, min(1.0, 2.0*(qw*qy - qz*qx)))
        pitch = math.asin(sp)
        yaw   = math.atan2(2.0*(qw*qz + qx*qy),
                           1.0 - 2.0*(qy*qy + qz*qz))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @property
    def quaternion(self):
        return self.q.copy()


# -----------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------

class Phase(Enum):
    WAIT_FOR_DATA  = auto()
    WAIT_FOR_START = auto()
    FLYING         = auto()


# -----------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------

class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn       = sim_conn
        self.data           = data
        self.system_boot_ms = system_boot_ms

        self._was_armed  = False
        self._disarm_at  = None

        self._state_lock = threading.Lock()

        self._est_running = False
        self._est_thread  = None

        # Estimation state
        self._last_imu_ts_us = None
        self.ahrs        = GyroAHRS(initial_pitch_deg=LAUNCH_PITCH_DEG)
        self.vel_ned     = np.zeros(3)   # [vN, vE, vD]       m/s — NED world frame
        self.vel_body    = np.zeros(3)   # [vX, vY, vZ]       m/s — body frame
        self.pos_ned     = np.zeros(3)   # [pN, pE, pD]       m   — relative to arm
        self.rates_body  = np.zeros(3)   # [roll, pitch, yaw]  deg/s — sign-corrected gyro
        self._att_deg    = (0.0, LAUNCH_PITCH_DEG, 0.0)
        self._last_acc_norm   = 0.0
        self._last_cmd_thrust = 0.0   # last thrust commanded by the control loop, [0,1]
        self._last_max_net    = 0.0   # last computed acceleration cap, m/s²

        self._thrust_integral = 0

        # Vision hold state — persists across blackout windows (passthrough cooldown)
        self._last_elev_err       = 0.0   # world-frame gate elevation error (m), frozen on loss
        self._last_fused_frame_id = None  # frame_id of last velocity fusion update
        self._gate_tilt_ema       = None  # EMA-smoothed gate face normal angle (deg)
        # Vision-derived rate state — previous frame values for finite-difference D terms.
        # These replace IMU vY/vD in roll and thrust D terms, avoiding IMU drift errors.
        self._last_d_frame_id = None  # frame_id when D terms were last updated from vision
        self._vY_at_vision    = 0.0   # body lateral velocity snapshotted at last vision D update
        self._vD_at_vision    = 0.0   # NED down velocity snapshotted at last vision D update

        self._reset_flight_state()

    # ------------------------------------------------------------------
    # Reset — estimation thread keeps running across disarm/rearm
    # ------------------------------------------------------------------

    def _reset_flight_state(self):
        with self._state_lock:
            self.phase              = Phase.WAIT_FOR_DATA
            self._finished          = False
            self._last_arm_attempt  = 0.0
            self._tick              = 0
            self._wait_start_sim_ms = None
            self._last_imu_ts_us    = None
            self.ahrs           = GyroAHRS(initial_pitch_deg=LAUNCH_PITCH_DEG)
            self.vel_ned[:]     = 0.0
            self.vel_body[:]    = 0.0
            self.pos_ned[:]     = 0.0
            self.rates_body[:]  = 0.0
            self._att_deg        = (0.0, LAUNCH_PITCH_DEG, 0.0)
            self._last_acc_norm   = 0.0
            self._last_cmd_thrust = 0.0
            self._last_max_net    = 0.0
        self._last_elev_err       = 0.0
        self._last_fused_frame_id = None
        self._gate_tilt_ema       = None
        self._prev_bearing_body     = None
        self._prev_bearing_frame_id = None
        self._prev_gate_pD          = None
        self._prev_elev_frame_id    = None
        self._last_d_frame_id = None
        self._vY_at_vision    = 0.0
        self._vD_at_vision    = 0.0
        print('Controller state reset.', flush=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_finished(self):
        return self._finished

    def arm(self):
        self._send_arm()

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0
        )

    def reset_nav_state(self, pos_ned=None, vel_ned=None):
        """
        Inject ground-truth position and/or velocity from the vision pipeline.

        Call whenever the camera produces a reliable gate position estimate.
        The IMU propagates state accurately over the 33 ms between camera frames;
        vision corrects the absolute state each time a gate is detected.

        Args:
            pos_ned : [pN, pE, pD] metres NED, relative to arm point. None = unchanged.
            vel_ned : [vN, vE, vD] m/s NED. None = unchanged.
        """
        with self._state_lock:
            if pos_ned is not None:
                self.pos_ned[0], self.pos_ned[1], self.pos_ned[2] = pos_ned
            if vel_ned is not None:
                self.vel_ned[0], self.vel_ned[1], self.vel_ned[2] = vel_ned

    def _send_attitude_rate(self, roll_deg, pitch_deg, yaw_deg, thrust):
        # Record the commanded thrust so the estimation thread can derive
        # the physical acceleration cap from it (see THRUST_ACCEL_COEFF).
        with self._state_lock:
            self._last_cmd_thrust = thrust

        q = euler_to_quat(math.radians(roll_deg),
                          math.radians(pitch_deg),
                          math.radians(yaw_deg))
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0b00000111,
            q,
            0, 0, 0,
            thrust
        )

    # ------------------------------------------------------------------
    # Estimation thread
    # ------------------------------------------------------------------

    def _start_estimation_thread(self):
        if self._est_thread is not None and self._est_thread.is_alive():
            return
        self._est_running = True
        self._est_thread = threading.Thread(
            target=self._estimation_loop,
            daemon=True,
            name='imu-estimation',
        )
        self._est_thread.start()
        print('IMU estimation thread started.', flush=True)

    def _estimation_loop(self):
        """
        Runs at ESTIMATION_POLL_HZ (400 Hz), independently of the 60 Hz control loop.
        Processes every new IMU sample from the simulator (120 Hz physics rate).
        All state writes are under _state_lock.
        """
        interval = 1.0 / ESTIMATION_POLL_HZ
        while self._est_running:
            data_lock = self.data.get('lock')
            if not data_lock:
                time.sleep(interval)
                continue
            with data_lock:
                imu = self.data.get('imu')
            if imu is not None:
                self._process_imu(imu)
            time.sleep(interval)

    # ------------------------------------------------------------------
    # IMU processing — called only from the estimation thread
    # ------------------------------------------------------------------

    def _process_imu(self, imu):
        """Process one IMU sample under _state_lock. Skips stale timestamps."""
        ts_us = imu['time_usec']

        with self._state_lock:
            if self._last_imu_ts_us is None:
                self._last_imu_ts_us = ts_us
                return
            if ts_us == self._last_imu_ts_us:
                return
            dt_imu = max(0.0005, min(0.1,
                         (ts_us - self._last_imu_ts_us) * 1e-6))
            self._last_imu_ts_us = ts_us

            # Gyro signs are inverted vs NED convention in this simulator.
            gx = -imu['xgyro']
            gy = -imu['ygyro']
            gz = -imu['zgyro']

            # Attitude: pure gyro integration — no accelerometer correction.
            # The accelerometer is corrupted by thrust during manoeuvres (most
            # of race flight), so any correction would pull attitude toward the
            # apparent gravity vector (gravity + linear acceleration), which is
            # wrong. The gyro alone gives accurate attitude for an ideal sensor.
            self._att_deg = self.ahrs.update(gx, gy, gz, dt_imu)

            # Sign-corrected body rates in deg/s
            self.rates_body[0] = math.degrees(gx)
            self.rates_body[1] = math.degrees(gy)
            self.rates_body[2] = math.degrees(gz)

            # Velocity integration with physical acceleration cap.
            # See _integrate_kinematics for the cap derivation and application.
            ax, ay, az = imu['xacc'], imu['yacc'], imu['zacc']
            acc_norm = math.sqrt(ax*ax + ay*ay + az*az)
            self._last_acc_norm = acc_norm

            self._integrate_kinematics(ax, ay, az, dt_imu)

    def _integrate_kinematics(self, ax, ay, az, dt_imu):
        """
        Strapdown integration — called under _state_lock.

        Applies a hard physical cap to the net linear acceleration, derived
        from the COMMANDED thrust (self._last_cmd_thrust) rather than the
        measured accelerometer magnitude:

            max_accel     = THRUST_ACCEL_COEFF * cmd_thrust²
            max_net_accel = sqrt(max_accel² − G²)     (max_accel ≥ G)
                          = 0                            (max_accel < G)

        Using the commanded value means the cap reflects what the motors are
        being asked to do, independent of accelerometer noise or transient
        readings that don't match what was actually commanded.  At
        cmd_thrust=0 (e.g. before the first control command), max_net=0 and
        velocity cannot change — correct, since no thrust has been requested.
        """
        qw, qx, qy, qz = self.ahrs.quaternion

        # Rotate specific force from body to NED (R_body→world inline)
        sf_n = (1-2*(qy*qy+qz*qz))*ax + 2*(qx*qy-qw*qz)*ay + 2*(qx*qz+qw*qy)*az
        sf_e = 2*(qx*qy+qw*qz)*ax + (1-2*(qx*qx+qz*qz))*ay + 2*(qy*qz-qw*qx)*az
        sf_d = 2*(qx*qz-qw*qy)*ax + 2*(qy*qz+qw*qx)*ay + (1-2*(qx*qx+qy*qy))*az

        # Remove gravity to get true (claimed) linear acceleration in NED
        a_n = sf_n
        a_e = sf_e
        a_d = sf_d + G

        # Physical cap derived from commanded thrust.
        max_accel = THRUST_ACCEL_COEFF * self._last_cmd_thrust * self._last_cmd_thrust
        max_net   = math.sqrt(max(0.0, max_accel*max_accel - G*G))
        self._last_max_net = max_net   # exposed for debug display

        a_mag = math.sqrt(a_n*a_n + a_e*a_e + a_d*a_d)
        if a_mag > max_net and a_mag > 1e-9:
            scale = max_net / a_mag
            a_n *= scale
            a_e *= scale
            a_d *= scale

        self.vel_ned[0] += a_n * dt_imu
        self.vel_ned[1] += a_e * dt_imu
        self.vel_ned[2] += a_d * dt_imu

        self.pos_ned[0] += self.vel_ned[0] * dt_imu
        self.pos_ned[1] += self.vel_ned[1] * dt_imu
        self.pos_ned[2] += self.vel_ned[2] * dt_imu

        # Body-frame velocity (R^T · vel_ned)
        vN, vE, vD = self.vel_ned
        self.vel_body[0] = (1-2*(qy*qy+qz*qz))*vN + 2*(qx*qy+qw*qz)*vE + 2*(qx*qz-qw*qy)*vD
        self.vel_body[1] = 2*(qx*qy-qw*qz)*vN     + (1-2*(qx*qx+qz*qz))*vE + 2*(qy*qz+qw*qx)*vD
        self.vel_body[2] = 2*(qx*qz+qw*qy)*vN     + 2*(qy*qz-qw*qx)*vE + (1-2*(qx*qx+qy*qy))*vD

    # ------------------------------------------------------------------
    # Main update — called at CONTROL_HZ (60 Hz) from main loop
    # ------------------------------------------------------------------

    def update(self):
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            time.sleep(1.0 / CONTROL_HZ)
            return

        with lock:
            imu         = self.data.get('imu')
            race_status = self.data.get('race_status')
            armed       = self.data.get('armed', False)

        # -- Disarm / sim-restart detection --------------------------------
        if self._was_armed and not armed:
            if self._disarm_at is None:
                print('Disarm detected — waiting before re-arm.', flush=True)
                self._disarm_at = time.time()
                with lock:
                    self.data['imu']         = None
                    self.data['race_status'] = None
                self._reset_flight_state()
            self._was_armed = armed
            time.sleep(1.0 / CONTROL_HZ)
            return

        if not armed and self._disarm_at is not None:
            if time.time() - self._disarm_at >= POST_DISARM_WAIT:
                print('Post-disarm wait done. Ready to re-arm.', flush=True)
                self._disarm_at        = None
                self._last_arm_attempt = 0.0
            else:
                self._was_armed = armed
                time.sleep(1.0 / CONTROL_HZ)
                return

        self._was_armed = armed

        # -- WAIT_FOR_DATA -------------------------------------------------
        if self.phase == Phase.WAIT_FOR_DATA:
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print('Sending arm command...', flush=True)
                    self._send_arm()
                    self._last_arm_attempt = now
            elif imu is not None:
                self._start_estimation_thread()
                print('Armed and IMU ready. Moving to WAIT_FOR_START.', flush=True)
                self.phase = Phase.WAIT_FOR_START
            else:
                self._send_attitude_rate(0.0, 0.0, 0.0, 0)
            time.sleep(1.0 / CONTROL_HZ)
            return

        # -- WAIT_FOR_START ------------------------------------------------
        if self.phase == Phase.WAIT_FOR_START:
            # Drone sits on the launch pad held by the sim — send zero thrust.
            self._send_attitude_rate(0.0, 0.0, 0.0, 0.0)

            if race_status is not None:
                sim_ms   = race_status['sim_boot_time_ms']
                start_ms = race_status['race_start_boot_time_ms']

                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f'[WAIT] Anchor set: sim_ms={sim_ms}', flush=True)

                race_is_fresh  = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_is_fresh and sim_ms >= start_ms

                if self._tick % DEBUG_EVERY_N == 0:
                    print(f'[WAIT] sim_ms={sim_ms}  race_start={start_ms}'
                          f'  fresh={race_is_fresh}  go={countdown_done}', flush=True)

                if countdown_done:
                    print('Countdown complete! Flying!', flush=True)
                    self.phase = Phase.FLYING

            elif self._tick % DEBUG_EVERY_N == 0:
                print('[WAIT] No race_status yet — holding...', flush=True)

            time.sleep(1.0 / CONTROL_HZ)
            return

        # -- FLYING --------------------------------------------------------
        if self.phase == Phase.FLYING:
            # Snapshot all estimation state atomically; release before control
            # maths and sleeping so the estimation thread is never blocked.
            with self._state_lock:
                roll_deg, pitch_deg, yaw_deg = self._att_deg
                pN  = float(self.pos_ned[0]);  pE  = float(self.pos_ned[1]);  pD  = float(self.pos_ned[2])
                vN  = float(self.vel_ned[0]);  vE  = float(self.vel_ned[1]);  vD  = float(self.vel_ned[2])
                vX  = float(self.vel_body[0]); vY  = float(self.vel_body[1]); vZ  = float(self.vel_body[2])
                rr  = float(self.rates_body[0]); pr = float(self.rates_body[1]); yr = float(self.rates_body[2])
                acc_norm    = self._last_acc_norm
                cmd_thrust  = self._last_cmd_thrust
                max_net     = self._last_max_net
                quat        = self.ahrs.quaternion   # body→NED rotation for elevation error

            # ----------------------------------------------------------------
            # GUIDANCE — gate-relative position from vision
            # ----------------------------------------------------------------
            vision     = self.data.get('vision_gate_estimate')
            vision_vel = self.data.get('vision_velocity')
            vision_valid = False
            bx = by = bz = float('nan')
            # frame_id needed throughout this block for gating derivative terms
            vis_frame_id = vision.get('frame_id') if vision is not None else None
            if vision is not None:
                bx = vision.get('body_x_m', float('nan'))
                by = vision.get('body_y_m', float('nan'))
                bz = vision.get('body_z_m', float('nan'))
                if not any(math.isnan(v) for v in (bx, by, bz)) and bx > 0.1:
                    vision_valid = True

            # World-frame elevation error and its vision-derived rate.
            # gate_pD: NED-down component of gate position relative to drone.
            #   Positive = gate is below the drone in the world.
            # elev_rate: d(gate_pD)/dt from consecutive vision frames (m/s).
            #   Positive = gate moving further below (or drone ascending).
            #   Used as D term in thrust — replaces IMU vD which accumulates drift.
            # Guard bx > MIN_BX_FOR_ELEV: below this range the geometry is
            # unreliable and could corrupt the frozen hold value.
            MIN_BX_FOR_ELEV = 3.0
            elev_rate = 0.0
            if vision_valid and bx > MIN_BX_FOR_ELEV and vis_frame_id is not None:
                qw, qx, qy, qz = quat
                gate_pD = (  2*(qx*qz - qw*qy) * bx
                           + 2*(qy*qz + qw*qx) * by
                           + (1 - 2*(qx*qx + qy*qy)) * bz)
                if (self._prev_elev_frame_id is not None
                        and 0 < vis_frame_id - self._prev_elev_frame_id <= 3):
                    dt_e = (vis_frame_id - self._prev_elev_frame_id) / 30.0
                    elev_rate = (gate_pD - self._prev_gate_pD) / dt_e
                self._prev_gate_pD       = gate_pD
                self._prev_elev_frame_id = vis_frame_id
                self._last_elev_err      = gate_pD
            elif not vision_valid:
                self._prev_gate_pD       = None
                self._prev_elev_frame_id = None

            # ----------------------------------------------------------------
            # VELOCITY FUSION — blend IMU strapdown with vision-derived velocity.
            # Vision runs at 30 Hz; the controller runs at 60 Hz; the same
            # vision_velocity dict therefore sits in self.data for ~2 controller
            # ticks before a new frame arrives.  We gate on frame_id so each
            # vision measurement is blended in exactly once, not twice.
            # Only vY and vZ are corrected; vX (forward) stays IMU-only because
            # the forward velocity from range-derivative is already well-damped
            # by the pitch dynamics and less prone to lateral drift.
            # ----------------------------------------------------------------
            OF_ALPHA     = 0.7   # IMU weight; 0.3 goes to vision estimate
            vel_source   = 'imu'
            if (vision_vel is not None
                    and vis_frame_id is not None
                    and vis_frame_id != self._last_fused_frame_id):
                vY = OF_ALPHA * vY + (1.0 - OF_ALPHA) * vision_vel['vy_body_mps']
                vZ = OF_ALPHA * vZ + (1.0 - OF_ALPHA) * vision_vel['vz_body_mps']
                self._last_fused_frame_id = vis_frame_id
                vel_source = 'fused'

            # ----------------------------------------------------------------
            # TURN COORDINATION — shared quantities for roll and yaw
            #
            # bearing_body : heading angle to gate center (deg, body frame).
            #                Drives roll (primary turn) and is computed here
            #                for use in both sections below.
            # blend        : 1.0 = far from gate (full roll authority)
            #                0.0 = at gate plane (roll zeroed, yaw aligns)
            #
            # Architecture:
            #   Roll  → primary steering via coordinated bank on bearing angle
            #   Yaw   → passive; only activates close in for perpendicular crossing
            # ----------------------------------------------------------------
            PERP_BLEND_DIST = 6.0    # m: distance at which blend ramp starts
            TILT_EMA_ALPHA  = 0.25   # EMA weight on new gate_tilt sample
                                     # Raw noise ±10° → smoothed to ~±4°

            bearing_body  = 0.0      # heading angle to gate, degrees (default 0)
            blend         = 0.0      # far/close blend factor (default: close, no P)
            gate_tilt_deg = float('nan')
            yaw_err       = 0.0
            yaw_mode      = 'no-vision'

            if vision_valid:
                # Bearing angle to gate. Wider cap (±25°) than before — this
                # drives roll, not yaw, so larger corrections are appropriate.
                bearing_body = float(np.clip(
                    math.degrees(math.atan2(by, bx)), -25.0, 25.0))

                blend = float(np.clip(bx / PERP_BLEND_DIST, 0.0, 1.0))

                # Bearing rate: d(bearing_body)/dt from consecutive vision frames.
                # Used as D term for roll — replaces IMU vY which accumulates drift.
                # Positive = heading error growing (gate drifting further off-axis).
                # Negative = heading error closing (drone turning toward gate).
                bearing_rate = 0.0
                if (vis_frame_id is not None
                        and self._prev_bearing_frame_id is not None
                        and 0 < vis_frame_id - self._prev_bearing_frame_id <= 3):
                    dt_b = (vis_frame_id - self._prev_bearing_frame_id) / 30.0
                    bearing_rate = (bearing_body - self._prev_bearing_body) / dt_b
                if vis_frame_id is not None:
                    self._prev_bearing_body     = bearing_body
                    self._prev_bearing_frame_id = vis_frame_id

                # ── Gate face normal (PnP rvec → body frame) ──────────────
                # EMA smoothing reduces the raw ±10° measurement noise to ~±4°
                # before it feeds the yaw command. EMA resets whenever PnP
                # drops out so stale values from a different gate don't persist.
                pnp_ok = vision.get('pnp_ok') and vision.get('pnp_rvec') is not None
                if pnp_ok:
                    rvec  = np.array(vision['pnp_rvec'], dtype=np.float64)
                    R, _  = cv2.Rodrigues(rvec)
                    n_cam = R @ np.array([0.0, 0.0, 1.0])
                    if n_cam[2] > 0:          # resolve IPPE sign ambiguity
                        n_cam = -n_cam
                    ct20 = math.cos(math.radians(20.0))
                    st20 = math.sin(math.radians(20.0))
                    n_bx = ct20 * (-n_cam[2]) + st20 * (-n_cam[1])
                    n_by = -n_cam[0]
                    tilt_raw = float(np.clip(
                        math.degrees(math.atan2(n_by, n_bx)), -30.0, 30.0))
                    if self._gate_tilt_ema is None:
                        self._gate_tilt_ema = tilt_raw
                    else:
                        self._gate_tilt_ema = (TILT_EMA_ALPHA * tilt_raw
                                               + (1.0 - TILT_EMA_ALPHA) * self._gate_tilt_ema)
                    gate_tilt_deg = self._gate_tilt_ema
                else:
                    self._gate_tilt_ema = None

                # ── Yaw: track gate bearing far out, blend to gate normal close in ──
                # Keeps the gate in frame throughout the approach.
                # bearing_body is already capped at ±25° for roll; apply a
                # tighter ±12° cap for yaw to limit body yaw rate.
                yaw_bearing = float(np.clip(bearing_body, -12.0, 12.0))
                if not math.isnan(gate_tilt_deg):
                    yaw_err  = blend * yaw_bearing + (1.0 - blend) * gate_tilt_deg
                    yaw_mode = f'blend={blend:.2f}'
                else:
                    yaw_err  = yaw_bearing
                    yaw_mode = 'bearing'
            else:
                self._gate_tilt_ema         = None
                self._prev_bearing_body     = None
                self._prev_bearing_frame_id = None
                bearing_rate                = 0.0


            # ----------------------------------------------------------------
            # UNIFIED D TERMS  (lateral and vertical)
            #
            # D_lateral  (m/s, body-right positive):
            #   Damping signal for roll.  Positive = moving right.
            #   desired_roll = p_lat - K_LAT_D * D_lateral
            #   → reduces right bank when moving right (brakes overshoot).
            #
            # D_vertical  (m/s, NED-down positive):
            #   Damping signal for thrust.  Positive = moving down.
            #   thrust += K_D * D_vertical
            #   → adds thrust when descending (brakes descent overshoot).
            #   → reduces thrust when climbing (brakes climb overshoot).
            #
            # On a new vision frame: derived from the position derivative
            #   (bearing_rate → lateral velocity; elev_rate → vertical velocity).
            #   IMU velocity reference is snapshotted so between-frame integration
            #   always starts from zero.
            #
            # Between frames (ONLY): IMU velocity increment since the last
            #   vision frame.  Integration window is ~16–33 ms so drift is
            #   negligible; this keeps D active at every 60 Hz controller tick.
            #
            # Vision lost: reference is reset so delta starts fresh from 0.
            # ----------------------------------------------------------------
            is_new_d_frame = (vision_valid
                              and vis_frame_id is not None
                              and vis_frame_id != self._last_d_frame_id)

            if is_new_d_frame:
                # Convert bearing_rate (deg/s) → lateral velocity (m/s):
                # bearing ≈ atan(by/bx), so d(bearing)/dt ≈ -vy_body / bx (rad/s)
                # → vy_body ≈ -bearing_rate_rad * bx
                bearing_rate_rad = math.radians(bearing_rate)
                D_lateral  = -bearing_rate_rad * max(bx, 0.5)   # m/s, cap bx to avoid ÷0

                # elev_rate = d(gate_pD)/dt = -vD_drone  (gate moves relatively
                # lower when drone climbs).  D_vertical = vD_drone = -elev_rate.
                D_vertical = -elev_rate   # m/s, NED-down positive

                # Snapshot IMU velocities so between-frame delta starts at 0
                self._vY_at_vision  = vY
                self._vD_at_vision  = vD
                self._last_d_frame_id = vis_frame_id

            elif not vision_valid:
                # Vision lost — reset reference to current IMU so delta stays small
                D_lateral  = 0.0
                D_vertical = 0.0
                self._vY_at_vision  = vY
                self._vD_at_vision  = vD
                self._last_d_frame_id = None

            else:
                # Between vision frames: IMU-integrated increment since last frame
                D_lateral  = vY - self._vY_at_vision   # m/s
                D_vertical = vD - self._vD_at_vision   # m/s, NED-down positive

            # ----------------------------------------------------------------
            # ATTITUDE COMMANDS
            # ----------------------------------------------------------------

            DESIRED_PITCH_DEG = -3.0    # hardcoded forward pitch
            K_BEARING         =  4.5    # deg bank per deg of bearing error
            K_LAT_D           =  9    # deg bank per m/s of D_lateral
            MAX_BANK_DEG      = 25.0

            KP = 1
            KR = -1
            KY = -1

            pitchCommand = (DESIRED_PITCH_DEG - pitch_deg) * KP

            # Roll: P on bearing angle, D on lateral velocity (vision + IMU)
            p_lat = K_BEARING * bearing_body * blend
            d_lat = K_LAT_D   * D_lateral    * blend
            desired_roll_deg = float(np.clip(p_lat - d_lat, -MAX_BANK_DEG, MAX_BANK_DEG))
            rollCommand = (desired_roll_deg - roll_deg) * KR

            yawCommand = yaw_err * KY

            # ----------------------------------------------------------------
            # THRUST
            # NED-down convention: D_vertical > 0 → moving down → add thrust
            #                      D_vertical < 0 → moving up   → reduce thrust
            # ----------------------------------------------------------------
            K_P_thrust = 0.014
            K_D_thrust = 0.0175

            tiltFactor = max(0.01, math.cos(math.radians(roll_deg))
                                 * math.cos(math.radians(pitch_deg)))

            thrustCommand = (HOVER_THRUST
                             - self._last_elev_err * K_P_thrust
                             + D_vertical          * K_D_thrust) / tiltFactor

            thrustCommand = float(np.clip(thrustCommand, 0.0, 1.0))

            if self._tick % DEBUG_EVERY_N == 0:
                d_src = 'vis' if is_new_d_frame else ('imu' if vision_valid else 'rst')
                print(
                    f'[NAV] vel_{vel_source}=({vX:+6.1f}fwd {vY:+5.1f}right {vZ:+5.1f}down)m/s  '
                    f'att=({roll_deg:+5.1f}r {pitch_deg:+5.1f}p {yaw_deg:+5.1f}y)°  '
                    f'gate=({bx:+5.1f}fwd {by:+5.1f}right {bz:+5.1f}down)m  '
                    f'elev={self._last_elev_err:+5.2f}m  '
                    f'D[{d_src}]=(lat={D_lateral:+5.2f} vert={D_vertical:+5.2f})m/s  '
                    f'roll=(p={p_lat:+5.1f} d={-d_lat:+5.1f} des={desired_roll_deg:+5.1f})°  '
                    f'yaw={yaw_mode}  '
                    f'cmd=(r={rollCommand:+5.2f} p={pitchCommand:+5.2f} '
                    f'y={yawCommand:+5.2f} T={thrustCommand:.3f})',
                    flush=True
                )
                if thrustCommand >= 0.99:
                    print('[WARN] thrust saturated high', flush=True)


            self._send_attitude_rate(rollCommand, pitchCommand, yawCommand, thrustCommand)

            time.sleep(1.0 / CONTROL_HZ)

    # ------------------------------------------------------------------
    # MAVLink helpers
    # ------------------------------------------------------------------

    def _send_arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )