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
HOVER_THRUST     = 0.2635
MAVLINK_CMD_SIM_RESET = 31000

# Known launch attitude — drone always starts on an angled block.
LAUNCH_PITCH_DEG = -17.8    # NED: negative = nose down

G = 9.81   # m/s²

# Physical cap on net linear acceleration, derived from the COMMANDED
# thrust rather than the measured accelerometer magnitude.
#
# Thrust-to-acceleration model (identified from earlier parameter work):
#       max_accel = THRUST_ACCEL_COEFF * cmd_thrust²
#
# This is the total specific-force magnitude the motors are *commanded* to
# produce.  As with the gravity-removal geometry, the component fighting
# gravity is G, so the maximum possible net linear acceleration is:
#
#       max_net_accel = sqrt(max_accel² − G²)     (max_accel ≥ G)
#                     = 0                            (max_accel < G)
#
# Using the commanded value instead of the measured accelerometer magnitude
# means the cap reflects what the motors are being asked to do, independent
# of accelerometer noise or any residual attitude-estimation error in the
# measured specific force.  cmd_thrust is captured from the last attitude
# target sent to the simulator (control loop, 60 Hz) and read by the
# estimation thread (400 Hz) under _state_lock.
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
        self._last_elev_err     = 0.0   # body_z_m held when vision is dark
        self._last_yaw_des      = 0.0   # absolute yaw setpoint held when PnP unavailable
        self._gate_normal_ema   = None  # smoothed yaw offset from gate face normal (deg)
        self._flying_started    = False # True after first FLYING tick — seeds yaw_des

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
        self._last_elev_err   = 0.0
        self._last_yaw_des    = 0.0
        self._gate_normal_ema = None
        self._flying_started  = False
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

            # Seed yaw_des from actual AHRS yaw on the very first FLYING tick.
            # Prevents a 0° vs actual-yaw mismatch from causing an instant spin.
            if not self._flying_started:
                self._last_yaw_des   = yaw_deg
                self._flying_started = True
                print(f'[FLY] start: yaw={yaw_deg:.1f}° pitch={pitch_deg:.1f}° '
                      f'roll={roll_deg:.1f}° — yaw_des seeded', flush=True)

            # ----------------------------------------------------------------
            # GUIDANCE — gate-relative position from vision
            # ----------------------------------------------------------------
            # Guidance gains
            K_FWD_POS  = 0.25    # desired fwd speed (m/s) per metre of gate distance
            V_FWD_MAX  = 6.0     # m/s forward speed cap
            K_LAT_POS  = 1.2    # desired lateral speed (m/s) per metre of lateral offset
            V_LAT_MAX  = 3.0    # m/s lateral speed cap

            vision = self.data.get('vision_gate_estimate')
            vision_valid = False
            if vision is not None:
                bx = vision.get('body_x_m', float('nan'))
                by = vision.get('body_y_m', float('nan'))
                bz = vision.get('body_z_m', float('nan'))
                if not any(math.isnan(v) for v in (bx, by, bz)) and bx > 0.1:
                    vision_valid = True

            if vision_valid:
                # Forward distance → desired forward speed (auto-slows on close approach)
                v_fwd_des = float(np.clip(K_FWD_POS * bx, 0.0, V_FWD_MAX))
                # Lateral offset → desired lateral speed (positive by = gate right → fly right)
                v_lat_des = float(np.clip(K_LAT_POS * by, -V_LAT_MAX, V_LAT_MAX))
                # Vertical: bz > 0 = gate is below drone (NED); hold for blackout
                self._last_elev_err = bz
            else:
                # Vision dark (passthrough cooldown or no gate).
                # Maintain gentle forward speed so the drone keeps flying through
                # the gate and toward the next one instead of braking hard.
                v_fwd_des = 3.0
                v_lat_des = 0.0

            # ----------------------------------------------------------------
            # YAW — goal 3: approach gate perpendicular to its face.
            #
            # Primary: PnP gate face normal → perpendicular approach heading.
            #   IPPE returns 2 solutions. The correct one has the gate face
            #   pointing TOWARD the camera, so n_cam[2] < 0 (opposing +Z look
            #   direction). If n_cam[2] > 0 we got the flipped solution — negate.
            #   Gate approach direction = -n_cam (from drone toward gate center,
            #   perpendicular to gate face).
            #
            # Fallback: if PnP unavailable, steer toward gate center by bearing.
            # ----------------------------------------------------------------
            if vision_valid:
                # Gate-center bearing: points drone at the gate opening.
                # PnP face-normal (goal 3) is implemented but disabled until the
                # drone reliably passes gate 1 — PnP yaw was rotating the drone
                # off-center at close range, causing harder collisions than
                # the simpler bearing approach. Re-enable after gate 1 is cleared.
                bearing_body = math.degrees(math.atan2(by, bx))
                gate_world_bearing = yaw_deg + bearing_body
                yaw_err = (gate_world_bearing - self._last_yaw_des + 180.0) % 360.0 - 180.0
                self._last_yaw_des += 0.3 * yaw_err

            # ----------------------------------------------------------------
            # ATTITUDE SETPOINTS — velocity errors → desired tilt angles
            # ----------------------------------------------------------------
            K_VX_P    = 3.5     # deg pitch per m/s forward-speed error
            K_VY_P    = 6.0     # deg roll  per m/s lateral-speed error
            PITCH_MAX = 25.0    # deg
            ROLL_MAX  = 25.0    # deg

            # Forward: vX (body frame) vs desired; nose-down (negative pitch) to accelerate
            vx_err  = v_fwd_des - vX
            pitch_des = float(np.clip(-K_VX_P * vx_err, -PITCH_MAX, PITCH_MAX))

            # Lateral: vY (body frame, right positive) vs desired
            vy_err  = v_lat_des - vY
            roll_des = float(np.clip(K_VY_P * vy_err, -ROLL_MAX, ROLL_MAX))

            yaw_des = self._last_yaw_des   # held across blackout; reset on arm

            # ----------------------------------------------------------------
            # ATTITUDE COMMANDS — sent directly as quaternion setpoints.
            # typemask=7 means the autopilot's own IMU loop holds these angles;
            # we must NOT add an outer feedback loop using AHRS pitch_deg/roll_deg
            # because AHRS drifts under vibration and makes commands insane.
            # ----------------------------------------------------------------
            pitchCommand = pitch_des   # deg, autopilot drives to this
            rollCommand  = roll_des    # deg, autopilot drives to this
            yawCommand   = yaw_des     # deg, absolute world-frame yaw

            # ----------------------------------------------------------------
            # THRUST — driven by vision body_z_m; held during blackout.
            # bz > 0  → gate below drone → descend → less thrust.
            # vD > 0  → falling → brake → more thrust.
            # Tilt compensation uses COMMANDED angles (reliable), not AHRS.
            # ----------------------------------------------------------------
            K_P_thrust = 0.06   # was 0.08; reduced to cut altitude overshoot past gate top
            K_D_thrust = 0.04

            thrustCommand = (HOVER_THRUST
                             - self._last_elev_err * K_P_thrust
                             + vD * K_D_thrust)

            tiltFactor = max(0.01, math.cos(math.radians(roll_des))
                                 * math.cos(math.radians(pitch_des)))

            if self._tick % DEBUG_EVERY_N == 0:
                gate_str = (f'{bx:.1f}m fwd  {by:+.1f}m right  {bz:+.1f}m down'
                            if vision_valid else 'no-vision')
                print(
                    f'[NAV] vel_body=({vX:+5.1f}fwd {vY:+5.1f}R {vZ:+5.1f}D)m/s  '
                    f'att=({roll_deg:+5.1f}r {pitch_deg:+5.1f}p {yaw_deg:+5.1f}y)°  '
                    f'gate={gate_str}  '
                    f'des=(p={pitch_des:+5.1f} r={roll_des:+5.1f} '
                    f'y={yaw_des:+6.1f})°  '
                    f'cmd=(p={pitchCommand:+5.2f} r={rollCommand:+5.2f} '
                    f'y={yawCommand:+5.2f} T={thrustCommand:.3f})',
                    flush=True
                )
                if thrustCommand > 1.0:
                    print('[WARN] thrust clipped — over-angled', flush=True)

            thrustCommand = np.clip(thrustCommand, 0, 1)
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