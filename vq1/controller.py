"""
Telemetry-guided gate-to-gate controller (Virtual Qualifier 1).

VQ1 publishes every gate pose and the next gate index, so navigation is a
waypoint chase: gate position -> velocity setpoint -> attitude setpoint.
Altitude is a separate tilt-compensated PD on thrust; heading is fixed,
since the course is a straight corridor.

The camera runs too. Every tick the controller back-projects the detector's
bbox into a world-frame gate position, then overrides it with telemetry when
telemetry is available — so vision flies the whole course live without ever
being able to ruin a run. That is where the VQ2 perception stack was built and
checked; see `vision.py` and `analysis/`.

Frame is NED (X north, Y east, Z down). More negative Z is higher.
"""

import math
import time
from enum import Enum, auto

import numpy as np
from pymavlink import mavutil

from . import vision as vis

# --------------------------------------------------------------------------
# Loop timing
# --------------------------------------------------------------------------

CONTROL_HZ = 50               # spec caps the command rate below 100 Hz
DT         = 1.0 / CONTROL_HZ
DEBUG_EVERY_N = 50            # print one telemetry line per second

ARM_RETRY_S      = 1.0        # re-send arm command this often until it takes
POST_DISARM_WAIT = 0.25       # settle time before re-arming after a sim reset

# --------------------------------------------------------------------------
# Guidance — gate position to velocity setpoint
# --------------------------------------------------------------------------

V_MAX       = 5.0   # m/s cap on the proportional part of the setpoint
K_POS       = 1.0   # (m/s) of setpoint per metre of position error
V_MIN_CLOSE = 1.0   # m/s floor on the along-course setpoint (see _velocity_setpoint)

# Aim above gate centre, just past the 0.75 m half-opening. The altitude loop
# lags a descending course and settles low, so aiming high lands it centred.
GATE_RISE_M = 0.8

# --------------------------------------------------------------------------
# Attitude — velocity error to bank/pitch angle
# --------------------------------------------------------------------------

K_VX_P          = 5.0    # deg of pitch per m/s of north-velocity error
K_VY_P          = 5.0    # deg of roll  per m/s of east-velocity error
K_VY_D          = 0.4    # damping on east-velocity error only; north needs none
PITCH_LIMIT_DEG = 50.0
ROLL_LIMIT_DEG  = 50.0
COURSE_YAW_RAD  = math.pi   # face south down the corridor for the whole run

# --------------------------------------------------------------------------
# Vision fallback — see _vision_gate_position
# --------------------------------------------------------------------------

GATE_WIDTH_M          = 2.7    # outer frame, the pinhole range basis
CAM_TILT_DEG          = 20.0   # camera is pitched up from the body frame
VISION_ELEV_OFFSET_M  = 1.0    # empirical bias correction on the back-projection

# --------------------------------------------------------------------------
# Thrust — altitude PD
# --------------------------------------------------------------------------

HOVER_THRUST    = 0.264   # measured hover trim (see sysid/)
K_ALT_P         = 0.05    # thrust per metre of altitude error
K_ALT_D         = 0.08    # thrust per m/s of vertical speed
MIN_TILT_FACTOR = 0.01    # guards the divide when the drone is near-vertical


# --------------------------------------------------------------------------
# Math helpers
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


def quat_to_euler(qw, qx, qy, qz):
    """Quaternion to (roll, pitch, yaw) in degrees, ZYX convention."""
    roll = math.atan2(2.0 * (qw * qx + qy * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def body_to_world_velocity(odo):
    """
    Rotate body-frame odometry velocity into NED. The sim reports body frame;
    guidance compares against a world-frame setpoint.
    """
    qw, qx, qy, qz = odo['qw'], odo['qx'], odo['qy'], odo['qz']
    vx, vy, vz = odo['vx'], odo['vy'], odo['vz']
    return (
        (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz,
        2 * (qx * qy + qw * qz) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy + 2 * (qy * qz - qw * qx) * vz,
        2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (1 - 2 * (qx * qx + qy * qy)) * vz,
    )


def tilt_factor(qx, qy):
    """
    cos(roll)*cos(pitch) — the vertical share of thrust once tilted. Dividing
    the command by it keeps lift constant through a turn.
    """
    return max(MIN_TILT_FACTOR, 1.0 - 2.0 * (qx * qx + qy * qy))


class Phase(Enum):
    WAIT_FOR_DATA = auto()
    WAIT_FOR_START = auto()
    FLYING = auto()


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------

class Controller:
    """Flies the VQ1 course. `update()` paces itself; main.py just loops."""

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self._was_armed = False
        self._disarm_at = None
        self._reset_flight_state()

    def _reset_flight_state(self):
        self.phase = Phase.WAIT_FOR_DATA
        self._last_arm_attempt = 0.0
        self._tick = 0
        self._wait_start_sim_ms = None
        self._prev_vy_err = 0.0
        print('Controller state reset.', flush=True)

    # ---------------------------------------------------------------- public

    def arm(self):
        self._send_arm()

    def update(self):
        """One control tick. Dispatches on flight phase, then sleeps."""
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            return self._idle()

        with lock:
            odometry = self.data.get('odometry')
            race_status = self.data.get('race_status')
            gates = self.data.get('gates')
            armed = self.data.get('armed', False)

        if self._handle_disarm(armed, lock):
            return self._idle()

        if self.phase is Phase.WAIT_FOR_DATA:
            self._wait_for_data(armed, odometry)
        elif self.phase is Phase.WAIT_FOR_START:
            self._wait_for_start(race_status)
        elif self.phase is Phase.FLYING and odometry is not None:
            self._fly(odometry, race_status, gates)

        self._idle()

    def _idle(self):
        time.sleep(DT)

    # ------------------------------------------------------------ arm / reset

    def _handle_disarm(self, armed, lock):
        """
        Detect a sim reset and hold off re-arming until it settles. True means
        skip this tick. Keeps one process alive across many attempts.
        """
        if self._was_armed and not armed:
            if self._disarm_at is None:
                print('Disarm detected — waiting before re-arm.', flush=True)
                self._disarm_at = time.time()
                with lock:
                    self.data['odometry'] = None
                    self.data['race_status'] = None
                    self.data['gates'] = None
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

    def _wait_for_data(self, armed, odometry):
        """Retry the arm command until the sim arms us and sends odometry."""
        if not armed:
            now = time.time()
            if now - self._last_arm_attempt >= ARM_RETRY_S:
                print('Sending arm command...', flush=True)
                self._send_arm()
                self._last_arm_attempt = now
        elif odometry is not None:
            print('Armed and data ready. Waiting for race start.', flush=True)
            self.phase = Phase.WAIT_FOR_START

    def _wait_for_start(self, race_status):
        """
        Hold until the countdown elapses. `race_start_boot_time_ms` persists
        across runs, so it is only trusted once it is at or after the clock
        reading taken when this phase began.
        """
        if race_status is None:
            if self._tick % DEBUG_EVERY_N == 0:
                print('[WAIT] No race_status yet — holding...', flush=True)
            return

        sim_ms = race_status['sim_boot_time_ms']
        start_ms = race_status['race_start_boot_time_ms']

        if self._wait_start_sim_ms is None:
            self._wait_start_sim_ms = sim_ms

        is_fresh = start_ms > 0 and start_ms >= self._wait_start_sim_ms
        if is_fresh and sim_ms >= start_ms:
            print('Countdown complete — flying.', flush=True)
            self.phase = Phase.FLYING

    # ------------------------------------------------------------- navigation

    def _target_gate(self, race_status, gates):
        """Centre of the gate the sim wants next, as (north, east, down)."""
        if not gates or race_status is None:
            return None
        idx = race_status['active_gate_index']
        if idx >= len(gates):
            return None
        gate = gates[idx]
        return gate['pos_x'], gate['pos_y'], gate['pos_z']

    @staticmethod
    def _velocity_setpoint(error_n, error_e):
        """
        Velocity setpoint toward the gate, m/s NED.

        North carries a V_MIN_CLOSE floor: a pure P term decays to zero at the
        gate plane and never triggers the advance. East has no floor, because
        settling to zero laterally is the goal.
        """
        v_n = V_MIN_CLOSE * np.sign(error_n) + np.clip(K_POS * error_n, -V_MAX, V_MAX)
        v_e = np.clip(K_POS * error_e, -V_MAX, V_MAX)
        return float(v_n), float(v_e)

    @staticmethod
    def _vision_gate_position(vision, odometry, roll_deg, pitch_deg, yaw_deg):
        """
        Back-project a gate detection to a world-frame position, or None.

        Range comes from the bbox width against the known 2.7 m gate. The pixel
        offset gives a ray, which is rotated camera -> body (fixed 20 deg up
        tilt) -> world (drone attitude), scaled by range, and added to the
        drone's position.

        This is the whole VQ1 vision path, and it is why the detector could be
        developed here at all: it produces a real target every frame, so any
        error in it shows up immediately against the telemetry gate that
        actually does the flying.
        """
        if vision is None or not vision.get('bw'):
            return None

        # Pixel ray from the bbox centre. Bbox centre, not contour centroid —
        # the gate is hollow, so the centroid drifts toward the thicker side.
        vc_x = (vision['bx'] + vision['bw'] / 2.0) - vis.CX
        vc_y = (vision['by'] + vision['bh'] / 2.0) - vis.CY
        vc_z = vis.FX

        depth = (GATE_WIDTH_M * vis.FX) / vision['bw']
        ray_norm = math.sqrt(vc_x * vc_x + vc_y * vc_y + vc_z * vc_z)
        distance = depth * (ray_norm / vc_z)
        rc_x, rc_y, rc_z = vc_x / ray_norm, vc_y / ray_norm, vc_z / ray_norm

        # Camera -> body: undo the fixed upward tilt.
        ct, st = math.cos(math.radians(CAM_TILT_DEG)), math.sin(math.radians(CAM_TILT_DEG))
        rb_x = rc_z * ct + rc_y * st
        rb_y = rc_x
        rb_z = -rc_z * st + rc_y * ct

        # Body -> world: roll, then pitch, then yaw.
        phi, theta, psi = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
        r1_y = rb_y * math.cos(phi) - rb_z * math.sin(phi)
        r1_z = rb_y * math.sin(phi) + rb_z * math.cos(phi)
        r2_x = rb_x * math.cos(theta) + r1_z * math.sin(theta)
        r2_z = -rb_x * math.sin(theta) + r1_z * math.cos(theta)
        rw_x = r2_x * math.cos(psi) - r1_y * math.sin(psi)
        rw_y = r2_x * math.sin(psi) + r1_y * math.cos(psi)

        return (odometry['x'] + distance * rw_x,
                odometry['y'] + distance * rw_y,
                odometry['z'] + distance * r2_z + VISION_ELEV_OFFSET_M)

    def _fly(self, odometry, race_status, gates):
        """Run one guidance/control pass and send the resulting setpoint."""
        roll_deg, pitch_deg, yaw_deg = quat_to_euler(
            odometry['qw'], odometry['qx'], odometry['qy'], odometry['qz'])
        vel_n, vel_e, vel_d = body_to_world_velocity(odometry)

        # Vision runs first and produces a target on its own; telemetry then
        # overrides it whenever gate positions are available. That ordering is
        # deliberate and is how the detector got developed — it flew the full
        # course live on every run, steering nothing, while the telemetry
        # target beside it did the actual flying.
        target = self._vision_gate_position(
            self.data.get('vision_gate_estimate'), odometry,
            roll_deg, pitch_deg, yaw_deg)
        telemetry_target = self._target_gate(race_status, gates)
        if telemetry_target is not None:
            target = telemetry_target

        if target is None:
            self._hold(odometry, vel_d)
            return
        gate_n, gate_e, gate_d = target
        source = 'ODO' if telemetry_target is not None else 'VIS'

        # Guidance: position error -> velocity setpoint.
        v_des_n, v_des_e = self._velocity_setpoint(
            gate_n - odometry['x'], gate_e - odometry['y'])

        # Attitude: velocity error -> bank angle. Nose down (negative pitch)
        # accelerates north, so the pitch command is negated.
        err_vn = v_des_n - vel_n
        pitch_des = float(np.clip(-K_VX_P * err_vn,
                                  -PITCH_LIMIT_DEG, PITCH_LIMIT_DEG))

        err_ve = v_des_e - vel_e
        d_err_ve = (err_ve - self._prev_vy_err) / DT
        self._prev_vy_err = err_ve
        roll_des = float(np.clip(K_VY_P * err_ve + K_VY_D * d_err_ve,
                                 -ROLL_LIMIT_DEG, ROLL_LIMIT_DEG))

        # Thrust: PD on altitude error, then divided out by the tilt loss.
        alt_err = (gate_d - GATE_RISE_M) - odometry['z']
        thrust = (HOVER_THRUST - alt_err * K_ALT_P + vel_d * K_ALT_D)
        thrust /= tilt_factor(odometry['qx'], odometry['qy'])
        thrust = float(np.clip(thrust, 0.0, 1.0))

        if self._tick % DEBUG_EVERY_N == 0:
            print(f'[FLY/{source}] '
                  f'pos=({odometry["x"]:.1f},{odometry["y"]:.1f},{odometry["z"]:.1f}) '
                  f'err=({gate_n - odometry["x"]:+.1f},{gate_e - odometry["y"]:+.1f},'
                  f'{alt_err:+.1f}) '
                  f'att=({roll_deg:+.0f}r,{pitch_deg:+.0f}p) '
                  f'cmd=(r={roll_des:+.0f} p={pitch_des:+.0f} T={thrust:.3f})',
                  flush=True)

        # The sim's positive-roll convention is opposite to the NED sense used
        # above, so the roll command is negated on the way out.
        self._send_attitude_target(-roll_des, pitch_des, COURSE_YAW_RAD, thrust)

    def _hold(self, odometry, vel_d):
        """
        Hold altitude with no gate to chase. Hover thrust alone only cancels
        gravity — the damping term is what arrests an existing vertical rate.
        """
        thrust = (HOVER_THRUST + vel_d * K_ALT_D) / tilt_factor(
            odometry['qx'], odometry['qy'])
        self._send_attitude_target(0.0, 0.0, COURSE_YAW_RAD,
                                   float(np.clip(thrust, 0.0, 1.0)))

    # ---------------------------------------------------------------- mavlink

    def _send_attitude_target(self, roll_deg, pitch_deg, yaw_rad, thrust):
        """Type mask 7 = use the quaternion, ignore the body-rate fields."""
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            7,
            euler_to_quat(math.radians(roll_deg), math.radians(pitch_deg), yaw_rad),
            0, 0, 0,
            thrust)

    def _send_arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
