import time
import math
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CONTROL_HZ       = 50          # spec hard-limits < 100 Hz
ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25
DEBUG_EVERY_N    = 50          # ~1 s at 50 Hz
HOVER_THRUST     = 0.265       # neutral thrust to keep drone armed while waiting
MAVLINK_CMD_SIM_RESET = 31000

dt = 1.0 / CONTROL_HZ

# -----------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """
    Roll/pitch/yaw (radians, ZYX convention) → quaternion [w, x, y, z].
    """
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w, x, y, z]


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

        self._was_armed = False
        self._disarm_at = None
        self._reset_flight_state()

    def _reset_flight_state(self):
        self.phase              = Phase.WAIT_FOR_DATA
        self._finished          = False
        self._last_arm_attempt  = 0.0
        self._tick              = 0
        self._wait_start_sim_ms = None
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

    def _send_attitude_target(self, roll_deg, pitch_deg, yaw_deg, thrust):
        q = euler_to_quat(math.radians(roll_deg),
                          math.radians(pitch_deg),
                          math.radians(yaw_deg))
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0b00000111,   # ignore rates; use quaternion + thrust
            q,
            0, 0, 0,
            thrust
        )

    # ------------------------------------------------------------------
    # Main update — called at CONTROL_HZ from main loop
    # ------------------------------------------------------------------

    def update(self):
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            time.sleep(1.0 / CONTROL_HZ)
            return

        with lock:
            # Phase 2: ODOMETRY, ATTITUDE, LOCAL_POSITION_NED, and GATE_INFO
            # are blocked by the simulator and will never arrive.
            # We use HIGHRES_IMU (still available) as the "sensors alive" signal.
            imu         = self.data.get('imu')
            race_status = self.data.get('race_status')
            armed       = self.data.get('armed', False)

        # ------------------------------------------------------------------
        # Disarm / sim-restart detection
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # WAIT_FOR_DATA
        # Phase 2 fix: transition on IMU data rather than ODOMETRY (blocked).
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_DATA:
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print('Sending arm command...', flush=True)
                    self._send_arm()
                    self._last_arm_attempt = now
            elif imu is not None:
                print('Armed and IMU ready. Moving to WAIT_FOR_START.', flush=True)
                self.phase = Phase.WAIT_FOR_START
            else:
                # Armed but IMU not yet received — send a neutral hold so the
                # simulator watchdog does not disarm us during this brief window.
                self._send_attitude_target(0.0, 0.0, 0.0, 0)

            time.sleep(1.0 / CONTROL_HZ)
            return

        # ------------------------------------------------------------------
        # WAIT_FOR_START
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_START:
            # Send a neutral level-hold every tick so the simulator watchdog
            # does not disarm us during the countdown.
            self._send_attitude_target(0.0, 0.0, 0.0, 0)

            if race_status is not None:
                sim_ms   = race_status['sim_boot_time_ms']
                start_ms = race_status['race_start_boot_time_ms']

                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f'[WAIT] Anchor set: sim_ms={sim_ms}', flush=True)

                race_is_fresh  = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_is_fresh and sim_ms >= start_ms

                if self._tick % DEBUG_EVERY_N == 0:
                    print(
                        f'[WAIT] sim_ms={sim_ms}  race_start={start_ms} '
                        f' fresh={race_is_fresh}  go={countdown_done}',
                        flush=True
                    )

                if countdown_done:
                    print('Countdown complete! Flying!', flush=True)
                    self.phase = Phase.FLYING

            elif self._tick % DEBUG_EVERY_N == 0:
                print('[WAIT] No race_status yet — holding...', flush=True)

            time.sleep(1.0 / CONTROL_HZ)
            return

        # ------------------------------------------------------------------
        # FLYING
        # ------------------------------------------------------------------
        if self.phase == Phase.FLYING:

            self._send_attitude_target(0.0, 0.0, 0.0, HOVER_THRUST)


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