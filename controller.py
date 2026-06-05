import time
import math
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration — tune these once the drone is in the air
# -----------------------------------------------------------------------

CONTROL_HZ = 50          # spec hard-limits < 100 Hz

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25

DEBUG_EVERY_N = 50        # ~1 s at 50 Hz

MAVLINK_CMD_SIM_RESET = 31000

# -----------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """
    Roll/pitch/yaw (radians, ZYX convention) → quaternion [w, x, y, z].

    NED body-frame sign conventions:
      positive pitch = nose UP  → negative pitch = fly forward
      positive roll  = right side DOWN
      positive yaw   = clockwise from above (North→East)
    """
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w, x, y, z]


def quat_to_yaw(qw, qx, qy, qz):
    siny = 2.0 * (qw*qz + qx*qy)
    cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny, cosy)


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


    def _send_attitude_rates(self, roll_rad, pitch_rad, yaw_rad, thrust):
        """
        Sends raw roll/pitch/yaw RATES and thrust commands.
        Thrust is a value between 0.0 (motors off) and 1.0 (full throttle).
        """
        now_ms = int(time.time() * 1000)
        
        # typemask 7 (0b00000111) means "IGNORE body rates, USE attitude and thrust"
        typemask = 7 
        
        # Convert the desired Euler angles to a Quaternion
        q = euler_to_quat(roll_rad, pitch_rad, yaw_rad)
        
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            typemask,
            q,
            0, 0, 0,  # (Ignored by the typemask)
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
            odometry    = self.data.get('odometry')
            race_status = self.data.get('race_status')
            armed       = self.data.get('armed', False)
            gates = self.data.get('gates')

        # ------------------------------------------------------------------
        # Disarm / sim-restart detection
        # ------------------------------------------------------------------
        if self._was_armed and not armed:
            if self._disarm_at is None:
                print('Disarm detected — waiting before re-arm.', flush=True)
                self._disarm_at = time.time()
                with lock:
                    self.data['odometry']    = None
                    self.data['race_status'] = None
                    self.data['gates']       = None
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
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_DATA:
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print('Sending arm command...', flush=True)
                    self._send_arm()
                    self._last_arm_attempt = now
            elif odometry is not None:
                print('Armed and data ready. Moving to WAIT_FOR_START.', flush=True)
                self.phase = Phase.WAIT_FOR_START
                
            time.sleep(1.0 / CONTROL_HZ)
            return
        
        # ------------------------------------------------------------------
        # WAIT_FOR_START 
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_START:

            if race_status is not None:
                sim_ms   = race_status['sim_boot_time_ms']
                start_ms = race_status['race_start_boot_time_ms']

                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f'[WAIT] Anchor set: sim_ms={sim_ms}', flush=True)

                race_is_fresh  = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_is_fresh and sim_ms >= start_ms

                if self._tick % DEBUG_EVERY_N == 0:
                    print(f'[WAIT] sim_ms={sim_ms}  race_start={start_ms}  fresh={race_is_fresh}  go={countdown_done}', flush=True)

                if countdown_done:
                    print(f'Countdown complete! Flying!', flush=True)
                    self.phase = Phase.FLYING

            elif self._tick % DEBUG_EVERY_N == 0:
                print('[WAIT] No race_status yet — holding...', flush=True)

            time.sleep(1.0 / CONTROL_HZ)
            return

        # ------------------------------------------------------------------
        # FLYING
        # ------------------------------------------------------------------
        if self.phase == Phase.FLYING:
            if odometry is None:
                time.sleep(1.0 / CONTROL_HZ)
                return

            # EXTRACTING RELEVANT DATA FROM ODOMETRY
            yaw = quat_to_yaw(
                odometry['qw'], odometry['qx'],
                odometry['qy'], odometry['qz']
            )
            yaw_deg = math.degrees(yaw)

            roll_deg  = math.degrees(math.atan2(
                    2*(odometry['qw']*odometry['qx'] + odometry['qy']*odometry['qz']),
                    1 - 2*(odometry['qx']**2 + odometry['qy']**2)
                ))

            pitch_deg = math.degrees(math.asin(max(-1, min(1,
                    2*(odometry['qw']*odometry['qy'] - odometry['qz']*odometry['qx'])
                ))))

            yaw_rate = odometry['yawspeed']
            roll_rate = odometry['rollspeed']
            pitch_rate = odometry['pitchspeed']

            x_pos = odometry["x"]
            y_pos = odometry["y"]
            z_pos = odometry["z"]

            x_v = odometry["vx"]
            y_v = odometry["vy"]
            z_v = odometry["vz"]

            # ------------------------------------------------------------
            # FRAME CORRECTION — true world-frame vertical velocity
            # ------------------------------------------------------------
            # Odometry velocity (vx,vy,vz) is in the drone's BODY frame, but the
            # altitude controller needs the WORLD-down (NED) component. Rotate the
            # body velocity into world frame with the attitude quaternion and take
            # the down component (3rd row of the body->world rotation matrix).
            # This is SMOOTH because it uses the sim's own velocity — unlike
            # differentiating position, which was noisy. When level it reduces to
            # vz, exactly as you'd expect; under tilt it correctly mixes in vx.
            qw, qx, qy, qz = odometry['qw'], odometry['qx'], odometry['qy'], odometry['qz']
            vz_world = (2.0*(qx*qz - qw*qy) * x_v
                        + 2.0*(qy*qz + qw*qx) * y_v
                        + (1.0 - 2.0*(qx*qx + qy*qy)) * z_v)
            # World-frame x (north) velocity — same rotation, 1st row. Used by the
            # horizontal-hold term below to brake the forward coast.
            vx_world = ((1.0 - 2.0*(qy*qy + qz*qz)) * x_v
                        + 2.0*(qx*qy - qw*qz) * y_v
                        + 2.0*(qx*qz + qw*qy) * z_v)

            if self._tick % DEBUG_EVERY_N == 0:
                print(
                    f'[FLY] pos=({x_pos:.1f},{y_pos:.1f},'
                    f'{z_pos:.2f})  '
                    f'vel=({x_v:.2f},{y_v:.2f},'
                    f'{z_v:.2f})  '
                    f'roll={roll_deg:.1f}° pitch={pitch_deg:.1f}° yaw={yaw_deg:.1f}°',
                    flush=True
                )
            
            # ARRAY OF GATE COORDINATES, SHOULD BE IN ORDER
            if gates:
                gate_positions = [[g['pos_x'], g['pos_y'], g['pos_z']] for g in gates]


            # PITCH PID CONTROLLER 
            # ATTITUDE NOTE: the interface is RATE-like — commanding 0 HOLDS the
            # current tilt, it does NOT return to level (that's why zeroing
            # pitchCommand last run froze the launch tilt and the drone flew off).
            # Keeping an ACTIVE command with setpoint 0 commands a corrective rate
            # that decays the angle to level. Leveling is what lets the hover
            # thrust below (calibrated at level) actually hold altitude.
            # I own attitude on this branch (was pitch_des = 2).
            # HORIZONTAL HOLD (v1 — forward-velocity damping; this WILL evolve into
            # a proper position hold later). The sim has no aero drag, so once
            # level the drone coasts forward on the speed it built up at launch.
            # Command a small corrective pitch proportional to world-x speed to
            # actively brake it; when speed reaches ~0, pitch_des -> 0 (level), so
            # it halts and holds. Sign grounded in observed data: at launch
            # NEGATIVE pitch drove x negative (forward), so to push x back POSITIVE
            # we need POSITIVE pitch -> pitch_des = -K_VX * vx_world (vx_world < 0
            # while coasting -> nose up). Clamped to a small tilt so braking never
            # costs much altitude (vertical thrust ~ cos(tilt)).
            K_VX = 1.0           # deg of corrective pitch per m/s of world-x speed
            PITCH_LIMIT = 8.0    # deg, max braking tilt
            pitch_des = float(np.clip(-K_VX * vx_world, -PITCH_LIMIT, PITCH_LIMIT))

            K_P_pitch = 0.015    # raised from 0.005: levels the ~18° launch pitch in ~1s, not ~4s
            K_D_pitch = 0.001    # raised with K_P for damping (lower K_P if it oscillates)

            err_pitch = pitch_des - pitch_deg

            pitchCommand = K_P_pitch*err_pitch  -  K_D_pitch*pitch_rate



            # ROLL PID CONTROLLER 
            roll_des = 0.0  # level (was roll_des = 10)

            K_P_roll = 0.015    # raised from 0.005 to match pitch (roll starts ~level, but stay symmetric)
            K_D_roll = 0.001    # raised with K_P for damping

            err_roll = roll_des - roll_deg

            rollCommand = K_P_roll*err_roll  -  K_D_roll*roll_rate



            # THRUST PID (may want to add integral term...)

            elev_des = -3   # logic for this should be replaced later

            # HOVER 0.26567 only holds the drone up when it is LEVEL (see the
            # attitude block above) — a tilted drone needs thrust/cos(tilt).
            # The D-term now uses vz_world (frame-corrected) instead of body vz.
            thrust_trim = 0.26567  # experimentally determined, this is damn near correct +/- 0.0001
            
            K_P_thrust = 0.015    # similar tuning situation as pitch controller
            K_D_thrust = 0.022

            err_elev = elev_des - z_pos

            thrustCommand = thrust_trim - err_elev * K_P_thrust  + vz_world * K_D_thrust

            thrustCommand = np.clip(thrustCommand, 0, 1)

            # --- CSV logging (toggle: LOG_CSV = False to disable) -----------
            # Verify BOTH lanes at once: roll_deg/pitch_deg -> 0 (leveling) and
            # z -> elev_des with vz_world -> 0 (altitude). vz_body is kept beside
            # vz_world so you can see the frame correction at work under tilt.
            # File overwritten each run, line-buffered (survives Ctrl-C).
            LOG_CSV = True
            if LOG_CSV:
                alt_csv = getattr(self, '_alt_csv', None)
                if alt_csv is None:
                    alt_csv = open('altitude_log.csv', 'w', buffering=1)
                    alt_csv.write('tick,x,y,z,z_des,vx_body,vx_world,vz_body,vz_world,'
                                  'pitch_des,roll_deg,pitch_deg,yaw_deg,thrust\n')
                    self._alt_csv = alt_csv
                alt_csv.write(
                    f'{self._tick},{x_pos:.3f},{y_pos:.3f},{z_pos:.3f},{elev_des:.3f},'
                    f'{x_v:.3f},{vx_world:.3f},{z_v:.3f},{vz_world:.3f},{pitch_des:.2f},'
                    f'{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f},{thrustCommand:.4f}\n'
                )




            # THESE INPUTS ARE RATES FOR ROLL, PITCH, YAW   
            # units dont really work out cleanly but 0.05 --> 5-7 degrees per second roughly
            # Last input is thrust, 0-1
            self._send_attitude_rates(rollCommand, pitchCommand, 0.0, thrustCommand)


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