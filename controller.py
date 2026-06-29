import time
import math
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration — tune these once the drone is in the air
# -----------------------------------------------------------------------

CONTROL_HZ = 50          # spec hard-limits < 100 Hz
FX = FY      = 320.0

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25

DEBUG_EVERY_N = 50        # ~1 s at 50 Hz

MAVLINK_CMD_SIM_RESET = 31000

dt = 1.0 / CONTROL_HZ 

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
        self.prev_vy_err        = 0.0
        self.prev_vx_err        = 0.0
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
            0b00000111,   # 7 — ignore rates, use quaternion + thrust
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
            # FRAME CORRECTION — true world-frame velocities
            # ------------------------------------------------------------
            qw, qx, qy, qz = odometry['qw'], odometry['qx'], odometry['qy'], odometry['qz']
            vz_world = (2.0*(qx*qz - qw*qy) * x_v
                        + 2.0*(qy*qz + qw*qx) * y_v
                        + (1.0 - 2.0*(qx*qx + qy*qy)) * z_v)
            # World-frame x (north) velocity
            vx_world = ((1.0 - 2.0*(qy*qy + qz*qz)) * x_v
                        + 2.0*(qx*qy - qw*qz) * y_v
                        + 2.0*(qx*qz + qw*qy) * z_v)
            # World-frame y (east) velocity             
            vy_world = (2.0*(qx*qy + qw*qz) * x_v
                        + (1.0 - 2.0*(qx*qx + qz*qz)) * y_v
                        + 2.0*(qy*qz - qw*qx) * z_v)

            if self._tick % DEBUG_EVERY_N == 0:
                print(
                    f'[FLY] pos=({x_pos:.1f},{y_pos:.1f},'
                    f'{z_pos:.2f})  '
                    f'vel=({x_v:.2f},{y_v:.2f},'
                    f'{z_v:.2f})  '
                    f'roll={roll_deg:.1f}° pitch={pitch_deg:.1f}° yaw={yaw_deg:.1f}°',
                    flush=True
                )


            # ================================================================
            # VISUAL BASED GATE TARGETING
            # ----------------------------------------------------------------
            vision = self.data.get('vision_gate_estimate')    

            if vision is not None:
                # ==========================================
                # SLAM - CONTINUOUS MAPPING
                # ==========================================
                FX = 320.0
                FY = 320.0
                CX = 320.0
                CY = 180.0
                
                # 1. True Center (Immune to hollow-contour centroid shifting)
                true_cx = vision['bx'] + (vision['bw'] / 2.0)
                true_cy = vision['by'] + (vision['bh'] / 2.0)

                # Pixel vector in Camera Frame
                vc_x = true_cx - CX
                vc_y = true_cy - CY
                vc_z = FX 

                # 2. Exact Distance using Width (Immune to camera pitch distortion)
                # Real width of gate is 2.7m. Pin-hole horizontal distance:
                z_dist_cam = (2.7 * FX) / vision['bw']
                
                # Scale it out to the full 3D hypotenuse ray length
                ray_norm = math.sqrt(vc_x**2 + vc_y**2 + vc_z**2)
                est_distance_3d = z_dist_cam * (ray_norm / vc_z)

                # Normalize the pixel vector for pure rotation
                rc_x = vc_x / ray_norm
                rc_y = vc_y / ray_norm
                rc_z = vc_z / ray_norm

                # 3. Rotate to Body Frame (Apply +20 deg camera tilt)
                tilt_rad = math.radians(20.0)
                ctilt = math.cos(tilt_rad); stilt = math.sin(tilt_rad)
                
                rb_x = rc_z * ctilt + rc_y * stilt
                rb_y = rc_x
                rb_z = -rc_z * stilt + rc_y * ctilt

                # 4. Rotate to NED World Frame using Drone IMU
                phi = math.radians(roll_deg)
                theta = math.radians(pitch_deg)
                psi = math.radians(yaw_deg)

                c_phi = math.cos(phi); s_phi = math.sin(phi)
                c_the = math.cos(theta); s_the = math.sin(theta)
                c_psi = math.cos(psi); s_psi = math.sin(psi)

                # Roll (X-axis)
                r1_x = rb_x
                r1_y = rb_y * c_phi - rb_z * s_phi
                r1_z = rb_y * s_phi + rb_z * c_phi

                # Pitch (Y-axis)
                r2_x = r1_x * c_the + r1_z * s_the
                r2_y = r1_y
                r2_z = -r1_x * s_the + r1_z * c_the

                # Yaw (Z-axis) - This yields the final normalized World Vector
                rw_x = r2_x * c_psi - r2_y * s_psi
                rw_y = r2_x * s_psi + r2_y * c_psi
                rw_z = r2_z

                # 5. Apply strictly to map coordinates
                g_north = x_pos + (est_distance_3d * rw_x)
                g_east = y_pos + (est_distance_3d * rw_y)
                gate_pz = z_pos + (est_distance_3d * rw_z) + 1
            
            else:
                g_north = 0
                g_east = 0
                gate_pz = -3

            # ================================================================
            # ODOMETRY BASED GATE TARGETING
            # ----------------------------------------------------------------
            if gates:
                if gates is not getattr(self, '_gates_ref', None):
                    self._gates_ref = gates
                    print(f'[GATE] track packet: {len(gates)} gates, '
                          f'drone@receipt=({x_pos:.2f},{y_pos:.2f},{z_pos:.2f})', flush=True)


                active_idx = race_status['active_gate_index'] if race_status else 999
                if active_idx < len(gates):
                    ga = gates[active_idx]

                    # Active gate in WORLD-NED (all axes + signs resolved & frozen above).
                    g_north =  ga['pos_x']
                    g_east  =  ga['pos_y']
                    gate_pz =  ga['pos_z']


            # ================================================================
            # DESIRED PATH GENERATION
            # ----------------------------------------------------------------

            V_MAX     = 20.0     # m/s horizontal cap
            K_POS     = 1.7     # m/s per m of horizontal error

            v_des_north = 0.0           # setpoints default to hover / hold when no gate
            v_des_east  = 0.0
            elev_des    = -3.0

            vec_n = g_north - x_pos
            vec_e = g_east  - y_pos

            # Horizontal velocity setpoints (P on position, capped -> auto-slow in).
            v_des_north = 35 * np.sign(vec_n) + float(np.clip(K_POS * vec_n, -V_MAX, V_MAX)) * 0.15
            v_des_east  = float(np.clip(K_POS * vec_e, -V_MAX, V_MAX))

            # Altitude: 
            elev_des = gate_pz - 0.8

            
            # ================================================================
            # PID CONTROLLERS
            # ----------------------------------------------------------------

            # PITCH DES
            K_VX_P = 1.5           # deg of corrective pitch per m/s of north-speed error
            K_VX_D = 0
            PITCH_LIMIT = 50.0   # deg, max tilt

            vx_err = v_des_north - vx_world
            d_vx_err = (vx_err - self.prev_vx_err) / dt

            self.prev_vx_err = vx_err

            pitch_des_raw = (K_VX_P * vx_err) + (K_VX_D * d_vx_err)

            pitch_des = float(np.clip(pitch_des_raw, -PITCH_LIMIT, -20))


            # ROLL DES
            K_VY_P = 30         # Proportional: deg of tilt per m/s of error
            K_VY_D = 7.25        
            ROLL_LIMIT = 50.0    # deg, max tilt
            
            vy_err = v_des_east - vy_world
            d_vy_err = (vy_err - self.prev_vy_err) / dt
            
            self.prev_vy_err = vy_err
            
            roll_des_raw = (K_VY_P * vy_err) + (K_VY_D * d_vy_err)
            roll_des = float(np.clip(roll_des_raw, -ROLL_LIMIT, ROLL_LIMIT))


            yaw_des = 0
            

            # THRUST PID
            thrust_trim = 0.265  # experimentally determined, this is damn near correct +/- 0.0001
            
            K_P_thrust = 0.0925
            K_D_thrust = 0.05

            err_elev = elev_des - z_pos

            thrustCommand = thrust_trim - err_elev * K_P_thrust  + vz_world * K_D_thrust

            tilt_factor = 1.0 - 2.0 * (qx**2 + qy**2)
            if tilt_factor < 0.01: 
                tilt_factor = 0.01
            thrustCommand = thrustCommand / tilt_factor

            if thrustCommand > 1:
                print("Over Angled; thrust cannot maintain elevation")
            
            thrustCommand = np.clip(thrustCommand, 0, 1)


            
            self._send_attitude_target(roll_des, pitch_des, yaw_des, thrustCommand)

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
