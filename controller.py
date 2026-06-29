import time
import math
import csv
import os
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration — tune these once the drone is in the air
# -----------------------------------------------------------------------

CONTROL_HZ = 100          # spec hard-limits < 100 Hz
FX = FY      = 320.0

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25

DEBUG_EVERY_N = 50        # ~1 s at 50 Hz

# -----------------------------------------------------------------------
# Rotational surface sysid configuration
# -----------------------------------------------------------------------
ROT_AXES        = ['roll', 'pitch', 'yaw']
ROT_T_LEVELS    = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
ROT_AMPLITUDES  = [0.3, 0.6, 1.0]
ROT_N_REPS      = 2
ROT_HALF_DUR    = 0.3    # seconds per doublet half — increase to 0.5–1.0 if R² < 0.7
ROT_POST_SETTLE = 2.0    # seconds to log after doublet ends (captures lag decay)
ROT_SETTLE_TIME = 2.0    # minimum settle time before doublet fires
ROT_SETTLE_RATE = math.radians(5.0)   # all three rates must be below this
ROT_SETTLE_WIN  = 0.5    # seconds all rates must stay below threshold

ROT_LOG_DIR = "rot_sysid_logs"

ROT_CSV_HEADER = [
    't', 'x', 'y', 'z',
    'vx_b', 'vy_b', 'vz_b',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'ax_b', 'ay_b', 'az_b',
    'cmd_roll', 'cmd_pitch', 'cmd_yaw', 'cmd_thrust',
    'phase_tag',
]

# -----------------------------------------------------------------------
# Telemetry logging — sysid CSV
# -----------------------------------------------------------------------
# One CSV file per flight, written at every FLYING tick.
# Columns match the sysid state/input convention used in analyze_segment.m:
#   State  x = [u_b, v_b, w_b, p, q_raw, r, phi, theta, psi]
#   Input  u = [cmd_p, cmd_q, cmd_r, cmd_thrust]
# NOTE: q_raw is the raw pitchspeed from the simulator (= -thetadot).
#       analyze_segment.m negates it automatically for pitch tests.
#       Logging it raw here keeps the CSV unmodified from the sensor.
LOG_DIR = "sysid_logs"

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
# Module-level PID helpers (used inside FLYING / rotational sysid)
# -----------------------------------------------------------------------
# Gains match the existing tuning in the original controller.
_K_P_ROLL  = 0.015;  _K_D_ROLL  = 0.001025
_K_P_PITCH = 0.015;  _K_D_PITCH = 0.001
_K_P_YAW   = 0.03;   _K_D_YAW   = 0.002
_K_P_ALT   = 0.0925; _K_D_ALT   = 0.05
_THRUST_HOVER = 0.265

def _roll_pid(phi_rad, roll_rate_rads):
    """PD to hold level roll (phi_target = 0)."""
    err = -math.degrees(phi_rad)   # target 0 deg
    return _K_P_ROLL * err - _K_D_ROLL * math.degrees(roll_rate_rads)

def _pitch_pid(theta_rad, pitch_rate_rads):
    """PD to hold level pitch (theta_target = 0)."""
    err = -math.degrees(theta_rad)
    return _K_P_PITCH * err - _K_D_PITCH * math.degrees(pitch_rate_rads)

def _yaw_pid(psi_target, psi_now, yaw_rate_rads):
    """PD to hold a fixed heading."""
    err = math.degrees(psi_target - psi_now)
    err = (err + 180.0) % 360.0 - 180.0
    return _K_P_YAW * err - _K_D_YAW * math.degrees(yaw_rate_rads)

def _tilt_thrust_cmd(qx, qy, z_pos, alt_target, vz_b):
    """Tilt-compensated hover thrust with altitude hold (NED)."""
    tilt = max(1.0 - 2.0*(qx**2 + qy**2), 0.01)
    tilt_comp = _THRUST_HOVER / tilt
    alt_err   = z_pos - alt_target   # >0 → below target in NED → need more thrust
    return float(np.clip(tilt_comp + _K_P_ALT*alt_err + _K_D_ALT*vz_b, 0.0, 1.0))


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
        self._csv_file   = None
        self._csv_writer = None
        self._flight_t0  = None
        self._reset_flight_state()

    def _reset_flight_state(self):
        self.phase              = Phase.WAIT_FOR_DATA
        self._finished          = False
        self._last_arm_attempt  = 0.0
        self._tick              = 0
        self._wait_start_sim_ms = None
        self.prev_vy_err        = 0.0
        self.prev_vx_err        = 0.0

        # --- Rotational sysid state ---
        # Flat list of (axis, T_coll, amp, rep) built once on first FLYING tick.
        self._rot_grid          = None
        self._rot_grid_idx      = 0
        # Phase within current doublet: 'settle' | 'excitation' | 'post_settle' | 'done'
        self._rot_phase         = 'settle'
        self._rot_psi_target    = None    # heading locked on first FLYING tick
        self._rot_alt_target    = None    # NED z locked on first FLYING tick
        self._rot_settle_start  = None    # wall time when rates first stayed low
        self._rot_doublet_start = None    # wall time when excitation phase began
        self._rot_post_start    = None    # wall time when post_settle phase began
        self._rot_csv_file      = None
        self._rot_csv_writer    = None
        self._rot_t0            = None    # wall time at start of current doublet CSV
        print('Controller state reset.', flush=True)

    def _open_csv(self):
        """Open a new timestamped sysid CSV for this flight."""
        os.makedirs(LOG_DIR, exist_ok=True)
        fname = os.path.join(LOG_DIR, f"sysid_{int(time.time())}.csv")
        self._csv_file = open(fname, "w", newline="", buffering=1)  # line-buffered
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "t_s",
            "u_b", "v_b", "w_b",          # body-frame velocities (m/s)
            "p", "q_raw", "r",             # body rates (rad/s); q_raw = -thetadot
            "phi", "theta", "psi",         # Euler angles (rad)
            "x_ned", "y_ned", "z_ned",     # NED position (m) — useful context
            "cmd_p", "cmd_q", "cmd_r",     # attitude-rate commands sent to MAVLink
            "cmd_thrust",                  # normalised thrust [0,1]
        ])
        self._flight_t0 = time.time()
        print(f"[LOG] Opened sysid log: {fname}", flush=True)

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            self._flight_t0  = None
            print("[LOG] Sysid log closed.", flush=True)

    def _log_tick(self, t_s,
                  u_b, v_b, w_b,
                  p, q_raw, r,
                  phi, theta, psi,
                  x_ned, y_ned, z_ned,
                  cmd_p, cmd_q, cmd_r, cmd_thrust):
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            f"{t_s:.6f}",
            f"{u_b:.6f}",   f"{v_b:.6f}",   f"{w_b:.6f}",
            f"{p:.6f}",     f"{q_raw:.6f}",  f"{r:.6f}",
            f"{phi:.6f}",   f"{theta:.6f}",  f"{psi:.6f}",
            f"{x_ned:.4f}", f"{y_ned:.4f}",  f"{z_ned:.4f}",
            f"{cmd_p:.6f}", f"{cmd_q:.6f}",  f"{cmd_r:.6f}",
            f"{cmd_thrust:.6f}",
        ])

    # ------------------------------------------------------------------
    # Rotational sysid helpers
    # ------------------------------------------------------------------

    def _rot_build_grid(self):
        """Build flat list of (axis, T_coll, amp, rep), skipping existing CSVs."""
        os.makedirs(ROT_LOG_DIR, exist_ok=True)
        grid = []
        for axis in ROT_AXES:
            for T in ROT_T_LEVELS:
                for amp in ROT_AMPLITUDES:
                    for rep in range(1, ROT_N_REPS + 1):
                        tag = f'rot_{axis}_T{T:.2f}_A{amp:.2f}_r{rep}'
                        csv_path = os.path.join(ROT_LOG_DIR, tag + '.csv')
                        if os.path.isfile(csv_path):
                            print(f'[ROT] Skip {tag} — already exists.', flush=True)
                        else:
                            grid.append((axis, T, amp, rep))
        print(f'[ROT] Grid built: {len(grid)} doublets remaining.', flush=True)
        return grid

    def _rot_open_csv(self, axis, T_coll, amp, rep):
        """Open a new per-doublet CSV with line buffering."""
        self._rot_close_csv()   # safety — close any open file
        os.makedirs(ROT_LOG_DIR, exist_ok=True)
        tag = f'rot_{axis}_T{T_coll:.2f}_A{amp:.2f}_r{rep}'
        csv_path = os.path.join(ROT_LOG_DIR, tag + '.csv')
        self._rot_csv_file   = open(csv_path, 'w', newline='', buffering=1)
        self._rot_csv_writer = csv.writer(self._rot_csv_file)
        self._rot_csv_writer.writerow(ROT_CSV_HEADER)
        self._rot_t0 = time.time()
        print(f'[ROT] Opened {csv_path}', flush=True)

    def _rot_close_csv(self, axis=None, T_coll=None, amp=None, rep=None,
                        trim_state=None):
        """Flush CSV and write metadata JSON sidecar."""
        if self._rot_csv_file is None:
            return
        self._rot_csv_file.flush()
        self._rot_csv_file.close()
        self._rot_csv_file   = None
        self._rot_csv_writer = None

        if axis is not None:
            tag  = f'rot_{axis}_T{T_coll:.2f}_A{amp:.2f}_r{rep}'
            meta = {
                'axis'         : axis,
                'T_collective' : T_coll,
                'amplitude'    : amp,
                'half_duration': ROT_HALF_DUR,
                'rep'          : rep,
                'settle_time'  : ROT_SETTLE_TIME,
                'post_settle'  : ROT_POST_SETTLE,
                'control_hz'   : CONTROL_HZ,
                'trim_state'   : trim_state or {},
            }
            meta_path = os.path.join(ROT_LOG_DIR, tag + '_meta.json')
            import json
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f'[ROT] Closed {tag}.csv + wrote meta JSON.', flush=True)

    def _rot_log(self, t_s, x, y, z, vx_b, vy_b, vz_b,
                  phi, theta, psi, p, q, r, cmd_roll, cmd_pitch, cmd_yaw,
                  cmd_thrust, phase_tag):
        """Write one row to the per-doublet CSV."""
        if self._rot_csv_writer is None:
            return
        self._rot_csv_writer.writerow([
            f'{t_s:.5f}',
            f'{x:.5f}', f'{y:.5f}', f'{z:.5f}',
            f'{vx_b:.5f}', f'{vy_b:.5f}', f'{vz_b:.5f}',
            f'{phi:.6f}', f'{theta:.6f}', f'{psi:.6f}',
            f'{p:.6f}', f'{q:.6f}', f'{r:.6f}',
            'nan', 'nan', 'nan',          # ax_b/ay_b/az_b: IMU not wired in
            f'{cmd_roll:.6f}', f'{cmd_pitch:.6f}',
            f'{cmd_yaw:.6f}', f'{cmd_thrust:.6f}',
            phase_tag,
        ])

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
                self._close_csv()
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
                    self._open_csv()
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
            # ROTATIONAL SURFACE SYSID
            # ================================================================

            # --- Build grid once on first FLYING tick -----------------------
            if self._rot_grid is None:
                self._rot_grid = self._rot_build_grid()
                # Lock heading and altitude from the initial hover position
                self._rot_psi_target = math.atan2(
                    2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
                self._rot_alt_target = z_pos   # NED z to maintain
                print(f'[ROT] Psi target locked={math.degrees(self._rot_psi_target):.1f}deg'
                      f'  alt_target_z={z_pos:.2f}m', flush=True)

            # --- All tests done → hover -----------------------------------
            if self._rot_grid_idx >= len(self._rot_grid):
                if self._rot_phase != 'done':
                    self._rot_phase = 'done'
                    self._rot_close_csv()
                    print('[ROT] All doublets complete. Hovering.', flush=True)

                rollCommand   = 0.0
                pitchCommand  = 0.0
                yawCommand    = _yaw_pid(self._rot_psi_target, math.atan2(
                    2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2)), yaw_rate)
                thrustCommand = _tilt_thrust_cmd(
                    qx, qy, z_pos, self._rot_alt_target, z_v)

                self._log_tick(
                    t_s=time.time()-self._flight_t0,
                    u_b=x_v, v_b=y_v, w_b=z_v,
                    p=roll_rate, q_raw=pitch_rate, r=yaw_rate,
                    phi=math.radians(roll_deg), theta=math.radians(pitch_deg),
                    psi=math.radians(yaw_deg),
                    x_ned=x_pos, y_ned=y_pos, z_ned=z_pos,
                    cmd_p=rollCommand, cmd_q=pitchCommand,
                    cmd_r=yawCommand, cmd_thrust=thrustCommand,
                )
                self._send_attitude_rates(rollCommand, pitchCommand,
                                          yawCommand, thrustCommand)
                time.sleep(1.0 / CONTROL_HZ)
                return

            # --- Current test identity ------------------------------------
            axis, T_coll, amp, rep = self._rot_grid[self._rot_grid_idx]
            now = time.time()

            # Compute attitude error PIDs (always active on all axes)
            phi_rad   = math.radians(roll_deg)
            theta_rad = math.radians(pitch_deg)
            psi_now   = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))

            roll_pid  = _roll_pid(phi_rad,   roll_rate)
            pitch_pid = _pitch_pid(theta_rad, pitch_rate)
            yaw_pid_v = _yaw_pid(self._rot_psi_target, psi_now, yaw_rate)

            # Doublet command (zero outside excitation phase)
            excite_roll = excite_pitch = excite_yaw = 0.0

            # ---- Phase: settle ------------------------------------------
            if self._rot_phase == 'settle':
                # Open CSV for this doublet on first settle tick
                if self._rot_csv_writer is None:
                    self._rot_open_csv(axis, T_coll, amp, rep)
                    self._rot_settle_start = None
                    print(f'[ROT] Settling for {axis} T={T_coll:.2f}'
                          f' amp={amp:.2f} rep={rep}', flush=True)

                # Rate-based settle detection: all rates < threshold for ROT_SETTLE_WIN
                rates_ok = (abs(roll_rate) < ROT_SETTLE_RATE and
                            abs(pitch_rate) < ROT_SETTLE_RATE and
                            abs(yaw_rate)   < ROT_SETTLE_RATE)
                time_ok  = (now - self._rot_t0) >= ROT_SETTLE_TIME

                if rates_ok:
                    if self._rot_settle_start is None:
                        self._rot_settle_start = now
                    settled_long_enough = (now - self._rot_settle_start) >= ROT_SETTLE_WIN
                else:
                    self._rot_settle_start = None
                    settled_long_enough = False

                if time_ok and settled_long_enough:
                    # Capture trim state at end of settle
                    self._rot_trim = dict(
                        phi=phi_rad, theta=theta_rad, psi=psi_now,
                        p=roll_rate, q=pitch_rate, r=yaw_rate,
                        vx_b=x_v, vy_b=y_v, vz_b=z_v,
                        T_measured=T_coll,
                    )
                    self._rot_phase         = 'excitation'
                    self._rot_doublet_start = now
                    print(f'[ROT] Settled → excitation  phi={roll_deg:.1f}°'
                          f'  theta={pitch_deg:.1f}°'
                          f'  rates=({math.degrees(roll_rate):.1f},'
                          f'{math.degrees(pitch_rate):.1f},'
                          f'{math.degrees(yaw_rate):.1f}) deg/s', flush=True)

                phase_tag     = 'settle'
                thrustCommand = _tilt_thrust_cmd(qx, qy, z_pos,
                                                  self._rot_alt_target, z_v)

            # ---- Phase: excitation ---------------------------------------
            elif self._rot_phase == 'excitation':
                t_local = now - self._rot_doublet_start
                doublet_done = t_local >= 2 * ROT_HALF_DUR

                if doublet_done:
                    self._rot_phase      = 'post_settle'
                    self._rot_post_start = now
                    print(f'[ROT] Doublet done → post_settle', flush=True)
                else:
                    # Positive-first doublet: +amp then -amp
                    exc = amp if t_local < ROT_HALF_DUR else -amp
                    if axis == 'roll':
                        excite_roll  = exc
                    elif axis == 'pitch':
                        excite_pitch = exc
                    elif axis == 'yaw':
                        excite_yaw   = exc

                phase_tag     = 'excitation'
                # Hold commanded collective — do NOT let altitude hold override it
                thrustCommand = float(np.clip(T_coll, 0.0, 1.0))

            # ---- Phase: post_settle -------------------------------------
            elif self._rot_phase == 'post_settle':
                if (now - self._rot_post_start) >= ROT_POST_SETTLE:
                    # Advance to next test
                    self._rot_close_csv(axis, T_coll, amp, rep,
                                         trim_state=getattr(self, '_rot_trim', {}))
                    self._rot_grid_idx  += 1
                    self._rot_phase      = 'settle'
                    self._rot_settle_start  = None
                    self._rot_doublet_start = None
                    self._rot_post_start    = None
                    remaining = len(self._rot_grid) - self._rot_grid_idx
                    print(f'[ROT] Test done. {remaining} remaining.', flush=True)
                    time.sleep(1.0 / CONTROL_HZ)
                    return

                phase_tag     = 'post_settle'
                thrustCommand = _tilt_thrust_cmd(qx, qy, z_pos,
                                                  self._rot_alt_target, z_v)

            else:
                phase_tag     = 'settle'
                thrustCommand = _tilt_thrust_cmd(qx, qy, z_pos,
                                                  self._rot_alt_target, z_v)

            # Assemble final commands
            rollCommand  = float(np.clip(roll_pid  + excite_roll,  -1.0, 1.0))
            pitchCommand = float(np.clip(pitch_pid + excite_pitch, -1.0, 1.0))
            yawCommand   = float(np.clip(yaw_pid_v + excite_yaw,   -1.0, 1.0))
            thrustCommand= float(np.clip(thrustCommand,             0.0,  1.0))

            # Log to per-doublet CSV (rotational sysid format)
            if self._rot_t0 is not None:
                self._rot_log(
                    t_s=now - self._rot_t0,
                    x=x_pos, y=y_pos, z=z_pos,
                    vx_b=x_v, vy_b=y_v, vz_b=z_v,
                    phi=phi_rad, theta=theta_rad, psi=psi_now,
                    p=roll_rate, q=pitch_rate, r=yaw_rate,
                    cmd_roll=rollCommand, cmd_pitch=pitchCommand,
                    cmd_yaw=yawCommand, cmd_thrust=thrustCommand,
                    phase_tag=phase_tag,
                )

            # Log to existing sysid CSV (general telemetry)
            self._log_tick(
                t_s=time.time()-self._flight_t0,
                u_b=x_v, v_b=y_v, w_b=z_v,
                p=roll_rate, q_raw=pitch_rate, r=yaw_rate,
                phi=math.radians(roll_deg), theta=math.radians(pitch_deg),
                psi=math.radians(yaw_deg),
                x_ned=x_pos, y_ned=y_pos, z_ned=z_pos,
                cmd_p=rollCommand, cmd_q=pitchCommand,
                cmd_r=yawCommand, cmd_thrust=thrustCommand,
            )

            self._send_attitude_rates(rollCommand, pitchCommand, yawCommand, thrustCommand)


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