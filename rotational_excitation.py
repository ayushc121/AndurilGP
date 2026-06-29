"""
rotational_excitation.py
========================
Extension to sysid_excitation.py for identifying rotational dynamics surfaces.

Adds `run_rotational_doublet` and a grid runner that sweeps:
  axes × T_levels × amplitudes × reps

Output files per doublet:
  rot_{axis}_{T_str}_{amp_str}_{rep}.csv
  rot_{axis}_{T_str}_{amp_str}_{rep}_meta.json

CSV columns: t, x, y, z, vx_b, vy_b, vz_b, phi, theta, psi, p, q, r,
             ax_b, ay_b, az_b, cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust,
             phase_tag
phase_tag values: settle | excitation | post_settle

USAGE
-----
Import and call `run_doublet_grid()` from your flight loop, or call
`run_rotational_doublet()` directly for a single test.

    from rotational_excitation import run_doublet_grid
    controller = ...   # your object that has .send_attitude_rates and .get_state
    run_doublet_grid(controller)

The controller object must expose:
    controller.send_attitude_rates(roll, pitch, yaw, thrust)
    controller.get_odometry()   -> dict with qw,qx,qy,qz,x,y,z,vx,vy,vz,
                                          rollspeed,pitchspeed,yawspeed
    controller.get_imu()        -> dict with xacc,yacc,zacc  (or None)
    controller.sleep(seconds)   -> accurate sleep

SIGN CONVENTIONS
----------------
- NED frame: positive z is downward.
- Positive cmd_pitch → positive thetadot (nose up) in the sim.
- Raw pitchspeed from odometry = −thetadot.  The MATLAB script corrects for
  this; we do NOT negate in this Python file so the CSV stores raw values.
- body-frame velocities are delivered directly by the odometry.
"""

import math
import time
import csv
import json
import os
import numpy as np

# =============================================================================
# Test grid configuration
# =============================================================================
AXES        = ['roll', 'pitch', 'yaw']
T_LEVELS    = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
AMPLITUDES  = [0.3, 0.6, 1.0]
N_REPS      = 2

HALF_DURATION = 0.3    # seconds per doublet half  ← easy to change
POST_SETTLE   = 2.0    # seconds to log after doublet for lag decay observation
SETTLE_TIME   = 2.0    # seconds to wait after trim is reached before doublet

# Settled-rate threshold — all three body rates must be below this (rad/s)
SETTLE_RATE_THRESH = math.radians(5.0)   # 5 deg/s
SETTLE_WINDOW      = 0.5                 # seconds all rates must be below threshold

CONTROL_HZ = 50
DT         = 1.0 / CONTROL_HZ

# Altitude: hold constant with a simple P controller on NED z.
# During thrust variation tests the tilt compensator handles most of the work.
THRUST_HOVER = 0.265
KP_ALT       = 0.10
KD_ALT       = 0.05

# PID gains for attitude hold during doublet (same as sysid_excitation.py)
KP_PITCH = 0.015;  KD_PITCH = 0.001
KP_ROLL  = 0.015;  KD_ROLL  = 0.001025
KP_YAW   = 0.03;   KD_YAW   = 0.002

# Odometry keys (match sysid_excitation.py)
KEY_QW = 'qw'; KEY_QX = 'qx'; KEY_QY = 'qy'; KEY_QZ = 'qz'
KEY_X = 'x';  KEY_Y = 'y';  KEY_Z = 'z'
KEY_VX = 'vx'; KEY_VY = 'vy'; KEY_VZ = 'vz'
KEY_P = 'rollspeed'; KEY_Q = 'pitchspeed'; KEY_R = 'yawspeed'
KEY_AX = 'xacc'; KEY_AY = 'yacc'; KEY_AZ = 'zacc'

CSV_HEADER = [
    't', 'x', 'y', 'z',
    'vx_b', 'vy_b', 'vz_b',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'ax_b', 'ay_b', 'az_b',
    'cmd_roll', 'cmd_pitch', 'cmd_yaw', 'cmd_thrust',
    'phase_tag',
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers
# =============================================================================

def _unpack_odometry(odometry):
    """Returns (qw,qx,qy,qz, x,y,z, vx,vy,vz, p,q,r, phi,theta,psi)."""
    qw = odometry[KEY_QW]; qx = odometry[KEY_QX]
    qy = odometry[KEY_QY]; qz = odometry[KEY_QZ]
    x  = odometry[KEY_X];  y  = odometry[KEY_Y];  z  = odometry[KEY_Z]
    vx = odometry[KEY_VX]; vy = odometry[KEY_VY]; vz = odometry[KEY_VZ]
    p  = odometry[KEY_P];  q  = odometry[KEY_Q];  r  = odometry[KEY_R]

    phi   = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    theta = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))
    psi   = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

    return qw, qx, qy, qz, x, y, z, vx, vy, vz, p, q, r, phi, theta, psi


def _tilt_thrust(qx, qy):
    """Tilt-compensated hover thrust matching simulator formula."""
    tilt = 1.0 - 2.0 * (qx**2 + qy**2)
    tilt = max(tilt, 0.01)
    return THRUST_HOVER / tilt


def _doublet_cmd(t_local, amplitude, half_dur):
    """
    Positive-first doublet:
        +amplitude   for  0 <= t < half_dur
        -amplitude   for  half_dur <= t < 2*half_dur
        0            otherwise
    """
    if t_local < 0:
        return 0.0
    elif t_local < half_dur:
        return +amplitude
    elif t_local < 2 * half_dur:
        return -amplitude
    else:
        return 0.0


def _tag(axis, T_coll, amplitude, rep):
    """Filename-safe test tag."""
    return f'rot_{axis}_T{T_coll:.2f}_A{amplitude:.2f}_r{rep}'


# =============================================================================
# Single doublet runner
# =============================================================================

def run_rotational_doublet(controller, axis, T_collective, amplitude,
                            half_duration=HALF_DURATION,
                            n_reps=1, settle_time=SETTLE_TIME,
                            out_dir=OUT_DIR):
    """
    Fly a doublet excitation on the specified axis at the given collective
    thrust, for n_reps repetitions.

    Parameters
    ----------
    controller   : object exposing send_attitude_rates(), get_odometry(),
                   get_imu(), sleep()
    axis         : 'roll' | 'pitch' | 'yaw'
    T_collective : float, normalised collective thrust [0..1]
    amplitude    : float, doublet command amplitude [0..1]
    half_duration: float, seconds per doublet half
    n_reps       : int, number of repetitions at this (axis, T, amp)
    settle_time  : float, seconds to wait for rates to settle before/after
    out_dir      : str, directory to save CSV + JSON
    """
    assert axis in ('roll', 'pitch', 'yaw'), f'Unknown axis: {axis}'

    for rep in range(1, n_reps + 1):
        tag      = _tag(axis, T_collective, amplitude, rep)
        csv_path = os.path.join(out_dir, tag + '.csv')
        meta_path= os.path.join(out_dir, tag + '_meta.json')

        if os.path.isfile(csv_path):
            print(f'[ROT-SYSID] Skip {tag} — CSV already exists.')
            continue

        print(f'\n[ROT-SYSID] Starting {tag}')
        print(f'[ROT-SYSID]   axis={axis}  T={T_collective:.2f}  amp={amplitude:.2f}'
              f'  half_dur={half_duration:.3f}s  rep={rep}/{n_reps}')

        # --- Open CSV with line buffering so data survives abrupt exits ------
        csv_file = open(csv_path, 'w', newline='', buffering=1)
        writer   = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        t_start      = time.time()
        psi_target   = None
        alt_target   = None
        prev_pitch_e = 0.0
        prev_roll_e  = 0.0
        prev_yaw_e   = 0.0

        # Settle-detection state
        settle_start = None   # time when all rates first went below threshold

        # Trim state measured at end of settle (for OE initial conditions)
        trim_state = {}

        # Doublet starts at this wall-clock time (set after settle is confirmed)
        t_doublet_start = None
        phase           = 'settle'

        while True:
            t_now = time.time()
            t_el  = t_now - t_start

            odo = controller.get_odometry()
            imu = controller.get_imu()

            (qw, qx, qy, qz,
             xp, yp, zp,
             vx_b, vy_b, vz_b,
             p_r, q_r, r_r,
             phi, theta, psi) = _unpack_odometry(odo)

            if imu is not None:
                ax_b = imu[KEY_AX]; ay_b = imu[KEY_AY]; az_b = imu[KEY_AZ]
            else:
                ax_b = ay_b = az_b = float('nan')

            # Lock heading and altitude reference on first tick
            if psi_target is None:
                psi_target = psi
                alt_target = zp    # NED: maintain spawn altitude
                print(f'[ROT-SYSID] Psi locked={math.degrees(psi):.1f}deg  '
                      f'alt_target_z={zp:.2f}m (NED)')

            # --- Phase transitions -------------------------------------------
            if phase == 'settle':
                # Check settle condition: all rates < threshold for SETTLE_WINDOW
                all_settled = (abs(p_r) < SETTLE_RATE_THRESH and
                               abs(q_r) < SETTLE_RATE_THRESH and
                               abs(r_r) < SETTLE_RATE_THRESH)
                if all_settled:
                    if settle_start is None:
                        settle_start = t_now
                    elif (t_now - settle_start) >= SETTLE_WINDOW:
                        # Capture trim state
                        trim_state = dict(
                            phi=phi, theta=theta, psi=psi,
                            p=p_r, q=q_r, r=r_r,
                            vx_b=vx_b, vy_b=vy_b, vz_b=vz_b,
                            T_measured=T_collective,
                        )
                        phase = 'excitation'
                        t_doublet_start = t_now
                        print(f'[ROT-SYSID] Settled after {t_el:.1f}s — starting doublet.')
                        print(f'[ROT-SYSID]   Trim:  phi={math.degrees(phi):.1f}  '
                              f'theta={math.degrees(theta):.1f}  '
                              f'p={math.degrees(p_r):.1f}  q={math.degrees(q_r):.1f}')
                else:
                    settle_start = None  # reset if rates went back up

                # Enforce minimum settle time regardless of rate threshold
                if t_el < settle_time:
                    phase = 'settle'   # don't advance yet

            elif phase == 'excitation':
                t_local = t_now - t_doublet_start
                if t_local >= 2 * half_duration:
                    phase = 'post_settle'
                    t_post_start = t_now

            elif phase == 'post_settle':
                if (t_now - t_post_start) >= POST_SETTLE:
                    break  # done

            # --- Commands ---------------------------------------------------
            # Tilt-compensated thrust + altitude hold
            tilt_comp   = _tilt_thrust(qx, qy)
            alt_err     = zp - alt_target          # >0 = below target (NED)
            thrust_base = float(np.clip(
                tilt_comp + KP_ALT * alt_err + KD_ALT * vz_b, 0.0, 1.0))

            # Override thrust to commanded collective during excitation window
            # (and post-settle so the drone doesn't immediately lurch)
            if phase in ('excitation', 'post_settle'):
                thrust_cmd = float(np.clip(T_collective, 0.0, 1.0))
            else:
                thrust_cmd = thrust_base

            # Attitude PID — always active on all non-excited axes
            err_pitch    = math.degrees(0.0 - theta)  # hold level
            pitch_pid    = KP_PITCH * err_pitch - KD_PITCH * math.degrees(q_r)
            prev_pitch_e = err_pitch

            err_roll     = math.degrees(0.0 - phi)
            roll_pid     = KP_ROLL * err_roll - KD_ROLL * math.degrees(p_r)
            prev_roll_e  = err_roll

            err_yaw      = math.degrees(psi_target - psi)
            err_yaw      = (err_yaw + 180.0) % 360.0 - 180.0
            yaw_pid      = KP_YAW * err_yaw - KD_YAW * math.degrees(r_r)
            prev_yaw_e   = err_yaw

            # Doublet overlay on excited axis
            excite_roll = excite_pitch = excite_yaw = 0.0
            if phase == 'excitation':
                t_local  = t_now - t_doublet_start
                exc_val  = _doublet_cmd(t_local, amplitude, half_duration)
                if axis == 'roll':
                    excite_roll = exc_val
                elif axis == 'pitch':
                    excite_pitch = exc_val
                elif axis == 'yaw':
                    excite_yaw = exc_val

            roll_cmd  = float(np.clip(roll_pid  + excite_roll,  -1.0, 1.0))
            pitch_cmd = float(np.clip(pitch_pid + excite_pitch, -1.0, 1.0))
            yaw_cmd   = float(np.clip(yaw_pid   + excite_yaw,   -1.0, 1.0))

            controller.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

            # --- Log --------------------------------------------------------
            writer.writerow([
                f'{t_el:.5f}',
                f'{xp:.5f}', f'{yp:.5f}', f'{zp:.5f}',
                f'{vx_b:.5f}', f'{vy_b:.5f}', f'{vz_b:.5f}',
                f'{phi:.6f}', f'{theta:.6f}', f'{psi:.6f}',
                f'{p_r:.6f}', f'{q_r:.6f}', f'{r_r:.6f}',
                f'{ax_b:.5f}', f'{ay_b:.5f}', f'{az_b:.5f}',
                f'{roll_cmd:.6f}', f'{pitch_cmd:.6f}',
                f'{yaw_cmd:.6f}', f'{thrust_cmd:.6f}',
                phase,
            ])

            controller.sleep(DT)

        # --- Close CSV and write metadata ------------------------------------
        csv_file.flush()
        csv_file.close()
        print(f'[ROT-SYSID] Data saved → {csv_path}')

        meta = {
            'axis'          : axis,
            'T_collective'  : T_collective,
            'amplitude'     : amplitude,
            'half_duration' : half_duration,
            'rep'           : rep,
            'settle_time'   : settle_time,
            'post_settle'   : POST_SETTLE,
            'control_hz'    : CONTROL_HZ,
            'trim_state'    : trim_state,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'[ROT-SYSID] Metadata saved → {meta_path}')

        # Brief pause between repetitions
        if rep < n_reps:
            print(f'[ROT-SYSID] Pause 2s before rep {rep+1}...')
            time.sleep(2.0)


# =============================================================================
# Grid runner
# =============================================================================

def run_doublet_grid(controller,
                     axes=None,
                     T_levels=None,
                     amplitudes=None,
                     n_reps=N_REPS,
                     half_duration=HALF_DURATION,
                     settle_time=SETTLE_TIME,
                     out_dir=OUT_DIR):
    """
    Run the full doublet grid.  Skips tests whose CSV already exists so the
    run can be interrupted and resumed safely.

    Total tests = len(axes) × len(T_levels) × len(amplitudes) × n_reps
    Default grid: 3 × 6 × 3 × 2 = 108 doublets, ~8 minutes.

    Parameters
    ----------
    controller   : flight controller object (see module docstring)
    axes         : list of axis strings  (default: AXES)
    T_levels     : list of thrust levels (default: T_LEVELS)
    amplitudes   : list of amplitudes    (default: AMPLITUDES)
    n_reps       : int, reps per combination
    half_duration: float, doublet half-width in seconds
    settle_time  : float, seconds to settle between tests
    out_dir      : str, output directory
    """
    if axes      is None: axes      = AXES
    if T_levels  is None: T_levels  = T_LEVELS
    if amplitudes is None: amplitudes = AMPLITUDES

    total = len(axes) * len(T_levels) * len(amplitudes) * n_reps
    done  = 0

    print(f'\n[ROT-GRID] Starting doublet grid: {total} total tests')
    print(f'[ROT-GRID] half_duration={half_duration}s  settle_time={settle_time}s')
    print(f'[ROT-GRID] NOTE: If half_duration < τ_axis, rates will not build up fully.')
    print(f'[ROT-GRID]       Check MATLAB fits; if R² < 0.7 extend half_duration to 0.5–1.0s.\n')

    for axis in axes:
        for T_coll in T_levels:
            if T_coll >= 0.78:
                print(f'[ROT-GRID] WARNING: T={T_coll:.2f} may saturate thrust during '
                      f'attitude commands.  Watch for saturation artifacts in fits.')
            for amp in amplitudes:
                # Check how many reps already exist
                reps_needed = []
                for rep in range(1, n_reps + 1):
                    tag      = _tag(axis, T_coll, amp, rep)
                    csv_path = os.path.join(out_dir, tag + '.csv')
                    if os.path.isfile(csv_path):
                        print(f'[ROT-GRID] Skip {tag} — already exists.')
                        done += 1
                    else:
                        reps_needed.append(rep)

                if not reps_needed:
                    continue

                try:
                    run_rotational_doublet(
                        controller, axis, T_coll, amp,
                        half_duration=half_duration,
                        n_reps=len(reps_needed),
                        settle_time=settle_time,
                        out_dir=out_dir,
                    )
                    done += len(reps_needed)
                except KeyboardInterrupt:
                    print(f'\n[ROT-GRID] Interrupted after {done}/{total} tests.')
                    return
                except Exception as e:
                    print(f'[ROT-GRID] ERROR during {axis} T={T_coll} amp={amp}: {e}')
                    print('[ROT-GRID] Continuing to next test...')

                print(f'[ROT-GRID] Progress: {done}/{total}')

    print(f'\n[ROT-GRID] Complete.  {done}/{total} tests done.')
