"""
sysid_excitation.py  (v3 — per-regime, per-axis, gain-scheduled)
=================================================================
Structured excitation script for regime-specific grey-box sysid.
Replaces the entire FLYING phase control block for data collection.

USAGE
-----
Edit the TEST CONFIGURATION block below, then run your simulator.
The script handles stabilisation, trim hold, excitation, and recovery
automatically. One CSV + one JSON metadata file are saved per run.

OUTPUT FILES
------------
  sysid_{REGIME_ID}_{EXCITE_AXIS}_{REPETITION}.csv
  sysid_{REGIME_ID}_{EXCITE_AXIS}_{REPETITION}_meta.json

CSV columns:
  t, x, y, z,
  vx_b, vy_b, vz_b,
  phi, theta, psi,
  p, q, r,
  ax_b, ay_b, az_b,
  cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust,
  phase_tag, regime_id

phase_tag values:
  settle      — initial stabilisation from launch ramp
  hold        — trim condition being held, pre-excitation
  excitation  — 3211 / doublet active  ← MATLAB uses only this
  recovery    — returning to hover
  done        — flight complete

PHASE TIMING  (seconds from start)
-----------------------------------
  0   – SETTLE_DUR      : PID drives to target trim
  +HOLD_DUR             : PID holds trim, data tagged 'hold'
  +EXCITE_DUR           : excitation on target axis
  +RECOVERY_DUR         : PID returns to hover
"""

import math
import time
import csv
import json
import os
import numpy as np

# =============================================================================
# TEST CONFIGURATION — the ONLY section you edit between runs
# =============================================================================
REGIME_ID    = 'P1'       # P0 P1 P2 P3 P4 P5  B1 B2 B3  C1 C2
EXCITE_AXIS  = 'roll'     # roll | pitch | yaw | heave | drop
REPETITION   = 1          # increment for repeat runs of the same test

# Target trim angles (degrees).  Use the table in the plan doc.
THETA0_DEG   =   -20.0      # target pitch  (negative = nose-down / forward)
PHI0_DEG     =   0.0      # target roll   (positive = right-wing-down)
T0_NOMINAL   =   0.265    # baseline hover thrust at this trim (before tilt comp) - DO NOT CHANGE

NOTES        = ''         # optional free-text annotation saved in metadata JSON
# =============================================================================

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
TAG      = f'sysid_{REGIME_ID}_{EXCITE_AXIS}_{REPETITION}'
CSV_PATH = os.path.join(OUT_DIR, TAG + '.csv')
META_PATH= os.path.join(OUT_DIR, TAG + '_meta.json')

CSV_HEADER = [
    't', 'x', 'y', 'z',
    'vx_b', 'vy_b', 'vz_b',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'ax_b', 'ay_b', 'az_b',
    'cmd_roll', 'cmd_pitch', 'cmd_yaw', 'cmd_thrust',
    'phase_tag', 'regime_id'
]

# ── Odometry / IMU keys ───────────────────────────────────────────────────────
KEY_QW='qw'; KEY_QX='qx'; KEY_QY='qy'; KEY_QZ='qz'
KEY_X='x'; KEY_Y='y'; KEY_Z='z'
KEY_VX='vx'; KEY_VY='vy'; KEY_VZ='vz'     # body-frame direct
KEY_P='rollspeed'; KEY_Q='pitchspeed'; KEY_R='yawspeed'
KEY_AX='xacc'; KEY_AY='yacc'; KEY_AZ='zacc'

# ── Timing ────────────────────────────────────────────────────────────────────
CONTROL_HZ    = 100
DT            = 1.0 / CONTROL_HZ

SETTLE_DUR    = 4.0    # s — stabilise from launch ramp to target trim
HOLD_DUR      = 3.0    # s — hold trim, verify it was reached
EXCITE_DUR    = 10.0   # s — active excitation window (fed to MATLAB)
RECOVERY_DUR  = 4.0    # s — return to hover before we stop

TOTAL_DUR     = SETTLE_DUR + HOLD_DUR + EXCITE_DUR + RECOVERY_DUR

# ── Convert trim targets to radians ──────────────────────────────────────────
THETA0 = math.radians(THETA0_DEG)
PHI0   = math.radians(PHI0_DEG)

# ── PID gains (match your existing tuning) ────────────────────────────────────
# Pitch
KP_PITCH = 0.015;  KD_PITCH = 0.001
# Roll
KP_ROLL  = 0.015;  KD_ROLL  = 0.001025
# Yaw  — hold current heading, don't fight drift aggressively
KP_YAW   = 0.03;   KD_YAW   = 0.002
# Altitude / heave — feedback on body-down velocity
KP_HEAVE = 0.05    # thrust delta per (m/s) of w_b error

# ── 3211 parameters ───────────────────────────────────────────────────────────
UNIT = 0.35   # seconds — base duration of one 3211 step
              # 7*UNIT = 2.45 s per sequence; fits well in 10 s excitation window

# ── Excitation amplitudes per axis ────────────────────────────────────────────
# Sized to produce clearly measurable angular rate responses (~10-20 deg/s)
# without departing too far from trim (keeping the linear assumption valid).
AMP = {
    'roll' : 0.12,   # ~14 deg/s peak roll rate
    'pitch': 0.12,   # ~14 deg/s peak pitch rate
    'yaw'  : 0.15,   # larger because yaw authority is weaker
    'heave': 0.20,   # thrust doublet amplitude (fraction of [0,1])
    'drop' : 0.0,    # drop cuts thrust to near-zero — no amplitude needed
}

# ── Heave doublet timing ──────────────────────────────────────────────────────
HEAVE_STEP_DUR = 2.0   # seconds per thrust step in doublet


# =============================================================================
# Helper: 3211 signal
# =============================================================================
def _3211(t_local):
    """
    Returns +1/-1/0 following the 3-2-1-1 pattern scaled by UNIT.
    Zero outside the 7*UNIT active window.
    """
    if   t_local < 0:          return  0.0
    elif t_local < 3*UNIT:     return +1.0
    elif t_local < 5*UNIT:     return -1.0
    elif t_local < 6*UNIT:     return +1.0
    elif t_local < 7*UNIT:     return -1.0
    else:                      return  0.0


# =============================================================================
# Helper: tilt-compensated thrust
# =============================================================================
def _tilt_thrust(base_thrust, phi, theta, w_b):
    """
    Compensates thrust command for drone tilt so altitude is maintained.
    Also adds a small w_b damping term.
    """
    tilt = math.cos(phi) * math.cos(theta)
    tilt = max(tilt, 0.10)   # never divide by near-zero at extreme angles
    return float(np.clip(base_thrust / tilt + KP_HEAVE * w_b, 0.0, 1.0))


# =============================================================================
# Main class
# =============================================================================
class SysIdSegment:
    """
    One test segment: settle → hold → excite → recover.

    In your flight loop (replace entire PID block):

        # ── initialise once ──────────────────────────────────────────
        from sysid_excitation import SysIdSegment
        self._seg = SysIdSegment()
        self._seg_start = None

        # ── each tick ────────────────────────────────────────────────
        if self._seg_start is None:
            self._seg_start = time.time()
        t_el = time.time() - self._seg_start

        imu = self.data.get('imu')
        rc, pc, yc, tc = self._seg.step(odometry, imu, t_el)
        self._send_attitude_rates(rc, pc, yc, tc)

        if t_el > self._seg.total_dur:
            self._seg.close()
            # land / transition here

        time.sleep(1.0 / CONTROL_HZ)
    """

    def __init__(self):
        self.total_dur      = TOTAL_DUR
        self._csv_file      = open(CSV_PATH, 'w', newline='')
        self._writer        = csv.writer(self._csv_file)
        self._writer.writerow(CSV_HEADER)

        # PID state
        self._prev_pitch_err = 0.0
        self._prev_roll_err  = 0.0
        self._prev_yaw_err   = 0.0
        self._psi_target     = None   # locked on first tick

        print(f'[SYSID] Regime={REGIME_ID}  Axis={EXCITE_AXIS}  Rep={REPETITION}')
        print(f'[SYSID] Target trim: theta={THETA0_DEG:.1f} deg  phi={PHI0_DEG:.1f} deg'
              f'  T0={T0_NOMINAL:.3f}')
        print(f'[SYSID] Logging → {CSV_PATH}')
        print(f'[SYSID] Total duration: {TOTAL_DUR:.1f} s')

    # ── single control tick ───────────────────────────────────────────────────
    def step(self, odometry, imu, t):
        """
        Returns (roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd).
        Logs one row to CSV.
        """
        # ── unpack state ──────────────────────────────────────────────────────
        qw=odometry[KEY_QW]; qx=odometry[KEY_QX]
        qy=odometry[KEY_QY]; qz=odometry[KEY_QZ]

        x_pos=odometry[KEY_X]; y_pos=odometry[KEY_Y]; z_pos=odometry[KEY_Z]
        vx_b=odometry[KEY_VX]; vy_b=odometry[KEY_VY]; vz_b=odometry[KEY_VZ]

        phi   = math.atan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
        theta = math.asin(max(-1.0, min(1.0, 2*(qw*qy-qz*qx))))
        psi   = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))

        p_r=odometry[KEY_P]; q_r=odometry[KEY_Q]; r_r=odometry[KEY_R]

        if imu is not None:
            ax_b=imu[KEY_AX]; ay_b=imu[KEY_AY]; az_b=imu[KEY_AZ]
        else:
            ax_b=ay_b=az_b=float('nan')

        # Lock yaw target on first tick
        if self._psi_target is None:
            self._psi_target = psi
            print(f'[SYSID] Yaw target locked: {math.degrees(psi):.1f} deg')

        # ── determine phase ───────────────────────────────────────────────────
        if   t < SETTLE_DUR:
            phase = 'settle'
        elif t < SETTLE_DUR + HOLD_DUR:
            phase = 'hold'
        elif t < SETTLE_DUR + HOLD_DUR + EXCITE_DUR:
            phase = 'excitation'
        elif t < TOTAL_DUR:
            phase = 'recovery'
        else:
            phase = 'done'

        t_excite = t - (SETTLE_DUR + HOLD_DUR)   # local time within excitation

        # ── compute commands ──────────────────────────────────────────────────
        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = \
            self._commands(phase, t_excite, phi, theta, psi,
                           p_r, q_r, r_r, vz_b)

        # ── log ───────────────────────────────────────────────────────────────
        self._writer.writerow([
            f'{t:.5f}',
            f'{x_pos:.5f}', f'{y_pos:.5f}', f'{z_pos:.5f}',
            f'{vx_b:.5f}',  f'{vy_b:.5f}',  f'{vz_b:.5f}',
            f'{phi:.6f}',   f'{theta:.6f}',  f'{psi:.6f}',
            f'{p_r:.6f}',   f'{q_r:.6f}',   f'{r_r:.6f}',
            f'{ax_b:.5f}',  f'{ay_b:.5f}',  f'{az_b:.5f}',
            f'{roll_cmd:.6f}', f'{pitch_cmd:.6f}',
            f'{yaw_cmd:.6f}',  f'{thrust_cmd:.6f}',
            phase, REGIME_ID
        ])

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd

    # ── command computation ───────────────────────────────────────────────────
    def _commands(self, phase, t_ex, phi, theta, psi,
                  p_r, q_r, r_r, w_b):
        """
        Compute roll/pitch/yaw/thrust commands for current phase.

        Strategy:
          • ALL phases run PID on every axis.
          • During 'excitation', the excited axis gets PID_trim + 3211 on top.
          • During 'settle'/'hold', PID drives to THETA0/PHI0.
          • During 'recovery', PID drives back to 0/0.
        """

        # ── pitch target ──────────────────────────────────────────────────────
        if phase in ('settle', 'hold', 'excitation'):
            theta_target = THETA0
        else:
            theta_target = 0.0   # recovery: return to level

        # ── roll target ───────────────────────────────────────────────────────
        if phase in ('settle', 'hold', 'excitation'):
            phi_target = PHI0
        else:
            phi_target = 0.0

        # ── PID for pitch (drives theta to theta_target) ──────────────────────
        err_pitch       = math.degrees(theta_target - theta)
        d_err_pitch     = (err_pitch - self._prev_pitch_err) / DT
        self._prev_pitch_err = err_pitch
        pitch_pid       = KP_PITCH * err_pitch - KD_PITCH * math.degrees(q_r)

        # ── PID for roll (drives phi to phi_target) ───────────────────────────
        err_roll        = math.degrees(phi_target - phi)
        d_err_roll      = (err_roll - self._prev_roll_err) / DT
        self._prev_roll_err = err_roll
        roll_pid        = KP_ROLL * err_roll - KD_ROLL * math.degrees(p_r)

        # ── PID for yaw (hold locked heading) ────────────────────────────────
        err_yaw         = math.degrees(self._psi_target - psi)
        err_yaw         = (err_yaw + 180.0) % 360.0 - 180.0
        yaw_pid         = KP_YAW * err_yaw - KD_YAW * math.degrees(r_r)

        # ── Tilt-compensated baseline thrust ─────────────────────────────────
        thrust_base     = _tilt_thrust(T0_NOMINAL, phi, theta, w_b)

        # ── Excitation overlay ────────────────────────────────────────────────
        excite_roll  = 0.0
        excite_pitch = 0.0
        excite_yaw   = 0.0
        excite_thrust_delta = 0.0

        if phase == 'excitation':
            amp = AMP[EXCITE_AXIS]

            if EXCITE_AXIS == 'roll':
                # Two back-to-back 3211 sequences with inverted sign
                # to capture both positive and negative nonlinearity
                seq1 = _3211(t_ex)
                seq2 = -_3211(t_ex - 7*UNIT - 0.5)   # 0.5 s gap between seqs
                excite_roll = amp * (seq1 + seq2)

            elif EXCITE_AXIS == 'pitch':
                seq1 = _3211(t_ex)
                seq2 = -_3211(t_ex - 7*UNIT - 0.5)
                excite_pitch = amp * (seq1 + seq2)

            elif EXCITE_AXIS == 'yaw':
                # Single 3211 — yaw recovers slowly so one sequence is enough
                excite_yaw = amp * _3211(t_ex)

            elif EXCITE_AXIS == 'heave':
                # Alternating thrust doublets
                idx = int(t_ex / HEAVE_STEP_DUR)
                excite_thrust_delta = amp if (idx % 2 == 0) else -amp

            elif EXCITE_AXIS == 'drop':
                # Near-zero thrust free-fall excitation.
                # Commands T=0.02 (minimum) for the full excitation window so
                # the drone enters free fall and reaches high downward velocity.
                # The altitude hold baseline is completely overridden here.
                # Recovery phase PID will bring it back up.
                excite_thrust_delta = 0.0   # handled by override below

        # ── Assemble final commands ───────────────────────────────────────────
        roll_cmd   = roll_pid  + excite_roll
        pitch_cmd  = pitch_pid + excite_pitch
        yaw_cmd    = yaw_pid   + excite_yaw
        thrust_cmd = thrust_base + excite_thrust_delta

        # ── Drop axis thrust override ─────────────────────────────────────────
        # During drop excitation we bypass tilt-compensated hover thrust
        # entirely and command minimum thrust to achieve free fall.
        # Attitude PIDs still run to keep the drone level during the drop.
        if EXCITE_AXIS == 'drop' and phase == 'excitation':
            thrust_cmd = 0.02   # near-zero, not exactly 0 to keep motors armed

        # ── Safety clips ─────────────────────────────────────────────────────
        roll_cmd   = float(np.clip(roll_cmd,  -0.6,  0.6))
        pitch_cmd  = float(np.clip(pitch_cmd, -0.6,  0.6))
        yaw_cmd    = float(np.clip(yaw_cmd,   -0.4,  0.4))
        thrust_cmd = float(np.clip(thrust_cmd, 0.0,  1.0))

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd

    def close(self):
        """Flush CSV and write metadata JSON."""
        self._csv_file.flush()
        self._csv_file.close()
        print(f'[SYSID] Data saved → {CSV_PATH}')

        meta = {
            'regime_id'    : REGIME_ID,
            'excite_axis'  : EXCITE_AXIS,
            'repetition'   : REPETITION,
            'theta0_deg'   : THETA0_DEG,
            'phi0_deg'     : PHI0_DEG,
            'T0_nominal'   : T0_NOMINAL,
            'control_hz'   : CONTROL_HZ,
            'settle_dur'   : SETTLE_DUR,
            'hold_dur'     : HOLD_DUR,
            'excite_dur'   : EXCITE_DUR,
            'recovery_dur' : RECOVERY_DUR,
            'unit_3211'    : UNIT,
            'amp_used'     : AMP.get(EXCITE_AXIS, 0.0),
            'notes'        : NOTES,
        }
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'[SYSID] Metadata saved → {META_PATH}')


# =============================================================================
# Regime reference table  (printed at import for quick reference)
# =============================================================================
REGIME_TABLE = """
╔══════════╦═══════════╦════════════╦══════════════════════════════════════╗
║ RegimeID ║  theta0   ║   phi0     ║  T0_nominal   Notes                 ║
╠══════════╬═══════════╬════════════╬══════════════════════════════════════╣
║ P0       ║   0 deg   ║   0 deg    ║  0.265        Hover baseline         ║
║ P1       ║ -20 deg   ║   0 deg    ║  0.28         Mild forward flight    ║
║ P2       ║ -40 deg   ║   0 deg    ║  0.35         Moderate forward       ║
║ P3       ║ -60 deg   ║   0 deg    ║  0.50         Aggressive forward     ║
║ P4       ║ -75 deg   ║   0 deg    ║  0.75         Near-max speed         ║
║ P5       ║ -85 deg   ║   0 deg    ║  0.95         Max speed / near-inv.  ║
║ B1       ║ -30 deg   ║  ±30 deg   ║  0.32         Moderate banked turn   ║
║ B2       ║ -45 deg   ║  ±45 deg   ║  0.45         Aggressive bank        ║
║ B3       ║ -60 deg   ║  ±50 deg   ║  0.65         Max bank racing turn   ║
║ C1       ║ -40 deg   ║  ±30 deg   ║  0.38         Combined validation    ║
║ C2       ║ -60 deg   ║  ±50 deg   ║  0.70         Extreme combined val.  ║
║ D1       ║   0 deg   ║   0 deg    ║  0.02         Free-fall drop test    ║
╚══════════╩═══════════╩════════════╩══════════════════════════════════════╝
Axes:  roll | pitch | yaw | heave | drop
Note:  For B1/B2/B3, run once with +phi0 and once with -phi0 (separate reps)
       T0_nominal values are estimates; the tilt-compensation PID will adapt.
"""

print(REGIME_TABLE)
print(f'[SYSID] Current config:  {TAG}  '
      f'theta0={THETA0_DEG:.0f}deg  phi0={PHI0_DEG:.0f}deg')