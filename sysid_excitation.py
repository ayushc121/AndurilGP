"""
sysid_excitation.py
====================
Drop-in replacement for the FLYING phase control block.
Runs a structured excitation sequence to collect data for
grey-box nonlinear sysid (equation-error + output-error via SIDPAC).

Excitation schedule (all times in seconds from phase start):
  Phase 0  [ 0 –  2s]  Pitch stabilise         — flatten from 20-deg launch ramp
  Phase 1  [ 2 – 17s]  Hover perturbations      — low-speed drag + control effectiveness
  Phase 2  [17 – 37s]  Forward flight ramp      — high-speed drag + Coriolis coupling
  Phase 3  [37 – 57s]  3211 multi-step inputs   — broadband rotational dynamics
  Phase 4  [57 – 72s]  Heave doublets           — thrust mapping + Z-axis drag

Output CSV:  sysid_data.csv  (one row per control tick)
Columns:
  t, x, y, z,
  vx_b, vy_b, vz_b,          <- body-frame (direct from odometry)
  phi, theta, psi,
  p, q, r,
  ax_b, ay_b, az_b,          <- body-frame (from highres_imu)
  cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust,
  phase
"""

import math
import time
import csv
import os
import numpy as np

# ── Odometry dict keys ────────────────────────────────────────────────────────
KEY_QW  = 'qw';  KEY_QX = 'qx';  KEY_QY = 'qy';  KEY_QZ = 'qz'
KEY_X   = 'x';   KEY_Y  = 'y';   KEY_Z  = 'z'
KEY_VX  = 'vx';  KEY_VY = 'vy';  KEY_VZ = 'vz'   # already body-frame
KEY_P   = 'rollspeed';  KEY_Q = 'pitchspeed';  KEY_R = 'yawspeed'

# ── IMU dict keys (from on_highres_imu) ──────────────────────────────────────
KEY_AX  = 'xacc';  KEY_AY = 'yacc';  KEY_AZ = 'zacc'

# ── Timing ────────────────────────────────────────────────────────────────────
CONTROL_HZ = 50
DT         = 1.0 / CONTROL_HZ

# ── Output file ───────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sysid_data.csv')
CSV_HEADER = [
    't',
    'x', 'y', 'z',
    'vx_b', 'vy_b', 'vz_b',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'ax_b', 'ay_b', 'az_b',
    'cmd_roll', 'cmd_pitch', 'cmd_yaw', 'cmd_thrust',
    'phase'
]

# ── Excitation parameters ─────────────────────────────────────────────────────
THRUST_TRIM  = 0.265    # hover thrust (level flight)
UNIT         = 0.4      # seconds — 3211 base unit duration

# Rate-command amplitudes (native units, ~2 rad/s per unit based on calibration)
AMP_HOVER    = 0.04     # ~5 deg/s  — gentle hover perturbations
AMP_3211     = 0.10     # ~12 deg/s — 3211 steps
AMP_HEAVE    = 0.15     # thrust doublet delta (fraction of [0,1])

# Forward flight trim values
PITCH_FF     = -0.20    # sustained forward pitch command during phase 2
THRUST_FF    = 0.75     # thrust during forward flight

# Phase 0 pitch-settle PID gains (matches your existing tuning)
K_P_PITCH_SETTLE = 0.015
K_D_PITCH_SETTLE = 0.001
SETTLE_DURATION  = 2.0  # seconds


# ═════════════════════════════════════════════════════════════════════════════
# Helper: 3211 signal value at local time t_local
# ─────────────────────────────────────────────────────────────────────────────
def _3211_value(t_local, unit):
    """
    +A for 3*unit  →  -A for 2*unit  →  +A for 1*unit  →  -A for 1*unit
    Total duration: 7*unit.  Returns 0.0 outside the active window.
    """
    if   t_local < 0:
        return  0.0
    elif t_local < 3 * unit:
        return +1.0
    elif t_local < 5 * unit:
        return -1.0
    elif t_local < 6 * unit:
        return +1.0
    elif t_local < 7 * unit:
        return -1.0
    else:
        return  0.0


# ═════════════════════════════════════════════════════════════════════════════
# Main excitation class
# ─────────────────────────────────────────────────────────────────────────────
class SysIdExcitation:
    """
    Usage in your flight loop
    ─────────────────────────
    Initialise once (e.g. in __init__):
        self._sysid = SysIdExcitation()
        self._sysid_start = None

    Each tick in the FLYING phase (replace the entire PID block):
        if self._sysid_start is None:
            self._sysid_start = time.time()
        t_elapsed = time.time() - self._sysid_start

        imu = self.data.get('imu')          # may be None if packet not yet received
        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = \\
            self._sysid.step(odometry, imu, t_elapsed)
        self._send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

        if t_elapsed > 72.0:
            self._sysid.close()

        time.sleep(1.0 / CONTROL_HZ)
    """

    def __init__(self):
        self._csv_file = open(CSV_PATH, 'w', newline='')
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow(CSV_HEADER)
        self._prev_pitch_err = 0.0      # for phase-0 pitch PID derivative
        print(f'[SYSID] Logging to {CSV_PATH}')

    # ── single control tick ───────────────────────────────────────────────────
    def step(self, odometry, imu, t):
        """
        odometry : dict   — simulator odometry packet
        imu      : dict | None  — highres_imu packet (may lag by one tick)
        t        : float  — elapsed time since sysid start (seconds)

        Returns  : (roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)
        """

        # ── unpack odometry ───────────────────────────────────────────────────
        qw = odometry[KEY_QW];  qx = odometry[KEY_QX]
        qy = odometry[KEY_QY];  qz = odometry[KEY_QZ]

        x_pos = odometry[KEY_X]
        y_pos = odometry[KEY_Y]
        z_pos = odometry[KEY_Z]

        # Body-frame velocities — direct from odometry, no rotation needed
        vx_b = odometry[KEY_VX]
        vy_b = odometry[KEY_VY]
        vz_b = odometry[KEY_VZ]

        # Euler angles (radians)
        phi   = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        theta = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))
        psi   = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

        # Angular rates (rad/s)
        p = odometry[KEY_P]
        q = odometry[KEY_Q]
        r = odometry[KEY_R]

        # Body-frame accelerations from IMU (NaN if packet not yet available)
        if imu is not None:
            ax_b = imu[KEY_AX]
            ay_b = imu[KEY_AY]
            az_b = imu[KEY_AZ]
        else:
            ax_b = float('nan')
            ay_b = float('nan')
            az_b = float('nan')

        # ── compute excitation commands ───────────────────────────────────────
        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, phase = \
            self._excitation(t, phi, theta, psi, p, q, r, vx_b, vy_b, vz_b)

        # ── log row ───────────────────────────────────────────────────────────
        self._writer.writerow([
            f'{t:.4f}',
            f'{x_pos:.5f}',    f'{y_pos:.5f}',    f'{z_pos:.5f}',
            f'{vx_b:.5f}',     f'{vy_b:.5f}',     f'{vz_b:.5f}',
            f'{phi:.6f}',      f'{theta:.6f}',     f'{psi:.6f}',
            f'{p:.6f}',        f'{q:.6f}',         f'{r:.6f}',
            f'{ax_b:.5f}',     f'{ay_b:.5f}',      f'{az_b:.5f}',
            f'{roll_cmd:.6f}', f'{pitch_cmd:.6f}',
            f'{yaw_cmd:.6f}',  f'{thrust_cmd:.6f}',
            phase
        ])

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        print(f'[SYSID] Data saved → {CSV_PATH}')

    # ── excitation schedule ───────────────────────────────────────────────────
    def _excitation(self, t, phi, theta, psi, p, q, r, u, v, w):
        """
        Returns (roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, phase_label)

        Baseline throughout all phases:
          • Light yaw damping (no hard heading hold — yaw command contamination
            would corrupt the yaw channel identification in phase 3)
          • Altitude hold via body-down velocity feedback on thrust
        """

        # Baseline commands — overridden per phase as needed
        yaw_cmd    = -0.015 * r                          # light yaw damping only
        thrust_cmd =  THRUST_TRIM + 0.05 * w            # altitude hold (w = body-down vel)

        # ── Phase 0: pitch stabilise [0 – 2s] ────────────────────────────────
        # Drone launches from a ~20-deg angled block.  Use the same pitch PID
        # gains from your existing controller to drive theta to zero before
        # excitation begins.  Roll and yaw are left at neutral.
        if t < SETTLE_DURATION:
            phase = 'settle'

            pitch_deg    = math.degrees(theta)
            pitch_des    = 0.0                           # target: level
            err_pitch    = pitch_des - pitch_deg
            d_err_pitch  = (err_pitch - self._prev_pitch_err) / DT
            self._prev_pitch_err = err_pitch

            pitch_cmd = K_P_PITCH_SETTLE * err_pitch - K_D_PITCH_SETTLE * math.degrees(q)
            roll_cmd  = 0.0

            # Boost thrust slightly to counteract the ramp angle during settle
            tilt_factor = max(0.01, 1.0 - 2.0 * (qx_from_theta(theta)**2))
            thrust_cmd  = float(np.clip(THRUST_TRIM / max(tilt_factor, 0.5) + 0.05 * w,
                                        0.0, 1.0))

        # ── Phase 1: hover perturbations [2 – 17s] ───────────────────────────
        elif t < 17.0:
            phase = 'hover'
            t_loc = t - SETTLE_DURATION

            # Sinusoidal chirp: frequency sweeps 0.2 → 1.5 Hz over 15s
            # Two different frequencies on roll vs pitch to decorrelate axes
            f      = 0.2 + (1.3 / 15.0) * t_loc
            omega  = 2 * math.pi * f

            roll_cmd  =  AMP_HOVER * math.sin(omega * t_loc)
            pitch_cmd = -AMP_HOVER * math.cos(omega * t_loc * 0.7)

            # Yaw: independent slower chirp
            yaw_cmd   =  AMP_HOVER * 0.5 * math.sin(2 * math.pi * 0.15 * t_loc)

        # ── Phase 2: forward flight ramp [17 – 37s] ──────────────────────────
        elif t < 37.0:
            phase = 'forward_flight'
            t_loc = t - 17.0

            # Trapezoid ramp: 5s up, 10s hold, 5s back to hover
            if t_loc < 5.0:
                ramp = t_loc / 5.0
            elif t_loc < 15.0:
                ramp = 1.0
            else:
                ramp = max(0.0, 1.0 - (t_loc - 15.0) / 5.0)

            pitch_cmd = PITCH_FF * ramp

            # Roll perturbations scaled with ramp so they're zero at hover endpoints
            # — excites Coriolis cross-coupling (Γ1*q*r) only during high-speed flight
            roll_cmd  = AMP_HOVER * ramp * math.sin(2 * math.pi * 0.5 * t_loc)

            # Thrust: tilt-compensated so altitude is maintained during pitch
            cos_theta = math.cos(theta)
            cos_phi   = math.cos(phi)
            tilt_comp = max(0.01, cos_theta * cos_phi)
            thrust_cmd = float(np.clip(
                (THRUST_TRIM + (THRUST_FF - THRUST_TRIM) * ramp) / tilt_comp + 0.05 * w,
                0.0, 1.0
            ))

        # ── Phase 3: 3211 multi-step [37 – 57s] ──────────────────────────────
        elif t < 57.0:
            phase = '3211'
            t_loc = t - 37.0
            period = 7 * UNIT       # 2.8s per 3211 sequence
            gap    = 1.0            # 1s neutral gap between axes

            roll_cmd  = 0.0
            pitch_cmd = 0.0

            # ── Axis schedule ─────────────────────────────────────────────────
            # Block 0 [0.0  – 2.8s]   Roll  3211  (+first)
            # Block 1 [3.8  – 6.6s]   Pitch 3211  (+first)
            # Block 2 [7.6  – 10.4s]  Yaw   3211  (half amplitude)
            # Block 3 [11.4 – 14.2s]  Roll  3211  (-first, inverted)
            # Block 4 [15.2 – 18.0s]  Pitch 3211  (-first, inverted)
            # Remainder: neutral until phase ends at t=57s

            if t_loc < period:
                # Block 0: Roll
                roll_cmd  =  AMP_3211 * _3211_value(t_loc, UNIT)

            elif t_loc < 2 * period + gap:
                # Block 1: Pitch
                pitch_cmd =  AMP_3211 * _3211_value(t_loc - (period + gap), UNIT)

            elif t_loc < 3 * period + 2 * gap:
                # Block 2: Yaw
                yaw_cmd   =  AMP_3211 * 0.5 * _3211_value(
                                 t_loc - (2 * period + 2 * gap), UNIT)

            elif t_loc < 4 * period + 3 * gap:
                # Block 3: Roll inverted
                roll_cmd  = -AMP_3211 * _3211_value(
                                 t_loc - (3 * period + 3 * gap), UNIT)

            elif t_loc < 5 * period + 4 * gap:
                # Block 4: Pitch inverted
                pitch_cmd = -AMP_3211 * _3211_value(
                                 t_loc - (4 * period + 4 * gap), UNIT)
            # else: neutral, baseline commands only

        # ── Phase 4: heave doublets [57 – 72s] ───────────────────────────────
        elif t < 72.0:
            phase = 'heave'
            t_loc = t - 57.0

            # Keep attitude level so thrust→heave channel is uncontaminated
            roll_cmd  = 0.0
            pitch_cmd = 0.0

            # Alternating thrust steps every 2.5s
            # Gives clean Z-axis data for Tmax/m, Zw, Zww identification
            idx = int(t_loc / 2.5)
            thrust_delta = AMP_HEAVE if (idx % 2 == 0) else -AMP_HEAVE
            thrust_cmd   = float(np.clip(THRUST_TRIM + thrust_delta, 0.0, 1.0))

        # ── Done ──────────────────────────────────────────────────────────────
        else:
            phase     = 'done'
            roll_cmd  = 0.0
            pitch_cmd = 0.0
            thrust_cmd = THRUST_TRIM

        # ── safety clips ─────────────────────────────────────────────────────
        roll_cmd   = float(np.clip(roll_cmd,   -0.5,  0.5))
        pitch_cmd  = float(np.clip(pitch_cmd,  -0.5,  0.5))
        yaw_cmd    = float(np.clip(yaw_cmd,    -0.3,  0.3))
        thrust_cmd = float(np.clip(thrust_cmd,  0.0,  1.0))

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, phase


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helper — approximate cos(theta) tilt factor without carrying qx around
# ─────────────────────────────────────────────────────────────────────────────
def qx_from_theta(theta):
    """sin(theta/2) ≈ theta/2 for small angles — good enough for settle phase."""
    return math.sin(theta / 2.0)
