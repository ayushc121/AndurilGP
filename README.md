# AI Grand Prix — Autonomous Drone Racing Client
### Branch: `systemIdentification` — Grey-Box System Identification & Gain Schedule

This branch extends the baseline autonomy stack with a complete grey-box system identification pipeline. The output is `gain_schedule.mat`, a set of 11 linearized state-space models (A, B matrices) covering the drone's full flight envelope. These models are the foundation for the LTV-MPC outer-loop planner under development on the next branch.

The baseline PID flight controller (`controller.py`) remains fully functional as a reference and fallback. No changes to the existing control or vision files were made on this branch.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Reference](#file-reference)
- [System Identification Pipeline](#system-identification-pipeline)
  - [Model Structure](#model-structure)
  - [Flight Test Regimes](#flight-test-regimes)
  - [Identification Procedure](#identification-procedure)
  - [Identified Parameters](#identified-parameters)
  - [gain_schedule.mat Format](#gain_schedulemat-format)
  - [Sign Conventions](#sign-conventions)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running](#running)
- [Configuration & Tuning](#configuration--tuning)
- [Control System Design](#control-system-design)
- [Vision Pipeline](#vision-pipeline)
- [Coordinate Frames](#coordinate-frames)
- [Known Simulator Behaviours](#known-simulator-behaviours)
- [Troubleshooting](#troubleshooting)

---

## Overview

The simulator exposes two interfaces: a MAVLink endpoint on UDP port 14550 for telemetry and control commands, and a JPEG-over-UDP stream on port 5600 for the forward-facing FPV camera. This client consumes both, fuses them, and outputs `SET_ATTITUDE_TARGET` commands at 50 Hz to fly the drone through each gate in order.

**This branch adds** a full system identification pipeline that characterises the drone's aerodynamics across 11 trim conditions spanning hover through 75° nose-down forward flight and up to 50° banked turns. The identified models feed directly into the LTV-MPC controller being developed on the next branch.

---

## Architecture

All real-time components run on separate threads and communicate through a single shared dictionary (`shared_data`) protected by a `threading.Lock`. The sysid pipeline runs entirely offline in MATLAB after flight data has been collected using `sysid_excitation.py`.

```
┌─────────────────────────────────────────────────────────────────┐
│                         shared_data (dict)                      │
│  odometry · attitude · race_status · gates · imu · collision    │
│  clock_offset_ns · armed · motor_feedback · vision_gate_estimate│
└─────────────────────────────────────────────────────────────────┘
        ▲                  ▲                        ▲
        │                  │                        │
  MAVLinkRX          VisionRX                  Controller
  (MAVLink thread)   (UDP/camera thread)       (control thread)
        ▲                                          │
        │                                          ▼
  sim (UDP :14550)                    SET_ATTITUDE_TARGET → sim
  sim (UDP :5600)  ←─────────────────────────────────────────────

Offline sysid pipeline (MATLAB, runs separately):

  sysid_excitation.py  →  CSV + JSON per segment
          │
          ▼
  drone_sysid_main.m
    ├── analyze_segment.m  (equation-error + output-error OE per segment)
    │     └── drone_statespace_reduced.m  (minimal axis-specific ODE)
    └── gain_schedule.mat  (11 × [A(9×9), B(9×4), p(16)] regime structs)

Lateral drag (post-processing):
  fit_yv_m.m  →  Yv_m, Yvv_m  →  updated priors in analyze_segment.m
```

---

## File Reference

### Existing files (unchanged on this branch)

#### `main.py`
Entry point. Instantiates all components via `setup_components`, arms the drone, and runs the control loop by calling `controller.update()` in a tight `while not controller.is_finished()` loop.

#### `setup.py`
Wires all components together. Creates the `threading.Lock`, opens the MAVLink UDP connection, and launches `HeartbeatSender`, `MAVLinkRX`, `TimeSync`, `VisionRX`, and `Controller`.

#### `mavlink_rx.py`
Background thread that receives all inbound MAVLink messages and stores parsed data in `shared_data`.

| Message | Stored key |
|---|---|
| `HEARTBEAT` | `armed` |
| `ODOMETRY` | `odometry` (pos, vel, quaternion, angular rates) |
| `LOCAL_POSITION_NED` | `position`, `velocity` |
| `HIGHRES_IMU` | `imu` |
| `ENCAPSULATED_DATA` (type 1) | `race_status` |
| `ENCAPSULATED_DATA` (type 2) | `gates` |

#### `timesync.py`
Sends `TIMESYNC` requests at 10 Hz to keep the MAVLink session alive and measure the sim-to-client clock offset.

#### `vision_rx.py`
Background thread that receives chunked JPEG packets, reassembles frames, and runs HSV-threshold gate detection. See [Vision Pipeline](#vision-pipeline).

#### `controller.py`
Baseline PID flight controller running at 50 Hz. Implements WAIT_FOR_DATA → WAIT_FOR_START → FLYING state machine. See [Control System Design](#control-system-design).

---

### New files (added on this branch)

#### `sysid_excitation.py`
Flight test excitation script. Replaces the FLYING phase of `controller.py` during data collection. Runs a structured settle → hold → excitation → recovery sequence and logs all telemetry to a CSV + JSON metadata file pair.

**Usage:** edit the TEST CONFIGURATION block (6 lines), then run the existing `main.py` normally. The script self-terminates after `TOTAL_DUR` seconds.

```python
REGIME_ID    = 'P1'      # see regime table below
EXCITE_AXIS  = 'pitch'   # roll | pitch | yaw | heave | drop | lateral
REPETITION   = 1
THETA0_DEG   = -20.0
PHI0_DEG     = 0.0
EXCITE_DUR   = 10.0      # seconds (use 48.0 for lateral trim sweep)
```

**Output per run:**
```
sysid_{REGIME_ID}_{EXCITE_AXIS}_{REPETITION}.csv
sysid_{REGIME_ID}_{EXCITE_AXIS}_{REPETITION}_meta.json
```

**Regime reference table:**

| RegimeID | theta0 | phi0 | Notes |
|----------|--------|------|-------|
| P0 | 0° | 0° | Hover baseline |
| P1 | −20° | 0° | Mild forward flight |
| P2 | −40° | 0° | Moderate forward |
| P3 | −60° | 0° | Aggressive forward |
| P4 | −75° | 0° | Near-max speed (thrust-saturated) |
| B1 | −30° | ±30° | Moderate banked turn |
| B2 | −45° | ±45° | Aggressive bank |
| B3 | −60° | ±50° | Max bank (thrust-saturated) |
| C1 | −40° | ±30° | Combined validation |
| C2 | −60° | ±50° | Extreme combined (thrust-saturated) |
| D1 | 0° | 0° | Free-fall drop (Zww_m identification) |

**Axes:**

| Axis | Excitation | Identifies |
|------|-----------|------------|
| `roll` | 3211 doublet on cmd_roll | Lp, tau_p |
| `pitch` | 3211 doublet on cmd_pitch | Mq, tau_q, Xu_m |
| `yaw` | 3211 doublet on cmd_yaw | Nr, tau_r |
| `heave` | Thrust doublet | Tmax_m, Zw_m |
| `drop` | Near-zero thrust free-fall | Zww_m |
| `lateral` | Bank-angle step sweep ±10°/±18°/±26° | Yv_m, Yvv_m |

For `lateral`, set `EXCITE_DUR = 48.0` (6 steps × 8 s). The drone will drift ±25–40 m laterally per step.

**Segments excluded from aggregation** (edit `EXCLUDE_SEGMENTS` in `drone_sysid_main.m`):
```
sysid_P5_pitch_1.csv   sysid_P5_pitch_2.csv   (unstable at 85° pitch)
sysid_P0_heave_2/3/4.csv                       (nonlinear thrust range)
sysid_P0_lateral_1/2/3.csv                     (lateral uses fit_yv_m.m instead)
```

---

#### `analyze_segment.m`
MATLAB function. Runs equation-error (lesq) followed by output-error (OE via SIDPAC `oe.m`) on a single CSV segment. Returns a `result` struct and saves it as `*_result.mat` alongside the CSV.

```matlab
result = analyze_segment('sysid_P2_pitch_1.csv', ...
                         'sysid_P2_pitch_1_meta.json', ...
                         'plot', true, 'save', true)
```

Each axis identifies only its directly excited parameters; all others are held at priors from previous identification phases. This keeps each OE problem small (2–5 free parameters) and well-conditioned.

| Axis | Free parameters |
|------|----------------|
| roll | Lp, tau_p |
| pitch (hover) | Mq, tau_q |
| pitch (non-hover) | Mq, tau_q, Xu_m |
| yaw | Nr, Gamma3, tau_r |
| heave / drop | Tmax_m, Zw_m, Zww_m |
| lateral | Yv_m, Yvv_m |

---

#### `drone_statespace_reduced.m`
MATLAB function called by SIDPAC `oe.m`. Implements a minimal per-axis ODE that integrates only the directly excited states, using all other measured quantities as exogenous inputs. This keeps each OE problem to 1–3 integrated states.

| Axis code | States | Inputs (u columns) |
|-----------|--------|-------------------|
| 1 (roll) | p, phi | cmd_roll, q_meas, r_meas, theta_meas |
| 2 (pitch) | q, theta, vx_b | cmd_pitch, p_meas, r_meas, phi_meas, vz_b_meas, vy_b_meas |
| 3 (yaw) | r, psi | cmd_yaw, p_meas, q_meas, phi_meas, theta_meas |
| 4 (heave) | vz_b | cmd_thrust |
| 5 (lateral) | vy_b | phi_meas, theta_meas, p_meas, r_meas, vx_b_meas, vz_b_meas |

---

#### `drone_sysid_main.m`
MATLAB script. Batch-processes all `sysid_*.csv` files in a directory, calls `analyze_segment.m` on each, aggregates results per regime using inverse-covariance weighting, and saves `gain_schedule.mat`.

```matlab
% Run from the directory containing the CSV files
drone_sysid_main
```

Key behaviours:
- Loads cached `*_result.mat` files if present (skips re-identification)
- Processes unsaturated regimes first so thrust-saturated regimes (B3, C2, P4) have all unsaturated neighbours available for extrapolation
- Replaces tau_p, tau_q, Lp, Mq, **tau_r** for saturated regimes via linear interpolation from unsaturated neighbours
- Applies hard-coded phi0 overrides for B/C regimes (simulator metadata stored phi0=0 due to config omission during data collection)
- Calls `linearise_drone` to compute A/B Jacobians at each regime's trim state

---

#### `fit_yv_m.m`
Standalone MATLAB function for lateral drag identification. Uses a quasi-static trim-sweep approach rather than OE integration. Pools multiple repetitions and returns Yv_m, Yvv_m.

```matlab
[Yv_m, Yvv_m] = fit_yv_m( ...
    {'sysid_P0_lateral_1.csv', 'sysid_P0_lateral_2.csv'}, ...
    {'sysid_P0_lateral_1_meta.json', 'sysid_P0_lateral_2_meta.json'}, ...
    'skip_dur', 1.0)
```

After identification, update these four lines:

In `analyze_segment.m`:
```matlab
PRIOR_Yv_m   = -0.08899;
PRIOR_Yvv_m  = -0.03909;
```
In `drone_sysid_main.m`:
```matlab
HARD_PRIOR(3) = -0.08899;  % Yv_m
HARD_PRIOR(4) = -0.03909;  % Yvv_m
```
Then delete all `*_result.mat` files and re-run `drone_sysid_main.m`.

---

#### `gain_schedule.mat`
MATLAB v5 binary. The primary output of this branch. Load in Python with:

```python
import scipy.io
gs = scipy.io.loadmat('gain_schedule.mat', simplify_cells=True)['gs']
regimes = gs['regimes']   # list of 11 regime dicts
```

Each regime dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `regime_id` | str | `'P0'` … `'D1'` |
| `theta0` | float | Trim pitch angle (rad) |
| `phi0` | float | Trim roll angle (rad) |
| `T0` | float | Trim thrust (0–1) |
| `v0_trim` | float | Lateral trim velocity (m/s, NED, nonzero for B/C) |
| `saturated` | bool | True for B3, C2, P4 |
| `p` | (16,) | Full parameter vector |
| `param_names` | (16,) | Parameter name strings |
| `A` | (9,9) | Continuous-time linearized A matrix |
| `B` | (9,4) | Continuous-time linearized B matrix |

See [gain_schedule.mat Format](#gain_schedulemat-format) for the full parameter vector layout and A/B matrix key entries.

---

## System Identification Pipeline

### Model Structure

The drone is modelled as a 9-state, 4-input nonlinear system in the NED body frame. The full equations of motion are:

```
udot  = (r·v - q·w) - g·sin(θ) + Xu_m·u + Xuu_m·u·|u|
vdot  = (p·w - r·u) - g·cos(θ)·sin(φ) + Yv_m·v + Yvv_m·v·|v|
wdot  = (q·u - p·v) + g·cos(θ)·cos(φ) - Tmax_m·dT + Zw_m·w + Zww_m·w·|w|
pdot  = Lp·p + Gamma1·q·r + tau_p·cmd_roll
qdot  = Mq·q + Gamma2·p·r + tau_q·cmd_pitch
rdot  = Nr·r + Gamma3·p·q + tau_r·cmd_yaw
phidot   = p + (q·sin(φ) + r·cos(φ))·tan(θ)
thetadot = q·cos(φ) - r·sin(φ)
psidot   = (q·sin(φ) + r·cos(φ)) / cos(θ)
```

**State:** `x = [u_b, v_b, w_b, p, q, r, phi, theta, psi]`  
**Input:** `u = [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust]`

The 16-element parameter vector (0-based Python / 1-based MATLAB):

| Idx (PY) | Idx (MAT) | Name | Identified from |
|----------|-----------|------|----------------|
| 0 | 1 | Xu_m | Pitch tests (per-regime effective value) |
| 1 | 2 | Xuu_m | Multi-regime regression (sub-task 1) |
| 2 | 3 | Yv_m | Lateral trim sweep (fit_yv_m.m) |
| 3 | 4 | Yvv_m | Lateral trim sweep (fit_yv_m.m) |
| 4 | 5 | Tmax_m | P0 heave test |
| 5 | 6 | Zw_m | P0 heave test |
| 6 | 7 | Zww_m | D1 drop test |
| 7 | 8 | Gamma1 | Fixed to 0 (not identifiable) |
| 8 | 9 | tau_p | Roll tests (per-regime) |
| 9 | 10 | Gamma2 | Fixed to 0 (not identifiable) |
| 10 | 11 | tau_q | Pitch tests (per-regime) |
| 11 | 12 | Gamma3 | Fixed to 0 (not identifiable) |
| 12 | 13 | tau_r | Yaw tests (per-regime) |
| 13 | 14 | Lp | Roll tests (per-regime) |
| 14 | 15 | Mq | Pitch tests (per-regime) |
| 15 | 16 | Nr | Yaw tests (per-regime) |

---

### Flight Test Regimes

11 regimes were identified. 3 are thrust-saturated and use extrapolated rotational parameters.

| Regime | theta0 | phi0 | T0 | v0_trim | Status |
|--------|--------|------|----|---------|--------|
| P0 | 0° | 0° | 0.265 | 0 | ✓ |
| P1 | −20° | 0° | 0.265 | 0 | ✓ |
| P2 | −40° | 0° | 0.292 | 0 | ✓ |
| P3 | −60° | 0° | 0.464 | 0 | ✓ |
| P4 | −75° | 0° | 0.932 | 0 | Saturated* |
| B1 | −30° | 30° | 0.319 | +9.35 m/s | ✓ |
| B2 | −45° | 45° | 0.595 | +10.1 m/s | ✓ |
| B3 | −60° | 50° | 0.999 | +8.73 m/s | Saturated* |
| C1 | −40° | 30° | 0.370 | +8.73 m/s | ✓ |
| C2 | −60° | 50° | 0.999 | +8.73 m/s | Saturated* |
| D1 | 0° | 0° | 0.265 | 0 | Hover cal. |

\* tau_p, tau_q, tau_r, Lp, Mq replaced by linear extrapolation from unsaturated neighbours.

`v0_trim` is the steady-state lateral body-frame drift velocity at banked trim. It is nonzero for B/C regimes and appears in the A matrix Coriolis entries A[0,5] and A[2,3].

---

### Identification Procedure

The full procedure to reproduce `gain_schedule.mat` from scratch:

**Step 1 — Heave and drop tests (hover)**
```
P0_heave_1, D1_drop_1    → Tmax_m, Zw_m, Zww_m
```
Run `sysid_excitation.py` with EXCITE_AXIS='heave' (REGIME_ID='P0') and then EXCITE_AXIS='drop' (REGIME_ID='D1'). Use only rep 1 for heave; higher-amplitude heave reps produce Tmax_m inflation and should be excluded.

**Step 2 — Roll, pitch, yaw tests (P0 → P4, B1 → C2)**
Run 2 repetitions per axis per regime. The pipeline processes these automatically. Note: pitch tests at hover (|theta0| < 10°) do not identify Xu_m (vx_b ≈ 0); the prior is used instead.

**Step 3 — Lateral trim sweep (P0)**
```python
EXCITE_AXIS = 'lateral'
REGIME_ID   = 'P0'
EXCITE_DUR  = 48.0   # 6 steps × 8 s
```
Run 2–3 repetitions. Post-process with `fit_yv_m.m`, then update priors and re-run `drone_sysid_main.m`.

**Step 4 — Xuu_m regression (offline)**
From the per-regime effective Xu_m values and their trim speeds, solve:
```
Xu_m_eff(i) = Xu_m_true + Xuu_m * u0(i)
```
via `lsqlin` across P1/P2/P3. Update `PRIOR_Xu_m = Xu_m_true` and `PRIOR_Xuu_m` in `analyze_segment.m` and `HARD_PRIOR(1:2)` in `drone_sysid_main.m`.

**Step 5 — Final run**
Delete all `*_result.mat` files and run `drone_sysid_main.m`. This triggers a full re-identification of all segments with the updated priors.

---

### Identified Parameters

Final parameter values used in the current `gain_schedule.mat`:

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Xu_m_true | −0.0117 | m/s² / (m/s) | Multi-regime regression |
| Xuu_m | −0.0493 | m/s² / (m/s)² | Multi-regime regression |
| Yv_m | −0.0890 | m/s² / (m/s) | Lateral trim sweep |
| Yvv_m | −0.0391 | m/s² / (m/s)² | Lateral trim sweep |
| Tmax_m | 36.241 | m/s² | P0 heave |
| Zw_m | −0.1816 | m/s² / (m/s) | P0 heave |
| Zww_m | −0.0432 | m/s² / (m/s)² | D1 drop |
| tau_p | 37–53 | rad/s / cmd | Per-regime roll test |
| tau_q | 47–64 | rad/s / cmd | Per-regime pitch test |
| tau_r | 27–37 | rad/s / cmd | Per-regime yaw test |
| Lp | −13 to −20 | rad/s² / (rad/s) | Per-regime roll test |
| Mq | −22 to −26 | rad/s² / (rad/s) | Per-regime pitch test |
| Nr | −12 to −16 | rad/s² / (rad/s) | Per-regime yaw test |

**Physical constants (fixed, never estimated):**  
`g = 9.81 m/s²`, `hover T0 = 0.265`, thrust mapping: `aT = 139.7 · cmd_thrust²`

---

### gain_schedule.mat Format

The A/B matrices are (9×9) and (9×4) continuous-time Jacobians. Key entries for verification when loading in Python (0-based):

```
A[0,0] = Xu_m_eff + Xuu_m*|u0|     forward drag (Xu_m_eff is per-regime)
A[0,5] = v0_trim                    r→u_b Coriolis (nonzero for B/C)
A[1,1] = Yv_m + 2*Yvv_m*|v0_trim|  lateral drag at trim
A[1,5] = +u0                        r→v_b Coriolis (sign-corrected)
A[1,6] = -g*cos(θ0)*cos(φ0)        phi gravity coupling (v_b row)
A[2,3] = -v0_trim                   p→w_b Coriolis (nonzero for B/C)
A[3,3] = Lp                         roll damping
A[4,4] = Mq                         pitch damping
A[5,5] = Nr                         yaw damping
A[7,4] = cos(φ0)                    theta kinematic
A[8,8] = 0.0                        psi is a free integrator (by design)

B[2,3] = -Tmax_m                    thrust → w_b (negative = upward in NED)
B[3,0] = tau_p                      cmd_roll → p
B[4,1] = tau_q                      cmd_pitch → q
B[5,2] = tau_r                      cmd_yaw → r
```

**Trim forward velocity** is computed as:  
`u0 = g * sin(theta0) / Xu_m_eff` (Xu_m_eff is the per-regime effective value from `p[0]`)

**A matrix A[0,0] formula note:** `p[0]` (Xu_m) is the *effective* drag at trim speed (absorbs one factor of Xuu_m·|u0|). The Jacobian entry therefore uses `p[0] + Xuu_m*|u0|`, not `p[0] + 2*Xuu_m*|u0|`.

---

### Sign Conventions

Several simulator-specific sign conventions differ from standard NED textbook formulations. All are already absorbed into the stored A/B matrices. Do not correct for these when feeding raw odometry into the MPC.

| Quantity | Convention | Impact |
|----------|-----------|--------|
| `v_b` (vy_b) | **Opposite** to NED body-y. Positive = leftward in simulator. | A[1,x] entries have flipped signs vs standard NED. |
| `q` (pitchspeed) | **Opposite** to thetadot. Positive q → decreasing theta. | B[4,1] = tau_q is positive (corrected). Pitch OE negates q before fitting. |
| Gravity in v_b eq. | `−g·cos(θ)·sin(φ)` (minus sign) | A[1,6] = −g·cos(θ0)·cos(φ0) for phi coupling. |
| Thrust mapping | Quadratic: `aT = 139.7 · cmd²` | Linear approximation underestimates by ~2×. |
| Tilt compensation | `tilt = 1 − 2(qx² + qy²)` | Exact simulator formula; divides hover thrust. |

---

## Prerequisites

### Real-time flight (existing)
- Python 3.8+
- `pymavlink`, `opencv-python`, `numpy`
- DCL Simulator on Windows 11 with UDP ports 14550 and 5600 open

### System identification (this branch)
- MATLAB R2020b or later with Optimization Toolbox (`lsqlin`)
- [SIDPAC toolbox](https://software.nasa.gov/software/LAR-16100-1) — provides `oe.m` and `deriv.m`
- `scipy` (Python) — for `scipy.io.loadmat` when loading results in Python

---

## Installation

```bash
pip install pymavlink opencv-python numpy scipy
```

SIDPAC: place the SIDPAC `m-files` directory on the MATLAB path or in the same folder as the sysid `.m` files.

---

## Running

### Flight (baseline PID controller)

Start the DCL Simulator first, then:

```bash
python main.py
```

### Data collection for sysid

Edit the TEST CONFIGURATION block in `sysid_excitation.py`, then run:

```bash
python main.py
```

The script automatically handles the settle → hold → excitation → recovery sequence and saves CSV + JSON metadata. One run per axis per regime. Recommended collection order:

1. P0 heave (rep 1 only — exclude higher amplitude reps)
2. D1 drop (rep 1)
3. P0 roll/pitch/yaw (2 reps each)
4. P1 → P4 roll/pitch/yaw (2 reps each)
5. B1, B2, C1 roll/pitch/yaw (2 reps each)
6. B3, C2 roll/yaw only (pitch is thrust-saturated)
7. P0 lateral (2–3 reps, EXCITE_DUR=48.0)

### Running the MATLAB sysid pipeline

```matlab
% From the directory containing all sysid_*.csv files:
drone_sysid_main
```

This produces `gain_schedule.mat` and `param_vs_trim.png`. To force re-identification of all segments, delete `*_result.mat` files first.

To re-run a single segment:
```matlab
result = analyze_segment('sysid_P2_pitch_1.csv', ...
                         'sysid_P2_pitch_1_meta.json', ...
                         'plot', true)
```

To run lateral drag fitting after collecting lateral CSV files:
```matlab
[Yv_m, Yvv_m] = fit_yv_m( ...
    {'sysid_P0_lateral_1.csv', 'sysid_P0_lateral_2.csv'}, ...
    {'sysid_P0_lateral_1_meta.json', 'sysid_P0_lateral_2_meta.json'}, ...
    'skip_dur', 1.0, 'plot', true)
```

---

## Configuration & Tuning

All primary tuning parameters are constants at the top of `controller.py`.

### Thrust

```python
thrust_trim = 0.265   # hover thrust — increase if sinking, decrease if climbing
K_P_thrust  = 0.0925  # altitude P gain
K_D_thrust  = 0.05    # altitude D gain (damps vertical oscillation)
```

### Pitch (north velocity control)

```python
K_VX_P      = 1.5
PITCH_LIMIT = 50.0    # maximum pitch angle (deg)
K_P_pitch   = 0.015
K_D_pitch   = 0.001
```

### Roll (east velocity control)

```python
K_VY_P     = 30.0
K_VY_D     = 7.25
ROLL_LIMIT = 50.0
K_P_roll   = 0.015
K_D_roll   = 0.001025
```

### Yaw

```python
K_P_yaw = 0.03
K_D_yaw = 0.002
```

---

## Control System Design

The current baseline controller implements a two-stage cascade from position to attitude outputting `SET_ATTITUDE_TARGET` at 50 Hz.

```
NED position error
        │
        ▼  (K_POS · error, capped at V_MAX)
Desired world velocity (north / east)
        │
        ▼  (K_VX_P/D, K_VY_P/D)
Desired attitude (pitch / roll angles)
        │
        ▼  (K_P_pitch/D, K_P_roll/D — inner attitude loop)
Rate commands (rad/s) + thrust  →  SET_ATTITUDE_TARGET
```

**Tilt-compensated thrust:** `T = T_hover / (1 − 2(qx² + qy²))` preserves vertical lift at any bank/pitch angle.

**Next branch:** The PID outer loop will be replaced by an LTV-MPC planner using the `gain_schedule.mat` produced by this branch. The inner attitude rate loop and MAVLink interface are unchanged.

---

## Vision Pipeline

When `vision_gate_estimate` is populated by `VisionRX`, the controller back-projects the gate bounding box into a 3D NED position using pinhole range estimation (`z_dist = gate_width_m * fx / bx_width`), a camera-body rotation (camera tilted 20° nose-up), and a body-NED rotation from current IMU Euler angles.

See `vision_rx.py` for HSV threshold constants and contour filtering parameters.

---

## Coordinate Frames

| Frame | Convention |
|-------|-----------|
| World (NED) | X = North, Y = East, Z = Down |
| Body (FRD) | X = Forward, Y = Right\*, Z = Down |
| Camera | Tilted 20° upward from body X |

\* In this simulator, body-Y velocity (`vy_b`) is measured with **opposite** sign — positive = leftward. The A/B matrices in `gain_schedule.mat` are built with this convention. Do not negate `vy_b` before using it.

---

## Known Simulator Behaviours

- **Stale UDP packets** from previous sessions remain in the OS buffer. The controller guards with a sim-clock anchor in `WAIT_FOR_START`.
- **`SET_ATTITUDE_TARGET` is always a rate interface.** Regardless of typemask, the simulator decodes quaternion values back to Euler angles and applies them directly as body rates in rad/s. The inner attitude P loop (`angle_error × K_ATT`) is therefore mandatory.
- **Pitchspeed sign is inverted.** Positive `pitchspeed` from odometry corresponds to decreasing theta (nose-down). The sysid pipeline negates `q` internally; the A/B matrices account for this. Do not negate `q` before passing to the MPC.
- **Thrust mapping is quadratic:** `aT = 139.7 · cmd_thrust²`. A linear model underestimates thrust authority by ~2×.
- **Ground contact collisions** fire hundreds of times per second while on the pad. The collision logger throttles to one line per second.

---

## Troubleshooting

**`oe.m` not found in MATLAB**  
Add the SIDPAC `m-files` directory to the MATLAB path: `addpath('/path/to/sidpac/m-files')`.

**`drone_sysid_main` produces NaN in gain schedule**  
At least one parameter was never well-identified and the P0 aggregate also has NaN for that parameter. The `HARD_PRIOR` vector in `drone_sysid_main.m` fills these. Check that the final `HARD_PRIOR` values match those listed in [Identified Parameters](#identified-parameters).

**Tmax_m inflated (>40 m/s²) in gain schedule**  
One or more high-amplitude heave test reps are included. Add `sysid_P0_heave_2.csv` (and any higher reps) to `EXCLUDE_SEGMENTS` in `drone_sysid_main.m`. Only `P0_heave_1` should contribute.

**`fit_yv_m.m` returns Yv_m = −0.0001 (upper bound hit)**  
The `skip_dur` argument is too large — the full trajectory including the vy_b ≈ 0 zero-crossing is being excluded. Try `'skip_dur', 1.0`. If the warning persists, reduce to `0.5`.

**Drone climbs uncontrollably / sinks after takeoff**  
Adjust `thrust_trim` in `controller.py`. See [Thrust](#thrust).

**Hangs at "Waiting for heartbeat..."**  
Simulator not running, or Windows Firewall blocking UDP 14550/5600.
