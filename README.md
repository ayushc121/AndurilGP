# AI Grand Prix — Autonomous Drone Racing Client

A Python autonomy stack for the AI Grand Prix Virtual Qualifier. The client connects to the DCL race simulator over MAVLink/UDP, receives telemetry and a live FPV camera stream, and flies the drone through a sequential gate course with zero human input.

We developed a robust state estimation and gate detection pipeline that enables real-time autonomous course navigation by fusing live computer vision feeds with accelerometer and gyroscope sensor data. The system tracks gate geometry continuously through the approach, uses vision-derived position derivatives as damping signals, and blends optical-flow velocity estimates with IMU dead-reckoning to maintain stable closed-loop control throughout the course.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Reference](#file-reference)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running](#running)
- [Configuration & Tuning](#configuration--tuning)
- [Control System Design](#control-system-design)
- [State Estimation Pipeline](#state-estimation-pipeline)
- [Vision Pipeline](#vision-pipeline)
- [Coordinate Frames](#coordinate-frames)
- [Known Simulator Behaviours](#known-simulator-behaviours)
- [Troubleshooting](#troubleshooting)

---

## Overview

The simulator exposes two interfaces: a MAVLink endpoint on UDP port 14550 for telemetry and control commands, and a JPEG-over-UDP stream on port 5600 for the forward-facing FPV camera. This client consumes both, fuses them, and outputs `SET_ATTITUDE_TARGET` commands at 60 Hz to fly the drone through each gate in order.

The pipeline:

```
Vision (30 Hz) ──────────────────────────────────────────┐
                                                          ▼
IMU (120 Hz) → SMA filter → Gyro AHRS → Strapdown DR → Sensor Fusion → Controller (60 Hz) → Sim
```

---

## Architecture

All components run on separate threads and communicate through a single shared dictionary (`shared_data`) protected by a `threading.Lock`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           shared_data (dict)                            │
│  imu · armed · race_status · gates · vision_gate_estimate               │
│  vision_velocity · last_collision · clock_offset_ns · motor_feedback    │
└─────────────────────────────────────────────────────────────────────────┘
        ▲                    ▲                           ▲
        │                    │                           │
  MAVLinkRX            VisionRX                    Controller
  (MAVLink thread)     (UDP/camera thread)         (control + estimation)
        ▲                                               │
        │                                               ▼
  sim (UDP :14550)                       SET_ATTITUDE_TARGET → sim
  sim (UDP :5600)  ←────────────────────────────────────────────────────
```

The `TimeSync` and `HeartbeatSender` threads also run in the background, keeping the MAVLink session alive and measuring the sim-to-client clock offset.

---

## File Reference

### `main.py`

Entry point. Instantiates all components via `setup_components`, arms the drone, and runs the control loop by calling `controller.update()` in a tight `while not controller.is_finished()` loop. Joins all background threads on exit.

### `setup.py`

Wires all components together. Responsibilities:

- Creates the `threading.Lock` and inserts it into `shared_data` before any other component is constructed.
- Opens the MAVLink UDP connection and blocks on `wait_heartbeat()`.
- Launches `HeartbeatSender` (5 Hz) — the sim requires client heartbeats at ≥ 2 Hz or it may reject control commands.
- Launches `MAVLinkRX`, `TimeSync`, `VisionRX`, and `Controller`.

### `mavlink_rx.py`

Background thread that receives all inbound MAVLink messages and stores parsed data in `shared_data`. Handles:

| Message | Stored key |
|---|---|
| `HEARTBEAT` | `armed` |
| `ODOMETRY` | `odometry` (pos, vel, quaternion, angular rates) |
| `ATTITUDE` | `attitude` |
| `LOCAL_POSITION_NED` | `position`, `velocity` |
| `HIGHRES_IMU` | `imu` |
| `TIMESYNC` | `clock_offset_ns` (computed offset, responds to server requests) |
| `ENCAPSULATED_DATA` (type 1) | `race_status` (sim time, race start, finish, active gate index) |
| `ENCAPSULATED_DATA` (type 2) | `gates` (full gate list in NED, sorted by gate ID) |
| `COLLISION` | `last_collision` (throttled to one log line per second) |
| `ACTUATOR_OUTPUT_STATUS` | `motor_feedback` |

### `timesync.py`

Sends `TIMESYNC` requests at 10 Hz. The response is handled in `MAVLinkRX.on_timesync`, which computes `clock_offset_ns = tc1 - (ts1 + now_ns) / 2` and stores it in `shared_data`. This offset is available for correlating vision frame timestamps (`sim_time_ns`) with MAVLink telemetry timestamps.

### `vision_rx.py`

Background thread that receives chunked JPEG packets on UDP port 5600, reassembles frames, decodes them with OpenCV, and runs gate detection at 30 Hz. Each decoded frame is passed to `process_frame`, which:

1. Converts the image to HSV colour space.
2. Thresholds for the gate's red colour using two hue ranges (H 0–10 and H 170–180) combined with `cv2.bitwise_or`, because red wraps around the HSV hue circle.
3. Applies morphological close and open operations to clean the mask.
4. Finds the largest contour above a minimum area threshold.
5. Estimates gate range using the known outer gate width and the pinhole camera model.
6. Computes PnP pose estimation (`pnp_ok`, `pnp_rvec`) for gate-normal yaw alignment.
7. Derives an optical-flow body-frame velocity estimate (`vision_velocity`) from consecutive frame positions.
8. Stores bounding box, centroid, body-frame gate position, PnP result, and velocity in `shared_data`.

Partial frames that never complete (from dropped UDP packets) are pruned after `FRAME_BUFFER_DEPTH` newer frame IDs have arrived.

### `controller.py`

The core autonomy logic. Runs at 60 Hz on the main control thread, with a dedicated 400 Hz estimation thread processing every IMU sample. Implements a three-phase state machine:

**`WAIT_FOR_DATA`** — Retries the arm command every second until the heartbeat confirms armed status and an IMU packet has been received.

**`WAIT_FOR_START`** — Sends zero commands until the race has officially started. Uses a sim-clock anchor to reject stale `race_start_boot_time_ms` values from previous sessions.

**`FLYING`** — Full autonomy. See [Control System Design](#control-system-design), [State Estimation Pipeline](#state-estimation-pipeline), and [Vision Pipeline](#vision-pipeline).

---

## Prerequisites

- Python 3.8+
- The DCL Simulator running on Windows 11
- UDP ports 14550 and 5600 reachable from the Python host

---

## Installation

```bash
pip install pymavlink opencv-python numpy
```

---

## Running

Start the DCL Simulator first and let it reach its idle/ready state. Then:

```bash
python main.py
```

A healthy startup sequence:

```
Waiting for heartbeat...
Connected to system: 1
Starting heartbeat sender...
Setting up MAVLink rx...
Setting up timesync loop...
Listening for camera frames...
Arming drone...
Sending arm command...
IMU estimation thread started.
Armed and IMU ready. Moving to WAIT_FOR_START.
[WAIT] Anchor set: sim_ms=1157476
[WAIT] sim_ms=1157476  race_start=-1  fresh=False  go=False
...
Countdown complete! Flying!
[NAV] vel_fused=(  +3.1fwd  -0.0right  -0.0down)m/s  att=( -0.0r  -3.0p  -0.0y)°  gate=(+22.5fwd ...
```

If the simulator is on a different machine, update these two constants at the top of `main.py`:

```python
SIM_SERVER_UDP_IP   = "192.168.x.x"
SIM_SERVER_UDP_PORT = 14550
```

---

## Configuration & Tuning

All primary tuning parameters are constants at the top of `controller.py`.

### Accelerometer Filter

```python
ACC_SMOOTH_N = 5   # SMA window size over raw body-frame IMU readings
```

A 5-sample sliding window at 120 Hz corresponds to a ~42 ms smoothing window and ~17 ms group delay. Reduce for faster response; increase to suppress more noise. Set to 1 to disable.

### Vision–IMU Velocity Fusion

```python
VIS_VEL_EMA_ALPHA = 0.35   # EMA weight on each new raw optical-flow sample
OF_ALPHA          = 0.6    # IMU weight in final fused velocity (0 = all vision)
```

`VIS_VEL_EMA_ALPHA` pre-smooths the raw vision velocity before it is blended with the IMU estimate. `OF_ALPHA` controls how much the fused result leans on the IMU vs smoothed vision. Only lateral (vY) and body-down (vZ) channels are corrected; forward velocity (vX) stays IMU-only.

### D Term Clamps

```python
BEARING_RATE_CLAMP_DEG_S = 60.0   # max bearing-angle rate into D_lateral
ELEV_RATE_CLAMP_M_S      = 5.0    # max gate elevation rate into D_vertical
```

These bound the vision finite-difference D signals before they feed commands. Bad gate detections can produce apparent rates of 100+ deg/s; the clamps prevent those from reaching the actuators.

### Thrust

```python
HOVER_THRUST = 0.264   # hover trim — the tilt compensation divides by this
K_P_thrust   = 0.014   # gate elevation P gain
K_D_thrust   = 0.0175  # vertical velocity D gain
```

`HOVER_THRUST` is the physical trim point at level hover. The tilt compensation factor (`cos(roll) × cos(pitch)`) scales thrust automatically when banked. `K_P_thrust` drives toward the gate's world-frame elevation; `K_D_thrust` damps vertical velocity.

### Attitude and Steering

```python
DESIRED_PITCH_DEG = -3.0    # fixed forward pitch setpoint
K_BEARING         = 4.5     # deg bank per deg of bearing error
K_LAT_D           = 9.0     # deg bank per m/s of lateral D term
MAX_BANK_DEG      = 25.0    # roll authority hard limit
PERP_BLEND_DIST   = 6.0     # m: gate distance at which blend ramp starts
TILT_EMA_ALPHA    = 0.25    # EMA weight on PnP gate-normal tilt measurement
```

---

## Control System Design

The controller operates a rate-command interface: `SET_ATTITUDE_TARGET` with typemask `0b00000111`, interpreted by the sim as direct body rates (rad/s). An explicit outer attitude P loop converts angle errors to rate commands:

```
rate_cmd = angle_error × K_ATT   (clipped to max rate)
```

### Steering architecture

```
Gate bearing angle  ──(K_BEARING × blend)──▶ desired roll
Lateral D term      ──(K_LAT_D   × blend)──▶ desired roll  (subtracted)
                                                    │
                                                    ▼
                                          (desired_roll − roll) × KR  →  roll_cmd

Gate bearing (capped)  ──────────────────▶ yaw_err (far field)
Gate face normal (PnP) ──(1 − blend)────▶ yaw_err (near field, blended in)
                                                    │
                                                    ▼
                                         yaw_err × KY  →  yaw_cmd

HOVER_THRUST − elev_err × K_P_thrust + D_vertical × K_D_thrust
                                        ──────────────────────▶ thrust_cmd
                                         (divided by tiltFactor)
```

**Roll** is the primary steering actuator. The drone steers via coordinated bank, with bearing angle to the gate as the P term. The D term (`D_lateral`) damps overshoot. Both are scaled by `blend`, which ramps from 1.0 (far from gate, full authority) to 0.0 (at the gate plane, roll zeroed).

**Yaw** is passive: it tracks the bearing angle in the far field and blends to the PnP-derived gate face normal close in, ensuring the drone crosses the gate perpendicular to its plane.

**Thrust** is driven by `_last_elev_err`, the world-frame elevation of the gate center relative to the drone, with `D_vertical` providing damping. The elevation error is frozen at its last valid value whenever the gate is out of frame, giving stable hold through brief vision blackouts.

**Pitch** holds a fixed forward angle (`DESIRED_PITCH_DEG`), maintaining forward progress throughout the course.

---

## State Estimation Pipeline

State estimation runs on a dedicated 400 Hz background thread, independently of the 60 Hz control loop. All state writes are protected by `_state_lock`; the control loop takes a single atomic snapshot at the start of each tick and then releases the lock before any computation.

### Attitude — GyroAHRS

Attitude is tracked by a pure quaternion gyro integrator (`GyroAHRS`). At each 120 Hz IMU sample:

```
q_new = q + 0.5 × dt × Ω(gx, gy, gz) × q   (quaternion kinematics)
```

The quaternion is renormalised after each step. No accelerometer correction is applied to attitude — the accelerometer is dominated by thrust-induced specific force during manoeuvres, making complementary correction unreliable in a racing context. The gyro is seeded at arm time from the known launch-ramp geometry (`LAUNCH_PITCH_DEG = -17.8°`).

Gyro signs are inverted from NED convention in this simulator and are corrected on ingestion:

```python
gx = -imu['xgyro']
gy = -imu['ygyro']
gz = -imu['zgyro']
```

### Accelerometer — SMA Filter

Raw body-frame accelerometer readings pass through a sliding-window simple moving average (`ACC_SMOOTH_N = 5` samples) before use. This was chosen over a thrust-proportional physical cap because the cap was found to have a floor discontinuity at low thrust that produced spikes rather than suppressing them. The SMA operates in body frame, before the rotation to NED, preserving the natural frame of the measurement.

```python
self._acc_buf.append((ax_raw, ay_raw, az_raw))
ax = mean of buffer x-samples   # smoothed body-frame reading
```

### Strapdown Dead-Reckoning

After attitude update and accelerometer smoothing, specific force is rotated from body to NED using the current quaternion, gravity is subtracted, and the result is integrated into velocity and position:

```
a_NED = R(q) · f_body
a_NED[2] += G          # subtract NED-down gravity
vel_NED  += a_NED · dt
pos_NED  += vel_NED · dt
vel_body  = R(q)ᵀ · vel_NED
```

Dead-reckoning accumulates gyro-bias-induced drift over time and is not used as an absolute reference. It provides short-term velocity estimates between vision frames, which are corrected at each camera update.

### Vision–IMU Velocity Fusion

At each new 30 Hz camera frame, optical-flow body-frame velocity estimates are pre-smoothed with a per-axis EMA and then blended into the IMU velocity:

```
vy_ema  = α · vy_raw + (1−α) · vy_ema          # α = VIS_VEL_EMA_ALPHA = 0.35
vY_fused = β · vY_imu + (1−β) · vy_ema          # β = OF_ALPHA = 0.6
```

This corrects lateral and vertical IMU drift at 30 Hz while preserving IMU dynamics between frames. Each frame is fused exactly once, gated on `frame_id`. Forward velocity (vX) remains IMU-only; it is less prone to lateral drift and is damped by pitch dynamics.

### D Term Computation

Vision-derived D terms are computed on each new camera frame and held between frames using IMU velocity increments:

```
On new vision frame:
    D_lateral  = −clamp(bearing_rate, ±60°/s) × bx   [capped at ±60 °/s]
    D_vertical = −clamp(elev_rate, ±5 m/s)
    snapshot: vY_at_vision, vD_at_vision = vY, vD

Between frames (IMU fallback, ~33 ms window):
    D_lateral  = vY − vY_at_vision
    D_vertical = vD − vD_at_vision

Vision lost:
    D_lateral = D_vertical = 0.0   (reference reset)
```

The clamps on vision finite-difference rates prevent large position jumps between detections (e.g., gate switching) from generating actuator-saturating D spikes.

---

## Vision Pipeline

### Gate Detection

`VisionRX` detects the gate's red frame in each camera frame using HSV thresholding across both red hue ranges (0–10 and 170–180), morphological cleanup, and largest-contour selection.

### Body-Frame Gate Position

The gate's body-frame position `(bx, by, bz)` is estimated from the detected bounding box using the pinhole camera model and known gate outer width (2.7 m):

```
z_dist = (gate_width × fx) / bounding_box_width
ray_body = rotate_cam_to_body(pixel_ray)   # −20° tilt correction
(bx, by, bz) = z_dist × ray_body / |ray_body|
```

`bx > 0` means the gate is ahead in body frame; `vision_valid` is set only when `bx > 0.1 m`.

### Elevation Error

The world-frame elevation of the gate center relative to the drone is extracted by rotating the body-frame gate vector into NED using the current attitude quaternion:

```
gate_pD = R(q) · [bx, by, bz]  →  NED-down component
```

This is the elevation error fed to the thrust P term. It is updated only when `bx > MIN_BX_FOR_ELEV = 3.0 m` (closer in, the geometry becomes unreliable) and is frozen at the last valid value whenever the gate leaves frame.

### Bearing and Yaw Guidance

Bearing to the gate in the body frame:
```
bearing_body = atan2(by, bx)     capped at ±25°
blend        = clip(bx / PERP_BLEND_DIST, 0, 1)
```

Yaw blends from bearing-tracking (far) to gate-face-normal alignment (close):
```
yaw_err = blend × bearing_capped_12 + (1−blend) × gate_tilt_ema
```

Gate face normal is derived from PnP pose estimation (`cv2.Rodrigues`), with an EMA (`TILT_EMA_ALPHA = 0.25`) reducing the raw ±10° PnP noise to ~±4°.

---

## Coordinate Frames

| Frame | Convention |
|---|---|
| World (NED) | X = North, Y = East, Z = Down. Origin at drone arm point. |
| Body (FRD) | X = Forward, Y = Right, Z = Down. Co-origin with world at arm point. |
| Camera | Origin at body frame. Tilted 20° upward (nose-up) from body X axis. |

All MAVLink messages use NED. `SET_ATTITUDE_TARGET` sends a quaternion with typemask `0b00000111`; the simulator decodes this back to Euler angles and applies them directly as body rates in rad/s. There is no native attitude hold mode in the simulator — the outer attitude P loop in the controller provides this explicitly.

**Critical sign conventions** (confirmed empirically against simulator ground truth):

- Gyro signs are inverted vs NED: `gx = -imu['xgyro']` etc.
- Simulator pitch speed reports with opposite sign to `thetadot`: `q_e = -q_raw` in sysid.
- Tilt compensation: `tilt_factor = 1 - 2*(qx² + qy²)` (exact simulator formula).
- Thrust mapping is quadratic: `a_T = 139.7 × cmd_thrust²`.

---

## Known Simulator Behaviours

- **Rate interface only.** `SET_ATTITUDE_TARGET` quaternion values are decoded back to Euler angles and applied directly as body rates in rad/s. The sim has no native attitude hold mode.
- **Quadratic thrust mapping.** `a_T = 139.7 × cmd_thrust²`. Using a linear model underestimates thrust authority by ~2×.
- **Stale UDP packets** from previous sessions remain in the OS buffer after a restart. The controller guards against these by recording the sim clock at the moment it enters `WAIT_FOR_START` and rejecting any `race_start_boot_time_ms` that predates this anchor.
- **Ground contact collisions** fire hundreds of times per second while the drone is on the pad before takeoff. The collision logger throttles output to one summary line per second.
- **Hover trim** is exactly `cmd_thrust = 0.265` at level attitude.

---

## Troubleshooting

**Hangs at "Waiting for heartbeat..."**
The simulator is not running, or Windows Firewall is blocking UDP 14550. Add inbound UDP rules for ports 14550 and 5600.

**"Armed and IMU ready" never prints**
The drone is arming but no `HIGHRES_IMU` message is arriving. Check that the sim is in a state where telemetry is streaming (not paused or in menu).

**Drone climbs or sinks steadily**
`HOVER_THRUST` is off. The correct value is `0.264`–`0.265`. Adjust in increments of `0.001`. Note that altitude is controlled via gate elevation error (`_last_elev_err`) not IMU-integrated altitude — IMU dead-reckoning drifts over multi-second timescales and cannot be used as an absolute altitude reference.

**Drone rolls or yaws unexpectedly at launch**
The `LAUNCH_PITCH_DEG` seed for `GyroAHRS` may be incorrect for the current launch-ramp geometry. Adjust to match the actual ramp angle.

**D terms saturating (T=0.000 or extreme roll commands)**
Check the vision pipeline for detection jumps between gates. `BEARING_RATE_CLAMP_DEG_S` and `ELEV_RATE_CLAMP_M_S` should prevent these from reaching the actuators. If still occurring, reduce `K_LAT_D` or `K_D_thrust`.

**Gate bearing oscillates close in**
The blend ramp (`PERP_BLEND_DIST`) may be too wide, keeping roll authority active too close to the gate. Reduce from 6.0 m. Alternatively, the PnP gate-normal EMA (`TILT_EMA_ALPHA`) may need reducing if tilt estimates are noisy at close range.

**Early-start disqualification**
The `WAIT_FOR_START` phase should prevent this. If it still occurs, check the `[WAIT] go=True` log line — it should only appear after `race_start_boot_time_ms` has become positive and greater than the anchor.
