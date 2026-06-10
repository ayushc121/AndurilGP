# AI Grand Prix — Autonomous Drone Racing Client

A Python autonomy stack for the AI Grand Prix Virtual Qualifier. The client connects to the DCL race simulator over MAVLink/UDP, receives telemetry and a live FPV camera stream, and flies the drone through a sequential gate course with zero human input.

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
- [Vision Pipeline](#vision-pipeline)
- [Coordinate Frames](#coordinate-frames)
- [Known Simulator Behaviours](#known-simulator-behaviours)
- [Troubleshooting](#troubleshooting)

---

## Overview

The simulator exposes two interfaces: a MAVLink endpoint on UDP port 14550 for telemetry and control commands, and a JPEG-over-UDP stream on port 5600 for the forward-facing FPV camera. This client consumes both, fuses them, and outputs `SET_ATTITUDE_TARGET` commands at 50 Hz to fly the drone through each gate in order.

The intended pipeline, as described in the competition technical specification, is:

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Stabilised Controller
```

---

## Architecture

All components run on separate threads and communicate through a single shared dictionary (`shared_data`) protected by a `threading.Lock`.

```
┌─────────────────────────────────────────────────────────────────┐
│                         shared_data (dict)                       │
│  odometry · attitude · race_status · gates · imu · collision    │
│  clock_offset_ns · armed · motor_feedback · vision_gate_estimate │
└─────────────────────────────────────────────────────────────────┘
        ▲                  ▲                        ▲
        │                  │                        │
  MAVLinkRX          VisionRX                  Controller
  (MAVLink thread)   (UDP/camera thread)       (control thread)
        ▲                                          │
        │                                          ▼
  sim (UDP :14550)                    SET_ATTITUDE_TARGET → sim
  sim (UDP :5600)  ←─────────────────────────────────────────────
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

Background thread that receives chunked JPEG packets on UDP port 5600, reassembles frames, decodes them with OpenCV, and runs gate detection. Each decoded frame is passed to `process_frame`, which:

1. Converts the image to HSV colour space.
2. Thresholds for the gate's red colour using two hue ranges (H 0–10 and H 170–180) combined with `cv2.bitwise_or`, because red wraps around the HSV hue circle.
3. Applies morphological close and open operations to clean the mask.
4. Finds the largest contour above a minimum area threshold.
5. Stores bounding box and centroid data in `shared_data['vision_gate_estimate']`.

Partial frames that never complete (from dropped UDP packets) are pruned after `FRAME_BUFFER_DEPTH` newer frame IDs have arrived.

### `controller.py`

The core autonomy logic. Runs at 50 Hz. Implements a three-phase state machine:

**`WAIT_FOR_DATA`** — Retries the arm command every second until the heartbeat confirms armed status and an odometry packet has been received.

**`WAIT_FOR_START`** — Holds completely still (sends no commands) until the race has officially started. Uses a sim-clock anchor to reject stale `race_start_boot_time_ms` values left in the UDP buffer from previous sessions.

**`FLYING`** — Full autonomy. See [Control System Design](#control-system-design) and [Vision Pipeline](#vision-pipeline).

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
Armed and data ready. Moving to WAIT_FOR_START.
[WAIT] Anchor set: sim_ms=1157476
[WAIT] sim_ms=1157476  race_start=-1  fresh=False  go=False
...
Countdown complete! Flying!
Track data received: 6 gates
[GATE] track packet: 6 gates, drone@receipt=(0.00,0.00,0.02)
```

If the simulator is on a different machine, update these two constants at the top of `main.py`:

```python
SIM_SERVER_UDP_IP   = "192.168.x.x"
SIM_SERVER_UDP_PORT = 14550
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

`thrust_trim` is the most important value to tune first. Hold the drone at a fixed altitude command and adjust until it hovers level. The tilt compensation factor (`1 / (1 - 2*(qx² + qy²))`) automatically scales thrust up when the drone is banked to preserve vertical lift.

### Pitch (north velocity control)

```python
K_VX_P      = 1.5     # desired pitch per unit of north velocity error (deg / m·s⁻¹)
K_VX_D      = 0.0
PITCH_LIMIT = 50.0    # maximum pitch angle (deg)
K_P_pitch   = 0.015   # inner attitude P gain
K_D_pitch   = 0.001   # inner attitude D gain
```

### Roll (east velocity control)

```python
K_VY_P     = 30.0    # desired roll per unit of east velocity error (deg / m·s⁻¹)
K_VY_D     = 7.25
ROLL_LIMIT = 50.0
K_P_roll   = 0.015
K_D_roll   = 0.001025
```

### Yaw

```python
K_P_yaw = 0.03   # heading P gain
K_D_yaw = 0.002  # heading D gain
```

Yaw targets a fixed world heading (currently 180° — due south). For gate racing this can be replaced with a bearing-to-gate setpoint.

### Navigation

```python
V_MAX  = 20.0   # horizontal speed cap (m/s)
K_POS  = 1.7    # position-to-velocity P gain (m·s⁻¹ per m)
```

---

## Control System Design

The controller implements a two-stage cascade from position to attitude, outputting `SET_ATTITUDE_TARGET` commands to the flight controller.

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

**Body → world velocity transform.** The odometry message provides velocities in the sensor/body frame. Before use in the north/east controllers, these are rotated into the NED world frame using the full quaternion rotation matrix derived from the odometry quaternion. This ensures velocity errors are computed in consistent world-frame axes regardless of the drone's orientation.

**Tilt-compensated thrust.** Collective thrust is scaled by `1 / cos²(tilt)` — approximated as `1 / (1 - 2*(qx² + qy²))` — so that the vertical component of thrust remains equal to the altitude setpoint demand even when the drone is pitched or rolled significantly.

**Disarm / restart handling.** When the `armed` flag drops to False (sim reset or crash), the controller immediately stops sending commands, clears stale telemetry from `shared_data`, resets the state machine, and waits `POST_DISARM_WAIT` seconds before attempting to re-arm. This prevents "please lower throttle" warnings from the simulator.

---

## Vision Pipeline

When `vision_gate_estimate` is populated by `VisionRX`, the controller back-projects the gate bounding box into a 3D NED world position using the following chain:

**1. True bounding box centre.** The centroid of the detected red contour can be misleading for a hollow rectangular gate frame (the centroid falls at the empty interior). Instead, the geometric centre of the bounding rectangle is used: `true_cx = bx + bw/2`, `true_cy = by + bh/2`.

**2. Range estimation from known gate width.** The gate outer width is 2.7 m. Using the horizontal pinhole formula:
```
z_dist_cam = (2.7 × fx) / bw
```
This gives the perpendicular distance to the gate plane in camera-frame Z. The full 3D hypotenuse distance to the centre of the gate is then:
```
est_distance_3d = z_dist_cam × (|ray| / ray_z)
```

**3. Camera → body rotation.** The camera is tilted 20° upward from the body frame (per the spec). The ray direction is rotated by −20° around the body Y axis:
```
rb_x = rc_z·cos(20°) + rc_y·sin(20°)
rb_y = rc_x
rb_z = −rc_z·sin(20°) + rc_y·cos(20°)
```

**4. Body → NED rotation.** The body-frame ray is rotated to NED using the sequential roll → pitch → yaw rotation matrices derived from the current IMU Euler angles.

**5. Gate NED position.** The 3D range is projected along the world-frame ray and added to the drone's current NED position.

If vision is unavailable (no detection in the current frame), the controller falls back to the MAVLink track data gate positions, which are provided in NED coordinates by the simulator at session start.

---

## Coordinate Frames

| Frame | Convention |
|---|---|
| World (NED) | X = North, Y = East, Z = Down. Origin at drone arm point. |
| Body (FRD) | X = Forward, Y = Right, Z = Down. Co-origin with world at arm point. |
| Camera | Origin at body frame. Tilted 20° upward (nose-up) from body X axis. |

All MAVLink messages use NED. `SET_ATTITUDE_TARGET` quaternions are ZYX Euler convention: positive pitch = nose up, positive roll = right side down, positive yaw = clockwise from above.

---

## Known Simulator Behaviours

- **`race_start_boot_time_ms`** is set to the sim clock value at the moment the user clicks Race in the UI. It does not include any countdown — the value is valid for GO immediately.
- **Stale UDP packets** from previous sessions remain in the OS buffer after a restart. The controller guards against these by recording the sim clock at the moment it enters `WAIT_FOR_START` and rejecting any `race_start_boot_time_ms` that predates this anchor.
- **Ground contact collisions** fire hundreds of times per second while the drone is on the pad before takeoff. The collision logger throttles output to one summary line per second.
- **Z velocity setpoints** in `SET_POSITION_TARGET_LOCAL_NED` are ignored by the simulator's flight controller. Altitude must be controlled via the `thrust` field of `SET_ATTITUDE_TARGET`.

---

## Troubleshooting

**Hangs at "Waiting for heartbeat..."**
The simulator is not running, or Windows Firewall is blocking UDP 14550. Add inbound UDP rules for ports 14550 and 5600.

**"Armed and data ready" never prints**
The drone is arming but no `ODOMETRY` message is arriving. Check that the sim is in a state where telemetry is streaming (not paused or in menu).

**Drone climbs uncontrollably**
`thrust_trim` is too high. Decrease in increments of 0.01 until the drone hovers level at the target altitude.

**Drone sinks after takeoff**
`thrust_trim` is too low. Increase in increments of 0.01.

**Early-start disqualification**
The `WAIT_FOR_START` phase should prevent this. If it still occurs, check the `[WAIT] go=True` log line — it should only appear after `race_start_boot_time_ms` has become positive and greater than the anchor.

**Vision gate estimate is jittery**
Adjust the HSV thresholds in `vision_rx.py` (`LOWER_RED_1/2`, `UPPER_RED_1/2`) to better match the gate colour under the sim's lighting. Increasing `MIN_CONTOUR_AREA` filters out smaller false-positive detections.
