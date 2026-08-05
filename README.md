# AI Grand Prix — Autonomous Drone Racing

An autonomous racing client for the DCL drone simulator. It connects over
MAVLink, reads telemetry and a 30 Hz FPV camera stream, and flies a quadrotor
through a sequence of gates with no human input — any control input during a
timed run is an instant disqualification.

The competition ran in two qualifiers with very different constraints, and this
repository holds both.

## Results

**Virtual Qualifier 1 — completed the full course.** Gate positions and the
active gate index arrive over telemetry, so navigation is a waypoint chase:
a position→velocity→attitude cascade with a tilt-compensated altitude loop.

The camera runs the whole time regardless. VQ1 is where the perception stack
was built: the detector back-projects a gate position every tick and telemetry
overrides it, so vision flew the full course live on every run without being
able to cost one.

<!-- Replace the line below with the run video.
     Drop the file in docs/ and use:  https://github.com/<user>/<repo>/assets/... -->
📹 **[VQ1 full-course run — video to be attached]**

**Virtual Qualifier 2 — vision-only, best result 3 gates.** VQ2 blocks
`ODOMETRY`, `ATTITUDE` and gate positions. The drone has a camera and an IMU
and nothing else, so it has to detect the gates, estimate its own state, and
steer on what it sees. It flew, it cleared gates, and it did not finish the
course. The perception layer works and is measured; the guidance layer is where
it fell short, and [vq2/README.md](vq2/README.md) says why in detail.

![gate detection at a range of distances](docs/detection_overlays.jpg)

*Live detector output at 2.8 m through 32.6 m. Green is the tracked gate,
yellow the other candidate contours, and the label shows which estimator
produced the range. The cyan ribbon is the simulator's own course marking.*

Measured perception accuracy, from 934 frames diffed against true odometry:

| range | frames | median 3D error | p90 | PnP solve rate |
|---|---|---|---|---|
| < 5 m | 120 | 1.24 m | 1.42 m | 74 % |
| 5–15 m | 353 | 1.20 m | 1.60 m | 95 % |
| 15–25 m | 372 | 1.62 m | 3.96 m | 97 % |
| 25–40 m | 89 | 3.39 m | 11.45 m | 84 % |

Reproduce with `python -m analysis.accuracy`.

## Layout

```
core/       MAVLink transport, camera transport, telemetry decode, clock sync
vq1/        telemetry-guided controller + the first gate detector
vq2/        vision-only controller + the evolved detector with pose solving
analysis/   offline accuracy scoring and detector replay, with sample data
sysid/      system identification: thrust map and attitude dynamics
tests/      63 tests, no simulator required
```

## Running

Requires Python 3.9+ and the simulator listening on `127.0.0.1:14550`.

```bash
pip install -r requirements.txt

python -m vq1.main      # telemetry-guided
python -m vq2.main      # vision-only
python -m pytest        # tests, no sim needed
```

Then open the course in the simulator, wait for the drone to arm, and press
Race. The client holds on the pad until the countdown clears and survives a sim
reset, so one process covers many attempts.

## Architecture

Background threads share one dictionary behind a single lock. Nothing else
talks to the network.

```
sim :14550 ──► MAVLinkRX ──┐
                           ├──► shared_data ──► Controller ──► SET_ATTITUDE_TARGET
sim :5600  ──► VisionRX ───┘        ▲
                                    │
                          IMU estimation thread (VQ2 only)
```

`MAVLinkRX` decodes telemetry, including two non-standard messages the sim
wraps in `ENCAPSULATED_DATA`: race status and the chunked track layout.
`CameraRX` reassembles chunked JPEG frames; each qualifier subclasses it with
its own detector. The controller reads that state at a fixed rate and writes
attitude setpoints back.

Both qualifiers run four background threads — heartbeat, telemetry, clock sync
and the camera — with the control loop on the main thread. VQ2 adds a fifth: a
400 Hz estimation thread that catches every 120 Hz IMU sample, maintaining
attitude by gyro integration and velocity by strapdown dead-reckoning. It is
deliberately decoupled from the 60 Hz control loop, since tying state
estimation to the command rate throws away three quarters of the IMU data.

Both entry points run until interrupted. The sim disarms and re-arms between
attempts and the controllers handle that in place, so one process covers a
whole session; Ctrl-C is the intended exit and shuts every thread down.

## Conventions

Everything is NED: X north, Y east, Z **down**, so a more negative Z is higher
and negative pitch is nose-down. The origin is wherever the drone armed. The
camera is mounted pitched 20° up from the body frame, which is why converting a
pixel bearing into a body-frame direction is never just a projection.

## Known issues

These are real and unresolved. They are listed because a repository that claims
everything works is less useful than one that says where the edges are.

The vision pipeline carries a **systematic −1.12 m vertical bias** — the median
signed error in the down axis, across every frame, on both the PnP and the
bounding-box paths. Because it is common to both estimators it is almost
certainly a frame or offset error rather than a perception limit: a candidate
is the object-frame origin, the fixed gate-elevation offset, or a camera
extrinsic mismatch. It was never tracked down.

VQ2's roll and pitch commands are computed as error terms relative to the
estimated attitude, while the simulator consumes them as absolute setpoints
(type mask 7); yaw does not even follow that convention. The gains were tuned
empirically around it and the aircraft flies, but the inner loop is a P
controller wearing a setpoint's clothing. Resolving it properly would mean
re-tuning from scratch.

`vq1/controller.py`'s along-course velocity setpoint carries a constant
minimum-closing-speed floor that its cross-course counterpart does not. The
asymmetry is deliberate and documented in the code, but it only works because
the VQ1 course runs along a single axis.

## What I would do differently

VQ2 failed at the guidance layer, not the perception layer, and the measurements
say so clearly. The controller re-derives its command from a single gate every
frame, with no representation of the path it intends to fly, so errors at gate
handoffs and on terminal approach have nothing to correct against — the
dominant failure was sliding laterally past a gate it could see perfectly well.

The fix is a planning layer: map the course across attempts, fit a trajectory
through the gates, and track it. Published autonomous racing work is consistent
on this point — single-gate-lookahead reactive control is specifically the thing
that plateaus. The course was deterministic with unlimited practice attempts,
which is exactly the setting where mapping offline between runs pays off, and
that was never built.

Second, there was no fast offline simulator. Every tuning iteration cost a
real-time manual flight, which caps how much of the parameter space anyone can
explore. The `sysid/` work is the start of the fix — an identified dynamics
model is what a surrogate simulator needs — but it was never closed into a loop.
