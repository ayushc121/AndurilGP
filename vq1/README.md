# VQ1 — telemetry-guided racing, and where the CV was built

Completes the full course. Run with `python -m vq1.main`.

## The problem

VQ1 publishes every gate's 3D pose up front and continuously reports which gate
is next. That removes perception from the problem entirely, and what remains is
a control problem: given where we are and where the next gate is, produce an
attitude and a thrust.

## Design

A quadrotor cannot be commanded to move sideways — it can only tilt, and let
the horizontal component of thrust do the work. So the guidance chain has to
convert a position error into a tilt, and it does that in two stages rather
than one:

```
gate position ──► velocity setpoint ──► attitude setpoint ──► sim
                       (P)                     (PD)
```

Splitting it matters. A direct position-to-attitude law has to be tuned for a
specific distance-to-gate, because the same error means something different at
40 m and at 4 m. Going through a velocity setpoint decouples the two: the outer
loop decides how fast to close, the inner loop decides how hard to lean to
achieve that speed, and the inner loop's tuning holds regardless of range.

Altitude runs as a separate PD on thrust. Heading is held at a constant
south-facing 180°, because the VQ1 course is a straight corridor and yawing
would only add a coupling term for no benefit.

## The camera runs the whole time

VQ1 did not need vision, which is exactly why it is the right place to develop
it. Every tick the controller runs the detector, back-projects its bounding box
into a world-frame gate position, and *then* overwrites that result with the
telemetry gate whenever telemetry is available.

That ordering is the point. The detector flew the full course live on every
single run, producing a real target continuously, while the telemetry target
beside it did the actual flying. A bad detection could not lose a run, and the
gap between the two numbers was free ground truth on every frame.

The back-projection is bbox width → range through the pinhole model, then a
pixel ray rotated camera → body (fixed 20° tilt) → world (drone attitude),
scaled and added to the drone's position. It needs the drone's attitude, which
is why VQ2 could not reuse it: with attitude blocked, VQ2 had to stop at the
body frame and solve pose properly instead. See
[`vision.py`](vision.py) for the v1 detector and
[`../vq2/gate_detector.py`](../vq2/gate_detector.py) for what it became.

The `[FLY/VIS]` and `[FLY/ODO]` tags in the debug output say which source is
steering on any given tick.

## Two details worth calling out

**Tilt compensation.** Only the vertical component of the thrust vector fights
gravity, and that component is `cos(roll)·cos(pitch)` of the total. Without
dividing it back out, the drone loses altitude every time it turns, and the
altitude loop chases a disturbance the controller itself created. The factor
falls out of the attitude quaternion directly as `1 − 2(qx² + qy²)`, clamped
away from zero so a near-vertical attitude cannot produce a divide-by-zero.

**A minimum closing speed on the course axis.** The along-course velocity
setpoint carries a constant floor on top of the proportional term. A pure P law
decays to zero as the gate plane approaches, so the drone converges
asymptotically and never quite crosses — and the simulator only advances the
active gate index on an actual crossing. The floor guarantees a crossing. The
cross-course axis has no floor, because there settling to zero is exactly what
is wanted. The asymmetry is real, deliberate, and only correct because this
course runs along one axis.

## Gains

All at the top of `controller.py`. The ones that matter:

| constant | value | meaning |
|---|---|---|
| `V_MAX` | 5.0 m/s | cap on the proportional part of the setpoint |
| `V_MIN_CLOSE` | 1.0 m/s | along-course floor, per above |
| `K_VX_P`, `K_VY_P` | 5.0 | degrees of tilt per m/s of velocity error |
| `K_VY_D` | 0.4 | lateral damping; the course axis needs none |
| `HOVER_THRUST` | 0.265 | measured, see [`sysid/`](../sysid) |
| `GATE_RISE_M` | 0.8 m | aim above gate centre, biased high against loop lag |

## Sim reset handling

The simulator disarms between runs. The controller watches for the arm flag
dropping, clears its telemetry and integrator state, waits out a settling
period, and re-arms — so a single process covers an evening of attempts without
a restart. It also refuses to launch on a stale `race_start_boot_time_ms`,
which persists across runs and would otherwise fire the drone off the pad
before the countdown.
