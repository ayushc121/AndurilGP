# VQ2 — vision-only racing

Best result: Completed the course, but not fast enough to qualify for further competition.
Run with `python -m vq2.main`.

## The problem

VQ2 blocks `ODOMETRY`, `ATTITUDE`, `LOCAL_POSITION_NED` and the gate layout.
What remains is a 640×360 camera at 30 Hz and a 120 Hz IMU. Everything the
controller knows about where it is, how fast it is going, and where the gates
are has to be derived from those two streams.

That splits into perception, state estimation and guidance.

## Perception — works, and is measured

This is the second detector. The first is in [`../vq1/vision.py`](../vq1/vision.py)
— an HSV threshold and a bounding box, developed on VQ1 where telemetry gave
free ground truth on every frame. Everything below is what that grew into once
the bbox alone was not enough.

`gate_detector.py` holds the whole pipeline as pure functions with no I/O, so
the live flight path and the offline replay harness call literally the same
code. Tuning a threshold against recorded frames is therefore guaranteed to
mean the same thing in the air.

Gates are the only red objects in the scene, so detection starts with an HSV
mask across both sides of the hue wrap. Morphology uses a 3×3 kernel rather
than the more usual 7×7 because at 23 m the gate border is about four pixels
wide and a larger kernel erases it. From the largest contour, a convex hull and
a progressively-loosened polygon fit give four corners, refined to sub-pixel,
and `cv2.solvePnP` with `IPPE_SQUARE` solves the full 6-DoF pose against the
known 2.7 m outer square. When the gate's inner opening also resolves, four
more correspondences refine the solve; on the committed sample frames that
happens on roughly a third of reliable detections, and less than that at range,
since the opening is only ~16 px across at 30 m. When PnP fails altogether, a
pinhole estimate from bounding-box width gives range and bearing but no
orientation.

The output contract is three numbers: `body_x_m` forward, `body_y_m` right,
`body_z_m` down. That is the entire interface the controller steers on, which
is what makes it telemetry-free.

**Measured against 934 frames of true odometry** (`python -m analysis.accuracy`):

| range | frames | median 3D error | p90 | PnP solve rate |
|---|---|---|---|---|
| < 5 m | 120 | 1.24 m | 1.42 m | 74 % |
| 5–15 m | 353 | 1.20 m | 1.60 m | 95 % |
| 15–25 m | 372 | 1.62 m | 3.96 m | 97 % |
| 25–40 m | 89 | 3.39 m | 11.45 m | 84 % |

Two things fall out of that table that were not obvious beforehand.

**PnP's value is the tail, not the median.** PnP gives a median error of
1.30 m against the bounding box's 1.41 m — barely a difference. But the p90 is
3.27 m against 11.86 m. The expensive geometric solve is not buying typical
accuracy; it is buying the absence of occasional 12 m errors, which is a much
better thing to buy, because it is the outliers that fly a drone into a post.

**There is a systematic −1.12 m vertical bias.** The median signed error in the
down axis, across every frame. Noise cancels in a median; a bias does not. It
appears on both the PnP and bounding-box paths, which rules out the pose solver
and points at something shared — the object-frame origin, the fixed gate
elevation offset, or a camera extrinsic. It was never tracked down, and it is
the single highest-value unresolved bug here.

## State estimation — a gate-relative filter

`GyroAHRS` integrates attitude from the gyro alone, seeded at the known −17.8°
launch-block pitch. There is deliberately no accelerometer correction: a
complementary filter pulls attitude toward the apparent gravity vector, but
under thrust that vector is gravity *plus* linear acceleration, which describes
essentially all of a race.

Velocity and position come from `estimator.py`, a six-state filter over
`[position, velocity]` expressed **relative to the gate currently anchored**.
The IMU predicts at 400 Hz; each PnP detection is a position measurement that
corrects it. Working gate-relative means no map is needed, which is the whole
point — VQ2 does not give you one.

Three details carry most of the value:

**The acceleration cap bounds thrust, not total acceleration.** Capping `|a|`
goes to zero at and below hover, which forbids acceleration exactly while the
drone is falling: a real free fall got scaled to nothing and the vertical
velocity estimate inverted its sign on 91% of moving ticks. Bounding the
thrust-produced part instead took that to 4%.

**Measurement noise is anisotropic and grows with range.** PnP is tight in the
image plane and weak along the optical axis, so `R` is built diagonal in the
camera frame and rotated into NED, with the range term scaling as range². The
coefficient was fitted to 1259 ground-truth samples, not guessed.

**Handoff re-anchors on the first valid fix, not on two that agree.** Judging
agreement against the filter's own drifted prediction is circular, and once
produced permanent blindness — the filter rejected every fix with the gate in
plain view. Garbage is caught instead by state-independent range and aspect
filters, so a bad anchor is a transient rather than a lockout. Re-anchoring
keeps velocity and its covariance, since world velocity does not care which
gate you are measuring against.

**Measured on a VQ1 flight** with the telemetry controller flying and the
estimator running blind beside it (`python -m analysis.ekf_accuracy`):

| | median | vs baseline |
|---|---|---|
| position error | 1.04 m | **6.5× better** than raw strapdown (6.74 m) |
| velocity error | 1.50 m/s | **2.8× better** than raw CV (4.23 m/s) |

That is on the anchor-active rows, which is the honest filter — scoring against
whichever gate happens to be nearest flatters the result, and the tool prints
both and says so.

## Layout

```
gate_detector.py   detection, tracking, corner extraction, PnP — pure functions
vision_rx.py       temporal state and diagnostics on top of core.camera_rx
gate_ekf.py        six-state [position, velocity] filter, pure numpy
estimator.py       gate-relative VIO: IMU predicts, PnP corrects
controller.py      guidance and attitude commands
```

Offline tooling and the accuracy data live in [`analysis/`](../analysis).

## Guidance

Roll steers, banking on the gate's bearing with damping on lateral closure,
faded out by a blend factor as the gate plane approaches so the drone is not
still turning as it crosses. Thrust is PD on measured gate elevation,
tilt-compensated.

Yaw is a rate-limited nulling integrator: each new camera frame moves the
commanded heading a step toward the bearing, and with no gate in view the
setpoint is held rather than steered. The bearing comes from the raw camera ray,
not from the AHRS — routing it through estimated attitude closes a loop on the
filter's own error. The step is gated on frame ID, since one 30 Hz frame spans
two 60 Hz ticks and a relative step would otherwise apply twice.

Yaw being an absolute setpoint also makes the pad heading a real constant. It
was 90° out on VQ2, so the aircraft snapped to that false heading the instant
thrust came up. Fixing it dropped peak yaw rate in the first half-second from
over 200 °/s to about 2, and gate detection went from roughly a quarter of
frames to about two thirds. It was the single largest measured improvement in
the project.

## Known issues

The −1.12 m vertical bias, above.

Gate selection starts from the largest red contour. The tracker keeps that
choice stable once made, but acquisition can still latch the wrong gate when
several are in frame — the detection overlays in the root README show one where
the heuristic is visibly debatable.
