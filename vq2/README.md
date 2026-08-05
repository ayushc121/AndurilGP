# VQ2 — vision-only racing

Best result: Completed the course, but not fast enough to qualify for further competition.
Run with `python -m vq2.main`.

## The problem

VQ2 blocks `ODOMETRY`, `ATTITUDE`, `LOCAL_POSITION_NED` and the gate layout.
What remains is a 640×360 camera at 30 Hz and a 120 Hz IMU. Everything the
controller knows about where it is, how fast it is going, and where the gates
are has to be derived from those two streams.

That splits into three problems, and they went differently.

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

## State estimation — adequate, structurally incomplete

`GyroAHRS` integrates attitude from the gyro alone, seeded at the known −17.8°
launch-block pitch. There is deliberately no accelerometer correction: a
complementary filter pulls attitude toward the apparent gravity vector, but
under thrust that vector is gravity *plus* linear acceleration, which describes
essentially all of a race. The correction would be wrong exactly when it
mattered most.

Velocity comes from strapdown integration of accelerometer specific force,
rotated to NED through the attitude estimate, on a dedicated 400 Hz thread so
no 120 Hz IMU sample is dropped. Vision corrects the lateral and vertical
channels by differencing gate position between frames; forward velocity stays
on the IMU, because the gate's apparent growth rate is a far weaker range
signal than its lateral motion is a bearing signal.

The structural gap: position is integrated but never corrected by anything. It
drifts without bound. That was survivable only because the controller never
uses absolute position — it steers entirely on relative geometry. A proper
visual-inertial filter with vision as a measurement update is the right answer
and was scoped but not built.

## Layout

```
gate_detector.py   detection, corner extraction, PnP, pose conversion — pure
vision_rx.py       temporal state and diagnostics on top of core.camera_rx
controller.py      estimation thread, guidance, attitude commands
```

Offline tooling and the accuracy data live in [`analysis/`](../analysis).

## Known issues

The −1.12 m vertical bias, above.
Gate selection is "largest red contour". With several gates in frame this picks
the nearest, which is usually but not always the active one. A tracker that
maintains identity across frames would be more robust, and the detection
overlays in the root README show a frame where the heuristic is visibly
debatable.
