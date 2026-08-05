# Offline analysis

Two questions this answers without touching the simulator: how accurate is the
gate estimate, and did a change to the detector help.

## Scoring perception against truth

Comparing the camera's gate estimate against the IMU's own estimate cannot tell
you whether either is right — both are uncertain, and agreement between two
wrong things looks identical to agreement between two right ones.

So the measurement runs on a **VQ1** flight, where the simulator still publishes
true odometry and true gate positions. The proven telemetry controller flies;
the vision pipeline runs alongside as a passive observer whose output nothing
steers on. Every camera frame, the vision estimate is diffed against measured
truth and written to CSV. Neither side of the comparison depends on the other,
and a bad detection cannot crash the flight.

```bash
python -m analysis.collect_ground_truth   # fly VQ1, writes cv_ground_truth.csv
python -m analysis.accuracy               # score it
```

`accuracy.py` reports 3D position error by range bucket, split by which
estimator produced it, plus the median signed per-axis error. Run with no
arguments it scores the committed sample in `data/`, so the numbers in the
READMEs are reproducible from a fresh clone.

Read the p90 column, not the median. A perception system whose typical error is
1.3 m but whose tail is 12 m will occasionally fly into a gate post, and
occasionally is enough to end a run.

## Replaying the detector

```bash
python -m analysis.replay                 # over data/frames/
python -m analysis.replay path/to/frames  # over your own dump
```

Runs `gate_detector.detect_gate` over recorded frames, writes an annotated
overlay per frame plus a per-frame CSV, and prints detection, reliability and
PnP solve rates alongside the thresholds currently in effect. Change a constant
in `vq2/gate_detector.py`, re-run, watch the rate move. Because the live
pipeline calls the same function, a change that helps here helps in the air.

## Sample data

`data/frames/` holds 24 raw frames spanning 2.1 m to 32.6 m of range, and
`data/cv_ground_truth.csv` holds 934 scored comparisons from one VQ1 flight.
Both are committed so the tooling runs from a fresh clone with no simulator.
Live runs write to the repository root and are gitignored.
