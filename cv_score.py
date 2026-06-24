#!/usr/bin/env python3
"""
cv_score.py — OFFLINE computer-vision ACCURACY scorer (no flight needed).

Purpose: turn captured frames + logged pose into a NUMBER for detection quality,
so the CV can be tuned at the desk instead of by repeated sim runs. It is the
"automatic accuracy scoring" half of the offline toolchain (cv_replay.py is the
detection-rate / visual half; this is the geometric-accuracy half).

How it works
------------
For every captured raw frame (vision_dump/NNNNNN_raw.jpg) we know the drone pose
at that instant (vision_log.csv: dx,dy,dz + quaternion). We also know the TRUE
static NED position of all 6 course gates (the oracle below). So we can:

  1. Project each known gate's 3D NED centre into the image using the EXACT
     inverse of the controller's camera->body->world back-projection. A gate is
     "visible" if it lands in front of the camera and inside the frame.
  2. Pick the gate the detector SHOULD lock onto = the nearest visible gate
     (detect_gate takes the largest red contour, i.e. the closest gate).
  3. Run the real detect_gate() on the frame and compare:
        - centre error (px): detected bbox centre vs projected gate centre
        - range error (m):   detected pinhole range (FX*2.7/bw) vs true depth
        - matched:           did the detection land on the expected gate?

This isolates perception error from control error: if centre/range error is
small, the camera math is trustworthy and remaining crashes are a control
problem; if it's large, tune detection (HSV/morph/width) and re-run — instantly,
over hundreds of real frames, zero sim flights.

Usage:
  python cv_score.py                 # score all frames in vision_dump/
  python cv_score.py <dir>           # frames from another directory
  python cv_score.py --pose <csv>    # pose log other than vision_log.csv

Pairs with cv_replay.py (same detect_gate, same dump dir).
"""

import os
import sys
import csv as csvmod
import glob
import math

import cv2
import numpy as np

import vision_rx as V

DUMP_DIR  = 'vision_dump'
POSE_CSV  = 'vision_log.csv'
OUT_CSV   = 'cv_score.csv'

CAM_TILT_DEG = 20.0        # camera tilted UP from body (matches controller)
GATE_WIDTH_M = 2.7         # real gate opening width (pinhole range basis)
MARGIN_PX    = 8           # how far outside the frame still counts as "visible-ish"
MIN_DEPTH_M  = 0.5         # gate must be at least this far in front of the camera

# --- Static course gate centres, NED world frame (the oracle). Source of truth
#     is DESIGN_NOTES.md; keep in sync if the course changes. -------------------
GATES_NED = [
    (-23.30,  -0.40,  -0.03),   # 0
    (-46.89,  -2.50,   5.07),   # 1
    (-74.59,   1.20,  13.67),   # 2
    (-111.49, -5.10,  24.57),   # 3
    (-135.49, -0.80,  25.36),   # 4
    (-159.19, -4.40,  25.97),   # 5
]


def quat_to_rpy(qw, qx, qy, qz):
    """Roll/pitch/yaw (radians) using the EXACT formulas the controller uses, so
    the projection is consistent with how the drone actually back-projects."""
    roll = math.atan2(2.0 * (qw * qx + qy * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def project_gate(gate_ned, drone_xyz, roll, pitch, yaw):
    """
    Project a gate's NED centre into the image. Returns (px, py, depth) where
    depth is metres along the optical axis (>0 = in front). This is the strict
    inverse of the controller's forward path:
        camera ray --(tilt)--> body --Rz*Ry*Rx--> world

    Forward: world = Rz(yaw) Ry(pitch) Rx(roll) * (M_tilt * cam)
    Inverse: cam = M_tilt^T * (Rx(-roll) Ry(-pitch) Rz(-yaw) * world_vec)
    M_tilt and the R's are orthonormal, so inverse == transpose.
    """
    gx, gy, gz = gate_ned
    dx, dy, dz = drone_xyz
    w = np.array([gx - dx, gy - dy, gz - dz], dtype=float)   # world vector to gate

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyaw, syaw = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]])

    R_world_from_body = Rz @ Ry @ Rx          # forward body->world
    body = R_world_from_body.T @ w            # world->body

    t = math.radians(CAM_TILT_DEG)
    ct, st = math.cos(t), math.sin(t)
    # Forward M_tilt (camera->body), from controller:
    #   rb_x = st*rc_y + ct*rc_z ; rb_y = rc_x ; rb_z = ct*rc_y - st*rc_z
    M = np.array([[0, st, ct], [1, 0, 0], [0, ct, -st]])
    cam = M.T @ body                          # body->camera

    cam_x, cam_y, cam_z = cam[0], cam[1], cam[2]
    if cam_z <= 0:
        return None, None, cam_z              # behind camera
    px = V.CX + V.FX * cam_x / cam_z
    py = V.CY + V.FY * cam_y / cam_z
    return px, py, cam_z


def load_pose(pose_path):
    """frame_id -> (x, y, z, qw, qx, qy, qz), last row wins for a given id."""
    poses = {}
    if not os.path.exists(pose_path):
        return poses
    with open(pose_path, newline='') as f:
        for row in csvmod.DictReader(f):
            try:
                fid = int(row['frame_id'])
                vals = [row['dx'], row['dy'], row['dz'],
                        row['qw'], row['qx'], row['qy'], row['qz']]
                if any(v == '' or v is None for v in vals):
                    continue
                poses[fid] = tuple(float(v) for v in vals)
            except (KeyError, ValueError):
                continue
    return poses


def frame_id_from_path(path):
    tag = os.path.splitext(os.path.basename(path))[0].replace('_raw', '')
    try:
        return int(tag)
    except ValueError:
        return None


def main():
    args = sys.argv[1:]
    pose_path = POSE_CSV
    if '--pose' in args:
        i = args.index('--pose')
        pose_path = args[i + 1]
        del args[i:i + 2]
    dirs = [a for a in args if not a.startswith('--')]
    dump_dir = dirs[0] if dirs else DUMP_DIR

    poses = load_pose(pose_path)
    if not poses:
        print(f'No usable pose rows in {pose_path} — need a flight with per-frame '
              f'pose logging (vision_rx INSTRUMENT=True). Cannot score.')
        return

    frames = sorted(glob.glob(os.path.join(dump_dir, '*_raw.jpg')))
    if not frames:
        print(f'No *_raw.jpg frames in {dump_dir}/ — fly once to capture some.')
        return

    out = open(OUT_CSV, 'w', buffering=1)
    out.write('frame,has_pose,n_visible,exp_gate,exp_cx,exp_cy,exp_depth_m,'
              'detected,det_cx,det_cy,det_bw,center_err_px,'
              'det_range_perp_m,true_depth_m,range_err_m,matched\n')

    n = n_pose = n_visible = n_det_when_visible = n_matched = 0
    center_errs = []
    range_errs = []

    for path in frames:
        fid = frame_id_from_path(path)
        if fid is None or fid not in poses:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        n += 1
        n_pose += 1
        dx, dy, dz, qw, qx, qy, qz = poses[fid]
        roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)

        # Project every gate; keep the visible ones (in front + within frame+margin).
        visible = []
        for gi, g in enumerate(GATES_NED):
            px, py, depth = project_gate(g, (dx, dy, dz), roll, pitch, yaw)
            if depth is None or depth < MIN_DEPTH_M:
                continue
            if (-MARGIN_PX <= px <= V.IMG_W + MARGIN_PX and
                    -MARGIN_PX <= py <= V.IMG_H + MARGIN_PX):
                visible.append((gi, px, py, depth))

        exp = min(visible, key=lambda v: v[3]) if visible else None  # nearest visible
        if exp:
            n_visible += 1

        estimate, _, _ = V.detect_gate(img)
        # Only RELIABLE detections are meant for back-projection ranging; weak hints
        # (low/clipped/small, used only for visual-servo descent) would inflate the
        # centre/range error, so they don't count as "detected" for scoring.
        detected = estimate is not None and estimate.get('reliable', True)

        det_cx = det_cy = det_bw = ''
        center_err = range_err = det_range = ''
        matched = ''
        exp_gate = exp_cx = exp_cy = exp_depth = ''

        if exp:
            gi, gpx, gpy, gdepth = exp
            exp_gate, exp_cx, exp_cy, exp_depth = gi, f'{gpx:.1f}', f'{gpy:.1f}', f'{gdepth:.2f}'
            if detected:
                n_det_when_visible += 1
                dcx = estimate['bx'] + estimate['bw'] / 2.0
                dcy = estimate['by'] + estimate['bh'] / 2.0
                det_cx, det_cy, det_bw = f'{dcx:.1f}', f'{dcy:.1f}', estimate['bw']
                ce = math.hypot(dcx - gpx, dcy - gpy)
                center_err = f'{ce:.1f}'
                center_errs.append(ce)
                dr = (V.FX * GATE_WIDTH_M) / estimate['bw'] if estimate['bw'] else float('nan')
                det_range = f'{dr:.2f}'
                re = dr - gdepth
                range_err = f'{re:.2f}'
                range_errs.append(re)
                # "matched" = detection centre lands within ~half a gate-width (in px
                # at this range) of the expected gate centre -> locked onto right gate.
                gate_px = (V.FX * GATE_WIDTH_M) / gdepth
                if ce <= 0.75 * gate_px:
                    matched = 1
                    n_matched += 1
                else:
                    matched = 0

        out.write(f'{fid},1,{len(visible)},{exp_gate},{exp_cx},{exp_cy},{exp_depth},'
                  f'{int(detected)},{det_cx},{det_cy},{det_bw},{center_err},'
                  f'{det_range},{exp_depth},{range_err},{matched}\n')

    out.close()

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    def med(xs):
        if not xs:
            return float('nan')
        s = sorted(xs)
        return s[len(s) // 2]

    print(f'Scored {n} frames with pose (of {len(frames)} dumped) from {dump_dir}/')
    print(f'  gate visible (projected in-frame): {n_visible}/{n} ({pct(n_visible, n):.0f}%)')
    print(f'  detected when a gate was visible:  {n_det_when_visible}/{n_visible} '
          f'({pct(n_det_when_visible, n_visible):.0f}%)')
    print(f'  detection matched expected gate:   {n_matched}/{n_det_when_visible} '
          f'({pct(n_matched, n_det_when_visible):.0f}%)')
    if center_errs:
        ce = sorted(center_errs)
        print(f'  centre error px:  median {med(center_errs):.1f}  '
              f'p90 {ce[int(0.9 * (len(ce) - 1))]:.1f}  max {ce[-1]:.1f}')
    if range_errs:
        abs_re = sorted(abs(x) for x in range_errs)
        print(f'  range error m:    median {med([abs(x) for x in range_errs]):.2f}  '
              f'p90 {abs_re[int(0.9 * (len(abs_re) - 1))]:.2f}  '
              f'(signed median {med(range_errs):+.2f}, + = over-estimates distance)')
    print(f'  per-frame CSV -> {OUT_CSV}')
    print('\nInterpretation: low centre-err + high match% => camera math is sound, '
          'remaining misses are CONTROL. High centre-err / low match% => tune '
          'detection in vision_rx.py and re-run (no flight needed).')


if __name__ == '__main__':
    main()
