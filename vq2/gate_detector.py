"""
Gate detection and pose estimation. Pure functions, no I/O and no state, so
the live path and the offline replay harness run identical code.

Range comes from `solvePnP` against the gate's known 2.7 m square, falling
back to a pinhole estimate from bbox width. Accuracy for both: VQ2 README.
"""

import math

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Camera model (spec §3.8) — 640x360, no distortion, pitched 20 deg up
# --------------------------------------------------------------------------

IMG_W, IMG_H = 640, 360
CX, CY = 320.0, 180.0
FX = FY = 320.0
CAM_TILT_DEG = 20.0

_CAM_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]], dtype=np.float32)
_CAM_DIST = np.zeros((4, 1), dtype=np.float32)
_HAS_IPPE = hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE')

# --------------------------------------------------------------------------
# Gate geometry (spec §3.7) — object frame is gate-centred, X right, Y up
# --------------------------------------------------------------------------

GATE_WIDTH_M = 2.7          # outer frame, used for the pinhole range fallback
_OUTER_HALF = GATE_WIDTH_M / 2.0
_INNER_HALF = 0.75          # 1.5 m inner opening

def _square(half):
    """Four corners of a centred square in [TL, TR, BR, BL] — IPPE_SQUARE order."""
    return np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)

_OBJ_OUTER = _square(_OUTER_HALF)
_OBJ_8PT = np.vstack([_OBJ_OUTER, _square(_INNER_HALF)])

PNP_MAX_REPROJ_PX = 12.0    # reject solves worse than this RMS reprojection error

# --------------------------------------------------------------------------
# Detection thresholds
# --------------------------------------------------------------------------

# Gates are the only red objects in the scene. Red straddles the hue wrap in
# OpenCV's 0-180 scale, so it takes two ranges.
LOWER_RED_1 = np.array([0, 120, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 50])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 300      # below this a detection is a hint, not a pose source
EDGE_MARGIN_PX = 4          # bbox this close to a border counts as clipped
CENTER_REJECT_FRAC = 0.25   # and this far off-centre makes the clip disqualifying

# Bbox smoothing, reset hard on discontinuity — otherwise passing one gate and
# acquiring the next blends them into a phantom gate in between.
EMA_ALPHA = 0.5
EMA_RESET_DCX = 120         # px of centre jump that means "different gate"
EMA_RESET_WR = 1.6          # or this ratio of bbox-width change


# --------------------------------------------------------------------------
# Corner extraction and PnP
# --------------------------------------------------------------------------

def _order_quad(pts):
    """Sort four points into [TL, TR, BR, BL] by coordinate sums and differences."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _approx_quad(contour, epsilons=(0.03, 0.05, 0.08, 0.12)):
    """Fit a 4-gon to a contour, loosening tolerance until one fits, else None."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in epsilons:
        cand = cv2.approxPolyDP(hull, eps * peri, True)
        if len(cand) == 4:
            return cand.reshape(4, 2).astype(np.float32)
    return None


def extract_corners(img, contour):
    """
    Sub-pixel outer and inner gate corners, [TL, TR, BR, BL]; either may be
    None. A border-touching contour is rejected — its hull is missing corners.

    The inner opening needs its own mask: the detection mask's morphology
    fills it at range, where it is only ~16 px across.
    """
    gx, gy, gw, gh = cv2.boundingRect(contour)
    if gx <= 1 or gy <= 1 or (gx + gw) >= IMG_W - 2 or (gy + gh) >= IMG_H - 2:
        return None, None

    outer = _approx_quad(contour)
    if outer is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    try:
        cv2.cornerSubPix(gray, outer, (7, 7), (-1, -1), criteria)
    except cv2.error:
        pass                       # refinement is a bonus; the raw quad still works
    outer = _order_quad(outer)

    try:
        inner = _find_inner_quad(img, gray, contour, (gx, gy, gw, gh), criteria)
    except cv2.error:
        inner = None
    return outer, inner


def _find_inner_quad(img, gray, outer_contour, outer_box, criteria):
    """Locate the gate's inner opening as a hole contour inside the red frame."""
    gx, gy, gw, gh = outer_box
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
                          cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None

    outer_area = cv2.contourArea(outer_contour)
    for contour, node in zip(contours, hierarchy[0]):
        if node[3] == -1:                       # top-level contour, not a hole
            continue
        ix, iy, iw, ih = cv2.boundingRect(contour)
        if ix < gx or iy < gy or ix + iw > gx + gw or iy + ih > gy + gh:
            continue
        if not 0.10 * outer_area < cv2.contourArea(contour) < 0.70 * outer_area:
            continue
        quad = _approx_quad(contour)
        if quad is None:
            continue
        try:
            cv2.cornerSubPix(gray, quad, (5, 5), (-1, -1), criteria)
        except cv2.error:
            pass
        return _order_quad(quad)
    return None


def solve_pose(corners):
    """
    Pose from four outer corners: (tvec, rvec, reproj_err) in the camera frame
    (x right, y down, z forward), or (None, None, inf).

    IPPE_SQUARE returns two solutions for a planar square; the one behind the
    camera is discarded and the lower reprojection error wins.
    """
    try:
        if _HAS_IPPE:
            n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                _OBJ_OUTER, corners, _CAM_K, _CAM_DIST,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            best, best_err = None, float('inf')
            for i in range(n):
                err = float(errors[i].flat[0])
                if float(tvecs[i][2, 0]) > 0.3 and err < best_err:
                    best, best_err = i, err
            if best is None:
                return None, None, float('inf')
            return tvecs[best].flatten(), rvecs[best].flatten(), best_err

        ok, rvec, tvec = cv2.solvePnP(_OBJ_OUTER, corners, _CAM_K, _CAM_DIST,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or float(tvec[2]) <= 0.3:
            return None, None, float('inf')
        return tvec.flatten(), rvec.flatten(), _reproj_error(_OBJ_OUTER, rvec, tvec, corners)
    except cv2.error:
        return None, None, float('inf')


def refine_pose_8pt(outer, inner, rvec, tvec):
    """Polish a 4-corner solve using the inner opening as four more points."""
    try:
        image_pts = np.vstack([outer, inner])
        ok, rvec_r, tvec_r = cv2.solvePnP(
            _OBJ_8PT, image_pts, _CAM_K, _CAM_DIST,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or float(tvec_r[2, 0]) <= 0.3:
            return None, None, float('inf')
        return (tvec_r.flatten(), rvec_r.flatten(),
                _reproj_error(_OBJ_8PT, rvec_r, tvec_r, image_pts))
    except cv2.error:
        return None, None, float('inf')


def _reproj_error(obj_pts, rvec, tvec, image_pts):
    """Mean reprojection error in pixels."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, _CAM_K, _CAM_DIST)
    return float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def detect_gate(img):
    """
    Find the gate in one BGR frame. Returns (estimate, mask, contours), with
    `estimate` None when nothing usable is present.

    `reliable` separates "clean enough to derive a pose from" from "real, but
    only good enough to steer toward" — the case on descending sections, where
    the next gate sits below the tilted camera as a clipped band.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
                          cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2))

    # 3x3 rather than 7x7: at 23 m the gate border is only ~4 px wide and a
    # larger kernel erases it entirely.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not candidates:
        return None, mask, contours

    best = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(best)
    if moments['m00'] == 0.0:
        return None, mask, contours

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    bx, by, bw, bh = cv2.boundingRect(best)
    area = cv2.contourArea(best)

    # Bbox centre, not centroid: the gate is hollow, so the centroid drifts
    # toward whichever side is thicker in view.
    box_cx, box_cy = bx + bw / 2.0, by + bh / 2.0
    clipped_lr = bx <= EDGE_MARGIN_PX or bx + bw >= IMG_W - EDGE_MARGIN_PX
    clipped_tb = by <= EDGE_MARGIN_PX or by + bh >= IMG_H - EDGE_MARGIN_PX
    off_centre_x = abs(box_cx - CX) > CENTER_REJECT_FRAC * IMG_W
    off_centre_y = abs(box_cy - CY) > CENTER_REJECT_FRAC * IMG_H

    # Passing through a gate leaves a sliver against one edge whose width
    # implies a wildly wrong range. A gate legitimately filling the frame also
    # touches edges but stays centred, so rejection needs both conditions.
    if clipped_lr and off_centre_x:
        return None, mask, contours

    reliable = area >= MIN_CONTOUR_AREA and not (clipped_tb and off_centre_y and area < 5000)

    estimate = {
        'cx': cx, 'cy': cy,
        'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
        'area': area,
        'reliable': reliable,
        'pnp_ok': False,
    }

    if reliable:
        _attach_pose(estimate, img, best)
    return estimate, mask, contours


def _attach_pose(estimate, img, contour):
    """Run the PnP chain and write the result into `estimate` if it holds up."""
    outer, inner = extract_corners(img, contour)
    if outer is None:
        return

    tvec, rvec, reproj = solve_pose(outer)
    if tvec is None or reproj >= PNP_MAX_REPROJ_PX:
        return    # a bad fit is worse than no pose: the bbox fallback is honest

    if inner is not None:
        tvec8, rvec8, reproj8 = refine_pose_8pt(
            outer, inner, rvec.reshape(3, 1), tvec.reshape(3, 1))
        if tvec8 is not None and reproj8 < PNP_MAX_REPROJ_PX:
            tvec, rvec = tvec8, rvec8

    estimate.update(pnp_ok=True, pnp_tvec=tvec, pnp_rvec=rvec)


# --------------------------------------------------------------------------
# Pose conversion and smoothing
# --------------------------------------------------------------------------

def body_relative_pose(estimate):
    """
    Add body_x_m (forward), body_y_m (right), body_z_m (down) in place — the
    entire interface the controller steers on, and why it needs no telemetry.
    """
    if estimate is None:
        return None

    if estimate.get('pnp_ok'):
        cam_x, cam_y, cam_z = (float(v) for v in estimate['pnp_tvec'])
    else:
        bw = estimate.get('bw', 0)
        if not bw:
            nan = float('nan')
            estimate.update(body_x_m=nan, body_y_m=nan, body_z_m=nan)
            return estimate
        cam_z = (FX * GATE_WIDTH_M) / bw
        cam_x = (estimate.get('cx', CX) - CX) * cam_z / FX
        cam_y = (estimate.get('cy', CY) - CY) * cam_z / FY

    # Camera to body: undo the fixed 20 deg upward tilt.
    tilt = math.radians(CAM_TILT_DEG)
    ct, st = math.cos(tilt), math.sin(tilt)
    estimate.update(
        body_x_m=ct * cam_z + st * cam_y,
        body_y_m=cam_x,
        body_z_m=-st * cam_z + ct * cam_y)
    return estimate


def ema_smooth(prev_bbox, estimate, alpha=EMA_ALPHA):
    """
    Smooth the bbox across frames -> (smoothed_estimate, new_prev_bbox).
    Stateless so live and replay smooth identically. Resets on a gap or a
    discontinuity in centre or width.
    """
    if estimate is None:
        return None, None

    current = (estimate['bx'], estimate['by'], estimate['bw'], estimate['bh'])
    if prev_bbox is None:
        smoothed = current
    else:
        centre_jump = abs((current[0] + current[2] / 2.0)
                          - (prev_bbox[0] + prev_bbox[2] / 2.0))
        width_ratio = (current[2] / prev_bbox[2]) if prev_bbox[2] else 99.0
        discontinuous = (centre_jump > EMA_RESET_DCX
                         or width_ratio > EMA_RESET_WR
                         or width_ratio < 1.0 / EMA_RESET_WR)
        smoothed = current if discontinuous else tuple(
            alpha * c + (1.0 - alpha) * p for c, p in zip(current, prev_bbox))

    bx, by, bw, bh = smoothed
    out = dict(estimate)
    out.update(bx=bx, by=by, bw=bw, bh=bh,
               cx=bx + bw / 2.0, cy=by + bh / 2.0)
    return out, smoothed
