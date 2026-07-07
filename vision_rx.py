import math
import os
import socket
import struct
import threading

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------
SIM_SERVER_UDP_IP   = '0.0.0.0'
SIM_SERVER_UDP_PORT = 5600

# --------------------------------------------------------------------------------------
# Camera intrinsics (from spec §3.8)
# --------------------------------------------------------------------------------------
IMG_W, IMG_H = 640, 360
CX, CY       = 320.0, 180.0
FX = FY      = 320.0
# Camera is tilted 20° upward from body frame — the controller accounts for
# this when converting pixel offsets into NED velocity corrections.
CAM_TILT_DEG = 20.0

# Gate outer width used for pinhole range (spec §3.7)
GATE_WIDTH_M = 2.7

# --------------------------------------------------------------------------------------
# PnP: known gate geometry for solvePnP (spec §3.7)
# Object frame: gate centre = origin, X right, Y up, Z toward camera.
# IPPE_SQUARE point order: TL, TR, BR, BL.
# --------------------------------------------------------------------------------------
_GATE_HALF    = GATE_WIDTH_M / 2.0          # 1.35 m
_GATE_OBJ_PTS = np.array([
    [-_GATE_HALF,  _GATE_HALF, 0.0],        # TL
    [ _GATE_HALF,  _GATE_HALF, 0.0],        # TR
    [ _GATE_HALF, -_GATE_HALF, 0.0],        # BR
    [-_GATE_HALF, -_GATE_HALF, 0.0],        # BL
], dtype=np.float32)

# Inner opening corners (spec §3.7: 1500 × 1500 mm inner square, half = 0.75 m).
# Same [TL, TR, BR, BL] order as outer.
_GATE_INNER_HALF = 0.75
_GATE_INNER_OBJ  = np.array([
    [-_GATE_INNER_HALF,  _GATE_INNER_HALF, 0.0],
    [ _GATE_INNER_HALF,  _GATE_INNER_HALF, 0.0],
    [ _GATE_INNER_HALF, -_GATE_INNER_HALF, 0.0],
    [-_GATE_INNER_HALF, -_GATE_INNER_HALF, 0.0],
], dtype=np.float32)

# 8-point combined object array: outer 4 followed by inner 4.
_GATE_OBJ_8   = np.vstack([_GATE_OBJ_PTS, _GATE_INNER_OBJ])

_CAM_K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]], dtype=np.float32)
_CAM_DIST = np.zeros((4, 1), dtype=np.float32)          # spec: no distortion
_HAS_IPPE = hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE')        # OpenCV >= 3.4.5

# Reject a PnP solve whose RMS reprojection error exceeds this threshold.
PNP_MAX_REPROJ_PX = 12.0

# --------------------------------------------------------------------------------------
# Gate colour — red on dark/grey background.
# Red wraps around the HSV hue circle in OpenCV (0–180 scale), so two
# ranges are needed: one near 0° and one near 180°.
# --------------------------------------------------------------------------------------
LOWER_RED_1 = np.array([0,   120,  50])
UPPER_RED_1 = np.array([10,  255, 255])

LOWER_RED_2 = np.array([170, 120,  50])
UPPER_RED_2 = np.array([180, 255, 255])

# Minimum contour area (px²) to be considered a reliable gate detection.
# Lowered from 800 → 500 to catch gate frames that are split into smaller
# blobs by the 7×7 morphology at oblique approach angles (15-25m range).
# The simulator has no non-gate red objects, so false positives are minimal.
MIN_CONTOUR_AREA = 300

# --- Partial / edge-exit rejection -----------------------------------------
# As the drone passes THROUGH a gate, the red frame breaks into a partial sliver
# against an image edge (e.g. bw=45 in the top-left corner). That fragment is a
# valid red contour but a GARBAGE geometry source: a clipped width gives a wildly
# wrong pinhole range and an off-centre fragment back-projects to a phantom gate
# tens of metres up/sideways (observed: a corner sliver -> range ~19 m -> a full-
# thrust 18 m climb command right at the gate plane). We reject a detection whose
# bbox touches a frame edge AND is OFF-CENTRE on that axis: a gate that legitimately
# fills the frame on close approach touches edges too, but stays centred, so it is
# kept; only the off-centre clipped slivers are dropped. Horizontal clipping
# corrupts width->range; vertical clipping corrupts the vertical estimate.
EDGE_MARGIN_PX     = 4      # bbox within this many px of a border = "touching" it
CENTER_REJECT_FRAC = 0.25   # off-centre by more than this fraction of W/H = reject

# A real-but-weak gate hint (small/far/low) usable for visual-servo descent but not
# trusted for back-projection steering (which needs MIN_CONTOUR_AREA + not clipped).
SERVO_AREA_FLOOR   = 300

# --- Temporal smoothing (EMA) of the published gate estimate -------------------
# The per-frame bbox jitters, and that jitter feeds the controller's lateral/altitude
# wander. An exponential moving average steadies the "middle point" the controller
# chases. Reset on a detection gap or a big discontinuity (gate switch) so two
# different gates are never blended together.
EMA_ALPHA     = 0.5    # weight of the newest frame (1.0 = no smoothing)
EMA_RESET_DCX = 120    # px centre jump -> treat as a new gate, reset (no smoothing)
EMA_RESET_WR  = 1.6    # bbox-width ratio (new/prev) beyond this -> reset

# Discard partially-received frames older than this many frame IDs
FRAME_BUFFER_DEPTH = 10

# --------------------------------------------------------------------------------------
# Gate-position velocity estimation
# Gate is stationary. body_relative_pose() gives its body-frame position each frame.
# Derivative between consecutive frames: vX=-d(bx)/dt, vY=-d(by)/dt, vZ=-d(bz)/dt.
# Published to data['vision_velocity'] = {'vx_body_mps', 'vy_body_mps', 'vz_body_mps'}
# --------------------------------------------------------------------------------------
_GATE_VEL_FPS = 30.0   # camera frame rate

# Socket timeout so the thread can notice is_running=False cleanly
SOCKET_TIMEOUT_S = 1.0

# --------------------------------------------------------------------------------------
# INSTRUMENTATION (diagnostics only, no behaviour change). Captures what the camera
# actually sees so detection can be evaluated against real frames + numbers. The
# published vision_gate_estimate is UNCHANGED.
#   * vision_dump/   : raw frame, HSV red mask, detection overlay, every N frames.
#   * vision_log.csv : per-frame contour count + the largest contour's area/aspect/
#                      fill_ratio/centroid/bbox (the metrics a shape filter would use).
# Set INSTRUMENT = False to disable entirely.
# --------------------------------------------------------------------------------------
INSTRUMENT       = True
DUMP_DIR         = 'vision_dump'
DUMP_EVERY_N     = 15      # save the image trio every this many PROCESSED frames
DUMP_MAX_SETS    = 400     # stop dumping after this many sets (disk-flood guard)
INSTR_AREA_FLOOR = 100     # log/draw contours down to this area (below MIN_CONTOUR_AREA)


def _order_quad(pts):
    """Return (4, 2) float32 in [TL, TR, BR, BL] order for IPPE_SQUARE."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest y−x
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest y−x
    ], dtype=np.float32)


def _approx_quad(contour, epsilons=(0.03, 0.05, 0.08, 0.12)):
    """Try increasing approxPolyDP epsilons; return (4,2) float32 or None."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in epsilons:
        cand = cv2.approxPolyDP(hull, eps * peri, True)
        if len(cand) == 4:
            return cand.reshape(4, 2).astype(np.float32)
    return None


def _extract_gate_corners(img, contour, mask):
    """
    Extract outer + inner gate corners from the outer red contour and the mask.
    Returns (outer_4, inner_4_or_None). outer_4 is None if the gate is clipped
    or no quad can be found.

    Outer corners  → ±1.35 m object points (outer gate frame).
    Inner corners  → ±0.75 m object points (inner opening edge), found via the
                     RETR_CCOMP hole contour inside the red-frame mask.
    Both ordered [TL, TR, BR, BL] for IPPE_SQUARE / 8-pt solvePnP.
    """
    # Reject clipped gates — the convex hull is missing corners.
    gx, gy, gw, gh = cv2.boundingRect(contour)
    if gx <= 1 or gy <= 1 or (gx + gw) >= IMG_W - 2 or (gy + gh) >= IMG_H - 2:
        return None, None

    outer_pts = _approx_quad(contour)
    if outer_pts is None:
        return None, None

    # Sub-pixel refinement on grayscale edges.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    try:
        cv2.cornerSubPix(gray, outer_pts, (7, 7), (-1, -1), crit)
    except Exception:
        pass
    outer_4 = _order_quad(outer_pts)

    # Inner hole: RETR_CCOMP on a *fine* mask (3×3 morphology instead of 7×7).
    # The standard detection mask uses 7×7 MORPH_CLOSE which fills the inner
    # hole at range (inner opening is only ~16 px at 30 m).  Re-computing with
    # a 3×3 kernel preserves the hole without affecting detection area / reliable.
    inner_4 = None
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _m1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        _m2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        mask_fine = cv2.bitwise_or(_m1, _m2)
        _k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_fine = cv2.morphologyEx(mask_fine, cv2.MORPH_CLOSE, _k3)
        mask_fine = cv2.morphologyEx(mask_fine, cv2.MORPH_OPEN,  _k3)
        all_cnts, hier = cv2.findContours(mask_fine, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            hier = hier[0]
            gate_area = cv2.contourArea(contour)
            for i, h in enumerate(hier):
                if h[3] == -1:           # skip outer boundaries
                    continue
                ix, iy, iw, ih = cv2.boundingRect(all_cnts[i])
                # Must sit inside the outer gate bbox.
                if ix < gx or iy < gy or ix + iw > gx + gw or iy + ih > gy + gh:
                    continue
                # Inner area should be 10–70 % of outer area.
                iarea = cv2.contourArea(all_cnts[i])
                if not (0.10 * gate_area < iarea < 0.70 * gate_area):
                    continue
                inner_pts = _approx_quad(all_cnts[i])
                if inner_pts is None:
                    continue
                try:
                    cv2.cornerSubPix(gray, inner_pts, (5, 5), (-1, -1), crit)
                except Exception:
                    pass
                inner_4 = _order_quad(inner_pts)
                break
    except Exception:
        pass

    return outer_4, inner_4


def _solve_gate_pnp(corners):
    """
    Run PnP on 4 ordered outer-gate corners. Returns (tvec, rvec, reproj_err)
    in camera frame (x=right, y=down, z=forward), metres, or (None, None, inf).

    IPPE_SQUARE returns 2 solutions; we pick the one with positive depth (gate
    in front of camera) and lower reprojection error. Falls back to ITERATIVE
    when IPPE is unavailable.
    """
    try:
        if _HAS_IPPE:
            n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                _GATE_OBJ_PTS, corners, _CAM_K, _CAM_DIST,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            best_i, best_err = None, float('inf')
            for i in range(n):
                depth = float(tvecs[i][2, 0])    # tvecs[i] is (3,1); scalar via [2,0]
                err   = float(errors[i].flat[0])  # errors[i] is (1,); scalar via .flat[0]
                if depth > 0.3 and err < best_err:
                    best_err, best_i = err, i
            if best_i is None:
                return None, None, float('inf')
            return tvecs[best_i].flatten(), rvecs[best_i].flatten(), best_err
        else:
            ok, rvec, tvec = cv2.solvePnP(
                _GATE_OBJ_PTS, corners, _CAM_K, _CAM_DIST,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or float(tvec[2]) <= 0.3:
                return None, None, float('inf')
            proj, _ = cv2.projectPoints(_GATE_OBJ_PTS, rvec, tvec, _CAM_K, _CAM_DIST)
            err = float(np.mean(np.linalg.norm(
                proj.reshape(4, 2) - corners.reshape(4, 2), axis=1)))
            return tvec.flatten(), rvec.flatten(), err
    except Exception:
        return None, None, float('inf')


def _refine_pnp_8pt(outer_corners, inner_corners, init_rvec, init_tvec):
    """
    Refine a 4-corner IPPE result using all 8 corners (outer + inner).
    The IPPE solution serves as the initial guess; ITERATIVE polishes it.
    Returns (tvec, rvec, reproj_err) or the original (tvec_flat, rvec_flat, inf)
    if refinement fails or produces a worse fit.
    """
    try:
        all_img = np.vstack([outer_corners, inner_corners])
        ok, rvec_r, tvec_r = cv2.solvePnP(
            _GATE_OBJ_8, all_img, _CAM_K, _CAM_DIST,
            rvec=init_rvec, tvec=init_tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or float(tvec_r[2, 0]) <= 0.3:
            return None, None, float('inf')
        proj, _ = cv2.projectPoints(_GATE_OBJ_8, rvec_r, tvec_r, _CAM_K, _CAM_DIST)
        err = float(np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - all_img.reshape(-1, 2), axis=1)))
        return tvec_r.flatten(), rvec_r.flatten(), err
    except Exception:
        return None, None, float('inf')


def body_relative_pose(estimate):
    """
    Augment a detection estimate with telemetry-free body-relative pose fields.
    Called after EMA smoothing so the smoothed bbox values are used.

    Adds to estimate (in-place and returns it):
      cam_x_m  — lateral offset in camera frame (right = +), metres
      cam_y_m  — vertical offset in camera frame (down = +), metres
      cam_z_m  — depth along optical axis (forward = +), metres
      body_x_m — forward component in body frame (body NED X), metres
      body_y_m — right component in body frame (body NED Y), metres
      body_z_m — down component in body frame (body NED Z), metres

    Convention matches the back-projection in controller._gate_world_direction and
    the oracle in cv_score.project_gate — same 20° tilt matrix, same sign choices.
    All values are NaN when bw == 0 (degenerate bbox).

    This is the Round-2 output contract: the controller can steer toward the gate
    using only these body-relative fields, with zero telemetry.
    """
    if estimate is None:
        return estimate

    if estimate.get('pnp_ok'):
        # PnP path: tvec is already in camera frame (x=right, y=down, z=forward).
        tvec    = estimate['pnp_tvec']
        cam_x_m = float(tvec[0])
        cam_y_m = float(tvec[1])
        cam_z_m = float(tvec[2])
    else:
        # Pinhole fallback: range from known gate width + bearing from centroid.
        bw = estimate.get('bw', 0)
        cx = estimate.get('cx', CX)
        cy = estimate.get('cy', CY)
        if not bw:
            nan = float('nan')
            estimate.update(cam_x_m=nan, cam_y_m=nan, cam_z_m=nan,
                            body_x_m=nan, body_y_m=nan, body_z_m=nan)
            return estimate
        cam_z_m = (FX * GATE_WIDTH_M) / bw
        cam_x_m = (cx - CX) * cam_z_m / FX   # right in camera
        cam_y_m = (cy - CY) * cam_z_m / FY   # down in camera

    # Camera → body: fixed 20° upward tilt (same matrix as controller).
    t   = math.radians(CAM_TILT_DEG)
    ct  = math.cos(t)
    st  = math.sin(t)
    body_x_m =  ct * cam_z_m + st * cam_y_m   # forward (body X)
    body_y_m =  cam_x_m                         # right   (body Y)
    body_z_m = -st * cam_z_m + ct * cam_y_m   # down    (body Z, NED)

    estimate.update(cam_x_m=cam_x_m, cam_y_m=cam_y_m, cam_z_m=cam_z_m,
                    body_x_m=body_x_m, body_y_m=body_y_m, body_z_m=body_z_m)
    return estimate


def detect_gate(img):
    """
    Pure gate detection — the SINGLE SOURCE OF TRUTH for the perception pipeline.

    Takes a BGR image and returns (estimate, mask, contours) where `estimate` is a
    dict of geometric fields (or None if no gate found). `frame_id` is added by the
    caller. Kept as a free function (no shared_data, no threads) so the offline
    replay harness (cv_replay.py) can run the EXACT same detection as the live
    VisionRX — so any threshold/param tuning done offline transfers to flight.

    Convention: cx_offset > 0 = gate right of centre; cy_offset > 0 = gate below
    centre (image Y-down).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps the hue circle — combine both ranges
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological clean-up — 3×3 preserves the ~4px gate border at 23m range
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Two-tier candidate set. SERVO_AREA_FLOOR catches a small/far gate low in the
    # frame so the controller can VISUAL-SERVO descend toward it (the descending
    # course drops the next gate below the 20°-up camera, where it shows only as a
    # weak/clipped band). MIN_CONTOUR_AREA marks a detection big enough to TRUST for
    # back-projection steering. Red mask is clean (only gates are red) so a small red
    # blob is almost certainly a real far gate, not noise.
    valid = [c for c in contours if cv2.contourArea(c) >= SERVO_AREA_FLOOR]
    if not valid:
        return None, mask, contours

    best    = max(valid, key=cv2.contourArea)
    moments = cv2.moments(best)
    if moments['m00'] == 0.0:
        return None, mask, contours

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    bx, by, bw, bh = cv2.boundingRect(best)
    area = cv2.contourArea(best)

    # Edge / off-centre analysis (bbox centre = immune to hollow-contour centroid shift).
    true_cx = bx + bw / 2.0
    true_cy = by + bh / 2.0
    touches_lr = (bx <= EDGE_MARGIN_PX) or (bx + bw >= IMG_W - EDGE_MARGIN_PX)
    touches_tb = (by <= EDGE_MARGIN_PX) or (by + bh >= IMG_H - EDGE_MARGIN_PX)
    off_centre_x = abs(true_cx - CX) > CENTER_REJECT_FRAC * IMG_W
    off_centre_y = abs(true_cy - CY) > CENTER_REJECT_FRAC * IMG_H

    # HARD reject only the horizontal exit-sliver garbage (gate breaking off to one
    # side as the drone passes THROUGH): clipped L/R AND off-centre horizontally. Its
    # width->range is wild and it is neither steerable nor a useful descend hint (it
    # caused the 18 m climb). Everything else is returned.
    if touches_lr and off_centre_x:
        return None, mask, contours

    # RELIABLE = big enough to trust for back-projection AND not a low/clipped band.
    # A centred gate that fills the frame on close approach touches edges but stays
    # centred -> still reliable. A low gate clipped at the bottom (touches T/B AND
    # off-centre vertically) is a WEAK hint: real, but its size/centre are unreliable
    # for ranging -> use it only to servo-descend, not to steer.
    # Large area = gate is close and definitely real; skip the low/clipped-band filter.
    reliable = (area >= MIN_CONTOUR_AREA) and not (touches_tb and off_centre_y and area < 5000)

    estimate = {
        'cx':        cx,
        'cy':        cy,
        'cx_offset': cx - CX,
        'cy_offset': cy - CY,
        'bbox_x': bx, 'bbox_y': by,
        'bbox_w': bw, 'bbox_h': bh,
        'area':      area,
        'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
        'reliable':  reliable,
        'pnp_ok':    False,
    }

    # PnP: extract corners + solve for full 6-DoF gate-relative pose.
    # Uses 8 points (outer + inner) when the inner hole is detectable; falls
    # back to 4-point IPPE when it isn't. Falls back to bbox silently.
    if reliable:
        outer_corners, inner_corners = _extract_gate_corners(img, best, mask)
        if outer_corners is not None:
            tvec, rvec, reproj = _solve_gate_pnp(outer_corners)
            if tvec is not None and reproj < PNP_MAX_REPROJ_PX:
                if inner_corners is not None:
                    tvec8, rvec8, reproj8 = _refine_pnp_8pt(
                        outer_corners, inner_corners,
                        rvec.reshape(3, 1), tvec.reshape(3, 1))
                    if tvec8 is not None and reproj8 < PNP_MAX_REPROJ_PX:
                        tvec, rvec, reproj = tvec8, rvec8, reproj8
                        estimate['pnp_8pt'] = True
                estimate['pnp_ok']      = True
                estimate['pnp_tvec']    = tvec
                estimate['pnp_rvec']    = rvec
                estimate['pnp_reproj']  = reproj
                estimate['pnp_corners'] = outer_corners

    return estimate, mask, contours


def ema_smooth(prev_bbox, estimate, alpha=EMA_ALPHA):
    """
    Temporally smooth the gate bbox. `prev_bbox` is the previous SMOOTHED
    (bx,by,bw,bh) or None; `estimate` is the current detect_gate() output (or None).
    Returns (smoothed_estimate, new_prev_bbox).

    Pure function (no instance state) so the live VisionRX and any offline replay
    smooth IDENTICALLY. Resets — returns the raw frame untouched — on a detection gap
    (estimate None) or a big jump (centre or width), so a gate SWITCH (pass one gate,
    acquire the next) is never blended into a phantom in-between gate.
    """
    if estimate is None:
        return None, None
    cur = (estimate['bx'], estimate['by'], estimate['bw'], estimate['bh'])
    if prev_bbox is None:
        sm = cur
    else:
        ccx = cur[0] + cur[2] / 2.0
        pcx = prev_bbox[0] + prev_bbox[2] / 2.0
        wr = (cur[2] / prev_bbox[2]) if prev_bbox[2] else 99.0
        if abs(ccx - pcx) > EMA_RESET_DCX or wr > EMA_RESET_WR or wr < 1.0 / EMA_RESET_WR:
            sm = cur                                  # discontinuity -> reset (new gate)
        else:
            sm = tuple(alpha * c + (1.0 - alpha) * p for c, p in zip(cur, prev_bbox))
    bx, by, bw, bh = sm
    out = dict(estimate)
    out['bx'], out['by'], out['bw'], out['bh'] = bx, by, bw, bh
    out['cx'] = bx + bw / 2.0
    out['cy'] = by + bh / 2.0
    out['cx_offset'] = out['cx'] - CX
    out['cy_offset'] = out['cy'] - CY
    return out, sm


class VisionRX:

    def __init__(self, data):
        self.data       = data
        self.is_running = True
        # Instrumentation state (diagnostics only — does not affect detection output).
        self._vlog       = None
        self._proc_count = 0
        self._dump_sets  = 0
        self._ema_prev   = None     # previous smoothed bbox (EMA state)
        self._pnp_ema    = None     # (tvec, gate_id) — PnP tvec EMA state
        self._prev_bxyz = None   # (bx, by, bz) from previous frame for velocity derivative
        # Gate-passthrough suppression state
        self._in_gate         = False  # True while gate area > threshold (inside gate)
        self._pass_cooldown   = 0      # frames left to suppress after passing through
        # Clear last run's dumped frames so vision_dump/ always matches the
        # freshly-rewritten vision_log.csv. Frame IDs restart at 0 each run, so
        # leftover frames would silently bind to the WRONG run's pose and corrupt
        # offline scoring (cv_score.py). Diagnostics-only; never affects detection.
        if INSTRUMENT:
            self._clear_dump_dir()
        self.thread     = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # ------------------------------------------------------------------
    # UDP receive and frame reassembly
    # ------------------------------------------------------------------

    def _vision_loop(self):
        header_fmt  = '<IHHIIQ'
        header_size = struct.calcsize(header_fmt)   # 24 bytes per spec
        frames      = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        sock.settimeout(SOCKET_TIMEOUT_S)
        print('Listening for camera frames...', flush=True)

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue

            header  = packet[:header_size]
            payload = packet[header_size:]

            (frame_id, chunk_id, total_chunks,
             jpeg_size, payload_size, sim_time_ns) = struct.unpack(header_fmt, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    'chunks': {},
                    'total':  total_chunks,
                    'size':   jpeg_size,
                    'time':   sim_time_ns,
                }

            frames[frame_id]['chunks'][chunk_id] = payload

            if len(frames[frame_id]['chunks']) == total_chunks:
                jpeg_bytes = bytearray()
                complete   = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]['chunks']:
                        complete = False
                        break
                    jpeg_bytes.extend(frames[frame_id]['chunks'][i])

                if complete:
                    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if image is not None:
                        self.process_frame(frame_id, image)
                    else:
                        print(f'Failed to decode frame {frame_id}', flush=True)

                del frames[frame_id]

            # Prune stale partial frames to prevent memory growth
            stale = [fid for fid in list(frames.keys())
                     if fid < frame_id - FRAME_BUFFER_DEPTH]
            for fid in stale:
                del frames[fid]

    # ------------------------------------------------------------------
    # Gate detection
    # ------------------------------------------------------------------

    # Physical velocity cap — readings above this indicate a bad PnP frame.
    _MAX_GATE_VEL_MPS = 25.0

    def _gate_velocity(self, estimate):
        """
        Estimate body-frame velocity from the rate of change of the gate's
        body-frame position. The gate is stationary, so:
            vX = -d(bx)/dt,  vY = -d(by)/dt,  vZ = -d(bz)/dt

        Uses the raw (pre-EMA) PnP tvec when available so the EMA lag on
        the navigation estimate doesn't attenuate the velocity signal.
        Falls back to the smoothed body pose when PnP is unavailable.
        Readings exceeding _MAX_GATE_VEL_MPS are rejected as PnP jumps.
        Publishes to data['vision_velocity']; sets it to None when invalid.
        """
        if estimate is None:
            self._prev_bxyz = None
            self.data['vision_velocity'] = None
            return

        # Raw tvec → body pose (bypasses EMA lag on the smoothed estimate).
        if estimate.get('pnp_ok') and estimate.get('pnp_tvec_raw') is not None:
            tmp = {**estimate, 'pnp_tvec': estimate['pnp_tvec_raw']}
            body_relative_pose(tmp)
            bx = tmp.get('body_x_m', float('nan'))
            by = tmp.get('body_y_m', float('nan'))
            bz = tmp.get('body_z_m', float('nan'))
        else:
            bx = estimate.get('body_x_m', float('nan'))
            by = estimate.get('body_y_m', float('nan'))
            bz = estimate.get('body_z_m', float('nan'))

        if any(math.isnan(v) for v in (bx, by, bz)) or bx <= 0.1:
            self._prev_bxyz = None
            self.data['vision_velocity'] = None
            return

        if self._prev_bxyz is not None:
            dt  = 1.0 / _GATE_VEL_FPS
            pbx, pby, pbz = self._prev_bxyz
            vx = -(bx - pbx) / dt
            vy = -(by - pby) / dt
            vz = -(bz - pbz) / dt
            # Reject frames where PnP jumped — physically impossible readings.
            if max(abs(vx), abs(vy), abs(vz)) > self._MAX_GATE_VEL_MPS:
                self._prev_bxyz = None
                self.data['vision_velocity'] = None
                return
            self.data['vision_velocity'] = {
                'vx_body_mps': round(vx, 2),
                'vy_body_mps': round(vy, 2),
                'vz_body_mps': round(vz, 2),
            }

        self._prev_bxyz = (bx, by, bz)

    def process_frame(self, frame_id, img):
        """
        Detect the red gate frame in the FPV image and write the centroid
        offset from image centre into shared_data['vision_gate_estimate'].

        The controller reads this each tick and applies a small lateral /
        vertical velocity nudge when on close approach.

        Stored convention:
          cx_offset > 0  →  gate is to the RIGHT of image centre
          cy_offset > 0  →  gate is BELOW image centre  (image Y-down)
        """
        # Detection logic lives in module-level detect_gate() so the offline replay
        # harness runs the EXACT same pipeline (tuning transfers to flight).
        estimate, mask, contours = detect_gate(img)

        # Diagnostics only — runs every frame, does not touch the raw detection.
        if INSTRUMENT:
            self._instrument(frame_id, img, mask, contours)

        # ---- Gate passthrough suppression ----------------------------------------
        # When the drone enters a gate's close-range zone (contour area grows very
        # large) the centroid is useless for steering.  Suppress the published
        # estimate while inside the gate, then hold that suppression for a short
        # cooldown after the area drops — preventing the controller from locking
        # back onto the gate it just flew through.
        #
        # Threshold 12000 px² ≈ gate width ~110 px ≈ range ~8 m.  Cooldown 25 frames
        # ≈ 0.8 s at 30 fps; at 4 m/s that's ~3 m of forward travel after passthrough.
        _PASS_AREA_PX     = 12000
        _PASS_COOLDOWN_FR = 25

        _raw_area = estimate['area'] if estimate is not None else 0

        if _raw_area > _PASS_AREA_PX:
            # Drone is inside / immediately in front of the gate.
            if not self._in_gate:
                self._in_gate = True   # transition: entered gate zone
            self._prev_bxyz = None
            self.data['vision_gate_estimate'] = None
            self.data['vision_velocity'] = None
            return None

        if self._in_gate:
            # Gate area just dropped — drone passed through.  Start cooldown and
            # reset EMA so the next gate starts tracking from scratch.
            self._in_gate       = False
            self._pass_cooldown = _PASS_COOLDOWN_FR
            self._ema_prev      = None
            self._pnp_ema       = None
            self._prev_bxyz     = None

        if self._pass_cooldown > 0:
            self._pass_cooldown -= 1
            self.data['vision_gate_estimate'] = None
            self.data['vision_velocity'] = None
            return None
        # --------------------------------------------------------------------------

        # Temporal smoothing of the PUBLISHED estimate (steadies the gate centre the
        # controller chases). State on the instance; ema_smooth resets it on gaps /
        # gate-switches. Instrumentation above logs the RAW detection, so offline
        # scoring still sees unsmoothed detections.
        estimate, self._ema_prev = ema_smooth(self._ema_prev, estimate)

        # PnP tvec EMA: light smoothing between frames when the same gate is tracked.
        # Resets on gate-switch (target_gate changes) or when PnP is not available.
        # Alpha=0.4 → heavier weight on new measurement than bbox EMA (0.5) because
        # PnP is already accurate enough that smoothing is just for jitter, not drift.
        PNP_EMA_ALPHA = 0.4
        if estimate is not None and estimate.get('pnp_ok'):
            cur_gate = estimate.get('target_gate')
            new_tvec = estimate['pnp_tvec'].copy()
            estimate['pnp_tvec_raw'] = new_tvec   # preserved for velocity derivative
            if (self._pnp_ema is not None
                    and self._pnp_ema[1] == cur_gate):
                prev_tvec = self._pnp_ema[0]
                smoothed  = PNP_EMA_ALPHA * new_tvec + (1 - PNP_EMA_ALPHA) * prev_tvec
                estimate['pnp_tvec'] = smoothed
            self._pnp_ema = (estimate['pnp_tvec'].copy(), cur_gate)
        else:
            self._pnp_ema = None

        # Add telemetry-free body-relative pose fields to the smoothed estimate.
        # These allow Round-2 controllers to steer without attitude/position data.
        body_relative_pose(estimate)

        # Velocity from gate-position derivative (vX, vY, vZ all three axes).
        self._gate_velocity(estimate)

        if estimate is None:
            self.data['vision_gate_estimate'] = None
            return None

        estimate['frame_id'] = frame_id
        self.data['vision_gate_estimate'] = estimate
        return estimate['cx'], estimate['cy']

    # ------------------------------------------------------------------
    # Instrumentation (diagnostics only, no behaviour change)
    # ------------------------------------------------------------------

    def _clear_dump_dir(self):
        """
        Delete prior runs' raw frames from DUMP_DIR at startup so the dump stays
        self-consistent with this run's vision_log.csv (frame IDs restart at 0 each
        run). Only removes our own '*_raw.jpg' artifacts. Wrapped so a filesystem
        error can never stop the vision pipeline from starting.
        """
        try:
            if not os.path.isdir(DUMP_DIR):
                return
            removed = 0
            for name in os.listdir(DUMP_DIR):
                if name.endswith('_raw.jpg'):
                    try:
                        os.remove(os.path.join(DUMP_DIR, name))
                        removed += 1
                    except OSError:
                        pass
            if removed:
                print(f'[VISION-INSTR] cleared {removed} old raw frame(s) from '
                      f'{DUMP_DIR}/', flush=True)
        except Exception as e:
            print(f'[VISION-INSTR] dump-clear non-fatal: {e}', flush=True)

    def _instrument(self, frame_id, img, mask, contours):
        """
        Record per-frame detection diagnostics so detection can be evaluated from real
        data. Writes vision_log.csv (per-frame contour count + the largest contour's
        area, aspect = bbox w/h, fill_ratio = red mask px / bbox area [hollow gate ->
        low, solid blob -> high], centroid + offsets, bbox) and, every DUMP_EVERY_N
        frames, an image trio to vision_dump/ (raw, HSV mask, overlay with all
        above-floor contours + the largest bbox/centroid + image-centre cross).
        Wrapped in try/except so a disk error can never kill the vision thread.
        """
        try:
            self._proc_count += 1

            stats = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < INSTR_AREA_FLOOR:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                m = cv2.moments(c)
                if m['m00'] == 0.0:
                    continue
                ccx = m['m10'] / m['m00']
                ccy = m['m01'] / m['m00']
                aspect = (w / h) if h else 0.0
                roi = mask[y:y + h, x:x + w]
                fill = (cv2.countNonZero(roi) / float(w * h)) if (w and h) else 0.0
                stats.append((area, x, y, w, h, ccx, ccy, aspect, fill))

            # Drone pose at this frame — the one missing ingredient for offline
            # auto-scoring (project a known gate's 3D NED position into this image and
            # compare to the detection). Read lock-free; odometry is replaced wholesale
            # so the snapshot is self-consistent. Pure diagnostics, no flight effect.
            odo = self.data.get('odometry') or {}
            pose = [odo.get('x'), odo.get('y'), odo.get('z'),
                    odo.get('qw'), odo.get('qx'), odo.get('qy'), odo.get('qz')]
            pose = ['' if v is None else f'{v:.5f}' for v in pose]

            if self._vlog is None:
                self._vlog = open('vision_log.csv', 'w', buffering=1)
                self._vlog.write('frame_id,n_contours,best_area,best_aspect,best_fill,'
                                 'best_cx,best_cy,best_cx_off,best_cy_off,'
                                 'best_bx,best_by,best_bw,best_bh,'
                                 'dx,dy,dz,qw,qx,qy,qz\n')
            if stats:
                area, x, y, w, h, ccx, ccy, aspect, fill = max(stats, key=lambda s: s[0])
                best = [frame_id, len(stats), f'{area:.0f}', f'{aspect:.3f}', f'{fill:.3f}',
                        f'{ccx:.1f}', f'{ccy:.1f}', f'{ccx - CX:.1f}', f'{ccy - CY:.1f}',
                        x, y, w, h]
            else:
                best = [frame_id, 0, '', '', '', '', '', '', '', '', '', '', '']
            self._vlog.write(','.join(str(v) for v in best + pose) + '\n')

            if (self._proc_count % DUMP_EVERY_N == 0
                    and self._dump_sets < DUMP_MAX_SETS):
                os.makedirs(DUMP_DIR, exist_ok=True)
                # Dump RAW frames only. Masks/overlays are fully regenerable offline
                # from raw via cv_replay.py, so writing them during flight is wasted
                # I/O and clutter (and needed manual cleanup). Raw is the source data.
                cv2.imwrite(os.path.join(DUMP_DIR, f'{frame_id:06d}_raw.jpg'), img)
                self._dump_sets += 1
        except Exception as e:
            print(f'[VISION-INSTR] non-fatal: {e}', flush=True)