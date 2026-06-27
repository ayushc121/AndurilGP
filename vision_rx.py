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
# Gate colour — red on dark/grey background.
# Red wraps around the HSV hue circle in OpenCV (0–180 scale), so two
# ranges are needed: one near 0° and one near 180°.
# --------------------------------------------------------------------------------------
LOWER_RED_1 = np.array([0,   120,  50])
UPPER_RED_1 = np.array([10,  255, 255])

LOWER_RED_2 = np.array([170, 120,  50])
UPPER_RED_2 = np.array([180, 255, 255])

# Minimum contour area (px²) to be considered a gate detection
MIN_CONTOUR_AREA = 800

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
    bw = estimate.get('bw', 0)
    cx = estimate.get('cx', CX)
    cy = estimate.get('cy', CY)

    if not bw:
        nan = float('nan')
        estimate.update(cam_x_m=nan, cam_y_m=nan, cam_z_m=nan,
                        body_x_m=nan, body_y_m=nan, body_z_m=nan)
        return estimate

    # Depth from known gate width via pinhole.
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

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
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
    reliable = (area >= MIN_CONTOUR_AREA) and not (touches_tb and off_centre_y)

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
    }
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

        # Temporal smoothing of the PUBLISHED estimate (steadies the gate centre the
        # controller chases). State on the instance; ema_smooth resets it on gaps /
        # gate-switches. Instrumentation above logs the RAW detection, so offline
        # scoring still sees unsmoothed detections.
        estimate, self._ema_prev = ema_smooth(self._ema_prev, estimate)

        # Add telemetry-free body-relative pose fields to the smoothed estimate.
        # These allow Round-2 controllers to steer without attitude/position data.
        body_relative_pose(estimate)

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