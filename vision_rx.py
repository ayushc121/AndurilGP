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


class VisionRX:

    def __init__(self, data):
        self.data       = data
        self.is_running = True
        # Instrumentation state (diagnostics only — does not affect detection output).
        self._vlog       = None
        self._proc_count = 0
        self._dump_sets  = 0
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red wraps around the hue circle — combine both ranges
        mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Morphological clean-up: close gaps in the gate frame, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Diagnostics only — runs every frame, does not touch the published estimate.
        if INSTRUMENT:
            self._instrument(frame_id, img, mask, contours)

        valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if not valid:
            self.data['vision_gate_estimate'] = None
            return None

        best    = max(valid, key=cv2.contourArea)
        moments = cv2.moments(best)
        if moments['m00'] == 0.0:
            self.data['vision_gate_estimate'] = None
            return None

        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        bx, by, bw, bh = cv2.boundingRect(best)

        self.data['vision_gate_estimate'] = {
            'cx':        cx,
            'cy':        cy,
            'cx_offset': cx - CX,   # positive = gate right of centre
            'cy_offset': cy - CY,   # positive = gate below centre
            'bbox_x': bx, 'bbox_y': by,
            'bbox_w': bw, 'bbox_h': bh,
            'area':      cv2.contourArea(best),
            'frame_id':  frame_id,
            'bx':        bx,
            'by':        by,
            'bw':        bw,
            'bh':        bh
        }

        return cx, cy

    # ------------------------------------------------------------------
    # Instrumentation (diagnostics only, no behaviour change)
    # ------------------------------------------------------------------

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

            if self._vlog is None:
                self._vlog = open('vision_log.csv', 'w', buffering=1)
                self._vlog.write('frame_id,n_contours,best_area,best_aspect,best_fill,'
                                 'best_cx,best_cy,best_cx_off,best_cy_off,'
                                 'best_bx,best_by,best_bw,best_bh\n')
            if stats:
                area, x, y, w, h, ccx, ccy, aspect, fill = max(stats, key=lambda s: s[0])
                self._vlog.write(
                    f'{frame_id},{len(stats)},{area:.0f},{aspect:.3f},{fill:.3f},'
                    f'{ccx:.1f},{ccy:.1f},{ccx - CX:.1f},{ccy - CY:.1f},'
                    f'{x},{y},{w},{h}\n')
            else:
                self._vlog.write(f'{frame_id},0,,,,,,,,,,,\n')

            if (self._proc_count % DUMP_EVERY_N == 0
                    and self._dump_sets < DUMP_MAX_SETS):
                os.makedirs(DUMP_DIR, exist_ok=True)
                tag = f'{frame_id:06d}'
                cv2.imwrite(os.path.join(DUMP_DIR, f'{tag}_raw.jpg'), img)
                cv2.imwrite(os.path.join(DUMP_DIR, f'{tag}_mask.png'), mask)
                overlay = img.copy()
                for (area, x, y, w, h, ccx, ccy, aspect, fill) in stats:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if stats:
                    area, x, y, w, h, ccx, ccy, aspect, fill = max(stats, key=lambda s: s[0])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.circle(overlay, (int(ccx), int(ccy)), 4, (255, 0, 0), -1)
                    cv2.putText(overlay, f'A{area:.0f} ar{aspect:.2f} f{fill:.2f}',
                                (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 1)
                cv2.drawMarker(overlay, (int(CX), int(CY)), (0, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
                cv2.imwrite(os.path.join(DUMP_DIR, f'{tag}_overlay.jpg'), overlay)
                self._dump_sets += 1
        except Exception as e:
            print(f'[VISION-INSTR] non-fatal: {e}', flush=True)