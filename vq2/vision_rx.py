"""
Runs each camera frame through `gate_detector` and publishes into
`shared_data`. Transport is inherited from `core.camera_rx`; the perception
maths lives in `gate_detector`. What remains here is temporal state.

Publishes `vision_gate_estimate` (smoothed detection with body-relative gate
position) and `vision_velocity`, both None when no gate is trustworthy.
"""

import math
import os

import cv2

from core.camera_rx import CameraRX
from . import gate_detector as gd

CAMERA_FPS = 30.0

# Passthrough suppression. Contour area is ~746000/range^2, so 80000 px means
# ~3.1 m, where the gate is 288 px across in a 360 px frame. Cooldown ~0.33 s.
PASS_AREA_PX = 80000
PASS_COOLDOWN_FRAMES = 10

PNP_EMA_ALPHA = 0.4    # jitter suppression on the PnP translation

# Diagnostics feeding the offline tooling in analysis/. No effect on flight.
INSTRUMENT = True
DUMP_DIR = 'vision_dump'
DUMP_EVERY_N = 15
DUMP_MAX_FRAMES = 400
LOG_PATH = 'vision_log.csv'
LOG_CONTOUR_FLOOR = 100       # log contours below the detection threshold too
LOG_HEADER = ('frame_id,n_contours,area,aspect,fill,cx,cy,cx_off,cy_off,'
              'bx,by,bw,bh,dx,dy,dz,qw,qx,qy,qz\n')


class VisionRX(CameraRX):
    """Owns the per-frame temporal state: smoothing, passthrough, velocity."""

    def __init__(self, data):
        self._bbox_ema = None      # previous smoothed bbox
        self._pnp_ema = None       # previous smoothed PnP translation
        self._prev_body_pos = None # previous body-frame gate position
        self._in_gate = False
        self._cooldown = 0

        self._log = None
        self._frames_seen = 0
        self._frames_dumped = 0
        if INSTRUMENT:
            self._clear_dump_dir()

        super().__init__(data)     # starts the receive thread — must be last

    # ------------------------------------------------------------- perception

    def process_frame(self, frame_id, img):
        """Detect, smooth, and publish. Returns the published estimate or None."""
        estimate, mask, contours = gd.detect_gate(img)

        if INSTRUMENT:
            self._log_frame(frame_id, img, mask, contours)

        if self._suppress_passthrough(estimate):
            return None

        estimate, self._bbox_ema = gd.ema_smooth(self._bbox_ema, estimate)
        self._smooth_pnp(estimate)
        gd.body_relative_pose(estimate)
        self._publish_velocity(estimate)

        if estimate is None:
            self.data['vision_gate_estimate'] = None
            return None

        estimate['frame_id'] = frame_id
        self.data['vision_gate_estimate'] = estimate
        return estimate

    def _suppress_passthrough(self, estimate):
        """
        Blank the estimate while crossing a gate; True means stop here. On exit
        the cooldown and EMA reset make the next gate acquire from scratch.
        """
        area = estimate['area'] if estimate is not None else 0

        if area > PASS_AREA_PX:
            self._in_gate = True
            self._blank()
            return True

        if self._in_gate:
            self._in_gate = False
            self._cooldown = PASS_COOLDOWN_FRAMES
            self._bbox_ema = None
            self._pnp_ema = None

        if self._cooldown > 0:
            self._cooldown -= 1
            self._blank()
            return True

        return False

    def _blank(self):
        self._prev_body_pos = None
        self.data['vision_gate_estimate'] = None
        self.data['vision_velocity'] = None

    def _smooth_pnp(self, estimate):
        """
        Smooth the PnP translation, keeping the raw value: velocity is
        differenced from the unsmoothed pose, or the EMA attenuates the signal
        being measured.
        """
        if estimate is None or not estimate.get('pnp_ok'):
            self._pnp_ema = None
            return

        raw = estimate['pnp_tvec'].copy()
        estimate['pnp_tvec_raw'] = raw
        if self._pnp_ema is not None:
            estimate['pnp_tvec'] = PNP_EMA_ALPHA * raw + (1 - PNP_EMA_ALPHA) * self._pnp_ema
        self._pnp_ema = estimate['pnp_tvec'].copy()

    def _publish_velocity(self, estimate):
        """
        The gate is stationary, so the drone's velocity is the negated rate of
        change of the gate's body-relative position.
        """
        if estimate is None:
            self._prev_body_pos = None
            self.data['vision_velocity'] = None
            return

        pose = estimate
        if estimate.get('pnp_ok') and estimate.get('pnp_tvec_raw') is not None:
            pose = gd.body_relative_pose(
                {**estimate, 'pnp_tvec': estimate['pnp_tvec_raw']})

        position = (pose.get('body_x_m', float('nan')),
                    pose.get('body_y_m', float('nan')),
                    pose.get('body_z_m', float('nan')))
        if any(math.isnan(v) for v in position) or position[0] <= 0.1:
            self._prev_body_pos = None
            self.data['vision_velocity'] = None
            return

        if self._prev_body_pos is not None:
            dt = 1.0 / CAMERA_FPS
            vx, vy, vz = (-(now - was) / dt
                          for now, was in zip(position, self._prev_body_pos))
            self.data['vision_velocity'] = {'vx_body_mps': round(vx, 2),
                                            'vy_body_mps': round(vy, 2),
                                            'vz_body_mps': round(vz, 2)}
        self._prev_body_pos = position

    # ------------------------------------------------------------ diagnostics

    def _clear_dump_dir(self):
        """
        Frame IDs restart at zero each run, so leftover frames would bind to
        the wrong run's pose in vision_log.csv and corrupt offline scoring.
        """
        if not os.path.isdir(DUMP_DIR):
            return
        for name in os.listdir(DUMP_DIR):
            if name.endswith('_raw.jpg'):
                try:
                    os.remove(os.path.join(DUMP_DIR, name))
                except OSError:
                    pass

    def _log_frame(self, frame_id, img, mask, contours):
        """
        One row of detection diagnostics, plus a raw frame every N.

        The pose columns are what make offline scoring possible, and only
        populate on a VQ1 flight — VQ2 is what blocks odometry. See
        `analysis/collect_ground_truth.py`.
        """
        try:
            self._frames_seen += 1
            best = self._largest_contour_stats(contours, mask)

            odo = self.data.get('odometry') or {}
            pose = ['' if odo.get(k) is None else f'{odo[k]:.5f}'
                    for k in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')]

            if self._log is None:
                self._log = open(LOG_PATH, 'w', buffering=1)
                self._log.write(LOG_HEADER)
            self._log.write(','.join(str(v) for v in [frame_id] + best + pose) + '\n')

            if (self._frames_seen % DUMP_EVERY_N == 0
                    and self._frames_dumped < DUMP_MAX_FRAMES):
                os.makedirs(DUMP_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(DUMP_DIR, f'{frame_id:06d}_raw.jpg'), img)
                self._frames_dumped += 1
        except (OSError, ValueError) as exc:
            print(f'[vision] diagnostics failed (non-fatal): {exc}', flush=True)

    @staticmethod
    def _largest_contour_stats(contours, mask):
        """
        Shape metrics for the largest contour. `fill` is the red fraction of
        the bbox: low for a hollow gate, high for a solid blob.
        """
        stats = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < LOG_CONTOUR_FLOOR:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments['m00'] == 0.0 or not (w and h):
                continue
            roi = mask[y:y + h, x:x + w]
            stats.append((area, x, y, w, h,
                          moments['m10'] / moments['m00'],
                          moments['m01'] / moments['m00'],
                          w / h,
                          cv2.countNonZero(roi) / float(w * h)))

        if not stats:
            return [0] + [''] * 11

        area, x, y, w, h, cx, cy, aspect, fill = max(stats, key=lambda s: s[0])
        return [len(stats), f'{area:.0f}', f'{aspect:.3f}', f'{fill:.3f}',
                f'{cx:.1f}', f'{cy:.1f}', f'{cx - gd.CX:.1f}', f'{cy - gd.CY:.1f}',
                x, y, w, h]
