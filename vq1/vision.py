"""
The first gate detector — HSV threshold, largest red contour, bounding box.

This is where the perception work started. VQ1 published true gate positions,
so the detector could run live on every flight and be checked against ground
truth without ever risking a run: the controller back-projects this bbox into a
world-frame gate position, then overwrites it with telemetry whenever telemetry
is available.

It has no pose solve, no temporal smoothing and no rejection rules. Everything
`vq2/gate_detector.py` grew was a response to something that broke here.
"""

import cv2
import numpy as np

from core.camera_rx import CameraRX

IMG_W, IMG_H = 640, 360
CX, CY = 320.0, 180.0
FX = FY = 320.0

# Red straddles the hue wrap in OpenCV's 0-180 scale, so it takes two ranges.
LOWER_RED_1 = np.array([0, 120, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 50])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 800    # VQ2 later dropped this to 300 to catch distant gates
MORPH_KERNEL = 7          # and to 3x3, once 7x7 was found to erase far gates


def detect_gate(img):
    """Largest red blob in a BGR frame -> bbox and centroid, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
                          cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not candidates:
        return None

    best = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(best)
    if moments['m00'] == 0.0:
        return None

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    bx, by, bw, bh = cv2.boundingRect(best)
    return {
        'cx': cx, 'cy': cy,
        'cx_offset': cx - CX,    # positive = gate right of centre
        'cy_offset': cy - CY,    # positive = gate below centre (image Y-down)
        'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
        'area': cv2.contourArea(best),
    }


class VisionRX(CameraRX):
    """Publishes each frame's detection to shared_data['vision_gate_estimate']."""

    def process_frame(self, frame_id, img):
        estimate = detect_gate(img)
        if estimate is not None:
            estimate['frame_id'] = frame_id
        self.data['vision_gate_estimate'] = estimate
        return estimate
