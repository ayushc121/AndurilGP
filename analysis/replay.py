"""
Re-run gate detection over recorded frames — a tuning loop with no simulator.
Change a threshold in `gate_detector`, replay, watch the rate move. The live
pipeline calls the same functions, so an improvement here transfers.

    python -m analysis.replay [frame_dir] [out_dir]

Writes replay.csv plus an annotated copy of each frame.
"""

import csv
import glob
import os
import sys

import cv2

from vq2 import gate_detector as gd

DEFAULT_FRAMES = 'analysis/data/frames'
DEFAULT_OUT = 'vision_replay'


def annotate(img, estimate, contours):
    """Draw the detection over a copy of the frame: mask contours, bbox, centre."""
    canvas = img.copy()
    cv2.drawContours(canvas, [c for c in contours if cv2.contourArea(c) >= 100],
                     -1, (0, 255, 255), 1)      # BGR: yellow
    cv2.drawMarker(canvas, (int(gd.CX), int(gd.CY)), (255, 255, 255),
                   cv2.MARKER_CROSS, 14, 1)

    if estimate is None:
        cv2.putText(canvas, 'no detection', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return canvas

    x, y = int(estimate['bx']), int(estimate['by'])
    w, h = int(estimate['bw']), int(estimate['bh'])
    colour = (0, 255, 0) if estimate['reliable'] else (0, 165, 255)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
    cv2.circle(canvas, (int(estimate['cx']), int(estimate['cy'])), 3, colour, -1)

    gd.body_relative_pose(estimate)
    source = 'pnp' if estimate.get('pnp_ok') else 'bbox'
    cv2.putText(canvas, f'{source} {estimate["body_x_m"]:.1f}m  area {estimate["area"]:.0f}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return canvas


def main(frame_dir=DEFAULT_FRAMES, out_dir=DEFAULT_OUT):
    frames = sorted(glob.glob(os.path.join(frame_dir, '*_raw.jpg')))
    if not frames:
        print(f'No frames in {frame_dir}/')
        return 1
    os.makedirs(out_dir, exist_ok=True)

    detected = reliable = solved = 0
    with open('replay.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'detected', 'reliable', 'pnp_ok',
                         'area', 'body_x_m', 'body_y_m', 'body_z_m'])

        for path in frames:
            name = os.path.basename(path)
            img = cv2.imread(path)
            estimate, _, contours = gd.detect_gate(img)
            cv2.imwrite(os.path.join(out_dir, name.replace('_raw', '_overlay')),
                        annotate(img, estimate, contours))

            if estimate is None:
                writer.writerow([name, 0, 0, 0, '', '', '', ''])
                continue

            detected += 1
            reliable += bool(estimate['reliable'])
            solved += bool(estimate.get('pnp_ok'))
            writer.writerow([name, 1, int(estimate['reliable']),
                             int(estimate.get('pnp_ok', False)),
                             f'{estimate["area"]:.0f}',
                             f'{estimate["body_x_m"]:.2f}',
                             f'{estimate["body_y_m"]:.2f}',
                             f'{estimate["body_z_m"]:.2f}'])

    total = len(frames)
    print(f'{total} frames from {frame_dir}/')
    print(f'  detected  {detected:>4}  ({100 * detected / total:.0f}%)')
    print(f'  reliable  {reliable:>4}  ({100 * reliable / total:.0f}%)')
    print(f'  PnP solve {solved:>4}  ({100 * solved / total:.0f}%)')
    print(f'\nOverlays in {out_dir}/, per-frame results in replay.csv')
    print(f'Thresholds: MIN_CONTOUR_AREA={gd.MIN_CONTOUR_AREA} '
          f'PNP_MAX_REPROJ_PX={gd.PNP_MAX_REPROJ_PX}')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
