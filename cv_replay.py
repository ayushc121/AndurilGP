#!/usr/bin/env python3
"""
cv_replay.py — OFFLINE gate-detection replay harness.

Purpose: tune the computer vision WITHOUT flying the sim. Runs the EXACT same
detection as the live client (vision_rx.detect_gate) over previously captured
frames in vision_dump/, so you can tweak thresholds/params and see the effect on
hundreds of real frames instantly.

Workflow:
  1. Fly once (the client writes raw frames to vision_dump/*_raw.jpg).
  2. Edit detection params in vision_rx.py (HSV ranges, MIN_CONTOUR_AREA, morph).
  3. Run:  python cv_replay.py
  4. Inspect vision_replay/ overlays + the printed summary. Repeat 2-4 freely.

Because it imports vision_rx.detect_gate, whatever you tune here is exactly what
flies — no drift between offline and live.

Usage:
  python cv_replay.py                # replay all frames in vision_dump/
  python cv_replay.py <dir>          # replay raw frames from another directory
  python cv_replay.py --no-images    # summary + CSV only, skip writing overlays
"""

import os
import sys
import glob

import cv2

import vision_rx as V

DUMP_DIR = 'vision_dump'
OUT_DIR  = 'vision_replay'


def make_overlay(img, estimate, contours):
    ov = img.copy()
    for c in contours:
        if cv2.contourArea(c) >= V.INSTR_AREA_FLOOR:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if estimate is not None:
        bx, by, bw, bh = estimate['bx'], estimate['by'], estimate['bw'], estimate['bh']
        cx = int(bx + bw / 2.0)
        cy = int(by + bh / 2.0)
        cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.circle(ov, (cx, cy), 4, (255, 0, 0), -1)
        # crude range estimate from the same pinhole the controller uses
        rng = (V.FX * 2.7) / bw if bw > 0 else -1
        cv2.putText(ov, f"A{estimate['area']:.0f} bw{bw} d~{rng:.1f}m",
                    (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    cv2.drawMarker(ov, (int(V.CX), int(V.CY)), (0, 255, 255),
                   cv2.MARKER_CROSS, 16, 1)
    return ov


def main():
    args = [a for a in sys.argv[1:]]
    write_images = '--no-images' not in args
    dirs = [a for a in args if not a.startswith('--')]
    dump_dir = dirs[0] if dirs else DUMP_DIR

    frames = sorted(glob.glob(os.path.join(dump_dir, '*_raw.jpg')))
    if not frames:
        print(f'No *_raw.jpg frames found in {dump_dir}/ — fly once to capture some.')
        return

    if write_images:
        os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = 'cv_replay.csv'
    csv = open(csv_path, 'w', buffering=1)
    csv.write('frame,detected,n_contours,area,bw,bh,cx_off,cy_off,est_range_m\n')

    n = 0
    n_det = 0
    areas = []
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        n += 1
        estimate, mask, contours = V.detect_gate(img)
        nc = sum(1 for c in contours if cv2.contourArea(c) >= V.INSTR_AREA_FLOOR)
        tag = os.path.splitext(os.path.basename(f))[0].replace('_raw', '')

        if estimate is not None:
            n_det += 1
            bw = estimate['bw']
            rng = (V.FX * 2.7) / bw if bw > 0 else -1
            areas.append(estimate['area'])
            csv.write(f"{tag},1,{nc},{estimate['area']:.0f},{bw},{estimate['bh']},"
                      f"{estimate['cx_offset']:.1f},{estimate['cy_offset']:.1f},{rng:.2f}\n")
        else:
            csv.write(f'{tag},0,{nc},,,,,,\n')

        if write_images:
            cv2.imwrite(os.path.join(OUT_DIR, f'{tag}_replay.jpg'),
                        make_overlay(img, estimate, contours))

    csv.close()
    print(f'Replayed {n} frames from {dump_dir}/')
    print(f'  detected gate in {n_det}/{n} ({100.0*n_det/max(n,1):.0f}%)')
    if areas:
        areas.sort()
        print(f'  detected area px^2: min {areas[0]:.0f} / '
              f'median {areas[len(areas)//2]:.0f} / max {areas[-1]:.0f}')
    print(f'  per-frame CSV -> {csv_path}')
    if write_images:
        print(f'  overlays      -> {OUT_DIR}/')
    print('\nCurrent params (edit in vision_rx.py, then re-run):')
    print(f'  LOWER_RED_1={V.LOWER_RED_1.tolist()} UPPER_RED_1={V.UPPER_RED_1.tolist()}')
    print(f'  LOWER_RED_2={V.LOWER_RED_2.tolist()} UPPER_RED_2={V.UPPER_RED_2.tolist()}')
    print(f'  MIN_CONTOUR_AREA={V.MIN_CONTOUR_AREA}')


if __name__ == '__main__':
    main()
