"""Perception tests: detection, rejection rules, pose conversion, smoothing."""

import math

import cv2
import numpy as np
import pytest

from vq2 import gate_detector as gd


def test_no_gate_returns_none():
    """An empty scene must produce no estimate, not a low-confidence guess."""
    blank = np.full((gd.IMG_H, gd.IMG_W, 3), 40, dtype=np.uint8)
    estimate, _, _ = gd.detect_gate(blank)
    assert estimate is None


def _project(tvec, rvec=(0.0, 0.0, 0.0)):
    """Project the true gate square through the camera model to get pixel corners."""
    pts, _ = cv2.projectPoints(gd._OBJ_OUTER, np.array(rvec, dtype=np.float64),
                               np.array(tvec, dtype=np.float64),
                               gd._CAM_K, gd._CAM_DIST)
    return pts.reshape(4, 2).astype(np.float32)


@pytest.mark.parametrize('depth', [6.0, 14.0, 25.0])
def test_pnp_inverts_the_camera_model(depth):
    """Project the known gate, solve it back: any failure here is geometry."""
    truth = (0.0, 0.0, depth)
    tvec, _, reproj = gd.solve_pose(_project(truth))
    assert tvec is not None
    assert reproj < 1e-3
    assert tvec == pytest.approx(np.array(truth), abs=1e-3)


def test_pnp_recovers_lateral_offset():
    """A gate offset to the right must solve to a positive camera-frame x."""
    tvec, _, _ = gd.solve_pose(_project((1.5, 0.0, 12.0)))
    assert tvec[0] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.parametrize('true_range', [8.0, 15.0, 22.0])
def test_detector_ranges_a_rendered_gate(gate_image, true_range):
    """
    End-to-end range check. Tolerance is loose: a synthetic hard-edged square
    refines differently to a rendered one. This catches sign and scale errors;
    real accuracy is measured in analysis/.
    """
    estimate, _, _ = gd.detect_gate(gate_image(true_range))
    assert estimate is not None
    assert estimate['pnp_ok']
    gd.body_relative_pose(estimate)
    assert estimate['body_x_m'] == pytest.approx(true_range, rel=0.20)


def test_estimated_range_grows_with_true_range(gate_image):
    """Ordering must hold even where absolute accuracy is imperfect."""
    ranges = []
    for true_range in (6.0, 12.0, 24.0):
        estimate, _, _ = gd.detect_gate(gate_image(true_range))
        assert estimate is not None, f'no detection at {true_range} m'
        gd.body_relative_pose(estimate)
        ranges.append(estimate['body_x_m'])
    assert ranges[0] < ranges[1] < ranges[2]


def test_clipped_off_centre_sliver_rejected():
    """
    Passing through a gate leaves a sliver at one edge whose width implies a
    wildly wrong range — once, an 18 m climb command at the gate plane.
    """
    img = np.full((gd.IMG_H, gd.IMG_W, 3), 40, dtype=np.uint8)
    img[100:260, 0:45] = (40, 40, 220)
    estimate, _, _ = gd.detect_gate(img)
    assert estimate is None


def test_centred_gate_filling_frame_is_kept(gate_image):
    """Edge contact only disqualifies when off-centre: a close gate fills the frame."""
    estimate, _, _ = gd.detect_gate(gate_image(3.2))
    assert estimate is not None
    assert estimate['reliable']


def test_pinhole_fallback_matches_pnp_scale():
    """Without a PnP solve, bbox width alone must still give a sane range."""
    estimate = {'bw': 100, 'cx': gd.CX, 'cy': gd.CY, 'pnp_ok': False}
    gd.body_relative_pose(estimate)
    expected = gd.FX * gd.GATE_WIDTH_M / 100
    # Camera is tilted, so forward distance is the cos component of depth.
    assert estimate['body_x_m'] == pytest.approx(
        expected * math.cos(math.radians(gd.CAM_TILT_DEG)), rel=1e-6)


def test_degenerate_bbox_yields_nan_not_zero():
    """A zero-width bbox must produce NaN, which the controller rejects."""
    estimate = {'bw': 0, 'pnp_ok': False}
    gd.body_relative_pose(estimate)
    assert math.isnan(estimate['body_x_m'])


def test_camera_tilt_puts_frame_centre_above_horizon():
    """A gate at image centre sits above the drone: camera points 20 deg up."""
    estimate = {'bw': 100, 'cx': gd.CX, 'cy': gd.CY, 'pnp_ok': False}
    gd.body_relative_pose(estimate)
    assert estimate['body_z_m'] < 0    # NED: negative down means higher


def _bbox(bx, bw):
    return {'bx': bx, 'by': 100, 'bw': bw, 'bh': 100}


def test_ema_smooths_small_motion():
    """Frame-to-frame jitter should be averaged, not chased."""
    prev = (100, 100, 100, 100)
    out, new = gd.ema_smooth(prev, _bbox(110, 100))
    assert 100 < out['bx'] < 110
    assert new == (out['bx'], out['by'], out['bw'], out['bh'])


def test_ema_resets_on_gate_switch():
    """A large centre jump is a different gate; blending them invents a target."""
    out, _ = gd.ema_smooth((100, 100, 100, 100),
                           _bbox(100 + gd.EMA_RESET_DCX + 50, 100))
    assert out['bx'] == 100 + gd.EMA_RESET_DCX + 50


def test_ema_resets_on_large_width_change():
    """Same rule for a sudden scale change — that is an acquisition, not motion."""
    out, _ = gd.ema_smooth((100, 100, 100, 100),
                           _bbox(100, int(100 * gd.EMA_RESET_WR) + 10))
    assert out['bw'] == int(100 * gd.EMA_RESET_WR) + 10


def test_ema_gap_clears_state():
    """Losing the gate must clear the filter, not freeze a stale box."""
    assert gd.ema_smooth((100, 100, 100, 100), None) == (None, None)


def test_collinear_corners_produce_no_pose():
    """A collapsed quad has no valid pose; the solver must say so, not guess."""
    corners = np.array([[100, 100], [200, 100], [300, 100], [400, 100]],
                       dtype=np.float32)
    tvec, _, reproj = gd.solve_pose(corners)
    assert tvec is None or reproj >= gd.PNP_MAX_REPROJ_PX


def test_contour_below_the_area_floor_is_discarded():
    """Too small to range from is too small to report."""
    img = np.full((gd.IMG_H, gd.IMG_W, 3), 40, dtype=np.uint8)
    img[170:182, 300:322] = (40, 40, 220)       # ~250 px, under the 300 floor
    estimate, _, contours = gd.detect_gate(img)
    assert contours                              # the blob is there...
    assert estimate is None                      # ...and is correctly ignored


def test_low_clipped_band_is_reported_but_not_trusted():
    """
    A gate clipped at the bottom is real but its geometry is not: on descents
    the next gate sits below the tilted camera as a band. Steer toward, do not
    range off.
    """
    img = np.full((gd.IMG_H, gd.IMG_W, 3), 40, dtype=np.uint8)
    img[330:gd.IMG_H, 180:260] = (40, 40, 220)   # touches bottom, off-centre
    estimate, _, _ = gd.detect_gate(img)
    assert estimate is not None
    assert not estimate['reliable']
    assert not estimate['pnp_ok']
