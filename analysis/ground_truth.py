"""
Ground-truth logger: is the camera's gate estimate accurate enough to correct
IMU drift with? Diffing it against the IMU cannot answer that — both are
uncertain — so this runs on a VQ1 flight and diffs against true odometry.

Logging only. Nothing here may be called from a control path.
"""

import csv
import math

CSV_HEADER = [
    't', 'gate_idx', 'range_m',
    'true_bx', 'true_by', 'true_bz',
    'cv_bx', 'cv_by', 'cv_bz',
    'err_x', 'err_y', 'err_z', 'err_total',
    'true_vx', 'true_vy', 'true_vz',
    'cv_vx', 'cv_vy', 'cv_vz',
    'verr_x', 'verr_y', 'verr_z', 'verr_total',
    'pnp_ok', 'reliable',
    # Raw orientation inputs; the error is derived offline so the frame
    # conventions can be checked against data rather than baked in here.
    'gate_qw', 'gate_qx', 'gate_qy', 'gate_qz',
    'odo_qw', 'odo_qx', 'odo_qy', 'odo_qz',
    'cv_rx', 'cv_ry', 'cv_rz',
]


def world_to_body(vn, ve, vd, qw, qx, qy, qz):
    """NED to body. Same rotation the controller uses, so truth and estimate
    are expressed identically."""
    return (
        (1 - 2 * (qy * qy + qz * qz)) * vn + 2 * (qx * qy + qw * qz) * ve + 2 * (qx * qz - qw * qy) * vd,
        2 * (qx * qy - qw * qz) * vn + (1 - 2 * (qx * qx + qz * qz)) * ve + 2 * (qy * qz + qw * qx) * vd,
        2 * (qx * qz + qw * qy) * vn + 2 * (qy * qz - qw * qx) * ve + (1 - 2 * (qx * qx + qy * qy)) * vd,
    )


class GroundTruthLogger:
    """Writes one CSV row per camera frame that has a usable detection."""

    def __init__(self, path='cv_ground_truth.csv'):
        self._file = open(path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)
        self._last_frame_id = None

    def close(self):
        self._file.close()

    def log(self, t, gate_idx, odo, gate_pos, vision, vision_velocity, gate_quat=None):
        """One comparison row. Skipped when there is no detection, which is normal
        during passthrough blanking."""
        if vision is None or gate_pos is None or odo is None:
            return

        frame_id = vision.get('frame_id')
        if frame_id is not None and frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id

        cv_pos = (vision.get('body_x_m'), vision.get('body_y_m'), vision.get('body_z_m'))
        if cv_pos[0] is None or any(math.isnan(v) for v in cv_pos):
            return

        quat = (odo['qw'], odo['qx'], odo['qy'], odo['qz'])
        offset = (gate_pos[0] - odo['x'], gate_pos[1] - odo['y'], gate_pos[2] - odo['z'])
        range_m = math.sqrt(sum(v * v for v in offset))
        true_pos = world_to_body(*offset, *quat)

        pos_err = tuple(c - t for c, t in zip(cv_pos, true_pos))
        pos_err_total = math.sqrt(sum(v * v for v in pos_err))

        true_vel = (odo['vx'], odo['vy'], odo['vz'])
        if vision_velocity is not None:
            cv_vel = (vision_velocity.get('vx_body_mps'),
                      vision_velocity.get('vy_body_mps'),
                      vision_velocity.get('vz_body_mps'))
            vel_err = tuple(c - t for c, t in zip(cv_vel, true_vel))
            vel_err_total = math.sqrt(sum(v * v for v in vel_err))
        else:
            cv_vel = vel_err = ('', '', '')
            vel_err_total = ''

        rvec = vision.get('pnp_rvec')
        cv_r = tuple(float(v) for v in rvec[:3]) if rvec is not None else ('', '', '')
        gate_q = gate_quat if gate_quat is not None else ('', '', '', '')

        def num(value, places=3):
            return '' if value is None or value == '' else f'{float(value):.{places}f}'

        self._writer.writerow(
            [f'{t:.3f}', gate_idx, f'{range_m:.3f}']
            + [num(v) for v in true_pos]
            + [num(v) for v in cv_pos]
            + [num(v) for v in pos_err] + [num(pos_err_total)]
            + [num(v) for v in true_vel]
            + [num(v) for v in cv_vel]
            + [num(v) for v in vel_err] + [num(vel_err_total)]
            + [vision.get('pnp_ok', False), vision.get('reliable', False)]
            + [num(v, 5) for v in gate_q]
            + [num(v, 5) for v in quat]
            + [num(v, 5) for v in cv_r])
