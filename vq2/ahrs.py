"""
Platform constants and the IMU-side attitude math.

Lives on its own because both the controller and the estimator need it, and
the controller now reads its state from the estimator — importing these from
controller.py would make that circular.

Frame is NED (X north, Y east, Z down). More negative Z is higher.
"""

import math

import numpy as np

G                  = 9.81
HOVER_THRUST       = 0.265   # measured, see sysid/
LAUNCH_PITCH_DEG   = -17.8   # pad is angled
THRUST_ACCEL_COEFF = G / HOVER_THRUST ** 2

# pad heading. yaw is an absolute setpoint so a wrong value here snaps the
# nose the moment thrust comes up — this was the "release kick", 90 deg out
LAUNCH_YAW_CMD_DEG = 90.0


def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler angles (radians, NED) to quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def quat_to_R_wb(q):
    """Body-to-world (NED) rotation matrix from a [w, x, y, z] quaternion."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def body_accel_to_ned(quat, ax, ay, az, cmd_thrust):
    """Gravity-removed, thrust-capped NED acceleration from one accelerometer sample."""
    qw, qx, qy, qz = quat
    sf_n = (1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az
    sf_e = 2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az
    sf_d = 2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az

    max_accel = THRUST_ACCEL_COEFF * cmd_thrust * cmd_thrust

    # cap the thrust part, not the total — capping |a| goes to zero at hover
    # and forbids acceleration while falling. inverted vD on 91% of ticks
    thrust_part = np.array([sf_n, sf_e, sf_d])
    magnitude = float(np.linalg.norm(thrust_part))
    saturated = magnitude > max_accel
    if saturated and magnitude > 1e-9:
        thrust_part *= max_accel / magnitude

    return thrust_part + np.array([0.0, 0.0, G]), max_accel, saturated


class GyroAHRS:
    """Attitude from gyro integration alone, seeded at the known launch pitch.
    No accelerometer correction on purpose: under thrust the apparent gravity
    vector is gravity plus linear acceleration, which is most of a race."""

    def __init__(self, initial_pitch_deg=0.0):
        self.q = np.array(euler_to_quat(0.0, math.radians(initial_pitch_deg), 0.0))

    def update(self, gx, gy, gz, dt):
        """Integrate one gyro sample (rad/s, sign-corrected). Returns Euler degrees."""
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        q = np.array([
            qw + h * (-qx * gx - qy * gy - qz * gz),
            qx + h * (qw * gx + qy * gz - qz * gy),
            qy + h * (qw * gy - qx * gz + qz * gx),
            qz + h * (qw * gz + qx * gy - qy * gx),
        ])
        self.q = q / np.linalg.norm(q)
        return self.euler_deg()

    def euler_deg(self):
        qw, qx, qy, qz = self.q
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @property
    def quaternion(self):
        return self.q.copy()
