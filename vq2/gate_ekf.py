"""
gate_ekf.py — 6-state gate-correction Kalman filter for vision-only drone racing.

Reference implementations: Azhari et al. (arXiv 2512.20475,
drift-corrected monocular VIO) and JamBoy `ekf.py` (6-state [p,v] NED skeleton).

State (world-NED, metres / m·s⁻¹):
    x = [pN, pE, pD, vN, vE, vD]

- ABSOLUTE mode: p is the drone's world position; gate detections localise it against a
  known course map (measurement z = gate_world − R_wb·[bx,by,bz]).
- RELATIVE mode: p is the drone's position relative to the current target-gate opening
  centre; no map needed (measurement z = −R_wb·[bx,by,bz]). The math below is identical
  either way — only the caller's construction of z differs.

Predict input is NED linear acceleration `a_ned` (gravity-removed, thrust-capped) — the
SAME quantity controller._integrate_kinematics already computes; this filter does NOT
recompute the rotation/gravity/cap. Attitude is NOT in the state (gyro AHRS owns it), so
with z precomputed this is a LINEAR KF — no state Jacobian, no linearisation bugs.

Pure numpy. No sim, no OpenCV, no controller import — unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np

# χ² 95% thresholds by DOF, for the Mahalanobis outlier gate on the innovation.
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815}


class GateEKF:
    """6-state [position, velocity] KF with position-only measurement updates."""

    def __init__(self, sigma_a=1.5, sigma_v_walk=0.05, chi2_dof3=CHI2_95[3]):
        """sigma_a       : accel process-noise std (m/s²) — drives Q via the input Jacobian."""
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[0:3, 0:3] *= 1.0     # position prior var (m²)
        self.P[3:6, 3:6] *= 9.0     # velocity prior var (m/s)² — loose, so first fixes pull v fast
        self.sigma_a = float(sigma_a)
        self.sigma_v_walk = float(sigma_v_walk)
        self.chi2 = float(chi2_dof3)
        self.initialised = False

    # init
    def init_position(self, p_ned, vel_ned=None, pos_var=0.5, vel_var=9.0):
        """Seed the filter from a first position fix (and optional velocity guess)."""
        self.x[0:3] = np.asarray(p_ned, dtype=float)
        self.x[3:6] = 0.0 if vel_ned is None else np.asarray(vel_ned, dtype=float)
        self.P = np.eye(6)
        self.P[0:3, 0:3] *= pos_var
        self.P[3:6, 3:6] *= vel_var
        self.initialised = True

    def reanchor(self, p_ned, R_pos, vel_inflate=1.1):
        """Gate handoff. Position jumps to the new fix, velocity and its covariance
        survive — world velocity doesn't care which gate we measure against."""
        self.x[0:3] = np.asarray(p_ned, dtype=float)
        self.P[0:3, 0:3] = np.asarray(R_pos, dtype=float)
        self.P[0:3, 3:6] = 0.0
        self.P[3:6, 0:3] = 0.0
        self.P[3:6, 3:6] *= vel_inflate
        self.initialised = True

    # predict
    def predict(self, a_ned, dt):
        """Constant-velocity kinematics with acceleration input."""
        dt = float(dt)
        a = np.asarray(a_ned, dtype=float)

        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)

        # x⁻ = F x + G a
        self.x[0:3] += self.x[3:6] * dt + 0.5 * dt * dt * a
        self.x[3:6] += dt * a

        # Q = G σ_a² Gᵀ + velocity random walk. G = [½dt²I ; dtI].
        g_pos = 0.5 * dt * dt
        g_vel = dt
        sa2 = self.sigma_a * self.sigma_a
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = sa2 * g_pos * g_pos * np.eye(3)
        Q[0:3, 3:6] = sa2 * g_pos * g_vel * np.eye(3)
        Q[3:6, 0:3] = sa2 * g_pos * g_vel * np.eye(3)
        Q[3:6, 3:6] = sa2 * g_vel * g_vel * np.eye(3)
        Q[3:6, 3:6] += (self.sigma_v_walk * self.sigma_v_walk * dt) * np.eye(3)

        self.P = F @ self.P @ F.T + Q

    # update
    def update(self, z, R, gate=True):
        """Position measurement update. H = [I₃, 0₃], so it observes position and corrects."""
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)

        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        y = z - self.x[0:3]                     # innovation (H x = position)
        S = self.P[0:3, 0:3] + R                # H P Hᵀ + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, float("inf")

        mahal = float(y @ S_inv @ y)
        # accept only if inside the gate, not reject if outside — `mahal > chi2`
        # is False for NaN, so the old form fused NaN and poisoned x and P
        if gate and not (mahal <= self.chi2):
            return False, mahal                 # outlier or non-finite — do not fuse

        K = self.P @ H.T @ S_inv                # (6,3)
        self.x = self.x + K @ y

        # Joseph form: P⁺ = (I−KH) P (I−KH)ᵀ + K R Kᵀ
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True, mahal

    # accessors
    @property
    def position(self):
        return self.x[0:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()

    def vector_to(self, gate_world):
        """ABSOLUTE mode: guidance vector to a gate's world opening centre = gate − p̂."""
        return np.asarray(gate_world, dtype=float) - self.x[0:3]


# Measurement construction helpers (kept out of the filter core so the KF stays
# linear and source-neutral — PnP/attitude/map math stays out of this file).

def build_R_anisotropic(R_wc, sigma_lat=0.3, sigma_range=1.2):
    """Anisotropic measurement covariance — our key adaptation over both references."""
    R_wc = np.asarray(R_wc, dtype=float)
    R_cam = np.diag([sigma_lat ** 2, sigma_lat ** 2, sigma_range ** 2])
    return R_wc @ R_cam @ R_wc.T


def z_absolute(gate_world, R_wb, gate_body):
    """ABSOLUTE mode measurement: drone world position implied by seeing a known-map gate."""
    R_wb = np.asarray(R_wb, dtype=float)
    gate_body = np.asarray(gate_body, dtype=float)
    return np.asarray(gate_world, dtype=float) - R_wb @ gate_body


def z_relative(R_wb, gate_body):
    """RELATIVE mode measurement: drone position relative to the target-gate centre."""
    R_wb = np.asarray(R_wb, dtype=float)
    gate_body = np.asarray(gate_body, dtype=float)
    return -(R_wb @ gate_body)
