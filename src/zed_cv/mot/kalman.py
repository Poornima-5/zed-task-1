"""Constant-velocity Kalman filter over (cx, cy, aspect a=w/h, h) with velocities (8-D state).

Process/measurement noise scale with the box height h (SORT/DeepSORT convention), so uncertainty is
scale-aware. ``update(..., r_scale)`` lets the tracker inflate measurement noise for unreliable
(low-score / occluded) detections. Gating uses the Mahalanobis distance in measurement space.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

CHI2_4DOF_95 = 9.4877


def tlbr_to_cxyah(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w / np.maximum(h, 1e-6), h], axis=-1)


def cxyah_to_tlbr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    w = z[..., 2] * z[..., 3]
    return np.stack([z[..., 0] - w / 2, z[..., 1] - z[..., 3] / 2, z[..., 0] + w / 2, z[..., 1] + z[..., 3] / 2], axis=-1)


class KalmanBoxFilter:
    def __init__(self, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.sp, self.sv = std_pos, std_vel
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.sp * h, 2 * self.sp * h, 1e-2, 2 * self.sp * h,
               10 * self.sv * h, 10 * self.sv * h, 1e-5, 10 * self.sv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        q = np.diag(np.square([self.sp * h, self.sp * h, 1e-2, self.sp * h, self.sv * h, self.sv * h, 1e-5, self.sv * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def project(self, mean: np.ndarray, cov: np.ndarray, r_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([self.sp * h, self.sp * h, 1e-1, self.sp * h]) * r_scale)
        return self.H @ mean, self.H @ cov @ self.H.T + r

    def update(self, mean, cov, z, r_scale: float = 1.0):
        pm, pc = self.project(mean, cov, r_scale)
        chol, low = scipy.linalg.cho_factor(pc, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve((chol, low), (cov @ self.H.T).T, check_finite=False).T
        new_mean = mean + gain @ (z - pm)
        new_cov = cov - gain @ pc @ gain.T
        return new_mean, new_cov

    def gating_distance(self, mean, cov, measurements: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of each measurement (N,4) to the projected state."""
        pm, pc = self.project(mean, cov)
        chol = np.linalg.cholesky(pc)
        d = measurements - pm
        y = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        return np.sum(y * y, axis=0)
