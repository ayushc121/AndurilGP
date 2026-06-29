"""
fit_tau_model.py

Fits a 2D power-law model for the effective vz time constant:

    tau_m = a * T_mid^b * dT^c

where T_mid = (T_start + T_end) / 2
      dT    = T_end - T_start

Log-linearised form: ln(tau) = ln(a) + b*ln(T_mid) + c*ln(dT)
This is an ordinary least-squares problem — no nonlinear solver needed.

Add rows to DATA as you collect more runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# =============================================================================
# DATA  [tau_m, T_start, T_end]
# =============================================================================
DATA = np.array([
    # fixed T_start = 0.265, vary T_end
    [0.4844,  0.265, 1.000],
    [0.5920,  0.265, 0.800],
    [0.7767,  0.265, 0.600],
    [1.3690,  0.265, 0.400],
    # fixed dT ≈ 0.4, vary T_start
    [0.4009,  0.600, 1.000],
    [0.5074,  0.400, 0.800],
    [0.7051,  0.265, 0.665],
    [0.9836,  0.100, 0.500],
    # fixed T_end = 0.8, vary T_start
    [0.4562,  0.600, 0.800],
    [0.4986,  0.400, 0.800],
    [0.5917,  0.265, 0.800],
    [0.6044,  0.100, 0.800],
])

tau     = DATA[:, 0]
T_start = DATA[:, 1]
T_end   = DATA[:, 2]
T_mid   = (T_start + T_end) / 2.0
dT      = T_end - T_start

# =============================================================================
# Fit: ln(tau) = ln(a) + b*ln(T_mid) + c*ln(dT)
# =============================================================================
ln_tau  = np.log(tau)
ln_Tmid = np.log(T_mid)
ln_dT   = np.log(dT)

# Design matrix for [ln(a), b, c]
A = np.column_stack([np.ones(len(tau)), ln_Tmid, ln_dT])
coeffs, residuals, rank, sv = np.linalg.lstsq(A, ln_tau, rcond=None)

ln_a, b, c = coeffs
a = np.exp(ln_a)

# Predictions and diagnostics
tau_pred = a * T_mid**b * dT**c
ss_res   = np.sum((tau - tau_pred)**2)
ss_tot   = np.sum((tau - tau.mean())**2)
r2       = 1.0 - ss_res / ss_tot

print(f"\n{'='*50}")
print(f"  Model:  tau_m = {a:.4f} * T_mid^({b:.4f}) * dT^({c:.4f})")
print(f"  R²    = {r2:.4f}")
print(f"{'='*50}")
print(f"\n  {'T_start':>8}  {'T_end':>6}  {'T_mid':>6}  {'dT':>6}  "
      f"{'tau_meas':>9}  {'tau_pred':>9}  {'err%':>7}")
print(f"  {'-'*65}")
for i in range(len(tau)):
    err_pct = 100 * (tau_pred[i] - tau[i]) / tau[i]
    print(f"  {T_start[i]:8.3f}  {T_end[i]:6.3f}  {T_mid[i]:6.4f}  {dT[i]:6.4f}  "
          f"{tau[i]:9.4f}  {tau_pred[i]:9.4f}  {err_pct:+7.2f}%")
print()

# =============================================================================
# Surface plot: tau_m over (T_mid, dT) grid
# =============================================================================
Tmid_grid = np.linspace(0.15, 0.90, 60)
dT_grid   = np.linspace(0.05, 0.75, 60)
TM, DT    = np.meshgrid(Tmid_grid, dT_grid)
TAU       = a * TM**b * DT**c

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: surface
cf = axes[0].contourf(TM, DT, TAU, levels=20, cmap="plasma_r")
fig.colorbar(cf, ax=axes[0], label="τ_m  (s)")
axes[0].scatter(T_mid, dT, c=tau, cmap="plasma_r",
                edgecolors="white", s=80, zorder=5, label="measured")
axes[0].set_xlabel("T_mid = (T_start + T_end) / 2")
axes[0].set_ylabel("ΔT = T_end − T_start")
axes[0].set_title(f"τ_m = {a:.3f}·T_mid^{b:.3f}·ΔT^{c:.3f}")
axes[0].legend()

# Right: measured vs predicted
lims = [0, max(tau.max(), tau_pred.max()) * 1.1]
axes[1].plot(lims, lims, "k--", lw=1, label="perfect fit")
axes[1].scatter(tau, tau_pred, s=70, zorder=5)
for i in range(len(tau)):
    axes[1].annotate(f"({T_start[i]:.2f}→{T_end[i]:.2f})",
                     (tau[i], tau_pred[i]), fontsize=7,
                     textcoords="offset points", xytext=(5, 3))
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_xlabel("τ_m measured  (s)")
axes[1].set_ylabel("τ_m predicted  (s)")
axes[1].set_title(f"Fit quality  R²={r2:.4f}")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("tau_model_fit.png", dpi=150)
plt.show()
print("Plot saved → tau_model_fit.png")
