"""
fit_vz_step.py
Fits vz(t) = vz_inf * (1 - exp(-t / tau_m)) to a thrust step response.
Edit the CSV path and STEP_* constants at the top, then run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# CONFIG — edit these per run
# =============================================================================
CSV_PATH     = "sysid_logs/thrust_4.csv"   # <-- change per run
HOVER_THRUST = 0.265
STEP_SIZE    = 0.4-0.265    # must match what was commanded
STEP_DURATION = 5.0   # seconds at each level (hover, then step, then back)
# =============================================================================

def model(t, vz_inf, tau_m):
    return vz_inf * (1.0 - np.exp(-t / tau_m))

df = pd.read_csv(CSV_PATH)

# Detect step onset: first tick where cmd_thrust > HOVER_THRUST + half the step
thresh = HOVER_THRUST + STEP_SIZE * 0.5
step_mask = df["cmd_thrust"] > thresh
if not step_mask.any():
    raise ValueError("No step detected — check STEP_SIZE or CSV_PATH.")

step_start_idx = step_mask.idxmax()
t0 = df.loc[step_start_idx, "t_s"]

# Crop to just the step window
end_t = t0 + STEP_DURATION
window = df[(df["t_s"] >= t0) & (df["t_s"] <= end_t)].copy()
window["t_rel"] = window["t_s"] - t0

t   = window["t_rel"].to_numpy()
vz  = window["w_b"].to_numpy()    # body-frame vz; positive = downward in NED

# Subtract initial vz so the model starts at zero
vz0 = vz[0]
vz_shifted = vz - vz0

# Initial guess: vz_inf from end of window, tau from 63% crossing
vz_inf_guess = vz_shifted[-1]
tau_guess    = 0.5

p0     = [vz_inf_guess, tau_guess]
bounds = ([-np.inf, 0.01], [np.inf, 10.0])

try:
    popt, pcov = curve_fit(model, t, vz_shifted, p0=p0, bounds=bounds, maxfev=10000)
except RuntimeError as e:
    raise RuntimeError(f"Fit failed: {e}")

vz_inf_fit, tau_m_fit = popt
vz_inf_std, tau_m_std = np.sqrt(np.diag(pcov))

vz_pred = model(t, *popt)
ss_res  = np.sum((vz_shifted - vz_pred) ** 2)
ss_tot  = np.sum((vz_shifted - vz_shifted.mean()) ** 2)
r2      = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"\n{'='*45}")
print(f"  CSV       : {CSV_PATH}")
print(f"  Step size : {STEP_SIZE:+.3f}  (hover={HOVER_THRUST})")
print(f"  Window    : {t[0]:.3f} – {t[-1]:.3f} s  ({len(t)} samples)")
print(f"{'='*45}")
print(f"  vz_inf    : {vz_inf_fit:+.4f} m/s  (±{vz_inf_std:.4f})")
print(f"  tau_m     : {tau_m_fit:.4f} s      (±{tau_m_std:.4f})")
print(f"  R²        : {r2:.4f}")
print(f"{'='*45}\n")

# Plot
t_fine = np.linspace(0, t[-1], 500)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].plot(t, vz_shifted, "b.", ms=3, label="measured vz (shifted)")
axes[0].plot(t_fine, model(t_fine, *popt), "r-", lw=2,
             label=f"fit: τ={tau_m_fit:.3f}s  vz∞={vz_inf_fit:.3f}m/s  R²={r2:.3f}")
axes[0].set_ylabel("vz − vz₀  (m/s)")
axes[0].legend(fontsize=9)
axes[0].grid(True)
axes[0].set_title(f"vz step fit  |  step={STEP_SIZE:+.2f}  |  {CSV_PATH}")

axes[1].plot(window["t_s"].to_numpy(), window["cmd_thrust"].to_numpy(),
             "k-", lw=1.5, label="cmd_thrust")
axes[1].set_ylabel("cmd_thrust")
axes[1].set_xlabel("t_rel  (s)")
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.tight_layout()
plt.savefig(CSV_PATH.replace(".csv", "_fit.png"), dpi=150)
plt.show()
print("Plot saved alongside CSV.")
