"""
identify_aT.py

Identifies C, Zw, Zww jointly from vertical step-thrust CSV.

Model: wdot = g  -  C·T_actual²  +  Zw·w_b  +  Zww·w_b·|w_b|

Three-parameter OLS pooled across all step windows.
Condition number of the design matrix is reported — if > 1e4, the
parameters are not well-separated and the sysid fallback values should
be used for Zw/Zww instead.

NED: w_b positive downward, g = +9.81.
At hover: C = g / T_hover² ≈ 9.81 / 0.265² ≈ 139.7 m/s²
B-matrix heave entry at trim T0: -2·C·T0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH     = "sysid_logs/sysid_1782693978.csv"
HOVER_THRUST = 0.265

# tau_m power-law:  tau_m = TAU_A · T_mid^TAU_B · dT^TAU_C
TAU_A =  0.2647
TAU_B = -0.9428
TAU_C = -0.2710

# Fallback values used if joint fit is ill-conditioned (cond > COND_WARN)
ZW_FALLBACK  = -0.18
ZWW_FALLBACK = -0.043
COND_WARN    = 1e4

STEP_THRESH = 0.03   # minimum cmd_thrust jump to count as a new step
HOVER_TOL   = 0.02   # steps within this of HOVER_THRUST are settle periods — skip
# =============================================================================

G = 9.81


def tau_model(T_start, T_end):
    T_mid = (T_start + T_end) / 2.0
    dT    = max(abs(T_end - T_start), 1e-3)
    return TAU_A * T_mid**TAU_B * dT**TAU_C


def reconstruct_T_actual(t, cmd, tau_m):
    T = np.empty_like(cmd)
    T[0] = cmd[0]
    for i in range(len(t) - 1):
        dt     = t[i+1] - t[i]
        alpha  = dt / (tau_m + dt)
        T[i+1] = T[i] + alpha * (cmd[i] - T[i])
    return T


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df  = pd.read_csv(CSV_PATH)
t   = df["t_s"].to_numpy()
w_b = df["w_b"].to_numpy()
cmd = df["cmd_thrust"].to_numpy()

dt_med = np.median(np.diff(t))
print(f"Loaded {len(t)} samples  |  dt={dt_med*1000:.2f} ms  |  dur={t[-1]:.1f} s")

wdot = np.gradient(w_b, t)

# ---------------------------------------------------------------------------
# Detect step onsets
# ---------------------------------------------------------------------------
dcmd = np.diff(cmd)
step_indices = [0]
for i in range(1, len(dcmd)):
    if abs(dcmd[i]) > STEP_THRESH and (i - step_indices[-1]) > int(0.05 / dt_med):
        step_indices.append(i)
step_indices.append(len(t))
print(f"Detected {len(step_indices)-1} thrust levels\n")

# ---------------------------------------------------------------------------
# Collect data from all step windows
# ---------------------------------------------------------------------------
segments = []   # list of (T_actual, w_b_seg, wdot_seg, T_cmd) per step

for k in range(len(step_indices) - 1):
    i0 = step_indices[k]
    i1 = step_indices[k + 1]

    T_cmd  = float(np.median(cmd[i0:i1]))
    T_prev = float(np.median(cmd[step_indices[k-1]:i0])) if k > 0 else T_cmd

    if abs(T_cmd - HOVER_THRUST) < HOVER_TOL:
        continue

    tau_m    = tau_model(T_prev, T_cmd)
    T_actual = reconstruct_T_actual(t[i0:i1], cmd[i0:i1], tau_m)

    segments.append(dict(
        T_cmd=T_cmd, tau_m=tau_m,
        T_actual=T_actual,
        w_b=w_b[i0:i1],
        wdot=wdot[i0:i1],
        i0=i0, i1=i1,
    ))

if not segments:
    raise RuntimeError("No valid step levels found.")

# ---------------------------------------------------------------------------
# C is exact from hover equilibrium — no fitting needed.
# At hover: wdot=0, w_b=0  →  0 = g - C·T_hover²  →  C = g/T_hover²
# Fitting C from flight data is ill-conditioned because T_actual and w_b
# always rise together — there's no window where thrust varies while
# velocity stays zero.
# ---------------------------------------------------------------------------
C_fixed  = G / HOVER_THRUST**2
print(f"C fixed from hover equilibrium: g / T_hover² = {C_fixed:.4f} m/s²\n")

# ---------------------------------------------------------------------------
# 2-parameter OLS with C fixed:  wdot - g + C·T_actual² = Zw·w_b + Zww·w_b·|w_b|
# Zw and Zww separate well: Zw dominates at low velocity, Zww at high velocity.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2-parameter OLS with C fixed:  wdot - g + C·T_actual² = Zw·w_b + Zww·w_b·|w_b|
# Zw and Zww separate well: Zw dominates at low velocity, Zww at high velocity.
# ---------------------------------------------------------------------------
Y_all   = np.concatenate([s["wdot"] - G + C_fixed*s["T_actual"]**2 for s in segments])
Zw_col  = np.concatenate([s["w_b"]                                  for s in segments])
Zww_col = np.concatenate([s["w_b"] * np.abs(s["w_b"])              for s in segments])

PHI  = np.column_stack([Zw_col, Zww_col])
cond = np.linalg.cond(PHI)

coeffs, _, _, _ = np.linalg.lstsq(PHI, Y_all, rcond=None)
Zw_fit, Zww_fit = coeffs

Y_pred      = PHI @ coeffs
rmse_global = np.sqrt(np.mean((Y_all - Y_pred)**2))

# Per-step residual check
print("Per-step RMSE with fitted Zw/Zww:")
T_cmds = []
for s in segments:
    y_seg    = s["wdot"] - G + C_fixed*s["T_actual"]**2
    y_pred_s = Zw_fit*s["w_b"] + Zww_fit*s["w_b"]*np.abs(s["w_b"])
    rmse_s   = np.sqrt(np.mean((y_seg - y_pred_s)**2))
    w_max    = np.min(s["w_b"])
    print(f"  T_cmd={s['T_cmd']:.3f}  rmse={rmse_s:.3f} m/s²  w_peak={w_max:.1f} m/s")
    T_cmds.append(s["T_cmd"])
T_cmds = np.array(T_cmds)

print(f"\n{'='*65}")
print(f"  C  fixed from hover equilibrium = {C_fixed:.4f} m/s²")
print(f"  {'':30s}  {'fitted':>10}  {'sysid (hover)':>13}")
print(f"  {'Zw  (s⁻¹)':30s}  {Zw_fit:10.4f}  {ZW_FALLBACK:13.4f}")
print(f"  {'Zww  (m⁻¹)':30s}  {Zww_fit:10.4f}  {ZWW_FALLBACK:13.4f}")
print(f"  {'-'*63}")
print(f"  Global RMSE      = {rmse_global:.4f} m/s²")
print(f"  Condition number = {cond:.2e}  ", end="")
if cond < COND_WARN:
    print(f"✓  (< {COND_WARN:.0e} — Zw/Zww well-separated)")
else:
    print(f"✗  (> {COND_WARN:.0e} — use sysid fallback values)")
print(f"  {'-'*63}")
print(f"  B-matrix heave at hover: -2·C·T_hover = {-2*C_fixed*HOVER_THRUST:.4f} m/s²")
print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"C/Zw/Zww identification  |  {CSV_PATH}", fontsize=10)

# Left: drag model — Zw·w + Zww·w|w| vs velocity
ax = axes[0]
w_line = np.linspace(-35, 0, 200)
ax.plot(w_line, Zw_fit*w_line + Zww_fit*w_line*np.abs(w_line),
        "r-", lw=2, label=f"fitted: Zw={Zw_fit:.3f}  Zww={Zww_fit:.3f}")
ax.plot(w_line, ZW_FALLBACK*w_line + ZWW_FALLBACK*w_line*np.abs(w_line),
        "k--", lw=1.5, label=f"sysid:  Zw={ZW_FALLBACK}  Zww={ZWW_FALLBACK}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("w_b (m/s)"); ax.set_ylabel("drag acceleration (m/s²)")
ax.set_title("Drag model comparison")
ax.legend(fontsize=8); ax.grid(True)

# Middle: measured vs predicted wdot
wdot_meas = Y_all + G - C_fixed * np.concatenate([s["T_actual"]**2 for s in segments]) + \
            C_fixed * np.concatenate([s["T_actual"]**2 for s in segments])
wdot_meas = np.concatenate([s["wdot"] for s in segments])
wdot_pred = G - C_fixed * np.concatenate([s["T_actual"]**2 for s in segments]) + Y_pred
lims = [min(wdot_meas.min(), wdot_pred.min()) - 2,
        max(wdot_meas.max(), wdot_pred.max()) + 2]
ax = axes[1]
ax.plot(lims, lims, "k--", lw=1.0, label="perfect")
ax.scatter(wdot_meas[::5], wdot_pred[::5], s=4, alpha=0.3)
ax.set_xlabel("wdot measured (m/s²)"); ax.set_ylabel("wdot predicted (m/s²)")
ax.set_title(f"Measured vs predicted\nRMSE={rmse_global:.2f} m/s²  cond={cond:.1e}")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(); ax.grid(True)

# Right: full w_b trace
ax  = axes[2]
ax2 = ax.twinx()
ax.plot(t, w_b, "b", lw=1.0, label="w_b (m/s)")
ax2.plot(t, cmd, "k--", lw=0.8, alpha=0.5, label="cmd_thrust")
for s in segments:
    ax.axvspan(t[s["i0"]], t[s["i1"]], alpha=0.12, color="green")
ax.set_xlabel("t (s)"); ax.set_ylabel("w_b (m/s)", color="b")
ax2.set_ylabel("cmd_thrust", color="k")
ax.set_title("Flight trace  (green = windows used)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8)

plt.tight_layout()
out = CSV_PATH.replace(".csv", "_aT.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Plot saved → {out}")