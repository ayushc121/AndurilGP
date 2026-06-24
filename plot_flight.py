#!/usr/bin/env python3
"""
plot_flight.py — visualize a flight from altitude_log.csv at a glance.

The CSV tables are unreadable by eye; this turns the LAST flight into two images:
  * side view  (north x  vs  altitude z, NED-down so the descending course goes
                DOWN on screen) — shows the altitude tracking / "flies high" / dives.
  * top view   (north x  vs  east y) — shows lateral tracking.
Gate centres are marked with their opening (+/- 1.36 m) so misses are obvious, and
trajectory points are coloured by controller STATE (approach / servo / acquire) so
the pitch-fight / acquire-toggle / stall show up immediately.

Usage:
  python plot_flight.py                 # plot the last flight in altitude_log.csv
  python plot_flight.py <csv>           # a different log
Output: flight_plot.png
"""

import sys
import csv as csvmod

import matplotlib
matplotlib.use('Agg')                      # headless: write a file, no GUI
import matplotlib.pyplot as plt

LOG      = sys.argv[1] if len(sys.argv) > 1 else 'altitude_log.csv'
OUT      = 'flight_plot.png'
OPENING  = 1.36                            # gate opening radius (m), 2.7 m wide

# Static course gate centres (NED): north x, east y, down z. Oracle (DESIGN_NOTES).
GATES = [
    (-23.30,  -0.40,  -0.03),
    (-46.89,  -2.50,   5.07),
    (-74.59,   1.20,  13.67),
    (-111.49, -5.10,  24.57),
    (-135.49, -0.80,  25.36),
    (-159.19, -4.40,  25.97),
]


def last_flight(rows):
    """Split on tick reset (sim restart) and return the last contiguous flight."""
    segs, cur, prev = [], [], None
    for r in rows:
        t = int(r['tick'])
        if prev is not None and t < prev:
            segs.append(cur); cur = []
        cur.append(r); prev = t
    if cur:
        segs.append(cur)
    return segs[-1] if segs else []


def state_color(r):
    # acquire (forced search) > servo (weak hint) > approach (normal)
    if r.get('acq', '0') == '1':
        return 'tab:red'        # ACQUIRE: forced -15 nose-down search
    if r.get('srv', '0') == '1':
        return 'tab:orange'     # SERVO: steering on a weak/low detection
    return 'tab:blue'           # APPROACH: reliable detection


def main():
    with open(LOG, newline='') as f:
        rows = list(csvmod.DictReader(f))
    flight = last_flight(rows)
    if not flight:
        print(f'No flight data in {LOG}'); return

    xs = [float(r['x']) for r in flight]
    ys = [float(r['y']) for r in flight]
    zs = [float(r['z']) for r in flight]
    cols = [state_color(r) for r in flight]

    fig, (ax_side, ax_top) = plt.subplots(2, 1, figsize=(14, 9))

    # ---- SIDE VIEW: north (x) vs altitude (z, NED-down) --------------------
    ax_side.scatter(xs, zs, c=cols, s=6, zorder=3)
    for i, (gx, gy, gz) in enumerate(GATES):
        ax_side.plot([gx, gx], [gz - OPENING, gz + OPENING], color='black', lw=2, zorder=2)
        ax_side.plot(gx, gz, 'kx', ms=8, zorder=4)
        ax_side.annotate(f'G{i}', (gx, gz - OPENING - 0.8), ha='center', fontsize=9)
    ax_side.invert_yaxis()                  # z down -> screen down
    ax_side.set_xlabel('north  x  (m)   [start 0 -> finish -159]')
    ax_side.set_ylabel('altitude  z  (NED down, m)')
    ax_side.set_title('SIDE VIEW — altitude tracking vs gates '
                      '(blue=approach, orange=servo, red=acquire; black bar=gate opening)')
    ax_side.grid(True, alpha=0.3)

    # ---- TOP VIEW: north (x) vs east (y) ----------------------------------
    ax_top.scatter(xs, ys, c=cols, s=6, zorder=3)
    for i, (gx, gy, gz) in enumerate(GATES):
        ax_top.plot([gx, gx], [gy - OPENING, gy + OPENING], color='black', lw=2, zorder=2)
        ax_top.plot(gx, gy, 'kx', ms=8, zorder=4)
        ax_top.annotate(f'G{i}', (gx, gy + OPENING + 0.3), ha='center', fontsize=9)
    ax_top.set_xlabel('north  x  (m)')
    ax_top.set_ylabel('east  y  (m)')
    ax_top.set_title('TOP VIEW — lateral tracking vs gates')
    ax_top.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f'Wrote {OUT}  ({len(flight)} ticks, x {min(xs):.1f}..{max(xs):.1f}, '
          f'z {min(zs):.1f}..{max(zs):.1f})')


if __name__ == '__main__':
    main()
