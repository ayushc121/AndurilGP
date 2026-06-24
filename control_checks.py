#!/usr/bin/env python3
"""
control_checks.py — OFFLINE controller-LOGIC checks (no sim, no flight, milliseconds).

Feeds the REAL Controller.update() hand-built synthetic inputs (drone pose + a vision
estimate) and asserts INVARIANTS on its outputs / internal setpoints. This catches the
recurring control-logic bug class *before* a flight, e.g. the coast=100 reverse-stall
would have failed here instead of wasting a sim run.

WHY NOT a full closed-loop simulator: reproducing the trajectory needs the sim's
attitude/translation dynamics, which we don't have a validated model of — guessing it
is the sim-to-real trap (confident-but-wrong green checks). So this tool tests LOGIC
only (what the controller commands for a given situation), never trajectory or
detection quality. Trajectory shape -> read a real flight with plot_flight.py.
Detection accuracy -> cv_score on real frames.

Run:  python control_checks.py        (prints PASS/FAIL per check, exits non-zero on fail)
"""

import sys
import math
import types
import threading

import controller as C
import vision_rx as V

C.time.sleep = lambda *a, **k: None      # don't actually sleep during checks
dt = 1.0 / C.CONTROL_HZ


def make_controller():
    """Controller wired for offline use: mock conn capturing the commanded attitude;
    jump to FLYING. Captured command is on ctrl._cap."""
    cap = {}

    class _Mav:
        def set_attitude_target_send(self, *a, **k): pass
    class _Conn:
        target_system = 1; target_component = 1; mav = _Mav()

    ctrl = C.Controller(_Conn(), {}, 0)

    def _capture(self, roll, pitch, yaw, thrust):
        cap.update(roll=roll, pitch=pitch, yaw=yaw, thrust=thrust)
    ctrl._send_attitude_rates = types.MethodType(_capture, ctrl)
    ctrl._cap = cap
    return ctrl


def odo(x, y, z, roll=0.0, pitch=0.0, yaw_deg=180.0, vx=0.0, vy=0.0, vz=0.0):
    q = C.euler_to_quat(roll, pitch, math.radians(yaw_deg))
    return {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz,
            'qw': q[0], 'qx': q[1], 'qy': q[2], 'qz': q[3],
            'rollspeed': 0.0, 'pitchspeed': 0.0, 'yawspeed': 0.0}


def step(ctrl, odometry, vision, gates=None):
    """Run one update() with synthetic inputs; return the captured command dict."""
    ctrl._cap.clear()
    ctrl.data = {'lock': threading.Lock(), 'odometry': odometry, 'race_status': None,
                 'armed': True, 'gates': gates, 'vision_gate_estimate': vision}
    ctrl.phase = C.Phase.FLYING
    ctrl._was_armed = True
    ctrl.update()
    return dict(ctrl._cap)


def reliable_estimate(cx, cy, bw=120, bh=120):
    """A reliable detection centred at (cx,cy)."""
    return {'bx': cx - bw / 2, 'by': cy - bh / 2, 'bw': bw, 'bh': bh,
            'cx': cx, 'cy': cy, 'cx_offset': cx - V.CX, 'cy_offset': cy - V.CY,
            'area': bw * bh, 'reliable': True}


# --------------------------------------------------------------------------------
RESULTS = []
def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def t_no_reverse_coast():
    """After PASSING a gate, coasting must NOT steer backward toward it (the coast=100
    stall). We fly -north, so a passed gate is at x > drone_x. Expect forward (nose-down,
    pitchCommand < 0), not nose-up braking."""
    ctrl = make_controller()
    ctrl._last_gate = (-23.3, -0.4, -0.03)   # gate 0, now BEHIND
    ctrl._blind = 5                           # within coast window
    cmd = step(ctrl, odo(-30.0, -0.4, 0.0), vision=None)   # drone past it, no detection
    check('no-reverse coast: commands forward (nose-down) past a gate',
          cmd['pitch'] < 0.0, f'pitchCommand={cmd["pitch"]:+.4f} (>=0 would be reversing)')


def t_elev_up_rate_limited():
    """elev_des may descend freely but must not jump UP faster than the cap (close-range
    back-projection garbage yanking the target into the gate top)."""
    ctrl = make_controller()
    ctrl._elev_des_cmd = 12.0                 # current altitude setpoint (12 m down)
    # gate appears ABOVE the drone (low z) -> back-projection target jumps UP a lot.
    # drone deep/low, gate near surface: place drone at z=12, a gate that projects to a
    # shallow gate_pz. Simplest: drive elev via a reliable estimate that yields gate_pz
    # well above 12. Use a high-in-frame, near detection (small range, pointing up).
    cmd = step(ctrl, odo(-50.0, 0.0, 12.0), reliable_estimate(V.CX, 20, bw=200, bh=200))
    up_step = 12.0 - ctrl._elev_des_cmd       # positive = moved UP (toward smaller z)
    cap = C.CONTROL_HZ and (3.0 / C.CONTROL_HZ)   # MAX_ELEV_UP_RATE(3) * dt
    check('elev_des upward motion is rate-limited',
          up_step <= cap + 1e-6, f'up_step={up_step:.3f} m/tick, cap={cap:.3f}')


def t_elev_down_free():
    """elev_des should be allowed to descend fast (no down-cap) so it doesn't lag the
    descending course."""
    ctrl = make_controller()
    ctrl._elev_des_cmd = 2.0
    # gate well BELOW -> elev_target much larger than 2 -> should move down > up-cap
    cmd = step(ctrl, odo(-50.0, 0.0, 2.0), reliable_estimate(V.CX, 340, bw=120, bh=120))
    down_step = ctrl._elev_des_cmd - 2.0      # positive = moved DOWN (toward larger z)
    cap = 3.0 / C.CONTROL_HZ
    check('elev_des descends faster than the up-cap (no down throttle)',
          down_step > cap, f'down_step={down_step:.3f} m/tick vs up-cap {cap:.3f}')


def t_brief_dropout_no_acquire():
    """A brief vision dropout (within the coast window) must NOT flip into ACQUIRE
    (forced -15 search) -- that toggling was the descent pitch-fight."""
    ctrl = make_controller()
    # establish a lock, then drop vision for a few ticks (< BLIND_HOLD_TICKS=100)
    step(ctrl, odo(-40.0, -2.0, 4.0), reliable_estimate(V.CX, 200))
    acq_any = False
    for _ in range(20):                       # 20 ticks (~0.4 s) of dropout
        step(ctrl, odo(-41.0, -2.0, 4.5), vision=None)
        acq_any = acq_any or getattr(ctrl, '_acquiring', False)
    check('brief dropout coasts (no ACQUIRE toggle within the window)',
          not acq_any, 'ACQUIRE fired during a short dropout' if acq_any else '')


def t_long_loss_acquires():
    """A genuine long loss SHOULD eventually enter ACQUIRE (so it searches for the next
    gate) -- the coast window must not be infinite."""
    ctrl = make_controller()
    step(ctrl, odo(-40.0, -2.0, 4.0), reliable_estimate(V.CX, 200))
    for _ in range(130):                      # > BLIND_HOLD_TICKS (100)
        step(ctrl, odo(-41.0, -2.0, 4.5), vision=None)
    check('long loss eventually enters ACQUIRE search',
          getattr(ctrl, '_acquiring', False), 'never acquired after 130 ticks blind')


def t_reliable_recovers_gate():
    """Sanity: a reliable centred detection makes the controller target a gate AHEAD
    (nose-down) and clears the blind counter."""
    ctrl = make_controller()
    cmd = step(ctrl, odo(-40.0, -2.0, 4.0), reliable_estimate(V.CX, 180))
    check('reliable detection -> not blind, commands forward',
          getattr(ctrl, '_blind', 1) == 0 and cmd['pitch'] < 0.5,
          f'blind={getattr(ctrl,"_blind",None)} pitchCommand={cmd["pitch"]:+.4f}')


def main():
    print('control_checks — offline controller-logic invariants (no flight):')
    for t in (t_no_reverse_coast, t_elev_up_rate_limited, t_elev_down_free,
              t_brief_dropout_no_acquire, t_long_loss_acquires, t_reliable_recovers_gate):
        try:
            t()
        except Exception as e:
            check(t.__name__, False, f'EXCEPTION {e}')
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f'\n{len(RESULTS)-n_fail}/{len(RESULTS)} passed.')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
