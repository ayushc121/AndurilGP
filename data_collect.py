#!/usr/bin/env python3
"""
data_collect.py — systematic gate-approach data collection for CV training.

Flies the drone to a series of standoff distances in front of each course gate
(40 m → 5 m), dwelling briefly at each stop while vision_rx.py captures frames
to vision_dump/.  Run in Training mode so GATE_INFO is available.

Usage:
    python data_collect.py

Do NOT use during a Competitive flight — that wastes a qualification attempt.
"""

import math
import time

import numpy as np

from setup import setup_components

# ── Connection ────────────────────────────────────────────────────────────────
SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550

# ── Tunable parameters ────────────────────────────────────────────────────────
# Standoff distances to visit for each gate (far → close).
STANDOFF_DISTANCES_M = [40, 30, 20, 15, 10, 7, 5]

# Seconds to hold each standoff position before stepping closer.
DWELL_S = 2.5

# Metres: "close enough" before declaring arrival and starting the dwell timer.
POSITION_TOL_M = 1.5

# Seconds: abort approaching the current standoff and move on if this elapses.
APPROACH_TIMEOUT_S = 20.0

# Seconds between arm retries.
ARM_RETRY_S = 1.0

# Seconds to wait for GATE_INFO before giving up (requires Training mode).
MAX_WAIT_GATE_S = 30.0

# ── MAVLink type_mask ─────────────────────────────────────────────────────────
# Bits: 3-5 = ignore vx/vy/vz, 6-8 = ignore ax/ay/az, 11 = ignore yaw_rate.
# Bit 10 (ignore yaw) is intentionally CLEAR so we can command yaw toward gate.
_MASK_POS_YAW = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 11)
# = 2552


def _gate_approach_dir(gates, idx):
    """
    Unit vector (horizontal NED) pointing FROM the previous gate (or origin)
    TOWARD gate[idx].  The drone hovers on the near side of this vector.
    """
    cur = np.array([gates[idx]['pos_x'], gates[idx]['pos_y'], 0.0])
    if idx > 0:
        prev = np.array([gates[idx - 1]['pos_x'], gates[idx - 1]['pos_y'], 0.0])
    else:
        prev = np.zeros(3)
    d = cur - prev
    n = float(np.linalg.norm(d))
    return d / n if n > 0.01 else np.array([1.0, 0.0, 0.0])


def _send_pos_target(sim_conn, target_sys, target_comp, x, y, z, yaw_rad, boot_ms):
    sim_conn.mav.set_position_target_local_ned_send(
        boot_ms & 0xFFFFFFFF,  # time_boot_ms
        target_sys,
        target_comp,
        1,                      # MAV_FRAME_LOCAL_NED
        _MASK_POS_YAW,
        x, y, z,
        0.0, 0.0, 0.0,         # vx, vy, vz (ignored)
        0.0, 0.0, 0.0,         # afx, afy, afz (ignored)
        yaw_rad,
        0.0,                   # yaw_rate (ignored)
    )


def _approach_and_dwell(sim_conn, shared_data, target_sys, target_comp,
                        tx, ty, tz, yaw_rad, system_boot_ms):
    """
    Fly to (tx, ty, tz) NED and hold for DWELL_S seconds once within
    POSITION_TOL_M.  Keeps sending position commands throughout so the
    flight controller doesn't revert to its own mode.
    """
    t0 = time.time()
    arrived = False

    while True:
        boot_ms = int(time.time() * 1000) - system_boot_ms
        _send_pos_target(sim_conn, target_sys, target_comp,
                         tx, ty, tz, yaw_rad, boot_ms)

        with shared_data['lock']:
            odo = shared_data.get('odometry')

        if odo and not arrived:
            dx = odo['x'] - tx
            dy = odo['y'] - ty
            dz = odo['z'] - tz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < POSITION_TOL_M:
                arrived = True
                t_arrive = time.time()

        if arrived and (time.time() - t_arrive) >= DWELL_S:
            break

        if not arrived and (time.time() - t0) > APPROACH_TIMEOUT_S:
            print('    TIMEOUT — moving on')
            break

        time.sleep(0.05)


def main():
    from pymavlink import mavutil as mavu

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    print('Connecting to simulator...', flush=True)
    components = setup_components(
        shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    )
    sim_conn    = components['sim_conn']
    target_sys  = sim_conn.target_system
    target_comp = sim_conn.target_component

    # ── Phase 1: Arm ──────────────────────────────────────────────────────────
    print('Arming drone...  (start a Training flight in the sim if stuck here)',
          flush=True)
    arm_ticks = 0
    while True:
        sim_conn.mav.command_long_send(
            target_sys, target_comp,
            mavu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,          # confirmation
            1,          # param1: arm
            0, 0, 0, 0, 0, 0,
        )
        time.sleep(ARM_RETRY_S)
        arm_ticks += 1
        with shared_data['lock']:
            armed = shared_data.get('armed', False)
            odo   = shared_data.get('odometry')
        if armed and odo is not None:
            print('Armed.', flush=True)
            break
        if arm_ticks % 5 == 0:
            print(f'  still waiting to arm... (armed={armed}, odo={'ok' if odo else 'none'})',
                  flush=True)

    # ── Phase 2: Wait for race start ──────────────────────────────────────────
    # race_status is None until the first RACE_STATUS MAVLink message arrives.
    # Use the same freshness anchor as controller.py: record sim_ms when we
    # enter this phase, then wait until race_start_boot_time_ms > that anchor.
    print('Waiting for race start...', flush=True)
    anchor_ms = None
    wait_ticks = 0
    while True:
        with shared_data['lock']:
            rs = shared_data.get('race_status')  # may be None initially
        if rs is not None:
            sim_ms   = rs['sim_boot_time_ms']
            start_ms = rs['race_start_boot_time_ms']
            if anchor_ms is None:
                anchor_ms = sim_ms
            race_is_fresh  = start_ms > 0 and start_ms >= anchor_ms
            countdown_done = race_is_fresh and sim_ms >= start_ms
            if countdown_done:
                print('Race started — beginning data collection.', flush=True)
                break
            if wait_ticks % 20 == 0:
                print(f'  sim_ms={sim_ms}  race_start={start_ms}  '
                      f'fresh={race_is_fresh}', flush=True)
        else:
            if wait_ticks % 20 == 0:
                print('  no race_status yet...', flush=True)
        wait_ticks += 1
        time.sleep(0.1)

    # ── Phase 3: Try to receive GATE_INFO (VQ2) or fall back to course sweep ──
    # GATE_INFO is only transmitted in VQ2.  In VQ1 training we skip straight
    # to the sweep after a short wait.
    GATE_INFO_WAIT_S = 10.0
    print(f'Waiting up to {GATE_INFO_WAIT_S:.0f} s for GATE_INFO (VQ2 only)...',
          flush=True)
    deadline = time.time() + GATE_INFO_WAIT_S
    gates = []
    while time.time() < deadline:
        with shared_data['lock']:
            raw_gates = shared_data.get('gates')
        if raw_gates:
            gates = list(raw_gates)
            print(f'  GATE_INFO received: {len(gates)} gate(s).', flush=True)
            break
        time.sleep(0.2)

    # ── Phase 4: Fly approach waypoints ───────────────────────────────────────
    if gates:
        # VQ2 path: approach each gate at multiple standoff distances
        print(f'\nApproaching {len(gates)} gate(s) at standoffs: '
              f'{STANDOFF_DISTANCES_M} m', flush=True)

        for gate_idx, gate in enumerate(gates):
            gx = gate['pos_x']
            gy = gate['pos_y']
            gz = gate['pos_z']
            approach = _gate_approach_dir(gates, gate_idx)
            yaw_rad  = math.atan2(approach[1], approach[0])

            print(f'\n--- Gate {gate_idx}  NED=({gx:.1f}, {gy:.1f}, {gz:.1f})  '
                  f'yaw={math.degrees(yaw_rad):.0f} deg ---', flush=True)

            for standoff in STANDOFF_DISTANCES_M:
                tx = gx - approach[0] * standoff
                ty = gy - approach[1] * standoff
                tz = gz
                print(f'  standoff {standoff:>3d} m  ({tx:.1f}, {ty:.1f}, {tz:.1f})',
                      flush=True)
                _approach_and_dwell(sim_conn, shared_data, target_sys, target_comp,
                                    tx, ty, tz, yaw_rad, system_boot_ms)
                print('    done', flush=True)

    else:
        # VQ1 path: no gate positions available — sweep the full course in
        # 8 m steps along the -x axis (southward) at spawn altitude.
        # vision_rx INSTRUMENT mode captures frames every 15 processed frames
        # throughout, giving coverage at all approach ranges for each gate.
        print('\nNo GATE_INFO — running VQ1 course sweep.', flush=True)

        with shared_data['lock']:
            odo = shared_data.get('odometry')
        spawn_x = odo['x'] if odo else 0.0
        spawn_y = odo['y'] if odo else 0.0
        spawn_z = odo['z'] if odo else -2.5  # NED: negative = above ground

        SWEEP_STEP_M  = 8      # metres between hover points along course
        SWEEP_TOTAL_M = 148    # total distance to cover (beyond all known gates)
        # Face south (-x direction = course heading)
        yaw_rad = math.atan2(0.0, -1.0)  # atan2(dy=0, dx=-1) = π ≈ south

        n_steps = SWEEP_TOTAL_M // SWEEP_STEP_M
        print(f'Sweep: {n_steps} waypoints, {SWEEP_STEP_M} m apart, '
              f'from x={spawn_x:.1f} to x={spawn_x - SWEEP_TOTAL_M:.1f}', flush=True)

        for step in range(1, n_steps + 1):
            tx = spawn_x - step * SWEEP_STEP_M
            ty = spawn_y
            tz = spawn_z
            print(f'  wp {step:>2d}/{n_steps}  x={tx:.1f}', flush=True)
            _approach_and_dwell(sim_conn, shared_data, target_sys, target_comp,
                                tx, ty, tz, yaw_rad, system_boot_ms)
            print('    done', flush=True)

    print('\nData collection complete.  Frames saved to vision_dump/', flush=True)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    for name in ('heartbeat', 'ts_loop', 'mavlink_rx', 'vision_rx'):
        components[name].get_thread_for_join().join(timeout=2.0)


if __name__ == '__main__':
    main()
