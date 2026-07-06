#!/usr/bin/env python3
"""
data_collect.py — systematic gate-approach data collection for CV training.

Flies the drone to a series of standoff distances in front of each course gate
(40 m → 5 m), dwelling briefly at each stop while vision_rx.py captures frames
to vision_dump/.  Run in Training mode so GATE_INFO is available.

Uses SET_ATTITUDE_TARGET (roll/pitch/yaw/thrust) with a minimal P-controller,
matching the control interface used by controller.py.  SET_POSITION_TARGET_LOCAL_NED
activates the sim's internal position controller which is uncalibrated for this
drone and oscillates uncontrollably.

Usage:
    python data_collect.py
"""

import math
import time

import numpy as np

from setup import setup_components

# ── Connection ────────────────────────────────────────────────────────────────
SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550

# ── Standoff parameters ───────────────────────────────────────────────────────
STANDOFF_DISTANCES_M = [40, 30, 20, 15, 10, 7, 5]
DWELL_S              = 2.5
POSITION_TOL_M       = 1.5
APPROACH_TIMEOUT_S   = 30.0
ARM_RETRY_S          = 1.0
MAX_WAIT_GATE_S      = 10.0

# ── Control constants (gentler than racing) ───────────────────────────────────
CONTROL_HZ       = 50
_dt              = 1.0 / CONTROL_HZ
V_MAX_HORIZ      = 3.0    # m/s cap — slow / safe for data collection
K_POS            = 1.0    # position error → velocity setpoint (P)
K_VX_P           = 1.5    # north velocity error → desired pitch (deg)
K_VY_P           = 3.0    # east  velocity error → desired roll  (deg)
PITCH_LIMIT      = 20.0   # deg
ROLL_LIMIT       = 20.0   # deg
K_P_PITCH        = 0.015
K_D_PITCH        = 0.001
K_P_ROLL         = 0.015
K_D_ROLL         = 0.001025
K_P_YAW          = 0.03
K_D_YAW          = 0.002
THRUST_TRIM      = 0.265  # experimentally validated hover trim from controller.py
K_P_THRUST       = 0.0925
K_D_THRUST       = 0.05
MAX_ELEV_UP_RATE = 3.0    # m/s cap on upward setpoint motion (mirrors controller.py)


# ── Math helpers (copied from controller.py) ──────────────────────────────────

def _euler_to_quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w, x, y, z]


def _quat_to_euler(qw, qx, qy, qz):
    roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))
    siny  = 2.0*(qw*qz + qx*qy)
    cosy  = 1.0 - 2.0*(qy*qy + qz*qz)
    yaw   = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _gate_approach_dir(gates, idx):
    cur = np.array([gates[idx]['pos_x'], gates[idx]['pos_y'], 0.0])
    prev = np.array([gates[idx-1]['pos_x'], gates[idx-1]['pos_y'], 0.0]) if idx > 0 else np.zeros(3)
    d = cur - prev
    n = float(np.linalg.norm(d))
    return d / n if n > 0.01 else np.array([1.0, 0.0, 0.0])


def _send_attitude(sim_conn, system_boot_ms, roll_r, pitch_r, yaw_r, thrust):
    q = _euler_to_quat(roll_r, pitch_r, yaw_r)
    now_ms = int(time.time() * 1000) - system_boot_ms
    sim_conn.mav.set_attitude_target_send(
        now_ms,
        sim_conn.target_system,
        sim_conn.target_component,
        7,           # typemask: ignore body rates, use attitude + thrust
        q,
        0.0, 0.0, 0.0,
        thrust,
    )


def _world_velocities(odo):
    """Body-frame velocities → world-NED frame using odometry quaternion."""
    qw, qx, qy, qz = odo['qw'], odo['qx'], odo['qy'], odo['qz']
    bx, by, bz = odo['vx'], odo['vy'], odo['vz']
    vx = (1-2*(qy**2+qz**2))*bx + 2*(qx*qy-qw*qz)*by + 2*(qx*qz+qw*qy)*bz
    vy = 2*(qx*qy+qw*qz)*bx + (1-2*(qx**2+qz**2))*by + 2*(qy*qz-qw*qx)*bz
    vz = 2*(qx*qz-qw*qy)*bx + 2*(qy*qz+qw*qx)*by + (1-2*(qx**2+qy**2))*bz
    return vx, vy, vz


def _approach_and_dwell(sim_conn, shared_data, system_boot_ms,
                        tx, ty, tz, target_yaw_rad, ctrl_state):
    """
    Fly to (tx, ty, tz) NED at target_yaw and hold for DWELL_S seconds once
    within POSITION_TOL_M.  Uses SET_ATTITUDE_TARGET with a P-cascade identical
    in structure to controller.py.

    ctrl_state is a dict that persists PD derivative terms and the rate-limited
    altitude setpoint across consecutive waypoints so there is no step discontinuity.
    """
    t0      = time.time()
    arrived = False
    t_arrive = None

    while True:
        with shared_data['lock']:
            odo = shared_data.get('odometry')

        if odo is None:
            time.sleep(_dt)
            continue

        x_pos, y_pos, z_pos = odo['x'], odo['y'], odo['z']
        roll_r, pitch_r, yaw_r = _quat_to_euler(odo['qw'], odo['qx'], odo['qy'], odo['qz'])
        roll_deg   = math.degrees(roll_r)
        pitch_deg  = math.degrees(pitch_r)
        vx_w, vy_w, vz_w = _world_velocities(odo)

        # ── Altitude: rate-limited setpoint → thrust PID ──────────────────
        if ctrl_state.get('elev_des_cmd') is None:
            ctrl_state['elev_des_cmd'] = z_pos
        delta = tz - ctrl_state['elev_des_cmd']
        max_up = MAX_ELEV_UP_RATE * _dt
        if delta < -max_up:
            delta = -max_up
        ctrl_state['elev_des_cmd'] += delta

        err_elev = ctrl_state['elev_des_cmd'] - z_pos
        qx, qy = odo['qx'], odo['qy']
        tilt_factor = max(0.01, 1.0 - 2.0*(qx**2 + qy**2))
        thrust_raw  = THRUST_TRIM - err_elev*K_P_THRUST + vz_w*K_D_THRUST
        thrust      = float(np.clip(thrust_raw / tilt_factor, 0.0, 1.0))

        # ── Horizontal: position → velocity → attitude ────────────────────
        vec_n = tx - x_pos
        vec_e = ty - y_pos
        v_des_n = float(np.clip(K_POS * vec_n, -V_MAX_HORIZ, V_MAX_HORIZ))
        v_des_e = float(np.clip(K_POS * vec_e, -V_MAX_HORIZ, V_MAX_HORIZ))

        # Desired pitch
        vx_err    = v_des_n - vx_w
        d_vx_err  = (vx_err - ctrl_state.get('prev_vx_err', 0.0)) / _dt
        ctrl_state['prev_vx_err'] = vx_err
        pitch_des = float(np.clip(K_VX_P * vx_err, -PITCH_LIMIT, PITCH_LIMIT))
        err_pitch  = pitch_des - pitch_deg
        pitch_cmd  = K_P_PITCH*err_pitch - K_D_PITCH*odo['pitchspeed']

        # Desired roll
        vy_err    = v_des_e - vy_w
        d_vy_err  = (vy_err - ctrl_state.get('prev_vy_err', 0.0)) / _dt
        ctrl_state['prev_vy_err'] = vy_err
        roll_des  = float(np.clip(K_VY_P * vy_err, -ROLL_LIMIT, ROLL_LIMIT))
        err_roll  = roll_des - roll_deg
        roll_cmd  = K_P_ROLL*err_roll - K_D_ROLL*odo['rollspeed']

        # Yaw toward target
        yaw_deg    = math.degrees(yaw_r)
        target_yaw_deg = math.degrees(target_yaw_rad)
        err_yaw    = (target_yaw_deg - yaw_deg + 180.0) % 360.0 - 180.0
        yaw_cmd    = K_P_YAW*err_yaw - K_D_YAW*odo['yawspeed']

        _send_attitude(sim_conn, system_boot_ms, roll_cmd, pitch_cmd, yaw_cmd, thrust)

        # ── Arrival / dwell / timeout ──────────────────────────────────────
        dist = math.sqrt((x_pos-tx)**2 + (y_pos-ty)**2 + (z_pos-tz)**2)
        if not arrived and dist < POSITION_TOL_M:
            arrived  = True
            t_arrive = time.time()
            print(f'    arrived  dist={dist:.2f} m', flush=True)

        if arrived and (time.time() - t_arrive) >= DWELL_S:
            break

        if not arrived and (time.time() - t0) > APPROACH_TIMEOUT_S:
            print(f'    TIMEOUT  dist={dist:.2f} m — moving on', flush=True)
            break

        time.sleep(_dt)


def main():
    from pymavlink import mavutil as mavu

    system_boot_ms = int(time.time() * 1000)
    shared_data    = {}

    print('Connecting to simulator...', flush=True)
    components = setup_components(
        shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    )
    sim_conn    = components['sim_conn']

    # Controller state persists across all waypoints
    ctrl_state = {}

    # ── Phase 1: Arm ──────────────────────────────────────────────────────────
    print('Arming drone...  (start a Training flight in the sim if stuck here)',
          flush=True)
    arm_ticks = 0
    while True:
        sim_conn.mav.command_long_send(
            sim_conn.target_system, sim_conn.target_component,
            mavu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
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
            print(f'  still waiting to arm... (armed={armed}, odo={"ok" if odo else "none"})',
                  flush=True)

    # ── Phase 2: Wait for race start ──────────────────────────────────────────
    print('Waiting for race start...', flush=True)
    anchor_ms  = None
    wait_ticks = 0
    while True:
        with shared_data['lock']:
            rs = shared_data.get('race_status')
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
                print(f'  sim_ms={sim_ms}  race_start={start_ms}  fresh={race_is_fresh}',
                      flush=True)
        else:
            if wait_ticks % 20 == 0:
                print('  no race_status yet...', flush=True)
        wait_ticks += 1
        time.sleep(0.1)

    # ── Phase 3: Try to receive GATE_INFO (VQ2) or fall back to course sweep ──
    print(f'Waiting up to {MAX_WAIT_GATE_S:.0f} s for GATE_INFO (VQ2 only)...',
          flush=True)
    deadline = time.time() + MAX_WAIT_GATE_S
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
            approach    = _gate_approach_dir(gates, gate_idx)
            target_yaw  = math.atan2(approach[1], approach[0])

            print(f'\n--- Gate {gate_idx}  NED=({gx:.1f}, {gy:.1f}, {gz:.1f})  '
                  f'yaw={math.degrees(target_yaw):.0f}° ---', flush=True)

            for standoff in STANDOFF_DISTANCES_M:
                tx = gx - approach[0] * standoff
                ty = gy - approach[1] * standoff
                tz = gz
                print(f'  standoff {standoff:>3d} m  ({tx:.1f}, {ty:.1f}, {tz:.1f})',
                      flush=True)
                _approach_and_dwell(sim_conn, shared_data, system_boot_ms,
                                    tx, ty, tz, target_yaw, ctrl_state)
                print('    done', flush=True)

    else:
        # VQ1 path: no gate positions — sweep the full course at spawn altitude.
        # vision_rx INSTRUMENT mode captures frames every 15 processed frames
        # throughout, giving coverage at all approach ranges for each gate.
        print('\nNo GATE_INFO — running VQ1 course sweep.', flush=True)

        with shared_data['lock']:
            odo = shared_data.get('odometry')
        spawn_x = odo['x'] if odo else 0.0
        spawn_y = odo['y'] if odo else 0.0
        spawn_z = odo['z'] if odo else -2.5

        SWEEP_STEP_M  = 8
        SWEEP_TOTAL_M = 148
        target_yaw    = math.atan2(0.0, -1.0)   # face south (−x = course direction)
        n_steps       = SWEEP_TOTAL_M // SWEEP_STEP_M

        print(f'Sweep: {n_steps} waypoints, {SWEEP_STEP_M} m apart, '
              f'from x={spawn_x:.1f} to x={spawn_x - SWEEP_TOTAL_M:.1f}', flush=True)

        for step in range(1, n_steps + 1):
            tx = spawn_x - step * SWEEP_STEP_M
            print(f'  wp {step:>2d}/{n_steps}  x={tx:.1f}', flush=True)
            _approach_and_dwell(sim_conn, shared_data, system_boot_ms,
                                tx, spawn_y, spawn_z, target_yaw, ctrl_state)
            print('    done', flush=True)

    print('\nData collection complete.  Frames saved to vision_dump/', flush=True)

    for name in ('heartbeat', 'ts_loop', 'mavlink_rx', 'vision_rx'):
        components[name].get_thread_for_join().join(timeout=2.0)


if __name__ == '__main__':
    main()
