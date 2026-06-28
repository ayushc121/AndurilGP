"""
train_controller.py — telemetry-guided training controller.

Flies the course using GATE_INFO coordinates so vision_rx INSTRUMENT mode
captures dense frames at all approach ranges.  Exits cleanly on gate collision
so the user can restart.

All FLYING gains are taken verbatim from the dfe7a74 commit that scored all
6 gates in one run.  Structural log written to train_log.jsonl.
"""

import json
import math
import os
import time

import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# ── Control constants (proven from dfe7a74 — do not change without testing) ──
CONTROL_HZ  = 50
_dt         = 1.0 / CONTROL_HZ
ARM_RETRY_S = 1.0

V_MAX       = 3.0    # m/s — slightly slower than original (4.0) for denser frames
K_POS       = 0.3    # pos error → velocity setpoint (full V_MAX beyond ~10 m)

K_VX        = 2.0    # deg of corrective pitch per m/s north-velocity error
K_VY        = 2.0    # deg of corrective roll  per m/s east-velocity error
PITCH_LIMIT = 12.0
ROLL_LIMIT  = 12.0
K_P_PITCH   = 0.015;  K_D_PITCH = 0.001
K_P_ROLL    = 0.015;  K_D_ROLL  = 0.001

THRUST_TRIM = 0.26567   # experimentally validated hover trim
K_P_THRUST  = 0.015
K_D_THRUST  = 0.022
ELEV_RAMP   = 1.5       # m/s — max rate of change of altitude setpoint (symmetric)
ELEV_CLEAR  = 1.0       # m above gate centre: fly upper half of the 2.72 m opening
ELEV_MIN    = -5.0      # NED clamp: 5 m up
ELEV_MAX    = 28.0      # NED clamp: 28 m down
ADVANCE_R   = 3.0       # m 3-D fallback-advance radius

GATE_WAIT_S      = 15.0
COLLISION_THRESH = 0.15

LOG_FILE = 'train_log.jsonl'

_FX = 320.0;  _CX = 320.0;  _CY = 180.0
_GATE_WIDTH_M = 2.7


class _Phase(Enum):
    WAIT_FOR_DATA  = auto()
    WAIT_FOR_START = auto()
    WAIT_FOR_GATES = auto()
    FLYING         = auto()


class TrainController:
    """Drop-in replacement for Controller, selected by main.py --train flag."""

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn       = sim_conn
        self.data           = data
        self.system_boot_ms = system_boot_ms
        self._finished      = False
        self._tick          = 0

        self._phase             = _Phase.WAIT_FOR_DATA
        self._last_arm_t        = 0.0
        self._wait_start_sim_ms = None
        self._gate_wait_t       = None

        self._wp               = 0        # local waypoint index (mirrors active_gate_index)
        self._gates_ref        = None     # reference for gate-packet change detection
        self._elev_des         = None     # seeded from z_pos on first FLYING tick
        self._current_gate_idx = None
        self._gate_approach_t  = None
        self._last_col_ts      = None
        self._last_vis_log_t   = 0.0

        self._log_fh = open(LOG_FILE, 'w', buffering=1)
        print(f'[TRAIN] Log → {LOG_FILE}', flush=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def is_finished(self):
        return self._finished

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, event, **kwargs):
        self._log_fh.write(json.dumps({'event': event, 'ts': time.time(), **kwargs}) + '\n')

    def _exit(self, reason):
        print(f'[TRAIN] Exiting: {reason}', flush=True)
        self._log('exit', reason=reason)
        self._log_fh.close()
        os._exit(0 if reason == 'course_complete' else 1)

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
        cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
        cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
        return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]

    @staticmethod
    def _world_vel(odo):
        qw, qx, qy, qz = odo['qw'], odo['qx'], odo['qy'], odo['qz']
        bx, by, bz     = odo['vx'], odo['vy'], odo['vz']
        vx = (1-2*(qy**2+qz**2))*bx + 2*(qx*qy-qw*qz)*by + 2*(qx*qz+qw*qy)*bz
        vy = 2*(qx*qy+qw*qz)*bx     + (1-2*(qx**2+qz**2))*by + 2*(qy*qz-qw*qx)*bz
        vz = 2*(qx*qz-qw*qy)*bx     + 2*(qy*qz+qw*qx)*by    + (1-2*(qx**2+qy**2))*bz
        return vx, vy, vz

    # ── Attitude output ───────────────────────────────────────────────────────

    def _send_attitude(self, roll_r, pitch_r, yaw_r, thrust):
        q      = self._euler_to_quat(roll_r, pitch_r, yaw_r)
        now_ms = int(time.time()*1000) - self.system_boot_ms
        self.sim_conn.mav.set_attitude_target_send(
            now_ms,
            self.sim_conn.target_system, self.sim_conn.target_component,
            7, q, 0.0, 0.0, 0.0, thrust,
        )

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self):
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            time.sleep(_dt)
            return

        with lock:
            odo         = self.data.get('odometry')
            race_status = self.data.get('race_status')
            armed       = self.data.get('armed', False)
            gates       = self.data.get('gates')
            collision   = self.data.get('last_collision')
            vision      = self.data.get('vision_gate_estimate')

        # ── WAIT_FOR_DATA ─────────────────────────────────────────────────────
        if self._phase == _Phase.WAIT_FOR_DATA:
            if not armed:
                now = time.time()
                if now - self._last_arm_t >= ARM_RETRY_S:
                    self.arm()
                    self._last_arm_t = now
            elif odo is not None:
                print('[TRAIN] Armed — waiting for race start.', flush=True)
                self._phase = _Phase.WAIT_FOR_START
            time.sleep(_dt)
            return

        # ── WAIT_FOR_START ────────────────────────────────────────────────────
        if self._phase == _Phase.WAIT_FOR_START:
            if race_status is not None:
                sim_ms   = race_status['sim_boot_time_ms']
                start_ms = race_status['race_start_boot_time_ms']
                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                race_is_fresh  = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_is_fresh and sim_ms >= start_ms
                if countdown_done:
                    print('[TRAIN] Race started — waiting for gate info.', flush=True)
                    self._log('race_start', sim_ms=sim_ms)
                    self._phase       = _Phase.WAIT_FOR_GATES
                    self._gate_wait_t = time.time()
            time.sleep(_dt)
            return

        # ── WAIT_FOR_GATES ────────────────────────────────────────────────────
        if self._phase == _Phase.WAIT_FOR_GATES:
            if gates:
                print(f'[TRAIN] Gate info received: {len(gates)} gates.', flush=True)
                self._log('gates_received',
                          count=len(gates),
                          gates=[{'id': g['gate_id'],
                                  'pos': [round(g['pos_x'],2),
                                          round(g['pos_y'],2),
                                          round(g['pos_z'],2)]}
                                 for g in gates])
                self._phase = _Phase.FLYING
            elif time.time() - self._gate_wait_t > GATE_WAIT_S:
                print('[TRAIN] No GATE_INFO received — cannot fly.', flush=True)
                self._exit('no_gate_info')
            time.sleep(_dt)
            return

        # ── FLYING ────────────────────────────────────────────────────────────
        if self._phase != _Phase.FLYING or odo is None:
            time.sleep(_dt)
            return

        # Collision check
        if collision is not None and collision['ts'] != self._last_col_ts:
            self._last_col_ts = collision['ts']
            if collision['type'] == 'gate' and collision['impulse'] > COLLISION_THRESH:
                print(f"[TRAIN] Gate collision impulse={collision['impulse']:.3f}", flush=True)
                self._log('collision', type=collision['type'],
                          impulse=round(collision['impulse'], 4),
                          gate_idx=self._current_gate_idx)
                self._exit('collision')

        x_pos = odo['x'];  y_pos = odo['y'];  z_pos = odo['z']
        qw, qx, qy, qz = odo['qw'], odo['qx'], odo['qy'], odo['qz']

        roll_deg  = math.degrees(math.atan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)))
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy-qz*qx)))))

        vx_world, vy_world, vz_world = self._world_vel(odo)

        # ── Gate targeting (verbatim from dfe7a74 guidance block) ─────────────
        if gates and gates is not self._gates_ref:
            self._gates_ref = gates
            print(f'[GATE] {len(gates)} gates, drone@receipt=({x_pos:.2f},{y_pos:.2f},{z_pos:.2f})',
                  flush=True)

        n = len(gates) if gates else 0
        v_des_north = 0.0
        v_des_east  = 0.0
        elev_des    = -3.0

        if gates:
            active_idx = race_status['active_gate_index'] if race_status else -1

            # Gate advanced — log pass
            if active_idx != self._current_gate_idx and active_idx >= 0:
                if self._current_gate_idx is not None:
                    elapsed = round(time.time() - self._gate_approach_t, 2)
                    print(f'[TRAIN] Gate {self._current_gate_idx} passed in {elapsed}s', flush=True)
                    self._log('gate_pass', gate_idx=self._current_gate_idx, elapsed_s=elapsed)
                self._current_gate_idx = active_idx
                self._gate_approach_t  = time.time()
                ga = gates[active_idx]
                dist = math.sqrt((x_pos-ga['pos_x'])**2 + (y_pos-ga['pos_y'])**2
                                 + (z_pos-ga['pos_z'])**2)
                print(f'[TRAIN] Targeting gate {active_idx}  dist={dist:.1f}m', flush=True)
                self._log('gate_approach',
                          gate_idx=active_idx,
                          gate_pos=[round(ga['pos_x'],2), round(ga['pos_y'],2), round(ga['pos_z'],2)],
                          drone_pos=[round(x_pos,2), round(y_pos,2), round(z_pos,2)],
                          dist_m=round(dist, 1))

            if active_idx > self._wp:
                self._wp = active_idx
            self._wp = max(0, min(self._wp, n-1))

            if active_idx >= n:
                print('[TRAIN] Course complete!', flush=True)
                self._log('course_complete', gates_passed=active_idx)
                self._exit('course_complete')

            ga      = gates[self._wp]
            g_north =  ga['pos_x']
            g_east  =  ga['pos_y']
            g_down  =  ga['pos_z']   # NED: positive = deeper down, same sign as z_pos

            vec_n      = g_north - x_pos
            vec_e      = g_east  - y_pos
            vec_d      = g_down  - z_pos
            horiz_dist = math.hypot(vec_n, vec_e)
            dist3d     = math.sqrt(vec_n**2 + vec_e**2 + vec_d**2)

            v_des_north = float(np.clip(K_POS * vec_n, -V_MAX, V_MAX))
            v_des_east  = float(np.clip(K_POS * vec_e, -V_MAX, V_MAX))

            # Altitude: ramp toward gate depth - ELEV_CLEAR (fly 1m above centre)
            elev_target = g_down - ELEV_CLEAR
            if self._elev_des is None:
                self._elev_des = z_pos
            step = float(np.clip(elev_target - self._elev_des,
                                 -ELEV_RAMP / CONTROL_HZ, ELEV_RAMP / CONTROL_HZ))
            self._elev_des = float(np.clip(self._elev_des + step, ELEV_MIN, ELEV_MAX))
            elev_des = self._elev_des

            # Local fallback advance: within ADVANCE_R or crossed gate plane
            passed_plane = (vec_n > 0.0 and horiz_dist < 5.0)
            if self._wp < n-1 and (dist3d < ADVANCE_R or passed_plane):
                self._wp += 1
                print(f'[WP] advance -> gate {self._wp} (dist3d={dist3d:.1f}m, plane={passed_plane})',
                      flush=True)

        # Vision log — at most once per second
        now = time.time()
        if vision is not None and now - self._last_vis_log_t >= 1.0:
            self._last_vis_log_t = now
            bw = vision.get('bw', 0)
            if bw > 0:
                cx  = vision.get('bx', 0) + bw/2
                cy  = vision.get('by', 0) + vision.get('bh', 0)/2
                ray = math.sqrt((_CX-cx)**2 + (_CY-cy)**2 + _FX**2)
                range_m = round(_GATE_WIDTH_M * ray / bw, 1)
            else:
                range_m = None
            self._log('vision_detect', gate_idx=self._current_gate_idx,
                      range_m=range_m,
                      bbox=[vision.get('bx'), vision.get('by'),
                            vision.get('bw'), vision.get('bh')])

        # ── Pitch PID ─────────────────────────────────────────────────────────
        pitch_des = float(np.clip(K_VX * (v_des_north - vx_world), -PITCH_LIMIT, PITCH_LIMIT))
        pitch_cmd = K_P_PITCH * (pitch_des - pitch_deg) - K_D_PITCH * odo['pitchspeed']

        # ── Roll PID ──────────────────────────────────────────────────────────
        roll_des = float(np.clip(K_VY * (v_des_east - vy_world), -ROLL_LIMIT, ROLL_LIMIT))
        roll_cmd = K_P_ROLL * (roll_des - roll_deg) - K_D_ROLL * odo['rollspeed']

        # ── Thrust PID ────────────────────────────────────────────────────────
        err_elev     = elev_des - z_pos
        thrust_cmd   = float(np.clip(
            THRUST_TRIM - err_elev * K_P_THRUST + vz_world * K_D_THRUST, 0.0, 1.0))

        self._send_attitude(roll_cmd, pitch_cmd, 0.0, thrust_cmd)
        time.sleep(_dt)
