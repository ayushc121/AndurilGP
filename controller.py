import time
import math
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

# -----------------------------------------------------------------------
# Configuration — tune these once the drone is in the air
# -----------------------------------------------------------------------

CONTROL_HZ = 50          # spec hard-limits < 100 Hz
FX = FY      = 320.0

ARM_RETRY_S      = 1.0
POST_DISARM_WAIT = 0.25

DEBUG_EVERY_N = 50        # ~1 s at 50 Hz

MAVLINK_CMD_SIM_RESET = 31000

dt = 1.0 / CONTROL_HZ 

# -----------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """
    Roll/pitch/yaw (radians, ZYX convention) → quaternion [w, x, y, z].

    NED body-frame sign conventions:
      positive pitch = nose UP  → negative pitch = fly forward
      positive roll  = right side DOWN
      positive yaw   = clockwise from above (North→East)
    """
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w, x, y, z]


def quat_to_yaw(qw, qx, qy, qz):
    siny = 2.0 * (qw*qz + qx*qy)
    cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny, cosy)


# -----------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------

class Phase(Enum):
    WAIT_FOR_DATA  = auto()
    WAIT_FOR_START = auto()
    FLYING         = auto()


# -----------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------

class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn       = sim_conn
        self.data           = data
        self.system_boot_ms = system_boot_ms

        self._was_armed = False
        self._disarm_at = None
        self._reset_flight_state()

    def _reset_flight_state(self):
        self.phase              = Phase.WAIT_FOR_DATA
        self._finished          = False
        self._last_arm_attempt  = 0.0
        self._tick              = 0
        self._wait_start_sim_ms = None
        self.prev_vy_err        = 0.0
        self.prev_vx_err        = 0.0
        self._elev_des_cmd      = None   # rate-limited altitude setpoint (Stage B)
        print('Controller state reset.', flush=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_finished(self):
        return self._finished

    def arm(self):
        self._send_arm()

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0
        )


    def _send_attitude_rates(self, roll_rad, pitch_rad, yaw_rad, thrust):
        """
        Sends raw roll/pitch/yaw RATES and thrust commands.
        Thrust is a value between 0.0 (motors off) and 1.0 (full throttle).
        """
        now_ms = int(time.time() * 1000)
        
        # typemask 7 (0b00000111) means "IGNORE body rates, USE attitude and thrust"
        typemask = 7 
        
        # Convert the desired Euler angles to a Quaternion
        q = euler_to_quat(roll_rad, pitch_rad, yaw_rad)
        
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            typemask,
            q,
            0, 0, 0,  # (Ignored by the typemask)
            thrust
        )

    # ------------------------------------------------------------------
    # Vision geometry
    # ------------------------------------------------------------------

    def _gate_world_direction(self, true_cx, true_cy, roll_deg, pitch_deg, yaw_deg):
        """
        Unit vector in NED world frame pointing from the drone toward the gate,
        derived from the gate's pixel position + the current attitude. It depends
        only on BEARING (pixel offset + attitude), NOT on the gate's apparent
        width -> it is trustworthy even for a weak/clipped detection whose
        width->range estimate is not. Camera is tilted 20° UP from the body frame.

        This is the single source of "which way is the gate"; the caller scales it
        by the true range (reliable detection) or a fixed look-ahead (weak hint).
        Generalizes to a gate in ANY direction (up/down/left/right) — there is no
        descend-only or straight-ahead assumption.
        """
        FX = 320.0
        CX = 320.0
        CY = 180.0

        # Pixel vector in camera frame, normalized to a pure direction.
        vc_x = true_cx - CX
        vc_y = true_cy - CY
        vc_z = FX
        ray_norm = math.sqrt(vc_x * vc_x + vc_y * vc_y + vc_z * vc_z)
        rc_x = vc_x / ray_norm
        rc_y = vc_y / ray_norm
        rc_z = vc_z / ray_norm

        # Camera -> body (apply +20° camera tilt).
        tilt_rad = math.radians(20.0)
        ctilt = math.cos(tilt_rad); stilt = math.sin(tilt_rad)
        rb_x = rc_z * ctilt + rc_y * stilt
        rb_y = rc_x
        rb_z = -rc_z * stilt + rc_y * ctilt

        # Body -> NED world (roll, then pitch, then yaw).
        phi = math.radians(roll_deg)
        theta = math.radians(pitch_deg)
        psi = math.radians(yaw_deg)
        c_phi = math.cos(phi); s_phi = math.sin(phi)
        c_the = math.cos(theta); s_the = math.sin(theta)
        c_psi = math.cos(psi); s_psi = math.sin(psi)

        r1_x = rb_x
        r1_y = rb_y * c_phi - rb_z * s_phi
        r1_z = rb_y * s_phi + rb_z * c_phi

        r2_x = r1_x * c_the + r1_z * s_the
        r2_y = r1_y
        r2_z = -r1_x * s_the + r1_z * c_the

        rw_x = r2_x * c_psi - r2_y * s_psi
        rw_y = r2_x * s_psi + r2_y * c_psi
        rw_z = r2_z
        return rw_x, rw_y, rw_z

    # ------------------------------------------------------------------
    # Main update — called at CONTROL_HZ from main loop
    # ------------------------------------------------------------------

    def update(self):
        self._tick += 1
        lock = self.data.get('lock')
        if lock is None:
            time.sleep(1.0 / CONTROL_HZ)
            return

        with lock:
            odometry    = self.data.get('odometry')
            race_status = self.data.get('race_status')
            armed       = self.data.get('armed', False)
            gates = self.data.get('gates')

        # ------------------------------------------------------------------
        # Disarm / sim-restart detection
        # ------------------------------------------------------------------
        if self._was_armed and not armed:
            if self._disarm_at is None:
                print('Disarm detected — waiting before re-arm.', flush=True)
                self._disarm_at = time.time()
                with lock:
                    self.data['odometry']    = None
                    self.data['race_status'] = None
                    self.data['gates']       = None
                self._reset_flight_state()
            self._was_armed = armed
            time.sleep(1.0 / CONTROL_HZ)
            return

        if not armed and self._disarm_at is not None:
            if time.time() - self._disarm_at >= POST_DISARM_WAIT:
                print('Post-disarm wait done. Ready to re-arm.', flush=True)
                self._disarm_at        = None
                self._last_arm_attempt = 0.0
            else:
                self._was_armed = armed
                time.sleep(1.0 / CONTROL_HZ)
                return

        self._was_armed = armed
        
        # ------------------------------------------------------------------
        # WAIT_FOR_DATA
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_DATA:
            if not armed:
                now = time.time()
                if now - self._last_arm_attempt >= ARM_RETRY_S:
                    print('Sending arm command...', flush=True)
                    self._send_arm()
                    self._last_arm_attempt = now
            elif odometry is not None:
                print('Armed and data ready. Moving to WAIT_FOR_START.', flush=True)
                self.phase = Phase.WAIT_FOR_START
                
            time.sleep(1.0 / CONTROL_HZ)
            return
        
        # ------------------------------------------------------------------
        # WAIT_FOR_START 
        # ------------------------------------------------------------------
        if self.phase == Phase.WAIT_FOR_START:

            if race_status is not None:
                sim_ms   = race_status['sim_boot_time_ms']
                start_ms = race_status['race_start_boot_time_ms']

                if self._wait_start_sim_ms is None:
                    self._wait_start_sim_ms = sim_ms
                    print(f'[WAIT] Anchor set: sim_ms={sim_ms}', flush=True)

                race_is_fresh  = start_ms > 0 and start_ms >= self._wait_start_sim_ms
                countdown_done = race_is_fresh and sim_ms >= start_ms

                if self._tick % DEBUG_EVERY_N == 0:
                    print(f'[WAIT] sim_ms={sim_ms}  race_start={start_ms}  fresh={race_is_fresh}  go={countdown_done}', flush=True)

                if countdown_done:
                    print(f'Countdown complete! Flying!', flush=True)
                    self.phase = Phase.FLYING

            elif self._tick % DEBUG_EVERY_N == 0:
                print('[WAIT] No race_status yet — holding...', flush=True)

            time.sleep(1.0 / CONTROL_HZ)
            return

        # ------------------------------------------------------------------
        # FLYING
        # ------------------------------------------------------------------
        if self.phase == Phase.FLYING:
            if odometry is None:
                time.sleep(1.0 / CONTROL_HZ)
                return

            # EXTRACTING RELEVANT DATA FROM ODOMETRY
            yaw = quat_to_yaw(
                odometry['qw'], odometry['qx'],
                odometry['qy'], odometry['qz']
            )
            yaw_deg = math.degrees(yaw)

            roll_deg  = math.degrees(math.atan2(
                    2*(odometry['qw']*odometry['qx'] + odometry['qy']*odometry['qz']),
                    1 - 2*(odometry['qx']**2 + odometry['qy']**2)
                ))

            pitch_deg = math.degrees(math.asin(max(-1, min(1,
                    2*(odometry['qw']*odometry['qy'] - odometry['qz']*odometry['qx'])
                ))))

            yaw_rate = odometry['yawspeed']
            roll_rate = odometry['rollspeed']
            pitch_rate = odometry['pitchspeed']

            x_pos = odometry["x"]
            y_pos = odometry["y"]
            z_pos = odometry["z"]

            x_v = odometry["vx"]
            y_v = odometry["vy"]
            z_v = odometry["vz"]

            # ------------------------------------------------------------
            # FRAME CORRECTION — true world-frame velocities
            # ------------------------------------------------------------
            qw, qx, qy, qz = odometry['qw'], odometry['qx'], odometry['qy'], odometry['qz']
            vz_world = (2.0*(qx*qz - qw*qy) * x_v
                        + 2.0*(qy*qz + qw*qx) * y_v
                        + (1.0 - 2.0*(qx*qx + qy*qy)) * z_v)
            # World-frame x (north) velocity
            vx_world = ((1.0 - 2.0*(qy*qy + qz*qz)) * x_v
                        + 2.0*(qx*qy - qw*qz) * y_v
                        + 2.0*(qx*qz + qw*qy) * z_v)
            # World-frame y (east) velocity             
            vy_world = (2.0*(qx*qy + qw*qz) * x_v
                        + (1.0 - 2.0*(qx*qx + qz*qz)) * y_v
                        + 2.0*(qy*qz - qw*qx) * z_v)

            if self._tick % DEBUG_EVERY_N == 0:
                print(
                    f'[FLY] pos=({x_pos:.1f},{y_pos:.1f},'
                    f'{z_pos:.2f})  '
                    f'vel=({x_v:.2f},{y_v:.2f},'
                    f'{z_v:.2f})  '
                    f'roll={roll_deg:.1f}° pitch={pitch_deg:.1f}° yaw={yaw_deg:.1f}°',
                    flush=True
                )


            # ================================================================
            # VISUAL BASED GATE TARGETING
            # ----------------------------------------------------------------
            vision = self.data.get('vision_gate_estimate')
            est_distance_3d = float('nan')   # set only by the reliable back-projection

            if vision is not None:
                # ==========================================
                # STEER TOWARD THE GATE ALONG ITS MEASURED BEARING
                # ------------------------------------------------------------------
                # Same logic for a strong or a weak detection: aim the target at a
                # point along the gate's bearing (pixel + attitude -> world unit
                # vector, reliable regardless of apparent width). The ONLY difference
                # is how far along that bearing we place the target:
                #   reliable  -> the true range from the known gate width (2.7 m).
                #   weak hint -> a fixed look-ahead (range from a clipped/small blob
                #                is untrustworthy; the BEARING still steers us toward
                #                it laterally + vertically until it grows reliable).
                # No descend-only / straight-ahead assumption -> generalizes to a gate
                # in any position. Forward, lateral AND vertical are all steered, which
                # is what was missing (frozen lateral wedged the drone into gate 2).
                FX = 320.0
                CX = 320.0
                CY = 180.0

                true_cx = vision['bx'] + (vision['bw'] / 2.0)
                true_cy = vision['by'] + (vision['bh'] / 2.0)

                rw_x, rw_y, rw_z = self._gate_world_direction(
                    true_cx, true_cy, roll_deg, pitch_deg, yaw_deg)

                # Range from the known gate width (2.7 m) via pin-hole, scaled to the
                # full 3D ray length (= 2.7 * |ray| / bbox_w). This is valid for ANY
                # detection that is not HORIZONTALLY clipped (those are rejected to None
                # upstream) -- vertical clipping/low gates keep a full width, so the
                # range is still good. Using it CONTINUOUSLY (incl. weak hints) avoids
                # the target-jump that a fixed look-ahead caused whenever a detection
                # flickered reliable<->weak (range snapping 10 m <-> true): that jump
                # wobbled the drone at the start and flip-flopped it into gate 1.
                vc_x = true_cx - CX
                vc_y = true_cy - CY
                ray_norm = math.sqrt(vc_x * vc_x + vc_y * vc_y + FX * FX)
                est_distance_3d = 2.7 * ray_norm / vision['bw']

                g_north = x_pos + (est_distance_3d * rw_x)
                g_east  = y_pos + (est_distance_3d * rw_y)
                gate_pz = z_pos + (est_distance_3d * rw_z) + 1
                self._servo = not vision.get('reliable', True)   # diagnostics/log only

                # Remember the last good vision target + reset the blind counter, so a
                # brief detection dropout coasts on it instead of immediately going blind.
                self._last_gate = (g_north, g_east, gate_pz)
                self._blind = 0
                self._acquiring = False   # have a gate -> steering to it, not acquiring

            else:
                # BLIND BEHAVIOR (pure-CV). Do NOT target the origin: the old
                # g=(0,0,-3) plus the 35 m/s north blast flew the drone clean off the
                # map the instant it lost sight of a gate (it clipped gate, rotated off
                # axis, never re-detected, blasted toward origin, over-angled). Instead:
                # COAST toward the last seen gate for a short window (ride through brief
                # dropouts), then HOLD position (target = current pose -> zero velocity
                # setpoint) so it stays controllable until it re-acquires a gate.
                self._blind = getattr(self, '_blind', 0) + 1
                last = getattr(self, '_last_gate', None)
                BLIND_HOLD_TICKS  = 100    # ~2 s at 50 Hz: COAST toward the last gate
                # through descent flicker (gate riding the bottom edge drops in/out of
                # view). Was 25 (~0.5 s): too short, so ACQUIRE kept firing mid-descent
                # -> forced -15 deg pitch + zeroed forward + held altitude, interrupting
                # the descent ("pitch fight", stayed high). Coasting keeps descending and
                # steering at the last back-projected gate until it re-locks.
                ACQUIRE_MAX_TICKS = 150    # ~3 s of active search before giving up
                self._acquiring = False
                self._servo = False
                if last is not None and self._blind <= BLIND_HOLD_TICKS:
                    # Coast on the last good gate target -- but NEVER reverse toward a
                    # gate we've already PASSED. We fly -north, so a gate behind us has
                    # g_north > x_pos -> that gave v_des_north > 0 (backward), pitching
                    # up and stalling after each crossing, then an acquire slam. Clamp
                    # the north target to stay at least FORWARD_COAST ahead so we glide
                    # forward while still tracking its lateral + altitude.
                    FORWARD_COAST = 2.0
                    g_north, g_east, gate_pz = last
                    g_north = min(g_north, x_pos - FORWARD_COAST)
                elif self._blind <= BLIND_HOLD_TICKS + ACQUIRE_MAX_TICKS:
                    # ACQUIRE (Phase 3): the next gate sits BELOW the 20°-up camera on
                    # the descending course (offline sweep: 12-17° below horizon ->
                    # py>360, out the bottom). HOLD altitude + no lateral, and let the
                    # pitch section force a nose-down camera aim so the lower gate rises
                    # into frame; the resulting forward glide closes on it. As soon as a
                    # gate is detected the vision branch takes over and brakes/descends.
                    g_north, g_east, gate_pz = x_pos, y_pos, z_pos + 0.8
                    self._acquiring = True
                else:
                    # Searched long enough with no gate -> HOLD (level + brake) so a
                    # failed search can't fly the drone off the map.
                    g_north, g_east, gate_pz = x_pos, y_pos, z_pos + 0.8

            # --- DIAG: capture the VISION-derived target before the telemetry override
            # so we can measure vision-localization accuracy against telemetry ground
            # truth. Pure diagnostics; does not affect steering.
            vision_present = vision is not None
            g_north_vis, g_east_vis, gate_pz_vis = g_north, g_east, gate_pz
            est_dist_vis = est_distance_3d if vision_present else float('nan')
            telem_present = False
            telem_idx = -1
            telem_n = telem_e = telem_pz = float('nan')

            # ================================================================
            # TELEMETRY GATE TARGET — LOGGING ONLY (PURE-CV: does NOT steer)
            # ----------------------------------------------------------------
            # Phase 0: the drone navigates on VISION ONLY. The telemetry track
            # packet is retained solely as a ground-truth ORACLE for nav_diag.csv
            # (so we can measure how far the vision back-projection is from truth).
            # It must NOT overwrite g_north/g_east/gate_pz — doing so silently flew
            # the drone on telemetry and masked the real CV performance.
            if gates:
                if gates is not getattr(self, '_gates_ref', None):
                    self._gates_ref = gates
                    print(f'[GATE] track packet: {len(gates)} gates, '
                          f'drone@receipt=({x_pos:.2f},{y_pos:.2f},{z_pos:.2f})', flush=True)


                active_idx = race_status['active_gate_index'] if race_status else 999
                if active_idx < len(gates):
                    ga = gates[active_idx]

                    # --- DIAG ONLY: record telemetry ground truth for the comparison
                    # log. Intentionally does NOT touch g_north/g_east/gate_pz.
                    telem_present = True
                    telem_idx = active_idx
                    telem_n, telem_e, telem_pz = ga['pos_x'], ga['pos_y'], ga['pos_z']


            # ================================================================
            # DESIRED PATH GENERATION
            # ----------------------------------------------------------------

            # CHANGE B — slow-into-gate profile (accuracy first; speed optimized later).
            # The old law floored north at 35 m/s (`35*sign(vec_n) + 0.15*...`), which
            # crossed each gate's plane before the east/altitude loops could converge ->
            # passed through 0 gates despite no collisions (misses 2-17 m).
            V_MAX       = 8.0    # m/s horizontal cap (was 20; lowered for accuracy)
            K_POS       = 1.7    # m/s per m of horizontal error (P slows as it nears)
            ALIGN_SCALE = 4.0    # m of LATERAL miss that halves north speed

            v_des_north = 0.0           # setpoints default to hover / hold when no gate
            v_des_east  = 0.0
            elev_des    = -3.0

            vec_n = g_north - x_pos
            vec_e = g_east  - y_pos
            vec_d = gate_pz - z_pos      # vertical miss to the gate (NED down)

            # Proper P-on-position velocity setpoints, capped -> auto-slow into the gate.
            v_des_north = float(np.clip(K_POS * vec_n, -V_MAX, V_MAX))
            v_des_east  = float(np.clip(K_POS * vec_e, -V_MAX, V_MAX))

            # SLOW INTO THE GATE: scale north speed down while not yet aligned LATERALLY
            # with the opening, so east CONVERGES before we cross the gate plane.
            # NOTE: the vertical miss (vec_d) was intentionally REMOVED from this throttle.
            # Including it zeroed forward speed whenever the gate was below -> the drone
            # dropped in place then crept forward ("flappy bird") on the descending
            # course. Vertical is handled separately by elev_des/thrust, so dropping it
            # here lets the drone descend WHILE flying forward (smooth diagonal glide).
            align = ALIGN_SCALE / (ALIGN_SCALE + abs(vec_e))
            v_des_north *= align

            # Altitude (Stage B) — RATE-LIMIT the setpoint toward the gate altitude.
            # The back-projected gate_pz steps ~8 m the instant a gate is detected and
            # jitters frame-to-frame, which slammed thrust (vz~6.8) and added a vertical
            # wobble. Ramp elev_des toward its target at <= MAX_ELEV_RATE so the descent
            # is gradual and frame jitter is filtered out. (Setpoint shaping only -- the
            # thrust PID is untouched, per the guidance/control split.)
            # ASYMMETRIC: descend as fast as the gate demands (down = no cap, so the
            # setpoint never lags the descending course), but rate-limit UPWARD motion
            # of the setpoint so close-range back-projection garbage can't yank the
            # target up into the gate's top bar. (down = increasing z in NED.)
            MAX_ELEV_UP_RATE = 3.0       # m/s cap on UPWARD setpoint motion only
            elev_target = gate_pz - 0.8
            if self._elev_des_cmd is None:
                self._elev_des_cmd = z_pos          # start from current altitude
            delta = elev_target - self._elev_des_cmd
            max_up = MAX_ELEV_UP_RATE * dt
            if delta < -max_up:          # target jumped UP faster than the cap
                delta = -max_up
            self._elev_des_cmd += delta
            elev_des = self._elev_des_cmd

            # --- DIAG: vision-vs-telemetry navigation log (no behaviour change). One
            # row per tick: drone pose, the VISION-derived gate target, the TELEMETRY
            # ground-truth target, presence flags, and the FINAL target actually used
            # (telemetry overrides vision when present). On a steady approach to the
            # same gate, vision (g_*_vis) should match telemetry (telem_*) within a few
            # metres if the back-projection is trustworthy. nan = that source absent.
            nav_csv = getattr(self, '_nav_csv', None)
            if nav_csv is None:
                nav_csv = open('nav_diag.csv', 'w', buffering=1)
                nav_csv.write('tick,drone_x,drone_y,drone_z,yaw_deg,'
                              'vision_present,est_dist_vis,gN_vis,gE_vis,gpz_vis,'
                              'telem_present,telem_idx,telem_n,telem_e,telem_pz,'
                              'gN_final,gE_final,gpz_final,vec_n,vec_e\n')
                self._nav_csv = nav_csv
            nav_csv.write(
                f'{self._tick},{x_pos:.3f},{y_pos:.3f},{z_pos:.3f},{yaw_deg:.1f},'
                f'{int(vision_present)},{est_dist_vis:.3f},{g_north_vis:.3f},{g_east_vis:.3f},{gate_pz_vis:.3f},'
                f'{int(telem_present)},{telem_idx},{telem_n:.3f},{telem_e:.3f},{telem_pz:.3f},'
                f'{g_north:.3f},{g_east:.3f},{gate_pz:.3f},{vec_n:.3f},{vec_e:.3f}\n')

            
            # ================================================================
            # PID CONTROLLERS
            # ----------------------------------------------------------------

            # PITCH PID CONTROLLER
            K_VX_P = 1.5           # deg of corrective pitch per m/s of north-speed error
            K_VX_D = 0
            PITCH_LIMIT = 30.0   # deg, max tilt (B: was 50; gentler for accuracy)

            vx_err = v_des_north - vx_world
            d_vx_err = (vx_err - self.prev_vx_err) / dt

            self.prev_vx_err = vx_err

            pitch_des_raw = (K_VX_P * vx_err) + (K_VX_D * d_vx_err)

            # CHANGE B — relax the pitch clamp. It was clip(raw, -50, -20), forcing a
            # permanent >=20 deg dive so the drone could never level or brake (it just
            # "slammed the gas"). Symmetric clamp lets it pitch up to brake / slow into
            # gates and hold altitude on the descent.
            pitch_des = float(np.clip(pitch_des_raw, -PITCH_LIMIT, PITCH_LIMIT))

            # PHASE 3 — camera-aim during ACQUIRE. The velocity loop alone re-levels at
            # cruise (vx_err->0), pointing the 20°-up camera back above the descending
            # course so the next (lower) gate never enters frame. While acquiring, force
            # at least AIM_PITCH_DEG of nose-down (negative = nose down) so the camera
            # looks down-forward; the offline sweep shows ~-15° brings every next gate
            # into frame (py 249-311 vs 382-423 at level). min() = "at least this much
            # nose-down" regardless of the velocity term. Bounded by ACQUIRE_MAX_TICKS.
            AIM_PITCH_DEG = -15.0
            if getattr(self, '_acquiring', False):
                pitch_des = float(np.clip(min(pitch_des_raw, AIM_PITCH_DEG),
                                          -PITCH_LIMIT, PITCH_LIMIT))

            K_P_pitch = 0.015
            K_D_pitch = 0.001

            err_pitch = pitch_des - pitch_deg

            pitchCommand = K_P_pitch*err_pitch  -  K_D_pitch*pitch_rate



            # ROLL PID CONTROLLER
            # Tamed for deliberate flight: was K_VY_P=30, K_VY_D=7.25, LIMIT=50 -- a
            # huge proportional gain and an enormous derivative gain on a noisy east
            # setpoint produced violent left-right banking (tilted-horizon frames).
            # Pitch works fine at K_VX_P=1.5; bring roll into the same regime.
            K_VY_P = 3.0         # deg of tilt per m/s of east-velocity error (was 30)
            K_VY_D = 0.5         # derivative (was 7.25 -> amplified noise into wobble)
            ROLL_LIMIT = 25.0    # deg, max tilt (was 50)
            
            vy_err = v_des_east - vy_world
            d_vy_err = (vy_err - self.prev_vy_err) / dt
            
            self.prev_vy_err = vy_err
            
            roll_des_raw = (K_VY_P * vy_err) + (K_VY_D * d_vy_err)
            roll_des = float(np.clip(roll_des_raw, -ROLL_LIMIT, ROLL_LIMIT))
            
            K_P_roll = 0.015    # match pitch
            K_D_roll = 0.001025    # raised with K_P for damping

            err_roll = roll_des - roll_deg

            rollCommand = K_P_roll*err_roll  -  K_D_roll*roll_rate


            # YAW PID
            K_P_yaw = 0.03
            K_D_yaw = 0.002

            err_yaw = 180 - yaw_deg
            err_yaw = (err_yaw + 180.0) % 360.0 - 180.0

            yawCommand = K_P_yaw*err_yaw  -  K_D_yaw*yaw_rate


            # THRUST PID
            thrust_trim = 0.265  # experimentally determined, this is damn near correct +/- 0.0001
            
            K_P_thrust = 0.0925
            K_D_thrust = 0.05

            err_elev = elev_des - z_pos

            thrustCommand = thrust_trim - err_elev * K_P_thrust  + vz_world * K_D_thrust

            tilt_factor = 1.0 - 2.0 * (qx**2 + qy**2)
            if tilt_factor < 0.01: 
                tilt_factor = 0.01
            thrustCommand = thrustCommand / tilt_factor

            if thrustCommand > 1:
                print("Over Angled; thrust cannot maintain elevation")
            
            thrustCommand = np.clip(thrustCommand, 0, 1)

            # --- altitude / attitude log (diagnostics; no behaviour change) ----------
            # Per-tick record of the attitude loops so the roll/pitch behaviour is
            # visible offline (the previous altitude_log.csv was stale -- this controller
            # never wrote it). Watch roll_deg vs roll_des to see the violent banking, and
            # z vs elev_des for altitude tracking.
            alt_csv = getattr(self, '_alt_csv', None)
            if alt_csv is None:
                alt_csv = open('altitude_log.csv', 'w', buffering=1)
                alt_csv.write('tick,x,y,z,elev_des,vx_world,vy_world,vz_world,'
                              'v_des_north,v_des_east,roll_deg,roll_des,'
                              'pitch_deg,pitch_des,yaw_deg,'
                              'rollCommand,pitchCommand,thrust,acq,srv\n')
                self._alt_csv = alt_csv
            alt_csv.write(
                f'{self._tick},{x_pos:.3f},{y_pos:.3f},{z_pos:.3f},{elev_des:.3f},'
                f'{vx_world:.3f},{vy_world:.3f},{vz_world:.3f},'
                f'{v_des_north:.3f},{v_des_east:.3f},{roll_deg:.2f},{roll_des:.2f},'
                f'{pitch_deg:.2f},{pitch_des:.2f},{yaw_deg:.2f},'
                f'{rollCommand:.4f},{pitchCommand:.4f},{thrustCommand:.4f},'
                f'{int(getattr(self, "_acquiring", False))},'
                f'{int(getattr(self, "_servo", False))}\n')

            self._send_attitude_rates(rollCommand, pitchCommand, yawCommand, thrustCommand)


            time.sleep(1.0 / CONTROL_HZ)

    # ------------------------------------------------------------------
    # MAVLink helpers
    # ------------------------------------------------------------------

    def _send_arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
