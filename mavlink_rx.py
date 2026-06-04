import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.lock = data['lock']
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

        # Throttle collision log to at most once per second
        self._last_collision_print = 0.0
        self._collision_count_since_last_print = 0

        with self.lock:
            self.data.update({
                'armed':                False,
                'attitude':             None,
                'odometry':             None,
                'position':             None,
                'velocity':             None,
                'imu':                  None,
                'race_status':          None,
                'gates':                None,
                'last_collision':       None,
                'motor_feedback':       None,
                'clock_offset_ns':      0,
                'vision_gate_estimate': None,
            })

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(target=rx.mavlink_receive_loop, daemon=False)
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # -----------------------------------------------------------------------
    # Receive loop
    # -----------------------------------------------------------------------

    def mavlink_receive_loop(self):
        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError. Stopping MAVLink listener.', flush=True)
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()
            if msg_type == 'BAD_DATA':
                continue

            if   msg_type == 'HEARTBEAT':                   self.on_heartbeat(msg)
            elif msg_type == 'TIMESYNC':                    self.on_timesync(msg)
            elif msg_type == 'ATTITUDE':                    self.on_attitude(msg)
            elif msg_type == 'LOCAL_POSITION_NED':          self.on_local_position_ned(msg)
            elif msg_type == 'ODOMETRY':                    self.on_odometry(msg)
            elif msg_type == 'HIGHRES_IMU':                 self.on_highres_imu(msg)
            elif msg_type == 'ENCAPSULATED_DATA':           self.on_encapsulated_data(msg)
            elif msg_type == 'ACTUATOR_OUTPUT_STATUS':      self.on_actuator_output_status(msg)
            elif msg_type == 'COLLISION':                   self.on_collision(msg)
            elif msg_type == 'DATA_TRANSMISSION_HANDSHAKE': self.on_data_transmission_handshake(msg)

    # -----------------------------------------------------------------------
    # Message handlers
    # -----------------------------------------------------------------------

    def on_heartbeat(self, msg):
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        with self.lock:
            self.data['armed'] = armed

    def on_timesync(self, msg):
        if msg.ts1 != 0:
            now_ns = int(time.time_ns())
            clock_offset_ns = msg.tc1 - (msg.ts1 + now_ns) // 2
            with self.lock:
                self.data['clock_offset_ns'] = clock_offset_ns
        else:
            now_ns = int(time.time_ns())
            self.mavlink_conn.mav.timesync_send(now_ns, msg.tc1)

    def on_attitude(self, msg):
        with self.lock:
            self.data['attitude'] = {
                'roll': msg.roll, 'pitch': msg.pitch, 'yaw': msg.yaw,
                'rollspeed': msg.rollspeed, 'pitchspeed': msg.pitchspeed,
                'yawspeed': msg.yawspeed,
                'time_boot_ms': msg.time_boot_ms, 'ts': time.time(),
            }

    def on_local_position_ned(self, msg):
        with self.lock:
            self.data['position'] = {
                'x': msg.x, 'y': msg.y, 'z': msg.z,
                'time_boot_ms': msg.time_boot_ms, 'ts': time.time(),
            }
            self.data['velocity'] = {
                'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
                'time_boot_ms': msg.time_boot_ms, 'ts': time.time(),
            }

    def on_odometry(self, msg):
        with self.lock:
            self.data['odometry'] = {
                'x': msg.x,   'y': msg.y,   'z': msg.z,
                'qw': msg.q[0], 'qx': msg.q[1], 'qy': msg.q[2], 'qz': msg.q[3],
                'vx': msg.vx,  'vy': msg.vy,  'vz': msg.vz,
                'rollspeed': msg.rollspeed, 'pitchspeed': msg.pitchspeed,
                'yawspeed': msg.yawspeed,
                'time_usec': msg.time_usec, 'reset_counter': msg.reset_counter,
                'ts': time.time(),
            }

    def on_highres_imu(self, msg):
        with self.lock:
            self.data['imu'] = {
                'xacc': msg.xacc, 'yacc': msg.yacc, 'zacc': msg.zacc,
                'xgyro': msg.xgyro, 'ygyro': msg.ygyro, 'zgyro': msg.zgyro,
                'time_usec': msg.time_usec, 'ts': time.time(),
            }

    def on_encapsulated_data(self, msg):
        raw_payload = bytes(msg.data)
        data_type = raw_payload[0]
        if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self.on_race_status(msg)
        elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        (_, sim_boot_time_ms, race_start_boot_time_ms,
         race_finish_time_ns, active_gate_index,
         last_gate_race_time) = struct.unpack_from('<BQqqIq', raw_payload)
        with self.lock:
            self.data['race_status'] = {
                'sim_boot_time_ms':        sim_boot_time_ms,
                'race_start_boot_time_ms': race_start_boot_time_ms,
                'race_finish_time_ns':     race_finish_time_ns,
                'active_gate_index':       int(active_gate_index),
                'last_gate_race_time':     last_gate_race_time,
                'ts':                      time.time(),
            }

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        _, transfer_id = struct.unpack_from('<BH', raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        expected = self.expected_num_track_chunks[transfer_id]
        if len(self.track_chunks[transfer_id]) == expected:
            full = b''.join(self.track_chunks[transfer_id][i] for i in range(expected))
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full)

    def on_track_data(self, payload):
        num_gates, = struct.unpack_from('<H', payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            (gate_id, pos_x, pos_y, pos_z,
             qw, qx, qy, qz,
             width, height) = struct.unpack_from('<Hfffffffff', payload)
            payload = payload[38:]
            gates.append({
                'gate_id': int(gate_id),
                'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'width': width, 'height': height,
            })
        gates.sort(key=lambda g: g['gate_id'])
        
        with self.lock:
            self.data['gates'] = gates
            
        print(f'Track data received: {num_gates} gates', flush=True)        
        for g in gates:
            print(f"  Gate {g['gate_id']}: X={g['pos_x']:.2f}, Y={g['pos_y']:.2f}, Z={g['pos_z']:.2f}")


    def on_data_transmission_handshake(self, msg):
        tid = msg.width
        self.track_chunks[tid] = {}
        self.expected_num_track_chunks[tid] = msg.packets

    def on_actuator_output_status(self, msg):
        with self.lock:
            self.data['motor_feedback'] = {
                'fl': msg.actuator[0], 'fr': msg.actuator[1],
                'bl': msg.actuator[2], 'br': msg.actuator[3],
                'time_usec': msg.time_usec, 'ts': time.time(),
            }

    def on_collision(self, msg):
        collision_type = 'gate' if msg.id == 1001 else 'environment'
        with self.lock:
            self.data['last_collision'] = {
                'id': msg.id, 'type': collision_type,
                'threat_level': msg.threat_level,
                'impulse': msg.horizontal_minimum_delta,
                'ts': time.time(),
            }

        # Throttle terminal output: print a summary at most once per second.
        # Ground-contact collisions at spawn can fire hundreds of times per
        # second and would otherwise flood the terminal.
        now = time.time()
        self._collision_count_since_last_print += 1
        if now - self._last_collision_print >= 1.0:
            count = self._collision_count_since_last_print
            print(
                f'COLLISION [{collision_type}] threat={msg.threat_level} '
                f'impulse={msg.horizontal_minimum_delta:.3f}'
                + (f'  (x{count} in last second)' if count > 1 else ''),
                flush=True
            )
            self._last_collision_print = now
            self._collision_count_since_last_print = 0