"""
MAVLink telemetry receiver. Decodes the sim's stream on a background thread
into `shared_data`; nothing else talks to the connection.

Only the messages the controllers consume are decoded — the sim also emits
attitude, local position and motor outputs, which nothing here reads.

Race status and track data are non-standard, arriving wrapped in
ENCAPSULATED_DATA with a leading type byte; track data is chunked.
"""

import struct
import threading
import time

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS = 1
ENCAPSULATED_TRACK_INFO = 2

COLLISION_ID_GATE = 1001
COLLISION_PRINT_INTERVAL_S = 1.0


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.lock = data['lock']
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}
        self._last_collision_print = 0.0
        self._collisions_since_print = 0

        with self.lock:
            self.data.update({
                'armed': False,
                'odometry': None,
                'imu': None,
                'race_status': None,
                'gates': None,
                'last_collision': None,
                'clock_offset_ns': 0,
                'vision_gate_estimate': None,
            })

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        """Construct and start the thread. Use this rather than the constructor."""
        rx = cls(mavlink_connection, data)
        rx.is_running = True
        rx.thread = threading.Thread(target=rx.receive_loop, daemon=False)
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def receive_loop(self):
        handlers = {
            'HEARTBEAT': self.on_heartbeat,
            'TIMESYNC': self.on_timesync,
            'ODOMETRY': self.on_odometry,
            'HIGHRES_IMU': self.on_highres_imu,
            'ENCAPSULATED_DATA': self.on_encapsulated_data,
            'COLLISION': self.on_collision,
            'DATA_TRANSMISSION_HANDSHAKE': self.on_data_transmission_handshake,
        }
        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('MAVLink connection reset — stopping listener.', flush=True)
                return
            if msg is None:
                time.sleep(0.001)
                continue
            handler = handlers.get(msg.get_type())
            if handler is not None:
                handler(msg)

    # handlers

    def on_heartbeat(self, msg):
        with self.lock:
            self.data['armed'] = bool(
                msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def on_timesync(self, msg):
        """Reply to the sim's probes; record the offset from its replies to ours."""
        now_ns = int(time.time_ns())
        if msg.ts1 == 0:
            self.mavlink_conn.mav.timesync_send(now_ns, msg.tc1)
            return
        with self.lock:
            self.data['clock_offset_ns'] = msg.tc1 - (msg.ts1 + now_ns) // 2

    def on_odometry(self, msg):
        with self.lock:
            self.data['odometry'] = {
                'x': msg.x, 'y': msg.y, 'z': msg.z,
                'qw': msg.q[0], 'qx': msg.q[1], 'qy': msg.q[2], 'qz': msg.q[3],
                'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
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

    def on_collision(self, msg):
        collision_type = 'gate' if msg.id == COLLISION_ID_GATE else 'environment'
        with self.lock:
            self.data['last_collision'] = {
                'id': msg.id, 'type': collision_type,
                'threat_level': msg.threat_level,
                'impulse': msg.horizontal_minimum_delta,
                'ts': time.time(),
            }

        # Ground contact at spawn fires hundreds of times a second, so the
        # terminal output is summarised rather than printed per event.
        self._collisions_since_print += 1
        now = time.time()
        if now - self._last_collision_print < COLLISION_PRINT_INTERVAL_S:
            return
        count = self._collisions_since_print
        suffix = f'  (x{count} in last second)' if count > 1 else ''
        print(f'COLLISION [{collision_type}] threat={msg.threat_level} '
              f'impulse={msg.horizontal_minimum_delta:.3f}{suffix}', flush=True)
        self._last_collision_print = now
        self._collisions_since_print = 0

    # encapsulated data

    def on_encapsulated_data(self, msg):
        payload = bytes(msg.data)
        if payload[0] == ENCAPSULATED_RACE_STATUS:
            self.on_race_status(payload)
        elif payload[0] == ENCAPSULATED_TRACK_INFO:
            self.on_track_chunk(msg, payload)

    def on_race_status(self, payload):
        (_, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns,
         active_gate_index, last_gate_race_time) = struct.unpack_from('<BQqqIq', payload)
        with self.lock:
            self.data['race_status'] = {
                'sim_boot_time_ms': sim_boot_time_ms,
                'race_start_boot_time_ms': race_start_boot_time_ms,
                'race_finish_time_ns': race_finish_time_ns,
                'active_gate_index': int(active_gate_index),
                'last_gate_race_time': last_gate_race_time,
                'ts': time.time(),
            }

    def on_data_transmission_handshake(self, msg):
        """Announces how many chunks the next track-data transfer will use."""
        self.track_chunks[msg.width] = {}
        self.expected_num_track_chunks[msg.width] = msg.packets

    def on_track_chunk(self, msg, payload):
        """Collect one chunk; decode the track once every chunk has arrived."""
        _, transfer_id = struct.unpack_from('<BH', payload)
        if transfer_id not in self.expected_num_track_chunks:
            return

        self.track_chunks[transfer_id][msg.seqnr] = payload[3:]
        expected = self.expected_num_track_chunks[transfer_id]
        if len(self.track_chunks[transfer_id]) < expected:
            return

        full = b''.join(self.track_chunks[transfer_id][i] for i in range(expected))
        del self.track_chunks[transfer_id]
        del self.expected_num_track_chunks[transfer_id]
        self.on_track_data(full)

    def on_track_data(self, payload):
        """Decode every gate's pose and dimensions, sorted by gate id."""
        num_gates, = struct.unpack_from('<H', payload)
        payload = payload[2:]

        gates = []
        for _ in range(num_gates):
            (gate_id, pos_x, pos_y, pos_z, qw, qx, qy, qz,
             width, height) = struct.unpack_from('<Hfffffffff', payload)
            payload = payload[38:]
            gates.append({'gate_id': int(gate_id),
                          'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
                          'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                          'width': width, 'height': height})
        gates.sort(key=lambda g: g['gate_id'])

        with self.lock:
            self.data['gates'] = gates

        print(f'Track data received: {num_gates} gates', flush=True)
        for gate in gates:
            print(f"  Gate {gate['gate_id']}: "
                  f"({gate['pos_x']:.2f}, {gate['pos_y']:.2f}, {gate['pos_z']:.2f})",
                  flush=True)
