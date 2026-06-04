import threading
import time

from pymavlink import mavutil

from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller

HEARTBEAT_HZ = 5  # spec requires >= 2 Hz; 5 Hz gives a comfortable margin


class HeartbeatSender:
    """
    Sends a MAVLink HEARTBEAT at a fixed rate on a background thread.
    The spec requires the client to maintain heartbeat at >= 2 Hz or the
    simulator may reject control commands.
    """

    def __init__(self, mavlink_conn):
        self.mavlink_conn = mavlink_conn
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=False)
        self.thread.start()

    def _loop(self):
        while self.is_running:
            self.mavlink_conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            time.sleep(1.0 / HEARTBEAT_HZ)

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # ------------------------------------------------------------------
    # Shared threading lock — stored in shared_data so every component
    # that receives a reference to shared_data can use the same lock.
    # Must be created before any component constructors run.
    # ------------------------------------------------------------------
    shared_data['lock'] = threading.Lock()

    # ------------------------------------------------------------------
    # MAVLink connection
    # ------------------------------------------------------------------
    sim_conn = mavutil.mavlink_connection(
        'udpin:%s:%s' % (server_ip, server_udp_port)
    )
    print('Waiting for heartbeat...', flush=True)
    sim_conn.wait_heartbeat()
    print(f'Connected to system: {sim_conn.target_system}', flush=True)

    # ------------------------------------------------------------------
    # Heartbeat sender  (new — was missing entirely from original)
    # ------------------------------------------------------------------
    print('Starting heartbeat sender...', flush=True)
    heartbeat = HeartbeatSender(sim_conn)

    # ------------------------------------------------------------------
    # MAVLink telemetry receiver
    # ------------------------------------------------------------------
    print('Setting up MAVLink rx...', flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # ------------------------------------------------------------------
    # Timesync loop
    # Bug fix: original code called TimeSync(…) directly which never
    # started the background thread. Must use the create_timesync
    # classmethod.
    # ------------------------------------------------------------------
    print('Setting up timesync loop...', flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # ------------------------------------------------------------------
    # Vision receiver
    # ------------------------------------------------------------------
    vision_rx = VisionRX(shared_data)

    # ------------------------------------------------------------------
    # Main controller
    # ------------------------------------------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'controller': controller,
        'mavlink_rx': mavlink_rx,
        'ts_loop':    ts_loop,
        'vision_rx':  vision_rx,
        'heartbeat':  heartbeat,
        'sim_conn':   sim_conn,
    }
