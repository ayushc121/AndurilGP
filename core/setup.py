"""
Wires the client together: MAVLink connection, background threads, controller.
Shared by both qualifiers, which differ only in the controller class and
whether they open the camera socket at all.
"""

import threading
import time

from pymavlink import mavutil

from .mavlink_rx import MAVLinkRX
from .timesync import TimeSync

__all__ = ['HeartbeatSender', 'run', 'setup_components', 'shutdown']

HEARTBEAT_HZ = 5   # spec requires >= 2 Hz; 5 leaves margin


class HeartbeatSender:
    """
    Emits a MAVLink HEARTBEAT. Not optional — the sim rejects commands from a
    client it has not heard from recently.
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
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            time.sleep(1.0 / HEARTBEAT_HZ)

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread


def setup_components(shared_data, system_boot_ms, server_ip, server_port,
                     controller_cls, vision_cls=None):
    """
    Build every component and start its thread. The shared lock goes into
    `shared_data` first, since every component takes its lock from there.
    """
    shared_data['lock'] = threading.Lock()

    sim_conn = mavutil.mavlink_connection(f'udpin:{server_ip}:{server_port}')
    print('Waiting for heartbeat...', flush=True)
    sim_conn.wait_heartbeat()
    print(f'Connected to system {sim_conn.target_system}.', flush=True)

    components = {
        'sim_conn': sim_conn,
        'heartbeat': HeartbeatSender(sim_conn),
        'mavlink_rx': MAVLinkRX.create_mavlink_rx(sim_conn, shared_data),
        'ts_loop': TimeSync.create_timesync(sim_conn),
    }
    if vision_cls is not None:
        components['vision_rx'] = vision_cls(shared_data)
    components['controller'] = controller_cls(sim_conn, shared_data, system_boot_ms)
    return components


def run(components):
    """
    Arm and fly until interrupted, then shut every thread down.

    No self-terminating condition: the sim re-arms between runs and the
    controller handles that in place, so Ctrl-C is the way out. The threads
    are non-daemon, so skipping the shutdown hangs the interpreter on exit.
    """
    controller = components['controller']

    print('Arming drone...', flush=True)
    controller.arm()

    print('Starting control loop. Ctrl-C to stop.', flush=True)
    try:
        while True:
            controller.update()
    except KeyboardInterrupt:
        print('\nInterrupted.', flush=True)
    finally:
        shutdown(components)


def shutdown(components):
    """Signal each background thread to stop and wait for it."""
    print('Shutting down background threads...', flush=True)
    for name in ('heartbeat', 'ts_loop', 'mavlink_rx', 'vision_rx'):
        component = components.get(name)
        if component is not None:
            component.get_thread_for_join().join(timeout=2.0)
    print('Client exited.', flush=True)
