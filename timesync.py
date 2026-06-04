import time
import threading

from pymavlink import mavutil

TIMESYNC_REQUEST_HZ = 10


class TimeSync:
    """
    Sends a MAVLink TIMESYNC request at a fixed rate so the simulator can
    report its clock offset back to us.  The response is handled in
    MAVLinkRX.on_timesync(), which stores the computed offset in
    shared_data['clock_offset_ns'].

    Note: the startup bug was in setup.py (it called TimeSync() directly
    instead of TimeSync.create_timesync()), not here. setup.py is now fixed.
    """

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

    @classmethod
    def create_timesync(cls, mavlink_connection, data):
        ts = cls(mavlink_connection, data)
        ts.thread = threading.Thread(
            target=ts.timesync_loop,
            daemon=False
        )
        ts.is_running = True
        ts.thread.start()
        return ts

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def timesync_loop(self):
        while self.is_running:
            now_ns = int(time.time_ns())
            # tc1 = our current time (ns), ts1 = 0 signals this is a request
            self.mavlink_conn.mav.timesync_send(
                now_ns,  # tc1
                0        # ts1 = 0 → request
            )
            time.sleep(1.0 / TIMESYNC_REQUEST_HZ)
