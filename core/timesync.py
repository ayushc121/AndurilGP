"""Clock synchronisation with the simulator."""

import threading
import time

TIMESYNC_REQUEST_HZ = 10


class TimeSync:
    """Requests TIMESYNC; `MAVLinkRX.on_timesync` stores the offset. Nothing
    reads it yet, but the sim expects the exchange."""

    def __init__(self, mavlink_connection):
        self.mavlink_conn = mavlink_connection
        self.thread = None
        self.is_running = False

    @classmethod
    def create_timesync(cls, mavlink_connection):
        """Construct and start the thread. Use this rather than the constructor."""
        ts = cls(mavlink_connection)
        ts.is_running = True
        ts.thread = threading.Thread(target=ts.timesync_loop, daemon=False)
        ts.thread.start()
        return ts

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def timesync_loop(self):
        while self.is_running:
            # tc1 = our clock now; ts1 = 0 marks this as a request, not a reply.
            self.mavlink_conn.mav.timesync_send(int(time.time_ns()), 0)
            time.sleep(1.0 / TIMESYNC_REQUEST_HZ)
