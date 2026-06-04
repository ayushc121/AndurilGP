#
# AI Grand Prix — autonomous drone racing client
#

import time

from setup import setup_components

# Modify if connecting to a remote simulator
SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550

# Wall-clock timestamp at process start — used as a local reference for
# time_boot_ms fields in outgoing MAVLink messages
system_boot_ms = int(time.time() * 1000)

# Shared state dictionary — all components read and write through this.
# setup_components() will add a threading.Lock under key 'lock' before
# any other component is constructed.
shared_data = {}

# -----------------------------------------------------------------------
# Initialise all components
# -----------------------------------------------------------------------
components  = setup_components(
    shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
)
controller  = components['controller']
ts_loop     = components['ts_loop']
mavlink_rx  = components['mavlink_rx']
vision_rx   = components['vision_rx']
heartbeat   = components['heartbeat']

# -----------------------------------------------------------------------
# Arm and run
# -----------------------------------------------------------------------
print('Arming drone...', flush=True)
controller.arm()

print('Starting control loop...', flush=True)
while not controller.is_finished():
    controller.update()

# -----------------------------------------------------------------------
# Clean shutdown — signal every background thread to stop, then join
# -----------------------------------------------------------------------
print('Shutting down background threads...', flush=True)
heartbeat.get_thread_for_join().join(timeout=2.0)
ts_loop.get_thread_for_join().join(timeout=2.0)
mavlink_rx.get_thread_for_join().join(timeout=2.0)
vision_rx.get_thread_for_join().join(timeout=2.0)

print('Client exited.', flush=True)
