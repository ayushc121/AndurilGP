"""
VQ2 entry point — vision-only racing. The controller owns the estimator.

Run from the repository root:

    python -m vq2.main
"""

import time

from core.setup import run, setup_components
from .controller import Controller
from .vision_rx import VisionRX

SIM_SERVER_UDP_IP = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550


def main():
    # Wall-clock reference for the time_boot_ms field on outgoing messages.
    system_boot_ms = int(time.time() * 1000)
    shared_data = {}
    components = setup_components(shared_data, system_boot_ms,
                                  SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
                                  controller_cls=Controller,
                                  vision_cls=VisionRX)

    run(components)


if __name__ == '__main__':
    main()
