"""
Score the VQ2 detector by flying VQ1 with it.

VQ1's controller steers on telemetry, so the detector under test cannot affect
the flight, and the sim's true gate positions give something real to diff
against. Note this deliberately swaps in `vq2.vision_rx` rather than VQ1's own
`vq1.vision` — the point is to measure the evolved detector on a course where
ground truth exists. Run from the repo root with the sim on the VQ1 course:

    python -m analysis.collect_ground_truth

Writes cv_ground_truth.csv. Score with `python -m analysis.accuracy`.
"""

import time

from core.setup import run, setup_components
from vq1.controller import Controller
from vq2.vision_rx import VisionRX
from .ground_truth import GroundTruthLogger

SIM_SERVER_UDP_IP = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550


class LoggingController(Controller):
    """VQ1's controller, unchanged, with a logging tap after each control pass."""

    def __init__(self, sim_conn, data, system_boot_ms):
        super().__init__(sim_conn, data, system_boot_ms)
        self.logger = GroundTruthLogger()
        self._t0 = time.time()

    def _fly(self, odometry, race_status, gates):
        super()._fly(odometry, race_status, gates)

        target = self._target_gate(race_status, gates)
        if target is None:
            return
        gate = gates[race_status['active_gate_index']]
        self.logger.log(
            t=time.time() - self._t0,
            gate_idx=race_status['active_gate_index'],
            odo=odometry,
            gate_pos=target,
            vision=self.data.get('vision_gate_estimate'),
            vision_velocity=self.data.get('vision_velocity'),
            gate_quat=(gate['qw'], gate['qx'], gate['qy'], gate['qz']))


def main():
    system_boot_ms = int(time.time() * 1000)
    shared_data = {}
    components = setup_components(shared_data, system_boot_ms,
                                  SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
                                  controller_cls=LoggingController,
                                  vision_cls=VisionRX)
    try:
        run(components)
    finally:
        components['controller'].logger.close()


if __name__ == '__main__':
    main()
