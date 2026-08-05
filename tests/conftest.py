"""Shared fixtures. No simulator, no sockets, no threads."""

import threading

import numpy as np
import pytest

from vq2.gate_detector import CX, CY, FX, GATE_WIDTH_M, IMG_H, IMG_W


class FakeMav:
    """Records outgoing MAVLink calls instead of sending them."""

    def __init__(self):
        self.attitude_targets = []
        self.commands = []

    def set_attitude_target_send(self, *args):
        # (time_ms, sys, comp, type_mask, quaternion, r, p, y, thrust)
        self.attitude_targets.append({'type_mask': args[3],
                                      'quaternion': args[4],
                                      'thrust': args[8]})

    def command_long_send(self, *args):
        self.commands.append(args)


class FakeSimConn:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.mav = FakeMav()


@pytest.fixture
def sim_conn():
    return FakeSimConn()


@pytest.fixture
def shared_data():
    return {'lock': threading.Lock()}


def render_gate(range_m, centre=(CX, CY), img_size=(IMG_W, IMG_H)):
    """
    Synthetic gate: a red square frame at a given range. Projected width is
    FX * GATE_WIDTH_M / range, so reading it back through the same pinhole
    model must recover `range_m`.
    """
    width = int(round(FX * GATE_WIDTH_M / range_m))
    border = max(2, int(round(width * 0.11)))
    img = np.full((img_size[1], img_size[0], 3), 40, dtype=np.uint8)

    x0 = int(round(centre[0] - width / 2))
    y0 = int(round(centre[1] - width / 2))
    img[max(0, y0):y0 + width, max(0, x0):x0 + width] = (40, 40, 220)   # BGR red
    img[max(0, y0 + border):y0 + width - border,
        max(0, x0 + border):x0 + width - border] = (40, 40, 40)
    return img


@pytest.fixture
def gate_image():
    return render_gate
