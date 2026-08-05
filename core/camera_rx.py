"""
Camera transport: reassembles the sim's chunked JPEG-over-UDP stream.

Subclass and implement `process_frame`. Both qualifiers do — they share this
loop but run different detectors, because the detector evolved between rounds.

Subclasses must set up their own state before calling `super().__init__`,
which starts the thread.
"""

import socket
import struct
import threading

import cv2
import numpy as np

UDP_IP = '0.0.0.0'
UDP_PORT = 5600
SOCKET_TIMEOUT_S = 1.0    # lets the thread notice is_running going False
FRAME_BUFFER_DEPTH = 10   # drop partial frames older than this many IDs

_HEADER_FMT = '<IHHIIQ'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class CameraRX:

    def __init__(self, data):
        self.data = data
        self.is_running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=False)
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def process_frame(self, frame_id, img):
        """Handle one decoded BGR frame. Implemented by the subclass."""
        raise NotImplementedError

    def _receive_loop(self):
        partial = {}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(SOCKET_TIMEOUT_S)
        print('Listening for camera frames...', flush=True)

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue

            frame_id, chunk_id, total, _, _, _ = struct.unpack(
                _HEADER_FMT, packet[:_HEADER_SIZE])
            chunks = partial.setdefault(frame_id, {})
            chunks[chunk_id] = packet[_HEADER_SIZE:]

            # check every index, don't trust the count — frame IDs restart
            # each run, so a stale partial can hit the total with a hole in it
            if len(chunks) >= total and all(i in chunks for i in range(total)):
                del partial[frame_id]
                image = cv2.imdecode(
                    np.frombuffer(b''.join(chunks[i] for i in range(total)),
                                  dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                if image is not None:
                    self._process_frame_guarded(frame_id, image)

            for stale in [f for f in partial if f < frame_id - FRAME_BUFFER_DEPTH]:
                del partial[stale]

    def _process_frame_guarded(self, frame_id, img):
        """Nothing supervises this thread — if it dies the controller flies on a
        stale estimate, so dropping a frame is the lesser failure."""
        try:
            self.process_frame(frame_id, img)
        except Exception as exc:                      # noqa: BLE001
            print(f'[vision] frame {frame_id} failed: {exc!r}', flush=True)
