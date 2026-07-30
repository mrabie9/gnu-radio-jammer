"""Chirp sweep jammer with LO stepping across two 100 MHz bands.

A linear chirp is generated across +/- samp_rate/2. The block also steps the
USRP local oscillator by publishing 'freq' command messages on its message
port; wire that port to a UHD Sink's command port and the LO tiles each band
and alternates between them, covering far more spectrum than one sample_rate
window. Sample rate, speed, dwell and band selection are all live: work()
rebuilds the LO plan whenever the sample rate or band selection changes.
"""

import math
import time

import numpy as np
import pmt
from gnuradio import gr

# The two target bands as (low_hz, high_hz), matching the anti-jam link's
# 2442 MHz and 5240 MHz allocations.
_BANDS = {
    "2g4": [(2_392_000_000, 2_492_000_000)],
    "5g2": [(5_190_000_000, 5_290_000_000)],
    "both": [(2_392_000_000, 2_492_000_000), (5_190_000_000, 5_290_000_000)],
    "5210": [(5_210_000_000, 5_210_000_000)],
}


class blk(gr.sync_block):
    def __init__(self, samp_rate=20e6, speed=1000.0, dwell=0.02, bands="both"):
        gr.sync_block.__init__(
            self, name="Chirp Sweep Jammer", in_sig=None,
            out_sig=[np.complex64],
        )
        self.samp_rate = float(samp_rate)
        self.speed = float(speed)
        self.dwell = float(dwell)
        self.bands = bands
        self.message_port_register_out(pmt.intern("command"))
        self._count = 0
        self._phase = 0.0
        self._index = 0
        self._next_step = 0.0
        self._built_for = None
        self._plan = [2_442_000_000.0]
        # The first real plan build and LO tune happen in start(), once the
        # command port is connected -- publishing here would go nowhere.

    # A live QT control writes straight to the attribute (GRC generates
    # e.g. self.chirp.samp_rate = ...), so no setters are needed: work()
    # reads speed and dwell fresh each call, and rebuilds the LO plan
    # whenever samp_rate or bands no longer match what it built for.
    def _rebuild_plan(self):
        centres = []
        for low, high in _BANDS.get(self.bands, _BANDS["both"]):
            width = high - low
            steps = max(1, math.ceil(width / self.samp_rate))
            step_width = width / steps
            for k in range(steps):
                centres.append(low + step_width * (k + 0.5))
        self._plan = centres
        self._index = 0
        self._built_for = (self.samp_rate, self.bands)
        self._retune(self._plan[0])
        self._next_step = time.monotonic() + self.dwell

    def _retune(self, freq):
        # Print the current centre frequency on every LO step, so the
        # console shows the sweep walking across the bands.
        print("[sweep jammer] centre frequency -> %.3f MHz" % (freq / 1e6),
              flush=True)
        self.message_port_pub(
            pmt.intern("command"),
            pmt.cons(pmt.intern("freq"), pmt.from_double(float(freq))),
        )

    def start(self):
        self._rebuild_plan()
        return True

    def work(self, input_items, output_items):
        if (self.samp_rate, self.bands) != self._built_for:
            self._rebuild_plan()

        out = output_items[0]
        n = len(out)
        period = max(2.0, self.samp_rate / self.speed)
        idx = self._count + np.arange(n)
        inst = (idx % period) / period - 0.5
        phase = self._phase + 2.0 * np.pi * np.cumsum(inst)
        out[:] = np.exp(1j * phase).astype(np.complex64)
        self._phase = float(phase[-1] % (2.0 * np.pi))
        self._count += n

        if len(self._plan) > 1 and time.monotonic() >= self._next_step:
            self._index = (self._index + 1) % len(self._plan)
            self._retune(self._plan[self._index])
            self._next_step = time.monotonic() + self.dwell
        return n
