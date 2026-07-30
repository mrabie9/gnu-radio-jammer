#!/usr/bin/env python3
"""Standalone sweep (chirp) jammer for two 100 MHz bands.

Bands jammed::

    2.4 GHz ISM   2392-2492 MHz   (centre 2442 MHz)
    5.2 GHz       5190-5290 MHz   (centre 5240 MHz)

A linear chirp sweeps back and forth across the instantaneous ``sample_rate``
window. On top of that the USRP local oscillator steps across each 100 MHz band
and alternates between the two bands, so the whole 2 x 100 MHz is covered even
when the radio only samples a fraction of 100 MHz at once. The two knobs that
matter most are exposed as command-line options::

    --sample-rate   the instantaneous bandwidth swept at each LO step
    --speed         chirp sweeps per second across that window

Usage::

    python3 examples/sweep_jammer.py                       # both bands, 20 Msps
    python3 examples/sweep_jammer.py --sample-rate 40e6 --speed 5000
    python3 examples/sweep_jammer.py --bands 5g_top --gain 70
    python3 examples/sweep_jammer.py --dry-run             # no radio, print plan

This is a standalone tool: it does not import the gpsk_comms link at all.

WARNING: this transmits. Only radiate in bands you are licensed to use, and only
into a dummy load or attenuator unless you understand the consequences. The two
default bands overlap licensed services; jamming them over the air is illegal in
most jurisdictions. ``--dry-run`` transmits nothing and is safe anywhere.
"""

import argparse
import math
import sys
import threading
import time

import numpy

# Band edges (low_hz, high_hz) and midpoints, matching the anti-jam link's
# 2442 MHz and 5240 MHz allocations. Each named preset is either a full 100 MHz
# band, its lower 50 MHz (*_bottom) or its upper 50 MHz (*_top).
BAND_4G = (2_392_000_000, 2_492_000_000)
BAND_5G = (5_190_000_000, 5_290_000_000)
_4G_MID = (BAND_4G[0] + BAND_4G[1]) // 2
_5G_MID = (BAND_5G[0] + BAND_5G[1]) // 2
BANDS = {
    "both": [BAND_4G, BAND_5G],
    "5g": [BAND_5G],
    "5g_top": [(_5G_MID, BAND_5G[1])],
    "5g_bottom": [(BAND_5G[0], _5G_MID)],
    "4g": [BAND_4G],
    "4g_top": [(_4G_MID, BAND_4G[1])],
    "4g_bottom": [(BAND_4G[0], _4G_MID)],
}


def lo_plan(band_list, sample_rate):
    """LO centre frequencies that tile every band in ``band_list``.

    Each LO step covers ``sample_rate`` Hz of instantaneous bandwidth. A 100 MHz
    band is split into ``ceil(width / sample_rate)`` equal steps, so adjacent
    windows overlap and the whole band is covered with no gaps. When
    ``sample_rate`` is 100 MHz or more a band collapses to a single step.
    """
    centres = []
    for low, high in band_list:
        width = high - low
        steps = max(1, math.ceil(width / sample_rate))
        step_width = width / steps
        for k in range(steps):
            centres.append(low + step_width * (k + 0.5))
    return centres


def make_chirp(sample_rate, speed, min_samples):
    """A repeating linear chirp across the full +/- sample_rate/2 window.

    ``speed`` is sweeps per second: the chirp ramps from -sample_rate/2 up to
    +sample_rate/2 and resets ``speed`` times a second. The returned buffer holds
    a whole number of sweeps so it can be cycled forever without accumulating a
    frequency error, and is at least ``min_samples`` long so the radio is fed in
    comfortable chunks.
    """
    period = max(2, int(round(sample_rate / speed)))
    reps = max(1, math.ceil(min_samples / period))
    n = numpy.arange(period)
    # Instantaneous frequency, in cycles/sample, ramps across [-0.5, +0.5).
    inst = n / period - 0.5
    # Phase is its running integral; 2*pi*inst is the per-sample phase advance.
    phase = 2.0 * numpy.pi * numpy.cumsum(inst)
    one_sweep = numpy.exp(1j * phase).astype(numpy.complex64)
    return numpy.tile(one_sweep, reps)


try:
    from gnuradio import gr

    class _chirp_source(gr.sync_block):
        """Emit a precomputed chirp buffer, cycling forever."""

        def __init__(self, waveform):
            gr.sync_block.__init__(
                self, name="chirp_source", in_sig=None, out_sig=[numpy.complex64]
            )
            self._waveform = numpy.asarray(waveform, dtype=numpy.complex64)
            self._offset = 0

        def work(self, input_items, output_items):
            output = output_items[0]
            count = len(output)
            indices = (self._offset + numpy.arange(count)) % len(self._waveform)
            output[:] = self._waveform[indices]
            self._offset = (self._offset + count) % len(self._waveform)
            return count

except ImportError:  # gnuradio not installed -- only --dry-run plan printing works
    gr = None
    _chirp_source = None


class _lo_stepper(threading.Thread):
    """Step the LO through ``plan`` every ``dwell`` seconds until stopped."""

    def __init__(self, plan, dwell, set_freq, announce=None):
        super().__init__(daemon=True)
        self._plan = plan
        self._dwell = dwell
        self._set_freq = set_freq
        self._announce = announce
        self._stop = threading.Event()
        self._index = 0

    def run(self):
        self._set_freq(self._plan[0])
        if self._announce:
            self._announce(0, self._plan[0])
        # Single-step LO plans (sample_rate covers the whole band) need no
        # retuning at all; just hold and wait to be stopped.
        if len(self._plan) == 1:
            self._stop.wait()
            return
        while not self._stop.wait(self._dwell):
            self._index = (self._index + 1) % len(self._plan)
            freq = self._plan[self._index]
            self._set_freq(freq)
            if self._announce:
                self._announce(self._index, freq)

    def stop(self):
        self._stop.set()


def describe_plan(plan, band_list, sample_rate, speed, dwell):
    """Human-readable summary of what the jammer will do."""
    lines = []
    lines.append(f"  sample rate   {sample_rate/1e6:g} Msps  "
                 f"(instantaneous swept bandwidth)")
    lines.append(f"  sweep speed   {speed:g} sweeps/s  "
                 f"({sample_rate*speed/1e6:g} MHz/s chirp rate)")
    lines.append(f"  LO steps      {len(plan)} total, "
                 f"{dwell*1e3:g} ms dwell each")
    lines.append(f"  band revisit  {len(plan)*dwell*1e3:g} ms per full cycle")
    for low, high in band_list:
        width = high - low
        steps = max(1, math.ceil(width / sample_rate))
        lines.append(f"  band          {low/1e6:g}-{high/1e6:g} MHz "
                     f"({width/1e6:g} MHz) in {steps} step(s)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample-rate", type=float, default=10e6,
                        help="instantaneous swept bandwidth, in Hz (default 10e6)")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="chirp sweeps per second (default 1000)")
    parser.add_argument("--bands", choices=sorted(BANDS), default="both",
                        help="which band(s) to jam: both, 5g/4g (full band), or a "
                             "*_top / *_bottom half (default both)")
    parser.add_argument("--dwell", type=float, default=0.02,
                        help="seconds to hold each LO step before retuning "
                             "(default 0.02)")
    parser.add_argument("--gain", type=float, default=60.0,
                        help="USRP transmit gain in dB (default 60)")
    parser.add_argument("--antenna", default="TX/RX", help="USRP antenna (default TX/RX)")
    parser.add_argument("--args", default="", help="UHD device address args")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run, or 0 to run until Ctrl-C (default 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and stream into a null sink; no radio, "
                             "transmits nothing")
    args = parser.parse_args()

    if args.sample_rate <= 0 or args.speed <= 0 or args.dwell <= 0:
        parser.error("--sample-rate, --speed and --dwell must all be positive")

    band_list = BANDS[args.bands]
    plan = lo_plan(band_list, args.sample_rate)

    print("=" * 72)
    print("Sweep jammer" + ("  [DRY RUN -- nothing is transmitted]" if args.dry_run else ""))
    print("=" * 72)
    print(describe_plan(plan, band_list, args.sample_rate, args.speed, args.dwell))

    if gr is None:
        print("\ngnuradio is not importable in this environment; showing the plan only.")
        return 0

    from gnuradio import blocks

    # A chirp buffer at least one dwell long, so the radio is fed smoothly.
    min_samples = max(int(args.sample_rate * max(args.dwell, 0.05)), 4096)
    chirp = make_chirp(args.sample_rate, args.speed, min_samples)

    top = gr.top_block()
    source = _chirp_source(chirp)

    if args.dry_run:
        sink = blocks.null_sink(gr.sizeof_gr_complex)
        top.connect(source, sink)

        def set_freq(freq):
            pass

        def announce(index, freq):
            print(f"    [dry-run] step {index}: LO -> {freq/1e6:.3f} MHz")
    else:
        try:
            from gnuradio import uhd
        except ImportError:
            print("\nERROR: gnuradio.uhd is not available, so no radio can be driven.",
                  file=sys.stderr)
            print("Install UHD support, or use --dry-run to preview the plan.",
                  file=sys.stderr)
            return 1

        usrp = uhd.usrp_sink(
            args.args, uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        usrp.set_samp_rate(args.sample_rate)
        usrp.set_center_freq(plan[0], 0)
        usrp.set_gain(args.gain, 0)
        usrp.set_antenna(args.antenna, 0)
        try:
            usrp.set_bandwidth(args.sample_rate, 0)
        except Exception:
            pass  # not every daughterboard exposes an analog bandwidth control
        top.connect(source, usrp)

        def set_freq(freq):
            usrp.set_center_freq(freq, 0)

        def announce(index, freq):
            print(f"    step {index}: centre frequency -> {freq/1e6:.3f} MHz",
                  flush=True)

    stepper = _lo_stepper(plan, args.dwell, set_freq, announce)

    print("\nStarting. Press Ctrl-C to stop.\n" if args.duration == 0
          else f"\nRunning for {args.duration:g} s.\n")
    top.start()
    stepper.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stepper.stop()
        top.stop()
        top.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
