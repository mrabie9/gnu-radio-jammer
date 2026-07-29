#!/usr/bin/env python3
"""Measure how much jamming the command link survives.

The acceptance metric is J/S: the jamming-to-signal power ratio in dB at which
command delivery stops working. Every layer's contribution is measured rather
than assumed, and the original unauthenticated GMSK link is measured alongside
as the baseline, because an improvement is only meaningful against a number.

Usage::

    python3 examples/jammer_harness.py                  # standard sweep
    python3 examples/jammer_harness.py --quick          # coarse and fast
    python3 examples/jammer_harness.py --jammer barrage # one jammer type

Everything runs in simulation. No radio is involved and nothing is transmitted.
"""

import argparse
import sys
import time

import numpy
import pmt
from gnuradio import analog, blocks, gr

from gpsk_comms import gmsk_command_rx, gmsk_command_tx
from gpsk_comms.aj_command import aj_command_rx, aj_command_tx
from gpsk_comms.levels import LEVEL_EXCISION, LEVEL_NAMES, profile
from gpsk_comms.security import generate_master_key

JAMMER_TYPES = ("barrage", "cw", "swept", "pulsed")


def _tone(count, amplitude, normalised_frequency):
    index = numpy.arange(count)
    return amplitude * numpy.exp(2j * numpy.pi * normalised_frequency * index)


def make_jammer(kind, amplitude, sample_rate, count, seed=0):
    """Return one jammer waveform of the requested kind.

    All kinds are normalised to the same *average* power, so a J/S figure means
    the same thing across them. A pulsed jammer therefore has a higher peak
    amplitude than a barrage one at equal J/S, which is exactly the trade a real
    pulsed jammer makes.
    """
    rng = numpy.random.default_rng(seed)
    if kind == "barrage":
        samples = (
            rng.normal(0, 1 / numpy.sqrt(2), count) + 1j * rng.normal(0, 1 / numpy.sqrt(2), count)
        )
    elif kind == "cw":
        samples = _tone(count, 1.0, 0.13)
    elif kind == "swept":
        # One sweep across the band per 10 ms, the sort of rate a cheap sweeper
        # manages. Constant envelope, so average power equals peak power.
        period = max(1, int(sample_rate * 0.01))
        phase = numpy.cumsum(
            2 * numpy.pi * (numpy.arange(count) % period) / period * 0.4 - 0.2
        )
        samples = numpy.exp(1j * phase)
    elif kind == "pulsed":
        # 10% duty cycle, so the same average power arrives in bursts 10 dB
        # stronger. This is the case interleaving exists to handle.
        duty = 0.1
        period = max(1, int(sample_rate * 0.005))
        gate = (numpy.arange(count) % period) < int(period * duty)
        samples = (
            rng.normal(0, 1 / numpy.sqrt(2), count) + 1j * rng.normal(0, 1 / numpy.sqrt(2), count)
        ) * gate
        samples = samples / numpy.sqrt(duty)
    else:
        raise ValueError(f"unknown jammer kind: {kind}")
    return (amplitude * samples).astype(numpy.complex64)


class _jammer_source(gr.sync_block):
    """Emit a precomputed jammer waveform, cycling forever."""

    def __init__(self, waveform):
        gr.sync_block.__init__(
            self, name="jammer_source", in_sig=None, out_sig=[numpy.complex64]
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


def measure_signal_power(build_transmitter, sample_rate, samples=200_000):
    """Average power of the transmitter output, for calibrating J/S."""
    top = gr.top_block()
    transmitter = build_transmitter()
    head = blocks.head(gr.sizeof_gr_complex, samples)
    sink = blocks.vector_sink_c()
    top.connect(transmitter, head, sink)
    top.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(sink.data()) < samples:
        time.sleep(0.05)
    top.stop()
    top.wait()
    captured = numpy.array(sink.data())
    if not len(captured):
        return 0.0
    return float(numpy.mean(numpy.abs(captured) ** 2))


def run_aj_trial(key, jam_kind, jam_amplitude, sample_rate, spreading_factor, duration,
                 seed, level=LEVEL_EXCISION):
    """Return the number of commands delivered under one jamming condition.

    Defaults to level 4: everything that improves jam resistance in a static
    channel. Hopping is deliberately excluded from the default because this
    harness models the jammer as fixed and in-band, which understates hopping and
    would credit it with nothing. Pass ``--level`` to measure a single rung.
    """
    top = gr.top_block()
    transmitter = aj_command_tx(
        key,
        sample_rate=sample_rate,
        repeat_rate=20.0,
        spreading_factor=spreading_factor,
        level=level,
    )
    receiver = aj_command_rx(
        key,
        sample_rate=sample_rate,
        spreading_factor=spreading_factor,
        level=level,
        watchdog_timeout=10.0,
    )
    jammer = _jammer_source(
        make_jammer(jam_kind, jam_amplitude, sample_rate, int(sample_rate * 0.05), seed)
    )
    combiner = blocks.add_cc()
    sink = blocks.message_debug()
    top.connect(transmitter, (combiner, 0))
    top.connect(jammer, (combiner, 1))
    top.connect(combiner, receiver)
    top.msg_connect((receiver, "command"), (sink, "store"))

    transmitter._source._handle_command(
        pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("forward"))
    )
    top.start()
    time.sleep(duration)
    top.stop()
    top.wait()
    return receiver.counts["valid"], receiver.counts, sink.num_messages()


def run_baseline_trial(jam_kind, jam_amplitude, sample_rate, duration, seed):
    """Same measurement against the original unauthenticated GMSK link."""
    top = gr.top_block()
    transmitter = gmsk_command_tx(
        sample_rate=sample_rate, samples_per_symbol=4, repeat_rate=20.0, session_id=1
    )
    receiver = gmsk_command_rx(
        sample_rate=sample_rate, samples_per_symbol=4, watchdog_timeout=10.0
    )
    jammer = _jammer_source(
        make_jammer(jam_kind, jam_amplitude, sample_rate, int(sample_rate * 0.05), seed)
    )
    combiner = blocks.add_cc()
    sink = blocks.message_debug()
    top.connect(transmitter, (combiner, 0))
    top.connect(jammer, (combiner, 1))
    top.connect(combiner, receiver)
    top.msg_connect((receiver, "command"), (sink, "store"))

    transmitter._source._handle_command(
        pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("forward"))
    )
    top.start()
    time.sleep(duration)
    top.stop()
    top.wait()
    return receiver._parser.counts["valid"], receiver._parser.counts, sink.num_messages()


def sweep(label, trial, signal_power, jam_levels, minimum_valid=1):
    """Run one system across a J/S sweep and report the breaking point."""
    print(f"\n  {label}")
    print("    J/S (dB)   valid frames   delivered")
    survived = None
    for jam_db in jam_levels:
        amplitude = numpy.sqrt(signal_power * 10 ** (jam_db / 10.0))
        valid, counts, delivered = trial(amplitude)
        marker = "" if valid >= minimum_valid else "   <-- link down"
        print(f"    {jam_db:8.0f}   {valid:12d}   {delivered:9d}{marker}")
        if valid >= minimum_valid:
            survived = jam_db
        else:
            break
    return survived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, default=2_000_000)
    parser.add_argument("--spreading-factor", type=int, default=255)
    parser.add_argument("--duration", type=float, default=1.2, help="seconds per J/S point")
    parser.add_argument("--quick", action="store_true", help="coarser sweep, fewer points")
    parser.add_argument(
        "--jammer", choices=JAMMER_TYPES, action="append", help="restrict to one jammer type"
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--level", type=int, default=LEVEL_EXCISION,
        help="feature level to measure (0-6); each level is the one below plus one "
             "mechanism, so sweeping it shows what each layer is worth",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    kinds = args.jammer or list(JAMMER_TYPES)
    step = 12 if args.quick else 6
    # Excision alone suppresses a tone by 50 dB or more, so the hardened link can
    # outlast a narrowband jammer far past the raw processing gain. Sweep high
    # enough to actually find its breaking point rather than running off the end
    # of the table.
    aj_levels = list(range(-6, 79, step))
    baseline_levels = list(range(-24, 25, step))

    key = generate_master_key() if profile(args.level).needs_key else None
    sample_rate = args.sample_rate

    print("=" * 72)
    print("Jam resistance measurement")
    print("=" * 72)
    print(f"sample rate       {sample_rate/1e6:g} Msps")
    print(f"spreading factor  {args.spreading_factor} "
          f"({10*numpy.log10(args.spreading_factor):.1f} dB of processing gain)")
    print(f"dwell per point   {args.duration:g} s")
    print(f"feature level     {args.level} ({LEVEL_NAMES[args.level]})")
    print("\nCalibrating transmitter power...")

    aj_power = measure_signal_power(
        lambda: aj_command_tx(
            key,
            sample_rate=sample_rate,
            repeat_rate=20.0,
            spreading_factor=args.spreading_factor,
            level=args.level,
        ),
        sample_rate,
    )
    print(f"  anti-jam link signal power: {aj_power:.4f}")

    baseline_power = None
    if not args.skip_baseline:
        baseline_power = measure_signal_power(
            lambda: gmsk_command_tx(
                sample_rate=sample_rate, samples_per_symbol=4, repeat_rate=20.0, session_id=1
            ),
            sample_rate,
        )
        print(f"  baseline GMSK signal power: {baseline_power:.4f}")

    results = {}
    for kind in kinds:
        print("\n" + "-" * 72)
        print(f"Jammer: {kind}")
        print("-" * 72)

        if baseline_power:
            results[(kind, "baseline")] = sweep(
                "baseline GMSK link (no spreading, no FEC, no authentication)",
                lambda amplitude, k=kind: run_baseline_trial(
                    k, amplitude, sample_rate, args.duration, args.seed
                ),
                baseline_power,
                baseline_levels,
            )

        enabled = ", ".join(profile(args.level).enabled) or "nothing"
        results[(kind, "aj")] = sweep(
            f"anti-jam link at level {args.level} ({LEVEL_NAMES[args.level]}): {enabled}",
            lambda amplitude, k=kind: run_aj_trial(
                key, k, amplitude, sample_rate, args.spreading_factor, args.duration,
                args.seed, args.level,
            )[:3],
            aj_power,
            aj_levels,
        )

    print("\n" + "=" * 72)
    print("Summary: highest J/S at which commands were still delivered")
    print("=" * 72)
    print(f"  {'jammer':10s} {'baseline':>12s} {'anti-jam':>12s} {'improvement':>13s}")
    for kind in kinds:
        base = results.get((kind, "baseline"))
        hardened = results.get((kind, "aj"))
        base_text = "link down" if base is None else f"{base:+d} dB"
        aj_text = "link down" if hardened is None else f"{hardened:+d} dB"
        if base is not None and hardened is not None:
            gain = f"{hardened - base:+d} dB"
        else:
            gain = "n/a"
        print(f"  {kind:10s} {base_text:>12s} {aj_text:>12s} {gain:>13s}")
    print(
        "\nNote: the baseline link is also trivially spoofable -- an adversary can "
        "\ninject commands at any J/S. The anti-jam link rejects forgeries outright, "
        "\nwhich no J/S figure captures."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
