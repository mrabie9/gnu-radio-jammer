#!/usr/bin/env python3
"""Walk the feature ladder and report the first level that fails.

Runs a real loopback flowgraph at each level from 0 upwards and checks that
commands actually arrive. Because each level is the one below plus exactly one
mechanism, the first failure names the layer at fault -- which is the whole
point of the ladder, and considerably faster than bisecting a stack of six.

Usage::

    python3 examples/level_ladder.py                # levels 0..6
    python3 examples/level_ladder.py --level 5      # just one level
    python3 examples/level_ladder.py --keep-going   # do not stop at the first failure
    python3 examples/level_ladder.py --verbose      # show per-stage counters

There is a GUI version of this in examples/aj_command_ladder_gui.grc, which runs
one level at a time with a live spectrum and waterfall, a keyboard command input
and an adjustable jammer::

    grcc examples/aj_command_ladder_gui.grc && python3 aj_command_ladder_gui.py --level 0

Use this script to find which level breaks and that flowgraph to watch it happen.
Over a real radio the same ladder is driven by the paired flowgraphs
examples/aj_command_tx_uhd.grc and examples/aj_command_rx_uhd.grc, launched at the
same --level on both ends.

Everything runs in simulation. No radio is involved and nothing is transmitted.
Passing here proves the layers agree with each other; it does not prove the link
works over the air, where clocks, gain and LO accuracy all still apply.
"""

import argparse
import sys
import time

import pmt
from gnuradio import gr

from gpsk_comms.aj_command import aj_command_rx, aj_command_tx
from gpsk_comms.levels import LEVEL_NAMES, profile
from gpsk_comms.security import generate_master_key

# Deliberately modest. The ladder checks that the layers agree with each other;
# jammer_harness.py measures processing gain, and that is what wants a high
# spreading factor. What matters here instead is *real-time headroom*: excision
# and the despreader are Python blocks and GNU Radio gives each its own thread
# contending for one GIL, so the whole chain has a few hundred kilosamples per
# second of budget on a typical host. Exceed it and the transmitter falls behind
# the wall clock its hop schedule is derived from, and level 5 fails for reasons
# that have nothing to do with hopping. Raise --sample-rate to find your own
# ceiling; the re-anchor count reports when you have passed it.
SAMPLE_RATE = 500_000
SPREADING_FACTOR = 127

#: Slow enough that a burst fits in a dwell at the sample rate above, with the
#: margin the receiver needs to search for one before the code changes.
DWELL_US = 200_000

#: One burst per interval at levels 0-4; level 5 and up send one per dwell.
REPEAT_RATE = 8.0

#: Commands sent in order. Repeats are suppressed by the receiver, so each one
#: must differ from the last for its delivery to be observable.
SEQUENCE = ("forward", "left", "right", "stop")


def command_message(command, pressed=True):
    message = pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern(command))
    return pmt.dict_add(message, pmt.intern("pressed"), pmt.from_bool(pressed))


def delivered_commands(sink):
    return [
        pmt.symbol_to_string(
            pmt.dict_ref(sink.get_message(index), pmt.intern("command"), pmt.PMT_NIL)
        )
        for index in range(sink.num_messages())
    ]


def run_level(level, master_key, timeout=12.0, dwell_us=DWELL_US,
              sample_rate=SAMPLE_RATE, spreading_factor=SPREADING_FACTOR):
    """Run one level end to end. Returns ``(ok, detail_dict)``."""
    from gnuradio import blocks

    link_profile = profile(level)
    # Levels 0 and 1 take no key at all, and passing one anyway would hide the
    # fact that they do not need key distribution to be working.
    key = master_key if link_profile.needs_key else None

    top = gr.top_block()
    transmitter = aj_command_tx(
        key,
        sample_rate=sample_rate,
        repeat_rate=REPEAT_RATE,
        spreading_factor=spreading_factor,
        dwell_us=dwell_us,
        level=level,
    )
    receiver = aj_command_rx(
        key,
        sample_rate=sample_rate,
        spreading_factor=spreading_factor,
        dwell_us=dwell_us,
        level=level,
        watchdog_timeout=timeout + 5.0,
    )
    sink = blocks.message_debug()
    top.connect(transmitter, receiver)
    top.msg_connect((receiver, "command"), (sink, "store"))

    top.start()
    deadline = time.monotonic() + timeout
    sent = []
    try:
        for command in SEQUENCE:
            transmitter._source._handle_command(command_message(command))
            sent.append(command)
            while time.monotonic() < deadline:
                if command in delivered_commands(sink):
                    break
                time.sleep(0.02)
            else:
                break
    finally:
        top.stop()
        top.wait()

    received = delivered_commands(sink)
    # The transmitter idles at 'stop' before any command is given, so a leading
    # 'stop' is expected and is not part of the sequence under test.
    missing = [command for command in sent if command not in received]
    detail = {
        "delivered": received,
        "missing": missing,
        "tx_bursts": transmitter.bursts,
        "reanchors": transmitter.reanchors,
        "despread": receiver.despread_counts,
        "decode": receiver.counts,
        "snr_db": receiver.snr_db,
    }
    return not missing and len(sent) == len(SEQUENCE), detail


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, action="append",
                        help="run only this level; repeatable")
    parser.add_argument("--timeout", type=float, default=12.0,
                        help="seconds to wait for the command sequence at each level")
    parser.add_argument("--dwell-us", type=int, default=DWELL_US)
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE)
    parser.add_argument("--spreading-factor", type=int, default=SPREADING_FACTOR)
    parser.add_argument("--keep-going", action="store_true",
                        help="continue past the first failing level")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    wanted = args.level if args.level else list(range(len(LEVEL_NAMES)))
    master_key = generate_master_key()

    print("=" * 72)
    print("Feature ladder: each level is the one below plus exactly one mechanism")
    print("=" * 72)

    failed = []
    for level in wanted:
        name = LEVEL_NAMES[level]
        adds = profile(level).enabled
        label = adds[-1] if adds else "bare point-to-point link"
        print(f"\nlevel {level} ({name}) -- adds {label}")
        try:
            ok, detail = run_level(level, master_key, args.timeout, args.dwell_us,
                                   args.sample_rate, args.spreading_factor)
        except Exception as error:
            print(f"  FAILED to build: {error}")
            failed.append(level)
            if not args.keep_going:
                break
            continue

        status = "ok" if ok else "FAILED"
        print(f"  {status}: delivered {detail['delivered']}")
        if detail["reanchors"]:
            # Distinguish "this host is too slow" from "this layer is broken",
            # because the two look identical in the delivery counters.
            print(f"  NOTE: the transmitter re-anchored {detail['reanchors']} times -- "
                  f"the flowgraph is not keeping up with {args.sample_rate/1e6:g} Msps.\n"
                  "        This is a host capacity limit, not a fault in this layer. "
                  "Lower --sample-rate.")
        if args.verbose or not ok:
            print(f"    tx bursts     {detail['tx_bursts']} "
                  f"(re-anchored {detail['reanchors']})")
            print(f"    despreader    {detail['despread']}")
            print(f"    decoder       {detail['decode']}")
            print(f"    last SNR      {detail['snr_db']:.1f} dB")
            if detail["missing"]:
                print(f"    never arrived {detail['missing']}")
        if not ok:
            failed.append(level)
            if not args.keep_going:
                print(f"\nStopping at level {level}. The mechanism this level adds "
                      f"({label}) is the one to look at:\nevery layer below it "
                      "carried traffic.")
                break

    print("\n" + "=" * 72)
    if failed:
        print(f"Failing levels: {', '.join(str(level) for level in failed)}")
        return 1
    print(f"All {len(wanted)} levels carried traffic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
