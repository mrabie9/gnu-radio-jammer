#!/usr/bin/env python3
"""Short over-the-air acceptance test for the pinned gpsk_comms USRPs."""

import argparse
import time

import numpy
import pmt
from gnuradio import blocks, gr, uhd

from gpsk_comms import gmsk_command_rx, gmsk_command_tx


TX_SERIAL = "337D3DF"
RX_SERIAL = "3449B99"
COMMANDS = ("forward", "backward", "left", "right", "stop")


def command_message(command):
    return pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern(command))


def message_field(message, field):
    return pmt.dict_ref(message, pmt.intern(field), pmt.PMT_NIL)


def decoded_messages(debug):
    result = []
    for index in range(debug.num_messages()):
        message = debug.get_message(index)
        result.append(
            {
                "command": pmt.symbol_to_string(message_field(message, "command")),
                "sequence": pmt.to_uint64(message_field(message, "sequence")),
                "session_id": pmt.to_uint64(message_field(message, "session_id")),
                "link_state": pmt.symbol_to_string(message_field(message, "link_state")),
                "rx_time_ns": pmt.to_uint64(message_field(message, "rx_time_ns")),
            }
        )
    return result


def make_usrp_sink(serial, sample_rate, center_freq, gain):
    sink = uhd.usrp_sink(
        f"serial={serial}",
        uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]),
        "",
    )
    sink.set_clock_source("internal", 0)
    sink.set_time_source("internal", 0)
    sink.set_samp_rate(sample_rate)
    sink.set_center_freq(center_freq, 0)
    sink.set_antenna("TX/RX", 0)
    sink.set_bandwidth(sample_rate, 0)
    sink.set_gain(gain, 0)
    return sink


def make_usrp_source(serial, sample_rate, center_freq, gain, antenna):
    source = uhd.usrp_source(
        f"serial={serial}",
        uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]),
        True,
    )
    source.set_clock_source("internal", 0)
    source.set_time_source("internal", 0)
    source.set_samp_rate(sample_rate)
    source.set_center_freq(center_freq, 0)
    source.set_antenna(antenna, 0)
    source.set_bandwidth(sample_rate, 0)
    source.set_gain(gain, 0)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transmit", action="store_true", help="required RF safety acknowledgement")
    parser.add_argument("--frequency", type=float, default=2_440_000_000)
    parser.add_argument("--sample-rate", type=float, default=1_000_000)
    parser.add_argument("--samples-per-symbol", type=int, default=8)
    parser.add_argument("--repeat-rate", type=float, default=100.0)
    parser.add_argument("--pace-factor", type=float, default=1.0)
    parser.add_argument("--tx-gain", type=float, default=0.0)
    parser.add_argument("--rx-gain", type=float, default=20.0)
    parser.add_argument("--rx-antenna", choices=("TX/RX", "RX2"), default="TX/RX")
    parser.add_argument("--tx-serial", default=TX_SERIAL)
    parser.add_argument("--rx-serial", default=RX_SERIAL)
    parser.add_argument("--command-timeout", type=float, default=3.0)
    parser.add_argument("--commands", nargs="+", choices=COMMANDS, default=list(COMMANDS))
    parser.add_argument("--preload-first", action="store_true")
    parser.add_argument("--verify-timeout", action="store_true")
    args = parser.parse_args()
    if not args.transmit:
        parser.error("--transmit is required; confirm frequency, EIRP, antennas, and local safety first")

    tx_tb = gr.top_block("gpsk_comms OTA TX")
    rx_tb = gr.top_block("gpsk_comms OTA RX")
    tx_tb.set_max_noutput_items(4096)
    rx_tb.set_max_noutput_items(4096)
    tx = gmsk_command_tx(
        sample_rate=args.sample_rate,
        samples_per_symbol=args.samples_per_symbol,
        repeat_rate=args.repeat_rate,
        session_id=None,
    )
    tx._source._byte_rate *= args.pace_factor
    rx = gmsk_command_rx(
        sample_rate=args.sample_rate,
        samples_per_symbol=args.samples_per_symbol,
        watchdog_timeout=0.5,
    )
    sink = make_usrp_sink(args.tx_serial, args.sample_rate, args.frequency, args.tx_gain)
    source = make_usrp_source(
        args.rx_serial,
        args.sample_rate,
        args.frequency,
        args.rx_gain,
        args.rx_antenna,
    )
    debug = blocks.message_debug()
    diagnostics_debug = blocks.message_debug()
    capture_head = blocks.head(gr.sizeof_gr_complex, int(args.sample_rate / 4))
    capture = blocks.vector_sink_c()
    carrier_corrector = blocks.rotator_cc(0.0)
    tx_tb.connect(tx, sink)
    rx_tb.connect(source, carrier_corrector, rx)
    rx_tb.connect(source, capture_head, capture)
    rx_tb.msg_connect((rx, "command"), (debug, "store"))
    rx_tb.msg_connect((rx, "diagnostics"), (diagnostics_debug, "store"))

    failures = []
    tx_running = False
    start_ns = time.time_ns()
    if args.preload_first and args.commands:
        tx._source._handle_command(command_message(args.commands[0]))
    rx_tb.start()
    tx_tb.start()
    tx_running = True
    try:
        time.sleep(1.0)
        samples = numpy.asarray(capture.data(), dtype=numpy.complex64)
        if len(samples) > 1:
            rms_dbfs = 20.0 * numpy.log10(max(numpy.sqrt(numpy.mean(numpy.abs(samples) ** 2)), 1e-12))
            phase_step = numpy.angle(numpy.mean(samples[1:] * numpy.conj(samples[:-1])))
            offset_hz = phase_step * args.sample_rate / (2.0 * numpy.pi)
            carrier_corrector.set_phase_inc(-phase_step)
            print(f"RX capture: {rms_dbfs:.1f} dBFS, estimated carrier offset {offset_hz:.0f} Hz")
        for command_index, command in enumerate(args.commands):
            is_preloaded = args.preload_first and command_index == 0
            before = 0 if is_preloaded else debug.num_messages()
            if not is_preloaded:
                tx._source._handle_command(command_message(command))
            print(
                f"TX active command={tx._source.command} sequence={tx._source.sequence} "
                f"queued_bytes={len(tx._source._pending)} elapsed={(time.time_ns() - start_ns) / 1e9:.2f}s"
            )
            deadline = time.monotonic() + args.command_timeout
            received = None
            while time.monotonic() < deadline:
                messages = decoded_messages(debug)
                received = next(
                    (
                        message
                        for message in messages[before:]
                        if message["command"] == command and message["link_state"] == "ok"
                    ),
                    None,
                )
                if received is not None:
                    break
                time.sleep(0.02)
            if received is None:
                failures.append(command)
                print(f"FAIL {command}: no valid packet")
            else:
                print(
                    f"PASS {command}: session={received['session_id']} "
                    f"sequence={received['sequence']} state={received['link_state']}"
                )
        if args.verify_timeout and not failures:
            sink.set_gain(0.0, 0)
            sink.set_center_freq(args.frequency + 10_000_000, 0)
            tx_tb.stop()
            tx_tb.wait()
            tx_running = False
            deadline = time.monotonic() + 1.0
            timeout_message = None
            while time.monotonic() < deadline:
                timeout_message = next(
                    (
                        message
                        for message in decoded_messages(debug)
                        if message["command"] == "stop" and message["link_state"] == "timeout"
                    ),
                    None,
                )
                if timeout_message is not None:
                    break
                time.sleep(0.01)
            if timeout_message is None:
                failures.append("watchdog timeout")
                print("FAIL watchdog: no timeout stop within 1 second")
            else:
                print("PASS watchdog: synthetic timeout stop received")
    finally:
        if tx_running:
            tx_tb.stop()
            tx_tb.wait()
        rx_tb.stop()
        rx_tb.wait()

    messages = decoded_messages(debug)
    for message in messages:
        message["elapsed_s"] = round((message["rx_time_ns"] - start_ns) / 1e9, 3)
    print(f"Published messages: {messages}")
    print(
        "RF diagnostics: "
        f"{[pmt.to_python(diagnostics_debug.get_message(i)) for i in range(diagnostics_debug.num_messages())]}"
    )
    ok_messages = [message for message in messages if message["link_state"] == "ok"]
    identities = [(message["session_id"], message["sequence"]) for message in ok_messages]
    if len(identities) != len(set(identities)):
        failures.append("duplicate publication")
    if failures:
        raise SystemExit(f"OTA smoke test failed: {', '.join(failures)}")
    print(f"OTA smoke test passed; {len(ok_messages)} unique commands published")


if __name__ == "__main__":
    main()
