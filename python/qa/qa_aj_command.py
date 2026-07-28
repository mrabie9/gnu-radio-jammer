#!/usr/bin/env python3
"""Integration tests for the assembled anti-jam link.

These run real flowgraphs in real time, so they use a low spreading factor and
short dwells to stay quick. The processing-gain figures that matter are measured
in qa_dsss.py and examples/jammer_harness.py; what is checked here is that the
layers are wired together correctly end to end.
"""

import time
import unittest

import numpy
import pmt
from gnuradio import blocks, gr

from gpsk_comms.aj_command import (
    PAYLOAD_BITS,
    aj_command_rx,
    aj_command_tx,
    bit_array_to_bytes,
    bytes_to_bit_array,
    hop_position,
)
from gpsk_comms.security import KeyError_, generate_master_key

SAMPLE_RATE = 2_000_000
SPREADING_FACTOR = 255
REPEAT_RATE = 20.0


def command_message(command, pressed=True):
    message = pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern(command))
    return pmt.dict_add(message, pmt.intern("pressed"), pmt.from_bool(pressed))


def field(message, name):
    return pmt.dict_ref(message, pmt.intern(name), pmt.PMT_NIL)


def published_commands(sink):
    return [
        pmt.symbol_to_string(field(sink.get_message(index), "command"))
        for index in range(sink.num_messages())
    ]


def wait_for(predicate, timeout=6.0, interval=0.02):
    """Poll until ``predicate`` holds, or the timeout expires.

    These flowgraphs run in real time, so a fixed sleep is a race: pipeline fill,
    burst boundaries and scheduler timing all shift how long the first frame
    takes. Polling keeps the tests both fast and reliable.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class Link:
    """A transmitter and receiver joined directly, with message sinks."""

    def __init__(self, tx_key, rx_key=None, **rx_kwargs):
        self.top = gr.top_block()
        self.tx = aj_command_tx(
            tx_key,
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
            hopping_enabled=False,
        )
        options = {"watchdog_timeout": 5.0, "hopping_enabled": False}
        options.update(rx_kwargs)
        self.rx = aj_command_rx(
            rx_key if rx_key is not None else tx_key,
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            **options,
        )
        self.commands = blocks.message_debug()
        self.diagnostics = blocks.message_debug()
        self.top.connect(self.tx, self.rx)
        self.top.msg_connect((self.rx, "command"), (self.commands, "store"))
        self.top.msg_connect((self.rx, "diagnostics"), (self.diagnostics, "store"))

    def send(self, command, pressed=True):
        self.tx._source._handle_command(command_message(command, pressed))

    def __enter__(self):
        self.top.start()
        return self

    def __exit__(self, *exception):
        self.top.stop()
        self.top.wait()
        return False


class BitPackingTest(unittest.TestCase):
    def test_round_trip(self):
        for payload in (b"\x00", b"\xff\x01", bytes(range(12))):
            bits = bytes_to_bit_array(payload)
            self.assertEqual(len(bits), len(payload) * 8)
            self.assertEqual(bit_array_to_bytes(bits), payload)

    def test_payload_bits_matches_the_v2_frame(self):
        self.assertEqual(PAYLOAD_BITS, 96)


class HopPositionTest(unittest.TestCase):
    def test_position_advances_once_per_dwell(self):
        dwell = 100_000
        base = 10 * dwell
        self.assertEqual(hop_position(base, dwell), hop_position(base + dwell - 1, dwell))
        self.assertNotEqual(hop_position(base, dwell), hop_position(base + dwell, dwell))

    def test_both_ends_agree_given_the_same_clock(self):
        dwell = 100_000
        for counter in (0, 12_345_678, 999_999_999_999):
            self.assertEqual(hop_position(counter, dwell), hop_position(counter, dwell))


class KeyHandlingTest(unittest.TestCase):
    def test_all_zero_key_is_refused(self):
        for factory in (aj_command_tx, aj_command_rx):
            with self.assertRaises(KeyError_):
                factory(bytes(32), sample_rate=SAMPLE_RATE, spreading_factor=SPREADING_FACTOR)

    def test_all_zero_key_is_allowed_only_when_explicitly_requested(self):
        transmitter = aj_command_tx(
            bytes(32),
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            allow_insecure_key=True,
        )
        self.assertIsNotNone(transmitter)


class LinkTest(unittest.TestCase):
    def test_link_idles_at_stop_before_any_command(self):
        """The transmitter sends 'stop' until told otherwise.

        Matching the plain GMSK source, whose command also defaults to 'stop'.
        A command link that idles at anything else would be unsafe, so the
        leading 'stop' the other tests skip past is deliberate, not a wart.
        """
        key = generate_master_key()
        with Link(key) as link:
            self.assertTrue(wait_for(lambda: link.commands.num_messages() > 0))
        self.assertEqual(published_commands(link.commands)[0], "stop")

    def test_commands_are_delivered_in_order(self):
        key = generate_master_key()
        expected = ["forward", "left", "stop", "right"]
        with Link(key) as link:
            # Let the idle 'stop' land first so it does not race the sequence.
            self.assertTrue(wait_for(lambda: link.commands.num_messages() > 0))
            baseline = link.commands.num_messages()
            for index, command in enumerate(expected):
                link.send(command)
                delivered = wait_for(
                    lambda n=index: link.commands.num_messages() > baseline + n
                )
                self.assertTrue(delivered, f"{command!r} was never delivered")
        self.assertEqual(published_commands(link.commands)[baseline:], expected)
        counts = link.rx.counts
        self.assertGreater(counts["valid"], 0)
        self.assertEqual(counts["auth_failure"], 0)

    def test_repeated_commands_are_suppressed(self):
        key = generate_master_key()
        with Link(key) as link:
            link.send("forward")
            self.assertTrue(wait_for(lambda: link.rx.counts["duplicate"] > 3))
        # However many frames carried it, 'forward' is published exactly once.
        delivered = published_commands(link.commands)
        self.assertEqual(delivered.count("forward"), 1)
        self.assertGreater(link.rx.counts["valid"], delivered.count("forward"))

    def test_link_reports_ok_while_receiving(self):
        key = generate_master_key()
        with Link(key) as link:
            link.send("forward")
            self.assertTrue(wait_for(lambda: link.rx.counts["valid"] > 0))
            state = link.rx.link_state
            snr = link.rx.snr_db
        self.assertEqual(state, "ok")
        self.assertGreater(snr, 10.0)

    def test_no_signal_is_reported_when_nothing_has_been_received(self):
        receiver = aj_command_rx(
            generate_master_key(),
            sample_rate=SAMPLE_RATE,
            spreading_factor=SPREADING_FACTOR,
            hopping_enabled=False,
        )
        self.assertEqual(receiver.link_state, "no_signal")


class SecurityTest(unittest.TestCase):
    def test_transmitter_with_the_wrong_key_delivers_nothing(self):
        """The central guarantee: an adversary cannot inject a command.

        The forged transmitter is otherwise identical -- same spreading factor,
        same structure, same power. Only the key differs.
        """
        with Link(generate_master_key(), rx_key=generate_master_key()) as link:
            link.send("forward")
            # Wait until the receiver has demonstrably seen and rejected traffic,
            # so a pass cannot come from the flowgraph having done nothing.
            self.assertTrue(wait_for(lambda: link.rx.counts["auth_failure"] > 5))
        self.assertEqual(published_commands(link.commands), [])
        counts = link.rx.counts
        self.assertEqual(counts["valid"], 0)
        self.assertGreater(counts["auth_failure"], 0)

    def test_unauthenticated_traffic_is_classified_as_jammed(self):
        with Link(generate_master_key(), rx_key=generate_master_key()) as link:
            link.send("forward")
            self.assertTrue(wait_for(lambda: link.rx.counts["auth_failure"] > 5))
            state = link.rx.link_state
        # Structure present but nothing authenticates: that is jamming, not
        # silence, and it is what should eventually drive a band escape.
        self.assertEqual(state, "jammed")


class WatchdogTest(unittest.TestCase):
    def test_stop_is_commanded_when_the_link_drops(self):
        key = generate_master_key()
        link = Link(key, watchdog_timeout=0.4)
        with link:
            link.send("forward")
            # Wait for 'forward' itself, not merely for a valid frame: the
            # transmitter idles at 'stop', so the valid count is already
            # non-zero before the command under test has been sent at all.
            self.assertTrue(
                wait_for(lambda: "forward" in published_commands(link.commands)),
                "forward was never delivered",
            )
            # Silence the transmitter and wait for the watchdog to fire.
            link.tx._source._active = False
            # Wait for the published message, not the counter. The decoder
            # increments its timeout count before publishing, so waiting on the
            # count can win the race and tear the flowgraph down in between.
            self.assertTrue(
                wait_for(lambda: published_commands(link.commands)[-1:] == ["stop"]),
                "watchdog never published a stop",
            )
        delivered = published_commands(link.commands)
        self.assertIn("forward", delivered)
        self.assertEqual(delivered[-1], "stop")
        states = [
            pmt.symbol_to_string(field(link.commands.get_message(index), "link_state"))
            for index in range(link.commands.num_messages())
        ]
        self.assertEqual(states[-1], "timeout")
        self.assertGreater(link.rx.counts["timeout"], 0)

    def test_grace_period_delays_the_failsafe(self):
        key = generate_master_key()
        link = Link(key, watchdog_timeout=0.3, grace_period=3.0)
        with link:
            link.send("forward")
            self.assertTrue(wait_for(lambda: link.rx.counts["valid"] > 0))
            link.tx._source._active = False
            # Well past the bare watchdog but well inside the grace period.
            time.sleep(1.0)
            timeouts_during_grace = link.rx.counts["timeout"]
        self.assertEqual(timeouts_during_grace, 0)

    def test_invalid_watchdog_configuration_is_rejected(self):
        key = generate_master_key()
        for kwargs in ({"watchdog_timeout": 0.0}, {"grace_period": -1.0}):
            with self.assertRaises(ValueError):
                aj_command_rx(
                    key,
                    sample_rate=SAMPLE_RATE,
                    spreading_factor=SPREADING_FACTOR,
                    hopping_enabled=False,
                    **kwargs,
                )


class HoldToSendTest(unittest.TestCase):
    def test_release_silences_the_transmitter(self):
        key = generate_master_key()
        top = gr.top_block()
        transmitter = aj_command_tx(
            key,
            sample_rate=SAMPLE_RATE,
            repeat_rate=REPEAT_RATE,
            spreading_factor=SPREADING_FACTOR,
            hopping_enabled=False,
            hold_to_send=True,
        )
        sink = blocks.vector_sink_c()
        head = blocks.head(gr.sizeof_gr_complex, 400_000)
        top.connect(transmitter, head, sink)
        top.start()
        time.sleep(0.4)
        transmitter._source._handle_command(command_message("forward", pressed=False))
        time.sleep(0.4)
        top.stop()
        top.wait()
        samples = numpy.array(sink.data())
        # The tail of the capture, after release, must be silent.
        if len(samples) > 100_000:
            self.assertLess(float(numpy.abs(samples[-50_000:]).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
