#!/usr/bin/env python3

import time
import unittest
import struct
import zlib

import pmt
from gnuradio import blocks, channels, gr

from gpsk_comms import gmsk_command_rx, gmsk_command_tx
from gpsk_comms.gmsk_command_rx import _command_frame_parser
from gpsk_comms.gmsk_command_tx import _command_frame_source
from gpsk_comms.protocol import (
    BODY_FORMAT,
    DEFAULT_ACCESS_CODE,
    PROTOCOL_VERSION,
    bytes_to_bits,
    encode_payload,
    normalise_access_code,
)


def _field(message, name):
    return pmt.dict_ref(message, pmt.intern(name), pmt.PMT_NIL)


def _stored_commands(debug):
    return [
        (
            pmt.symbol_to_string(_field(debug.get_message(index), "command")),
            pmt.to_uint64(_field(debug.get_message(index), "session_id")),
            pmt.to_uint64(_field(debug.get_message(index), "sequence")),
            pmt.symbol_to_string(_field(debug.get_message(index), "link_state")),
        )
        for index in range(debug.num_messages())
    ]


class GmskLoopbackTest(unittest.TestCase):
    def test_hold_to_send_stops_frames_on_key_release(self):
        source = _command_frame_source(
            byte_rate=10_000,
            repeat_rate=100,
            access_code=DEFAULT_ACCESS_CODE,
            session_id=12,
            hold_to_send=True,
        )
        filler = bytes([0x55]) * source._interval_bytes
        self.assertEqual(source._next_interval(), filler)

        press = pmt.make_dict()
        press = pmt.dict_add(press, pmt.intern("command"), pmt.intern("forward"))
        press = pmt.dict_add(press, pmt.intern("pressed"), pmt.PMT_T)
        source._handle_command(press)
        first_sequence = source.sequence
        self.assertTrue(source._next_interval().startswith(normalise_access_code(DEFAULT_ACCESS_CODE)))

        release = pmt.dict_add(press, pmt.intern("pressed"), pmt.PMT_F)
        source._handle_command(release)
        self.assertEqual(source._next_interval(), filler)

        source._handle_command(press)
        self.assertEqual(source.sequence, (first_sequence + 1) & 0xFFFF)
        self.assertTrue(source._next_interval().startswith(normalise_access_code(DEFAULT_ACCESS_CODE)))

    def test_modem_decodes_command_and_suppresses_repeats(self):
        tb = gr.top_block()
        tx = gmsk_command_tx(sample_rate=100_000, samples_per_symbol=4, repeat_rate=50, session_id=7)
        rx = gmsk_command_rx(sample_rate=100_000, samples_per_symbol=4, watchdog_timeout=1.0)
        command_debug = blocks.message_debug()
        message = pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("right"))
        command_strobe = blocks.message_strobe(message, 100)
        tb.connect(tx, rx)
        tb.msg_connect((command_strobe, "strobe"), (tx, "command"))
        tb.msg_connect((rx, "command"), (command_debug, "store"))

        tb.start()
        deadline = time.monotonic() + 2.0
        decoded_commands = []
        while time.monotonic() < deadline:
            decoded_commands = [
                pmt.symbol_to_string(
                    pmt.dict_ref(command_debug.get_message(index), pmt.intern("command"), pmt.PMT_NIL)
                )
                for index in range(command_debug.num_messages())
            ]
            if "right" in decoded_commands:
                break
            time.sleep(0.01)
        tb.stop()
        tb.wait()

        self.assertIn("right", decoded_commands)
        self.assertLessEqual(command_debug.num_messages(), 2)

    def test_modem_with_deterministic_channel_impairments(self):
        tb = gr.top_block()
        tx = gmsk_command_tx(sample_rate=100_000, samples_per_symbol=4, repeat_rate=40, session_id=8)
        channel = channels.channel_model(
            noise_voltage=0.01,
            frequency_offset=0.0005,
            epsilon=1.001,
            taps=[1.0 + 0.0j],
            noise_seed=42,
            block_tags=False,
        )
        rx = gmsk_command_rx(sample_rate=100_000, samples_per_symbol=4, watchdog_timeout=1.0)
        command_debug = blocks.message_debug()
        message = pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("left"))
        command_strobe = blocks.message_strobe(message, 100)
        tb.connect(tx, channel, rx)
        tb.msg_connect((command_strobe, "strobe"), (tx, "command"))
        tb.msg_connect((rx, "command"), (command_debug, "store"))

        tb.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if any(command == "left" for command, *_ in _stored_commands(command_debug)):
                break
            time.sleep(0.01)
        tb.stop()
        tb.wait()
        self.assertTrue(any(command == "left" for command, *_ in _stored_commands(command_debug)))

    def test_parser_suppresses_duplicates_accepts_new_session_and_rejects_corruption(self):
        access = normalise_access_code(DEFAULT_ACCESS_CODE)
        good = encode_payload(10, 4, "backward")
        corrupt = bytearray(encode_payload(10, 5, "right"))
        corrupt[1] ^= 0x20
        stream = access + good + access + good + access + corrupt + access + encode_payload(11, 4, "left")

        tb = gr.top_block()
        source = blocks.vector_source_b(bytes_to_bits(stream), False)
        parser = _command_frame_parser(DEFAULT_ACCESS_CODE, watchdog_timeout=2.0)
        command_debug = blocks.message_debug()
        tb.connect(source, parser)
        tb.msg_connect((parser, "command"), (command_debug, "store"))
        tb.run()

        self.assertEqual(
            _stored_commands(command_debug),
            [("backward", 10, 4, "ok"), ("left", 11, 4, "ok")],
        )
        self.assertEqual(parser.counts["valid"], 2)
        self.assertEqual(parser.counts["duplicate"], 1)
        self.assertEqual(parser.counts["crc_failure"], 1)

    def test_parser_counts_crc_valid_malformed_packet(self):
        body = struct.pack(BODY_FORMAT, PROTOCOL_VERSION, 2, 3, 255)
        malformed = body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        frame = normalise_access_code(DEFAULT_ACCESS_CODE) + malformed
        tb = gr.top_block()
        source = blocks.vector_source_b(bytes_to_bits(frame), False)
        parser = _command_frame_parser(DEFAULT_ACCESS_CODE, watchdog_timeout=2.0)
        command_debug = blocks.message_debug()
        tb.connect(source, parser)
        tb.msg_connect((parser, "command"), (command_debug, "store"))
        tb.run()
        self.assertEqual(command_debug.num_messages(), 0)
        self.assertEqual(parser.counts["malformed"], 1)

    def test_watchdog_emits_stop(self):
        tb = gr.top_block()
        frame = normalise_access_code(DEFAULT_ACCESS_CODE) + encode_payload(9, 3, "forward")
        bits = bytes_to_bits(frame) + [0] * 3000
        source = blocks.vector_source_b(bits, False)
        throttle = blocks.throttle(gr.sizeof_char, 10_000, True, 100)
        parser = _command_frame_parser(DEFAULT_ACCESS_CODE, watchdog_timeout=0.05)
        command_debug = blocks.message_debug()
        tb.connect(source, throttle, parser)
        tb.msg_connect((parser, "command"), (command_debug, "store"))
        tb.run()

        commands = [command for command, *_ in _stored_commands(command_debug)]
        states = [state for *_, state in _stored_commands(command_debug)]
        self.assertEqual(commands, ["forward", "stop"])
        self.assertEqual(states, ["ok", "timeout"])
        self.assertEqual(parser.counts["timeout"], 1)


if __name__ == "__main__":
    unittest.main()
