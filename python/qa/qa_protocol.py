#!/usr/bin/env python3

import unittest
import struct
import zlib

import pmt

from gpsk_comms.protocol import (
    BODY_FORMAT,
    COMMAND_TO_ID,
    PROTOCOL_VERSION,
    PacketCRCError,
    PacketFormatError,
    PacketLengthError,
    decode_payload,
    encode_payload,
    normalise_access_code,
)
from gpsk_comms.gmsk_command_tx import _command_frame_source, gmsk_command_tx


class ProtocolTest(unittest.TestCase):
    def test_all_commands_round_trip(self):
        for sequence, command in enumerate(COMMAND_TO_ID):
            encoded = encode_payload(0x1234, sequence, command)
            self.assertEqual(decode_payload(encoded), (0x1234, sequence, command))

    def test_crc_rejects_corruption(self):
        encoded = bytearray(encode_payload(1, 2, "left"))
        encoded[2] ^= 0x01
        with self.assertRaisesRegex(PacketCRCError, "CRC mismatch"):
            decode_payload(encoded)

    def test_truncated_payload_is_rejected(self):
        with self.assertRaises(PacketLengthError):
            decode_payload(encode_payload(1, 2, "left")[:-1])

    def test_crc_valid_invalid_fields_are_rejected(self):
        for version, command_id in ((PROTOCOL_VERSION + 1, 0), (PROTOCOL_VERSION, 255)):
            body = struct.pack(BODY_FORMAT, version, 1, 2, command_id)
            payload = body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
            with self.assertRaises(PacketFormatError):
                decode_payload(payload)

    def test_sequence_is_16_bit(self):
        self.assertEqual(decode_payload(encode_payload(1, 0x10001, "stop"))[1], 1)

    def test_sequence_changes_only_for_a_new_valid_command_and_rolls_over(self):
        source = _command_frame_source(10_000, 10, "D391DA26", session_id=5)
        source._sequence = 0xFFFF
        source._handle_command(
            pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("forward"))
        )
        self.assertEqual((source.command, source.sequence, source.session_id), ("forward", 0, 5))
        source._handle_command(
            pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("forward"))
        )
        self.assertEqual(source.sequence, 0)

    def test_invalid_command_is_ignored(self):
        source = _command_frame_source(10_000, 10, "D391DA26", session_id=5)
        source._handle_command(
            pmt.dict_add(pmt.make_dict(), pmt.intern("command"), pmt.intern("up"))
        )
        self.assertEqual((source.command, source.sequence), ("stop", 0))

    def test_stream_timed_w_s_d_a_cycle(self):
        source = _command_frame_source(
            8_000,
            20,
            "D391DA26",
            session_id=5,
            command_cycle=("forward", "backward", "right", "left"),
            cycle_period=0.5,
        )
        observed = []
        for _ in range(31):
            interval = source._next_interval()
            decoded = decode_payload(interval[4 : 4 + 10])
            identity = decoded[1:]
            if not observed or identity != observed[-1]:
                observed.append(identity)
        self.assertEqual(
            observed,
            [
                (0, "forward"),
                (1, "backward"),
                (2, "right"),
                (3, "left"),
            ],
        )

    def test_access_code_validation(self):
        self.assertEqual(normalise_access_code("0xD391_DA26"), bytes.fromhex("D391DA26"))
        with self.assertRaises(ValueError):
            normalise_access_code("1234")

    def test_impossible_repeat_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeat_rate is too high"):
            gmsk_command_tx(sample_rate=10_000, samples_per_symbol=4, repeat_rate=1000)


if __name__ == "__main__":
    unittest.main()
