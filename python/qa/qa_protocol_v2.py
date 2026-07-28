#!/usr/bin/env python3

import unittest

from gpsk_comms.protocol import (
    COMMAND_TO_ID,
    COUNTER_MODULUS,
    DEFAULT_MAX_ADVANCE,
    PAYLOAD_SIZE_V2,
    PROTOCOL_VERSION_V2,
    PacketAuthError,
    PacketFormatError,
    PacketLengthError,
    PacketReplayError,
    ReplayWindow,
    decode_frame_v2,
    encode_frame_v2,
    monotonic_counter,
)
from gpsk_comms.security import MAC_SIZE, derive_keys, frame_mac

KEYS = derive_keys(bytes(range(32)))
OTHER_KEYS = derive_keys(bytes(range(1, 33)))


class FrameV2Test(unittest.TestCase):
    def test_all_commands_round_trip(self):
        for index, command in enumerate(COMMAND_TO_ID):
            frame = encode_frame_v2(KEYS.mac, 1000 + index, command)
            self.assertEqual(len(frame), PAYLOAD_SIZE_V2)
            self.assertEqual(decode_frame_v2(KEYS.mac, frame), (1000 + index, command))

    def test_frame_is_twelve_bytes(self):
        self.assertEqual(PAYLOAD_SIZE_V2, 12)
        self.assertEqual(len(encode_frame_v2(KEYS.mac, 1, "stop")), 12)

    def test_unsupported_command_is_rejected_at_encode(self):
        with self.assertRaises(ValueError):
            encode_frame_v2(KEYS.mac, 1, "launch")

    def test_every_single_bit_flip_is_rejected(self):
        frame = encode_frame_v2(KEYS.mac, 4242, "left")
        for index in range(len(frame)):
            for bit in range(8):
                corrupted = bytearray(frame)
                corrupted[index] ^= 1 << bit
                with self.assertRaises(PacketAuthError):
                    decode_frame_v2(KEYS.mac, corrupted)

    def test_frame_from_another_key_is_rejected(self):
        frame = encode_frame_v2(OTHER_KEYS.mac, 99, "forward")
        with self.assertRaises(PacketAuthError):
            decode_frame_v2(KEYS.mac, frame)

    def test_truncated_and_padded_frames_are_rejected(self):
        frame = encode_frame_v2(KEYS.mac, 7, "right")
        for bad in (frame[:-1], frame + b"\x00"):
            with self.assertRaises(PacketLengthError):
                decode_frame_v2(KEYS.mac, bad)

    def test_authentic_frame_with_bad_fields_is_a_format_error(self):
        # Forge fields but re-MAC with the real key: proves field validation runs
        # after authentication rather than instead of it.
        for version, command_id in ((PROTOCOL_VERSION_V2 + 1, 0), (PROTOCOL_VERSION_V2, 200)):
            body = bytes([version]) + (5).to_bytes(6, "big") + bytes([command_id])
            with self.assertRaises(PacketFormatError):
                decode_frame_v2(KEYS.mac, body + frame_mac(KEYS.mac, body))

    def test_mac_is_the_expected_width(self):
        self.assertEqual(MAC_SIZE, 4)

    def test_counter_wraps_at_six_bytes(self):
        counter, command = decode_frame_v2(
            KEYS.mac, encode_frame_v2(KEYS.mac, COUNTER_MODULUS + 77, "stop")
        )
        self.assertEqual((counter, command), (77, "stop"))

    def test_monotonic_counter_advances(self):
        first = monotonic_counter()
        self.assertLessEqual(0, first)
        self.assertLess(first, COUNTER_MODULUS)
        self.assertGreaterEqual(monotonic_counter(), first)


class ReplayWindowTest(unittest.TestCase):
    def test_first_frame_is_accepted_then_strictly_increasing(self):
        window = ReplayWindow()
        self.assertEqual(window.check(500), 500)
        self.assertEqual(window.check(501), 501)
        self.assertEqual(window.last, 501)

    def test_replayed_and_stale_counters_are_rejected(self):
        window = ReplayWindow()
        window.check(500)
        for stale in (500, 499, 0):
            with self.assertRaises(PacketReplayError):
                window.check(stale)
        # A rejected frame must not move the window forward.
        self.assertEqual(window.last, 500)

    def test_far_future_counter_cannot_lock_out_the_transmitter(self):
        window = ReplayWindow(max_advance=1000)
        window.check(500)
        with self.assertRaises(PacketReplayError):
            window.check(500 + 1001)
        self.assertEqual(window.check(500 + 1000), 1500)

    def test_clock_bound_rejects_a_stale_first_frame(self):
        # Without a clock the first frame after boot is accepted on trust; with
        # one, a frame captured long ago is rejected even at startup.
        window = ReplayWindow(max_advance=1000, clock=lambda: 10_000_000)
        with self.assertRaises(PacketReplayError):
            window.check(500)
        self.assertIsNone(window.last)
        self.assertEqual(window.check(10_000_500), 10_000_500)

    def test_clock_bound_rejects_far_future_first_frame(self):
        window = ReplayWindow(max_advance=1000, clock=lambda: 10_000_000)
        with self.assertRaises(PacketReplayError):
            window.check(20_000_000)

    def test_reset_clears_history(self):
        window = ReplayWindow()
        window.check(900)
        window.reset()
        self.assertIsNone(window.last)
        self.assertEqual(window.check(1), 1)

    def test_invalid_max_advance_is_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                ReplayWindow(max_advance=bad)

    def test_default_window_is_five_seconds_of_microseconds(self):
        self.assertEqual(DEFAULT_MAX_ADVANCE, 5_000_000)


class EndToEndTest(unittest.TestCase):
    def test_captured_frame_cannot_be_replayed_into_a_live_receiver(self):
        window = ReplayWindow()
        captured = encode_frame_v2(KEYS.mac, 1_000_000, "forward")
        window.check(decode_frame_v2(KEYS.mac, captured)[0])

        # Operator has since commanded stop.
        window.check(decode_frame_v2(KEYS.mac, encode_frame_v2(KEYS.mac, 1_000_100, "stop"))[0])

        # Adversary retransmits the earlier 'forward' verbatim. It authenticates
        # -- it is a genuine frame -- but the counter has already been used.
        counter, command = decode_frame_v2(KEYS.mac, captured)
        self.assertEqual(command, "forward")
        with self.assertRaises(PacketReplayError):
            window.check(counter)


if __name__ == "__main__":
    unittest.main()
