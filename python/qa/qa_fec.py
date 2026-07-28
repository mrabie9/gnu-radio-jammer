#!/usr/bin/env python3

import unittest

import numpy

from gpsk_comms.dsss import bits_to_symbols
from gpsk_comms.fec import (
    RATE_DENOMINATOR,
    STATE_COUNT,
    TAIL_BITS,
    convolutional_encode,
    decode_frame,
    deinterleave,
    encode_frame,
    encoded_length,
    interleave,
    viterbi_decode,
)


class TrellisTest(unittest.TestCase):
    def test_code_parameters(self):
        self.assertEqual(STATE_COUNT, 64)
        self.assertEqual(TAIL_BITS, 6)
        self.assertEqual(RATE_DENOMINATOR, 2)

    def test_encoded_length_includes_the_tail(self):
        self.assertEqual(encoded_length(96), (96 + 6) * 2)
        self.assertEqual(len(convolutional_encode(numpy.zeros(96, dtype=numpy.uint8))), 204)

    def test_all_zero_input_encodes_to_all_zeros(self):
        coded = convolutional_encode(numpy.zeros(32, dtype=numpy.uint8))
        self.assertEqual(int(coded.sum()), 0)

    def test_a_single_set_bit_produces_the_generator_impulse_response(self):
        # Impulse response length is one constraint length; a shorter or longer
        # response means the shift register is being clocked incorrectly.
        bits = numpy.zeros(32, dtype=numpy.uint8)
        bits[0] = 1
        coded = convolutional_encode(bits).reshape(-1, 2)
        nonzero = numpy.nonzero(coded.any(axis=1))[0]
        self.assertEqual(int(nonzero.max()) + 1, 7)


class ViterbiTest(unittest.TestCase):
    def test_noiseless_round_trip(self):
        rng = numpy.random.default_rng(1)
        for length in (8, 96, 204):
            bits = rng.integers(0, 2, length).astype(numpy.uint8)
            soft = bits_to_symbols(convolutional_encode(bits))
            numpy.testing.assert_array_equal(viterbi_decode(soft, length), bits)

    def test_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            viterbi_decode(numpy.ones(10, dtype=numpy.float32), 96)

    def test_isolated_bit_errors_are_corrected(self):
        rng = numpy.random.default_rng(2)
        bits = rng.integers(0, 2, 96).astype(numpy.uint8)
        soft = bits_to_symbols(convolutional_encode(bits))
        # Well-separated flips are the easy case for a convolutional code.
        for position in (5, 40, 100, 160):
            soft[position] = -soft[position]
        numpy.testing.assert_array_equal(viterbi_decode(soft, 96), bits)

    def test_soft_decisions_outperform_hard_slicing(self):
        """Confirms the ~2 dB that hard-slicing throws away is real.

        Both decoders see the same channel; only the quantisation differs.
        """
        rng = numpy.random.default_rng(3)
        sigma = 10 ** (-1.0 / 20.0)
        soft_errors = hard_errors = 0
        for _ in range(200):
            bits = rng.integers(0, 2, 96).astype(numpy.uint8)
            received = bits_to_symbols(convolutional_encode(bits)) + rng.normal(0, sigma, 204)
            soft_errors += int(numpy.sum(viterbi_decode(received, 96) != bits))
            hard = numpy.sign(received).astype(numpy.float32)
            hard_errors += int(numpy.sum(viterbi_decode(hard, 96) != bits))
        self.assertLess(soft_errors, hard_errors)

    def test_coding_gain_against_awgn(self):
        """Coded BER must be far below uncoded at the same symbol SNR."""
        rng = numpy.random.default_rng(4)
        sigma = 10 ** (-2.0 / 20.0)
        coded_errors = uncoded_errors = 0
        trials = 200
        for _ in range(trials):
            bits = rng.integers(0, 2, 96).astype(numpy.uint8)
            received = bits_to_symbols(convolutional_encode(bits)) + rng.normal(0, sigma, 204)
            coded_errors += int(numpy.sum(viterbi_decode(received, 96) != bits))
            raw = bits_to_symbols(bits) + rng.normal(0, sigma, 96)
            uncoded_errors += int(numpy.sum((raw < 0).astype(numpy.uint8) != bits))
        total = trials * 96
        self.assertLess(coded_errors / total, 0.02)
        self.assertGreater(uncoded_errors / total, 0.05)

    def test_uncoded_reference_matches_theory(self):
        """Guards the measurement itself, not the code under test.

        A uint8 underflow in the symbol mapping once made this whole comparison
        meaningless by pinning every BER at chance; anchoring the uncoded curve
        to Q(2) catches that class of harness bug.
        """
        rng = numpy.random.default_rng(5)
        sigma = 10 ** (-6.0 / 20.0)
        bits = rng.integers(0, 2, 20000).astype(numpy.uint8)
        received = bits_to_symbols(bits) + rng.normal(0, sigma, len(bits))
        ber = numpy.mean((received < 0).astype(numpy.uint8) != bits)
        self.assertAlmostEqual(ber, 0.0228, delta=0.004)


class InterleaverTest(unittest.TestCase):
    def test_round_trip_for_various_depths(self):
        rng = numpy.random.default_rng(6)
        values = rng.integers(0, 2, 204).astype(numpy.uint8)
        for depth in (0, 1, 8, 16, 17, 204):
            restored = deinterleave(interleave(values, depth), depth, len(values))
            numpy.testing.assert_array_equal(restored, values, err_msg=f"depth {depth}")

    def test_interleaving_actually_reorders(self):
        values = numpy.arange(64, dtype=numpy.uint8)
        self.assertFalse(numpy.array_equal(interleave(values, 8), values))

    def test_a_contiguous_burst_is_scattered(self):
        """The property the layer exists for: bursts become isolated errors."""
        depth = 17
        values = numpy.zeros(204, dtype=numpy.uint8)
        values[40:52] = 1  # a 12-symbol burst, as a pulsed jammer would produce
        scattered = deinterleave(values, depth, 204)
        positions = numpy.nonzero(scattered)[0]
        gaps = numpy.diff(positions)
        self.assertEqual(len(positions), 12)
        self.assertGreater(int(gaps.min()), 1)

    def test_interleaving_rescues_a_burst_the_bare_code_cannot(self):
        rng = numpy.random.default_rng(7)
        depth = 17
        bits = rng.integers(0, 2, 96).astype(numpy.uint8)

        # A burst long enough to swamp a K=7 code's memory when contiguous.
        def punch(soft):
            damaged = soft.copy()
            damaged[60:76] = -damaged[60:76]
            return damaged

        bare = viterbi_decode(punch(bits_to_symbols(convolutional_encode(bits))), 96)
        self.assertGreater(int(numpy.sum(bare != bits)), 0)

        interleaved = decode_frame(punch(bits_to_symbols(encode_frame(bits, depth))), 96, depth)
        numpy.testing.assert_array_equal(interleaved, bits)


class FrameApiTest(unittest.TestCase):
    def test_encode_decode_round_trip_with_and_without_interleaving(self):
        rng = numpy.random.default_rng(8)
        bits = rng.integers(0, 2, 96).astype(numpy.uint8)
        for depth in (0, 16, 17):
            soft = bits_to_symbols(encode_frame(bits, depth))
            numpy.testing.assert_array_equal(decode_frame(soft, 96, depth), bits)

    def test_encoded_frame_length_is_stable(self):
        bits = numpy.zeros(96, dtype=numpy.uint8)
        self.assertEqual(len(encode_frame(bits, 0)), 204)
        # Interleaving pads to a whole block, so length may round up.
        self.assertGreaterEqual(len(encode_frame(bits, 17)), 204)


if __name__ == "__main__":
    unittest.main()
