#!/usr/bin/env python3

import unittest

import numpy
import pmt
from gnuradio import blocks, gr

from gpsk_comms.dsss import (
    PREAMBLE_SYMBOLS,
    SNR_ESTIMATE_LIMITS,
    bits_to_symbols,
    build_burst,
    burst_length,
    chip_phase,
    despread,
    differential_decode,
    differential_encode,
    dsss_despreader,
    estimate_frequency_offset,
    estimate_snr_db,
    matched_filter_taps,
    spread,
    symbols_to_bits,
)
from gpsk_comms.security import derive_keys, pn_code

KEYS = derive_keys(bytes(range(32)))


def make_code(length=255, hop=0):
    return pn_code(KEYS.pn, hop, length).astype(numpy.float32)


#: Trailing idle appended to every test stream, in symbols. The receiver only
#: searches once it holds a full window, so a burst within one window of the end
#: of a *finite* capture is never decoded. A live link always has more samples
#: arriving; a recorded one must be padded. Comfortably exceeds the window.
TRAILING_IDLE_SYMBOLS = 320


def pad_tail(stream, spreading_factor):
    """Append the trailing idle a finite capture needs to flush its last burst."""
    tail = numpy.zeros(TRAILING_IDLE_SYMBOLS * spreading_factor, dtype=numpy.complex64)
    return numpy.concatenate((numpy.asarray(stream, dtype=numpy.complex64), tail))


def run_receiver(stream, code, preamble, payload_bits, threshold=0.25):
    """Push ``stream`` through the despreader and return the recovered payloads."""
    stream = pad_tail(stream, len(code))
    top = gr.top_block()
    source = blocks.vector_source_c(numpy.asarray(stream, dtype=numpy.complex64).tolist(), False)
    receiver = dsss_despreader(
        code, preamble, payload_bits, symbol_rate=8e6 / len(code), threshold=threshold
    )
    sink = blocks.message_debug()
    top.connect(source, receiver)
    top.msg_connect((receiver, "payload"), (sink, "store"))
    top.run()
    recovered = [
        symbols_to_bits(numpy.array(pmt.f32vector_elements(pmt.cdr(sink.get_message(index)))))
        for index in range(sink.num_messages())
    ]
    return recovered, receiver


class SpreadingTest(unittest.TestCase):
    def test_round_trip_is_exact_for_every_supported_length(self):
        rng = numpy.random.default_rng(1)
        for length in (127, 255, 511, 1023):
            code = make_code(length)
            bits = rng.integers(0, 2, 40)
            chips = spread(bits_to_symbols(bits), code)
            self.assertEqual(len(chips), 40 * length)
            recovered = symbols_to_bits(despread(chips, code) / length)
            numpy.testing.assert_array_equal(recovered, bits)

    def test_despread_ignores_a_trailing_partial_symbol(self):
        code = make_code(127)
        chips = spread(bits_to_symbols([0, 1, 1]), code)
        self.assertEqual(len(despread(chips[:-5], code)), 2)
        self.assertEqual(len(despread(chips[:10], code)), 0)

    def test_matched_filter_taps_are_reversed_and_conjugated(self):
        code = make_code(127)
        numpy.testing.assert_array_equal(matched_filter_taps(code).real, code[::-1])

    def test_spreading_gain_appears_in_the_correlation_peak(self):
        code = make_code(1023)
        chips = spread(bits_to_symbols([0]), code)
        self.assertAlmostEqual(float(despread(chips, code)[0]), 1023.0, places=3)


class DifferentialTest(unittest.TestCase):
    def test_round_trip(self):
        rng = numpy.random.default_rng(2)
        bits = rng.integers(0, 2, 100)
        encoded = differential_encode(bits_to_symbols(bits))
        self.assertEqual(len(encoded), len(bits) + 1)
        numpy.testing.assert_array_equal(
            symbols_to_bits(differential_decode(encoded.astype(numpy.complex64))), bits
        )

    def test_decoding_is_invariant_to_carrier_phase(self):
        rng = numpy.random.default_rng(3)
        bits = rng.integers(0, 2, 64)
        encoded = differential_encode(bits_to_symbols(bits)).astype(numpy.complex64)
        for phase in (0.0, 0.7, numpy.pi / 2, numpy.pi, 2.5):
            rotated = encoded * numpy.exp(1j * phase)
            numpy.testing.assert_array_equal(
                symbols_to_bits(differential_decode(rotated)),
                bits,
                err_msg=f"phase {phase}",
            )


class TimingAndEstimatorTest(unittest.TestCase):
    def test_chip_phase_finds_the_symbol_boundary(self):
        code = make_code(255)
        burst = build_burst([1, 0, 1, 1], code, [0] * 8)
        for offset in (0, 37, 128, 254):
            padded = numpy.concatenate((numpy.zeros(offset, dtype=numpy.complex64), burst))
            correlation = numpy.convolve(padded, matched_filter_taps(code))[: len(padded)]
            self.assertEqual(chip_phase(correlation, 255), (offset + 254) % 255)

    def test_chip_phase_handles_a_short_buffer(self):
        self.assertEqual(chip_phase(numpy.zeros(10, dtype=numpy.complex64), 255), 0)

    def test_snr_estimate_tracks_added_noise(self):
        rng = numpy.random.default_rng(4)
        clean = numpy.full(32, 1.0 + 0j, dtype=numpy.complex64)
        previous = None
        for sigma in (0.01, 0.1, 0.3, 1.0):
            noisy = clean + rng.normal(0, sigma, 32) + 1j * rng.normal(0, sigma, 32)
            estimate = estimate_snr_db(noisy)
            self.assertGreaterEqual(estimate, SNR_ESTIMATE_LIMITS[0])
            self.assertLessEqual(estimate, SNR_ESTIMATE_LIMITS[1])
            if previous is not None:
                self.assertLess(estimate, previous)
            previous = estimate

    def test_snr_estimate_is_clamped_not_infinite(self):
        self.assertEqual(estimate_snr_db(numpy.ones(16, dtype=numpy.complex64)), 40.0)
        self.assertEqual(estimate_snr_db(numpy.zeros(1, dtype=numpy.complex64)), -30.0)

    def test_frequency_estimate_recovers_a_known_offset(self):
        symbol_rate = 8e6 / 255
        for offset_hz in (-40.0, 0.0, 25.0, 60.0):
            index = numpy.arange(PREAMBLE_SYMBOLS)
            rotated = numpy.exp(2j * numpy.pi * offset_hz * index / symbol_rate)
            estimate = estimate_frequency_offset(rotated, symbol_rate)
            self.assertAlmostEqual(estimate, offset_hz, delta=3.0)

    def test_frequency_estimate_is_safe_on_degenerate_input(self):
        self.assertEqual(estimate_frequency_offset(numpy.zeros(2), 1000.0), 0.0)
        self.assertEqual(estimate_frequency_offset(numpy.zeros(32), 1000.0), 0.0)


class BurstTest(unittest.TestCase):
    def test_burst_length_matches_the_generated_burst(self):
        code = make_code(255)
        for payload in (8, 96, 204):
            burst = build_burst([0] * payload, code, [0] * PREAMBLE_SYMBOLS)
            self.assertEqual(len(burst), burst_length(payload, 255))

    def test_receiver_recovers_every_burst_in_a_clean_stream(self):
        rng = numpy.random.default_rng(5)
        code = make_code(255)
        preamble = rng.integers(0, 2, PREAMBLE_SYMBOLS)
        payload = rng.integers(0, 2, 96)
        burst = build_burst(payload, code, preamble)
        gap = numpy.zeros(6 * 255, dtype=numpy.complex64)
        stream = numpy.concatenate([gap, burst, gap, burst, gap, burst, gap])

        recovered, receiver = run_receiver(stream, code, preamble, len(payload))
        self.assertEqual(len(recovered), 3)
        for bits in recovered:
            numpy.testing.assert_array_equal(bits, payload)
        self.assertEqual(receiver.counts["burst"], 3)

    def test_receiver_survives_a_carrier_phase_rotation(self):
        rng = numpy.random.default_rng(6)
        code = make_code(255)
        preamble = rng.integers(0, 2, PREAMBLE_SYMBOLS)
        payload = rng.integers(0, 2, 64)
        gap = numpy.zeros(6 * 255, dtype=numpy.complex64)
        burst = build_burst(payload, code, preamble)
        for phase in (0.9, numpy.pi):
            stream = numpy.concatenate([gap, burst * numpy.exp(1j * phase), gap])
            recovered, _ = run_receiver(stream, code, preamble, len(payload))
            self.assertEqual(len(recovered), 1, f"phase {phase}")
            numpy.testing.assert_array_equal(recovered[0], payload)

    def test_receiver_publishes_nothing_for_noise_only_input(self):
        rng = numpy.random.default_rng(7)
        code = make_code(255)
        preamble = rng.integers(0, 2, PREAMBLE_SYMBOLS)
        noise = (rng.normal(0, 1, 60_000) + 1j * rng.normal(0, 1, 60_000)).astype(numpy.complex64)
        recovered, _ = run_receiver(noise, code, preamble, 96, threshold=0.45)
        self.assertEqual(recovered, [])

    def test_wrong_code_isolation_is_modest_and_below_the_periodic_bound(self):
        """Effective isolation between codes is only about 14 dB at N=255.

        Three measurements that are easy to confuse, in increasing realism:

        1. At perfect alignment two Gold codes may nearly cancel -- 48 dB was
           observed here. Quoting this number would badly overstate isolation.
        2. The periodic Gold cross-correlation bound gives 255/31, or 18.3 dB.
        3. Across a *data-modulated* stream the correlator window straddles
           symbol transitions, so the relevant quantity is the aperiodic partial
           correlation, which exceeds the periodic bound. Measured: ~14 dB.

        The third is what governs, because a receiver's timing search settles on
        whichever lag carries the most energy. Assert against that.

        Note what this deliberately does not claim. The residual still carries
        the symbol signs, so in a noiseless simulation a wrong-code receiver can
        sync on a signal 14 dB down and decode it correctly. Code choice is an
        acquisition and isolation mechanism, never an access control -- the MAC
        in protocol.py is what makes a frame unforgeable.
        """
        rng = numpy.random.default_rng(8)
        bits = rng.integers(0, 2, 64)
        right, wrong = make_code(255, hop=0), make_code(255, hop=1)
        chips = spread(bits_to_symbols(bits), right)

        aligned_peak = numpy.abs(despread(chips, right)).mean()
        worst_lag_leakage = numpy.abs(
            numpy.convolve(chips, matched_filter_taps(wrong))
        ).max()
        isolation_db = 20 * numpy.log10(aligned_peak / worst_lag_leakage)
        self.assertGreater(isolation_db, 12.0)
        # Upper bound guards against a future change that accidentally makes the
        # measurement look better than it is -- for example by only ever
        # evaluating the aligned lag.
        self.assertLess(isolation_db, 20.0)

    def test_processing_gain_holds_at_positive_jam_to_signal(self):
        # N=255 gives 24.1 dB of gain, so a jammer 10 dB above the signal still
        # leaves a comfortable margin and the payload must survive intact.
        rng = numpy.random.default_rng(9)
        code = make_code(255)
        preamble = rng.integers(0, 2, PREAMBLE_SYMBOLS)
        payload = rng.integers(0, 2, 96)
        gap = numpy.zeros(6 * 255, dtype=numpy.complex64)
        stream = numpy.concatenate([gap, build_burst(payload, code, preamble), gap])
        amplitude = 10 ** (10.0 / 20.0)
        noise = rng.normal(0, amplitude / numpy.sqrt(2), len(stream)) + 1j * rng.normal(
            0, amplitude / numpy.sqrt(2), len(stream)
        )
        recovered, receiver = run_receiver(stream + noise, code, preamble, len(payload))
        # The correlator may also false-alarm on noise at this permissive
        # threshold. That is expected and harmless: a spurious detection carries
        # no valid MAC and is discarded by the frame decoder. What must hold is
        # that the genuine payload is recovered intact.
        self.assertTrue(
            any(numpy.array_equal(bits, payload) for bits in recovered),
            f"payload not recovered among {len(recovered)} detections",
        )


if __name__ == "__main__":
    unittest.main()
