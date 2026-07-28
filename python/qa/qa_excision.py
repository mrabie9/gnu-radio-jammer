#!/usr/bin/env python3

import unittest

import numpy
from gnuradio import blocks, gr

from gpsk_comms.excision import (
    DEFAULT_FFT_SIZE,
    MAX_EXCISED_FRACTION,
    narrowband_excision,
)

#: The block delays its output by one FFT block; see the priming comment in
#: excision.py for why that delay exists and why it is exactly this long.
DELAY = DEFAULT_FFT_SIZE


def run_excision(signal, **kwargs):
    top = gr.top_block()
    source = blocks.vector_source_c(numpy.asarray(signal, dtype=numpy.complex64).tolist(), False)
    block = narrowband_excision(**kwargs)
    sink = blocks.vector_sink_c()
    top.connect(source, block, sink)
    top.run()
    return numpy.array(sink.data()), block


def flat_signal(count, seed=0):
    rng = numpy.random.default_rng(seed)
    return (rng.normal(0, 1, count) + 1j * rng.normal(0, 1, count)).astype(numpy.complex64)


def tone(count, amplitude, normalised_frequency=0.13):
    index = numpy.arange(count)
    return (amplitude * numpy.exp(2j * numpy.pi * normalised_frequency * index)).astype(
        numpy.complex64
    )


def tone_amplitude(samples, normalised_frequency=0.13):
    kernel = numpy.exp(-2j * numpy.pi * normalised_frequency * numpy.arange(len(samples)))
    return float(numpy.abs(numpy.dot(samples, kernel)) / len(samples))


class ConstructionTest(unittest.TestCase):
    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({"fft_size": 8}, {"fft_size": 1023}, {"threshold": 1.0}):
            with self.assertRaises(ValueError):
                narrowband_excision(**kwargs)


class ReconstructionTest(unittest.TestCase):
    def test_clean_signal_passes_through_essentially_unchanged(self):
        """Overlap-add must reconstruct exactly when nothing is excised.

        This is the regression guard for stream alignment. An earlier version
        primed its buffer with only half a block, which reconstructed perfectly
        for several thousand samples and then silently desynchronised once the
        buffer ran dry -- so a short test would have passed. Check a long run.
        """
        signal = flat_signal(1024 * 60, seed=1)
        output, block = run_excision(signal)
        self.assertEqual(len(output), len(signal))
        recovered = output[DELAY : DELAY + 50_000]
        numpy.testing.assert_allclose(recovered, signal[:50_000], atol=1e-5)
        self.assertEqual(block.statistics["excised_bins"], 0)

    def test_alignment_holds_over_a_long_stream(self):
        signal = flat_signal(1024 * 60, seed=2)
        output, _ = run_excision(signal)
        for start in (0, 10_000, 30_000, 50_000):
            segment = output[DELAY + start : DELAY + start + 2000]
            reference = signal[start : start + 2000]
            error = numpy.abs(segment - reference).max()
            self.assertLess(error, 1e-4, f"misaligned at sample {start}")

    def test_disabled_block_is_a_pure_passthrough(self):
        signal = flat_signal(8192, seed=3)
        output, block = run_excision(signal, enabled=False)
        numpy.testing.assert_allclose(output, signal, atol=1e-6)
        self.assertEqual(block.statistics["blocks"], 0)


class SuppressionTest(unittest.TestCase):
    def test_continuous_wave_jammer_is_deeply_suppressed(self):
        count = 1024 * 60
        signal = flat_signal(count, seed=4)
        for jam_db in (10.0, 20.0, 30.0, 40.0):
            amplitude = 10 ** (jam_db / 20.0)
            output, _ = run_excision(signal + tone(count, amplitude))
            residual = tone_amplitude(output[4096:20480])
            suppression = 20 * numpy.log10(amplitude / max(residual, 1e-12))
            self.assertGreater(suppression, 40.0, f"J/S {jam_db} dB")

    def test_only_a_few_bins_are_sacrificed(self):
        count = 1024 * 40
        signal = flat_signal(count, seed=5)
        _, block = run_excision(signal + tone(count, 100.0))
        # A handful of bins out of 1024 costs the wanted signal almost nothing.
        self.assertLess(block.statistics["mean_excised_bins"], 32)
        self.assertGreater(block.statistics["mean_excised_bins"], 0)

    def test_broadband_jammer_triggers_no_excision(self):
        """Excision must not fire on barrage noise -- that is spreading's job.

        A flat jammer raises the median along with every bin, so nothing stands
        out. Firing here would blank the band for no benefit.
        """
        count = 1024 * 40
        rng = numpy.random.default_rng(6)
        barrage = (rng.normal(0, 30, count) + 1j * rng.normal(0, 30, count)).astype(
            numpy.complex64
        )
        _, block = run_excision(flat_signal(count, seed=7) + barrage)
        self.assertEqual(block.statistics["excised_bins"], 0)
        self.assertEqual(block.statistics["saturated_blocks"], 0)

    def test_excision_is_capped_for_a_wideband_interferer(self):
        """Many strong bins must degrade gracefully, not gut the spectrum."""
        count = 1024 * 20
        rng = numpy.random.default_rng(8)
        signal = flat_signal(count, seed=9)
        # Energy confined to half the band: enough bins stand above the median to
        # push past the cap.
        wide = numpy.fft.ifft(
            numpy.fft.fft(rng.normal(0, 1, count) + 1j * rng.normal(0, 1, count))
            * (numpy.arange(count) < count // 2)
        ).astype(numpy.complex64)
        _, block = run_excision(signal + 50.0 * wide)
        statistics = block.statistics
        limit = int(MAX_EXCISED_FRACTION * DEFAULT_FFT_SIZE)
        self.assertLessEqual(statistics["mean_excised_bins"], limit)

    def test_multiple_tones_are_all_removed(self):
        count = 1024 * 40
        signal = flat_signal(count, seed=10)
        jammed = signal.copy()
        for frequency in (0.11, 0.23, -0.31):
            jammed = jammed + tone(count, 50.0, frequency)
        output, _ = run_excision(jammed)
        for frequency in (0.11, 0.23, -0.31):
            residual = tone_amplitude(output[4096:20480], frequency)
            self.assertLess(residual, 1.0, f"tone at {frequency} survived")


if __name__ == "__main__":
    unittest.main()
