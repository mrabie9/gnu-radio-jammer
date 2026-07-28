"""Frequency-domain excision of narrowband jammers.

A continuous-wave or slowly-swept jammer concentrates all of its power into a
handful of FFT bins, while a spread signal is close to flat across the band.
Zeroing the outlying bins therefore removes almost all of the jammer's energy
and only a small fraction of the wanted signal -- if 4 bins out of 1024 are
excised, the signal loses 0.02 dB while a tone sitting in those bins loses
everything.

This composes with the processing gain in :mod:`gpsk_comms.dsss` rather than
competing with it: spreading handles jammers that are spread out, excision
handles jammers that are concentrated, and a jammer cannot be both at once for a
fixed total power.

The detection threshold is set against the *median* bin magnitude, not the mean.
A mean is dragged upwards by the very outliers being detected, so a strong
enough jammer would raise the threshold above itself and escape; the median is
insensitive to a minority of large values.
"""

import threading

import numpy
from gnuradio import gr

DEFAULT_FFT_SIZE = 1024

#: Bins whose magnitude exceeds this multiple of the median are excised. 6.0 is
#: about 15 dB, comfortably above the peaks of a flat spread signal while still
#: catching any tone worth worrying about.
DEFAULT_THRESHOLD = 6.0

#: Refuse to blank more than this fraction of the spectrum in one block. Past
#: this point the interferer is not narrowband and excision is the wrong tool --
#: blanking half the band would destroy the signal while barely denting a
#: barrage jammer. Hitting this limit is a strong hint the jammer is broadband,
#: which is what the processing gain is for.
MAX_EXCISED_FRACTION = 0.25


class narrowband_excision(gr.sync_block):
    """Overlap-add FFT excision of outlying spectral bins.

    Uses a Hann window with 50% overlap, which sums to a constant and so
    reconstructs the unexcised signal without block-edge artefacts. Everything
    is whole-array numpy, one FFT per half-block.
    """

    def __init__(
        self,
        fft_size=DEFAULT_FFT_SIZE,
        threshold=DEFAULT_THRESHOLD,
        max_fraction=MAX_EXCISED_FRACTION,
        enabled=True,
    ):
        gr.sync_block.__init__(
            self,
            name="narrowband_excision",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64],
        )
        self._fft_size = int(fft_size)
        if self._fft_size < 16 or self._fft_size % 2:
            raise ValueError("fft_size must be even and at least 16")
        self._threshold = float(threshold)
        if self._threshold <= 1.0:
            raise ValueError("threshold must exceed 1.0")
        self._max_fraction = float(max_fraction)
        self._enabled = bool(enabled)

        self._hop = self._fft_size // 2
        # Hann with 50% overlap sums to exactly 1.0 everywhere.
        self._window = numpy.hanning(self._fft_size + 1)[:-1].astype(numpy.float32)

        # Prime the input buffer with a full FFT block of zeros. Without priming
        # the first call cannot produce as many samples as it consumes, and a
        # sync block that zero-fills the shortfall injects those zeros *into* the
        # stream, shifting everything downstream forever.
        #
        # A full block, not one hop, is required. After priming with P zeros the
        # loop has produced at least ``P + consumed - fft_size`` samples, so
        # production only provably keeps pace when ``P >= fft_size``. Priming
        # with one hop satisfies it only when the work-call sizes happen to be
        # multiples of the hop; with arbitrary sizes the buffer runs dry a few
        # thousand samples in and the output silently desynchronises from there.
        self._pending = numpy.zeros(self._fft_size, dtype=numpy.complex64)
        self._tail = numpy.zeros(self._hop, dtype=numpy.complex64)
        self._ready = numpy.zeros(0, dtype=numpy.complex64)
        self._lock = threading.Lock()
        self._blocks = 0
        self._excised_bins = 0
        self._saturated_blocks = 0

    @property
    def statistics(self):
        """Excision activity, for the link-state classifier to read."""
        with self._lock:
            blocks = max(self._blocks, 1)
            return {
                "blocks": self._blocks,
                "excised_bins": self._excised_bins,
                "mean_excised_bins": self._excised_bins / blocks,
                "saturated_blocks": self._saturated_blocks,
            }

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)

    def _excise_block(self, block):
        spectrum = numpy.fft.fft(block * self._window)
        magnitude = numpy.abs(spectrum)
        median = numpy.median(magnitude)
        if median <= 0.0:
            return numpy.fft.ifft(spectrum)

        mask = magnitude > self._threshold * median
        count = int(mask.sum())
        limit = int(self._max_fraction * self._fft_size)
        saturated = count > limit
        if saturated:
            # Excise only the worst offenders up to the cap, so a broadband
            # jammer degrades this stage gracefully instead of gutting the band.
            order = numpy.argsort(magnitude)[::-1]
            mask = numpy.zeros_like(mask)
            mask[order[:limit]] = True
            count = limit

        if count:
            spectrum = spectrum.copy()
            spectrum[mask] = 0.0

        with self._lock:
            self._blocks += 1
            self._excised_bins += count
            self._saturated_blocks += int(saturated)
        return numpy.fft.ifft(spectrum)

    def work(self, input_items, output_items):
        output = output_items[0]
        count = len(output)

        if not self._enabled:
            output[:] = input_items[0][:count]
            return count

        self._pending = numpy.concatenate((self._pending, input_items[0][:count]))

        produced = [self._ready]
        while len(self._pending) >= self._fft_size:
            block = self._pending[: self._fft_size]
            processed = self._excise_block(block)
            # Overlap-add: the first half completes the previous block's tail.
            produced.append((processed[: self._hop] + self._tail).astype(numpy.complex64))
            self._tail = processed[self._hop :].astype(numpy.complex64)
            self._pending = self._pending[self._hop :]
        self._ready = numpy.concatenate(produced)

        # Priming above guarantees production keeps pace with consumption, so
        # the shortfall branch should never run. It is retained only so that an
        # unexpected buffering edge case degrades to a gap rather than to a
        # stream-wide misalignment that would be far harder to diagnose.
        emit = min(count, len(self._ready))
        output[:emit] = self._ready[:emit]
        output[emit:] = 0.0
        self._ready = self._ready[emit:]
        return count
