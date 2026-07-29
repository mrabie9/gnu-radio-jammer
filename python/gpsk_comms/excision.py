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
from numpy.lib.stride_tricks import sliding_window_view
from gnuradio import gr

# SciPy's transforms keep single precision end to end; NumPy's promote complex64
# to complex128 and double the memory traffic of every stage downstream of them.
# The fallback is exact, only slower, so SciPy stays an optional dependency.
try:
    from scipy.fft import fft as _fft, ifft as _ifft
except ImportError:  # pragma: no cover - exercised only where SciPy is absent
    from numpy.fft import fft as _fft, ifft as _ifft

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
        self._threshold_power = self._threshold ** 2
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
        # Whether ``_tail`` still equals the plain windowed input, which is the
        # precondition for the reconstruction-free path in _process_batch. The
        # primed zero block satisfies it, and only an actual excision breaks it.
        self._tail_clean = True
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

    def _process_batch(self, source, windows):
        """Excise a whole batch of blocks. Returns ``(output samples, new tail)``.

        Batched rather than looped because this is the only stage that touches
        every sample from Python, and GNU Radio runs each Python block in its own
        thread contending for one GIL -- so what limits the receiver is the *sum*
        of every Python block's cost, not any one of them. Falling under real
        time is a correctness problem and not merely a slow one: the transmitter
        anchors its chip stream to the wall clock, so a receiver that cannot keep
        up drags the whole flowgraph behind the hop schedule it is trying to
        follow, and level 5 stops decoding for reasons no counter explains.

        Every decision stays strictly per-block: median, threshold and cap are
        all computed along the row axis, so a batch produces exactly what the
        same blocks would have produced one at a time.
        """
        count = len(windows)
        spectrum = _fft(windows * self._window, axis=1)
        # Compare powers rather than magnitudes. The median commutes with
        # squaring, so ``|S| > t * median(|S|)`` is ``P > t**2 * median(P)`` --
        # the same decision without a square root over every bin.
        power = spectrum.real ** 2 + spectrum.imag ** 2
        median = numpy.median(power, axis=1)
        # A silent block has a zero median and every bin is "above" it; excising
        # there would blank pure silence and charge it to the statistics.
        mask = (power > self._threshold_power * median[:, None]) & (median[:, None] > 0.0)
        counts = mask.sum(axis=1)

        limit = max(1, int(self._max_fraction * self._fft_size))
        saturated = counts > limit
        if saturated.any():
            # Excise only the worst offenders up to the cap, so a broadband
            # jammer degrades this stage gracefully instead of gutting the band.
            rows = numpy.flatnonzero(saturated)
            strongest = numpy.argpartition(-power[rows], limit - 1, axis=1)[:, :limit]
            mask[rows] = False
            mask[rows[:, None], strongest] = True
            counts[rows] = limit

        excised = int(counts.sum())
        with self._lock:
            self._blocks += count
            self._excised_bins += excised
            self._saturated_blocks += int(saturated.sum())

        if not excised and self._tail_clean:
            # Nothing stood out, which is the ordinary case whenever no
            # narrowband jammer is present. A Hann pair at 50% overlap sums to
            # unity, so reconstructing here would return the input bit for bit:
            # skip the inverse transform and the overlap-add entirely and hand
            # back the input. This is what keeps excision close to free when it
            # has nothing to do, and so what makes leaving it switched on cheap.
            #
            # Only valid while the carried-over tail is itself unexcised. If the
            # previous batch removed anything, its tail overlaps the first hop of
            # this one and must be added in properly, so that batch is
            # reconstructed the long way even though it excises nothing itself.
            emitted = source[: count * self._hop].astype(numpy.complex64)
            tail = (windows[-1][self._hop :] * self._window[self._hop :]).astype(
                numpy.complex64
            )
            return emitted, tail

        self._tail_clean = not excised
        spectrum[mask] = 0.0
        processed = _ifft(spectrum, axis=1)
        # Overlap-add: each block's first half completes its predecessor's tail,
        # so the tails are the same array shifted by one block.
        leading = processed[:, : self._hop]
        trailing = processed[:, self._hop :]
        tails = numpy.empty_like(trailing)
        tails[0] = self._tail
        tails[1:] = trailing[:-1]
        return (
            (leading + tails).ravel().astype(numpy.complex64),
            trailing[-1].astype(numpy.complex64),
        )

    def work(self, input_items, output_items):
        output = output_items[0]
        count = len(output)

        if not self._enabled:
            output[:] = input_items[0][:count]
            return count

        self._pending = numpy.concatenate((self._pending, input_items[0][:count]))

        produced = [self._ready]
        if len(self._pending) >= self._fft_size:
            # Every block start, one hop apart, that has a full block behind it.
            windows = sliding_window_view(self._pending, self._fft_size)[:: self._hop]
            emitted, self._tail = self._process_batch(self._pending, windows)
            produced.append(emitted)
            self._pending = self._pending[len(windows) * self._hop :]
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
