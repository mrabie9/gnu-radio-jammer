"""Direct-sequence spreading and despreading.

This is the layer that buys processing gain against a broadband jammer. Each
data symbol is multiplied by a length-N spreading code, so a jammer that does
not know the code must spread its power across the whole occupied bandwidth and
loses ``10*log10(N)`` dB relative to the wanted signal -- 30 dB at N=1023.

Two design choices are worth stating because they shape the whole receiver:

**The matched filter is one code period long, not one preamble long.** Because
every symbol is the same code multiplied by +/-1, correlating against a single
period produces a peak once per symbol for the entire burst rather than only at
the preamble. That keeps the filter at N taps (cheap, and it runs in C++ inside
``filter.fft_filter_ccc``), gives chip timing for free from the peak positions,
and turns frame sync into a short search over *symbol signs* instead of a
32k-tap correlation.

**Data symbols are differentially encoded.** The receiver has no carrier
recovery loop, so an unknown carrier phase would otherwise flip every bit.
Differential encoding removes the absolute phase, and a coarse frequency
estimate taken across the preamble removes most of the residual rotation. See
:func:`estimate_frequency_offset` for the offset range this actually tolerates.
"""

import threading

import numpy
import pmt
from gnuradio import filter as gr_filter
from gnuradio import gr

#: Default number of chips per data symbol. 30.1 dB of processing gain.
DEFAULT_SPREADING_FACTOR = 1023

#: Symbols of preamble prefixed to every burst. Long enough to give a reliable
#: frame-start correlation and a usable frequency estimate, short enough that it
#: stays a small fraction of the burst.
PREAMBLE_SYMBOLS = 32


def bits_to_symbols(bits):
    """Map 0/1 bits to +1/-1 BPSK symbols."""
    return (1 - 2 * numpy.asarray(bits, dtype=numpy.int16)).astype(numpy.float32)


def symbols_to_bits(symbols):
    """Map +/-1 symbols (or soft values) back to 0/1 bits."""
    return (numpy.asarray(symbols) < 0).astype(numpy.uint8)


def differential_encode(symbols):
    """Return the running product of ``symbols``, starting from a +1 reference.

    The reference symbol is prepended, so the output is one longer than the
    input. The receiver recovers each data symbol as the product of adjacent
    received symbols, which cancels any constant phase.
    """
    symbols = numpy.asarray(symbols, dtype=numpy.float32)
    return numpy.concatenate(([numpy.float32(1.0)], numpy.cumprod(symbols)))


def differential_decode(symbols):
    """Recover data symbols from a differentially encoded sequence.

    Accepts complex soft values. The product of a symbol with the conjugate of
    its predecessor has a phase of 0 or pi regardless of the common carrier
    phase, so the real part is the soft decision.
    """
    symbols = numpy.asarray(symbols)
    return numpy.real(symbols[1:] * numpy.conj(symbols[:-1]))


def spread(symbols, code):
    """Expand each symbol into ``len(code)`` chips.

    Vectorised with an outer product: no per-chip Python loop ever runs, which
    is what keeps the transmitter viable at multi-megachip rates.
    """
    symbols = numpy.asarray(symbols, dtype=numpy.float32)
    code = numpy.asarray(code, dtype=numpy.float32)
    return numpy.outer(symbols, code).ravel()


def despread(chips, code):
    """Collapse whole chip periods back into symbols.

    ``chips`` must be aligned to a symbol boundary and a whole multiple of the
    code length; :class:`dsss_receiver` establishes that alignment first.
    """
    code = numpy.asarray(code, dtype=numpy.float32)
    length = len(code)
    usable = (len(chips) // length) * length
    if usable == 0:
        return numpy.zeros(0, dtype=numpy.complex64)
    grid = numpy.asarray(chips[:usable]).reshape(-1, length)
    return grid @ code


def matched_filter_taps(code):
    """Taps for a despreading matched filter.

    Time-reversed and conjugated so that ``filter.fft_filter_ccc`` performs
    correlation rather than convolution, placing the peak at the *last* chip of
    each symbol.
    """
    code = numpy.asarray(code, dtype=numpy.complex64)
    return numpy.conj(code[::-1]).astype(numpy.complex64)


def chip_phase(correlation, spreading_factor):
    """Return the chip offset of the symbol boundary within ``correlation``.

    Folds the correlator magnitude into one code period and takes the strongest
    column. Fully vectorised, and averaging over many symbol periods makes the
    estimate robust well below 0 dB SNR.
    """
    usable = (len(correlation) // spreading_factor) * spreading_factor
    if usable == 0:
        return 0
    grid = numpy.abs(numpy.asarray(correlation[:usable])).reshape(-1, spreading_factor)
    return int(numpy.argmax(grid.sum(axis=0)))


def estimate_frequency_offset(preamble_symbols, symbol_rate):
    """Estimate carrier offset in Hz from the phase drift across the preamble.

    Splits the preamble in half and measures the phase advance between the two
    halves. The unambiguous range is +/- ``symbol_rate / (2 * P)`` for a
    P-symbol preamble: at N=1023 chips, 8 Mcps and a 32-symbol preamble that is
    about +/-122 Hz, which is why the transmitter and receiver still need to be
    within a few hundred Hz before this correction is applied. Beyond that range
    the estimate wraps and makes matters worse, so treat it as fine correction
    on top of a decent LO, not as acquisition.
    """
    symbols = numpy.asarray(preamble_symbols)
    if len(symbols) < 4:
        return 0.0
    half = len(symbols) // 2
    first = symbols[:half].sum()
    second = symbols[half : 2 * half].sum()
    if first == 0 or second == 0:
        return 0.0
    phase_step = numpy.angle(second * numpy.conj(first))
    return float(phase_step * symbol_rate / (2.0 * numpy.pi * half))


#: Range the SNR estimate is clamped to. A noiseless simulated signal would
#: otherwise report an absurd figure, and the jam classifier downstream reads
#: this number to decide whether the link is degraded or denied.
SNR_ESTIMATE_LIMITS = (-30.0, 40.0)


def estimate_snr_db(derotated_preamble):
    """Data-aided SNR estimate from a preamble already multiplied by its symbols.

    With the known symbols removed every sample should equal the same complex
    constant, so the mean is the signal and the scatter about it is the noise.
    The result is clamped to :data:`SNR_ESTIMATE_LIMITS`.
    """
    samples = numpy.asarray(derotated_preamble)
    if len(samples) < 2:
        return SNR_ESTIMATE_LIMITS[0]
    mean = samples.mean()
    signal_power = float(numpy.abs(mean) ** 2)
    noise_power = float(numpy.mean(numpy.abs(samples - mean) ** 2))
    if noise_power <= 0.0:
        return SNR_ESTIMATE_LIMITS[1]
    if signal_power <= 0.0:
        return SNR_ESTIMATE_LIMITS[0]
    estimate = 10.0 * numpy.log10(signal_power / noise_power)
    return float(numpy.clip(estimate, *SNR_ESTIMATE_LIMITS))


def build_burst(payload_bits, code, preamble_bits):
    """Assemble one complete burst as complex chips.

    Layout is ``[preamble | differentially encoded payload]``, all spread by the
    same code, so the receiver's per-symbol correlator covers the whole burst.
    """
    preamble = bits_to_symbols(preamble_bits)
    payload = differential_encode(bits_to_symbols(payload_bits))
    return spread(numpy.concatenate((preamble, payload)), code).astype(numpy.complex64)


def burst_length(payload_bit_count, spreading_factor, preamble_bit_count=PREAMBLE_SYMBOLS):
    """Chips occupied by one burst, including the differential reference symbol."""
    return (preamble_bit_count + payload_bit_count + 1) * spreading_factor


class dsss_receiver(gr.sync_block):
    """Despread, frame-synchronise, and publish payload soft bits.

    Sits downstream of a ``filter.fft_filter_ccc`` matched to one code period and
    consumes only that correlator output -- the despread symbols are simply the
    correlator sampled on the symbol boundary, so the raw signal is not needed
    again. All work is whole-array numpy, so although this is Python it carries
    no per-sample loop and sustains multi-Msps input.

    Payloads are published on the ``payload`` message port as a PMT uniform
    vector of soft bit values, for a downstream decoder to consume.
    """

    def __init__(
        self,
        code,
        preamble_bits,
        payload_bit_count,
        symbol_rate,
        threshold=0.25,
        correct_frequency=True,
    ):
        gr.sync_block.__init__(
            self, name="dsss_receiver", in_sig=[numpy.complex64], out_sig=None
        )
        self._code = numpy.asarray(code, dtype=numpy.float32)
        self._spreading_factor = len(self._code)
        self._preamble = bits_to_symbols(preamble_bits)
        self._payload_bit_count = int(payload_bit_count)
        self._symbol_rate = float(symbol_rate)
        self._threshold = float(threshold)
        self._correct_frequency = bool(correct_frequency)

        self._burst_symbols = len(self._preamble) + self._payload_bit_count + 1
        self._burst_chips = self._burst_symbols * self._spreading_factor
        # One burst plus a search margin. The margin sets how many candidate
        # frame offsets each pass can consider; sizing the window to barely more
        # than a burst leaves only a handful and the receiver then syncs only
        # when a burst happens to arrive nearly frame-aligned. On a failed
        # search the buffer slides forward by exactly the margin, never by a
        # whole burst, so a burst straddling two passes is never discarded.
        self._search_margin_symbols = 64
        self._search_margin_chips = self._search_margin_symbols * self._spreading_factor
        self._window_chips = self._burst_chips + self._search_margin_chips

        self._correlation = numpy.zeros(0, dtype=numpy.complex64)
        self._lock = threading.Lock()
        self._counts = {"burst": 0, "no_sync": 0}
        self._last_snr_db = float("-inf")

        self._payload_port = pmt.intern("payload")
        self.message_port_register_out(self._payload_port)

    @property
    def spreading_factor(self):
        return self._spreading_factor

    @property
    def window_chips(self):
        """Chips that must be buffered before a burst can be searched for.

        The hop controller reads this: a code may only be changed once the burst
        sent under the previous one has had time to be buffered *and* searched.
        """
        return self._window_chips

    @property
    def counts(self):
        with self._lock:
            return dict(self._counts)

    @property
    def last_snr_db(self):
        with self._lock:
            return self._last_snr_db

    def set_preamble(self, preamble_bits):
        """Install the sync word to search for from now on.

        Called from the hop controller between bursts, never during one. The
        buffered correlator output is deliberately left alone: it holds the
        previous hop's quiet tail, which will simply fail to sync and be slid
        past, and discarding it would throw away a burst that had already
        arrived but not yet been searched.
        """
        symbols = bits_to_symbols(preamble_bits)
        if len(symbols) != len(self._preamble):
            raise ValueError("sync word length must not change while running")
        with self._lock:
            self._preamble = symbols

    def _publish(self, soft_bits, snr_db):
        # The per-burst SNR travels with its payload. Reading the receiver's
        # "most recent" SNR instead would attribute a false alarm's SNR to the
        # last genuine frame, and the link-state classifier would then downgrade
        # a healthy link because the correlator happened to twitch at noise.
        metadata = pmt.dict_add(pmt.make_dict(), pmt.intern("snr_db"), pmt.from_double(snr_db))
        self.message_port_pub(
            self._payload_port,
            pmt.cons(metadata, pmt.init_f32vector(len(soft_bits), soft_bits.tolist())),
        )

    def _find_frame_start(self, symbols, preamble):
        """Locate the preamble within a run of despread symbols.

        Scores each candidate offset by the normalised inner product between the
        symbol window and the preamble -- cosine similarity, which is 1.0 for a
        clean match and about ``1/sqrt(P)`` for noise. Normalising per window
        rather than against the whole buffer is what makes the score independent
        of how much idle time surrounds the burst, and dividing by the window's
        own energy stops a single loud sample in an otherwise empty window from
        scoring highly.

        ``abs`` on the numerator keeps the score independent of carrier phase.
        Vectorised via a sliding energy sum, so no Python loop runs per offset.

        The sliding energy is accumulated in float64, which is required rather
        than tidy. Despread symbols have magnitude N, so their squares reach
        N**2 -- 65025 at N=1023 -- and a cumulative sum over a few hundred of
        them exceeds the roughly seven significant digits float32 carries. The
        window energy is then a difference of two large nearly-equal numbers, and
        in an idle stretch that difference collapses to numerical noise. The
        score becomes a small numerator over a near-zero denominator, producing
        values in the millions instead of at most one, and the receiver locks
        onto whatever idle sample happened to round the least favourably.
        """
        count = len(preamble)
        if len(symbols) < count:
            return -1, 0.0
        wide = numpy.asarray(symbols, dtype=numpy.complex128)
        numerator = numpy.abs(
            numpy.correlate(wide, preamble.astype(numpy.float64), mode="valid")
        )
        cumulative = numpy.concatenate(([0.0], numpy.cumsum(numpy.abs(wide) ** 2)))
        window_energy = numpy.maximum(cumulative[count:] - cumulative[:-count], 0.0)
        if not len(window_energy) or window_energy.max() <= 0.0:
            return -1, 0.0
        # Ignore windows carrying negligible energy rather than dividing by them.
        floor = window_energy.max() * 1e-9
        scores = numpy.where(
            window_energy > floor,
            numerator / numpy.sqrt(numpy.maximum(window_energy, floor) * count),
            0.0,
        )
        best = int(numpy.argmax(scores))
        return best, float(scores[best])

    def _process(self):
        """Try to extract one burst from the buffered window.

        The receiver needs ``window_chips`` buffered before it will search, so a
        burst arriving within one window of the end of a finite capture is not
        recovered. That is harmless on a live link, which always has more samples
        coming, but a recorded capture must be padded with at least that much
        trailing idle if its last burst is to be decoded. Draining the tail at
        ``stop()`` is not an option: messages published during teardown are not
        delivered downstream, so it would advance the counters without ever
        emitting the payload.
        """
        if len(self._correlation) < self._window_chips:
            return False

        # Snapshot the sync word once. The hop controller may swap it between
        # any two statements here, and scoring a candidate against one sync word
        # while derotating against another would silently corrupt both the
        # frequency estimate and the SNR.
        with self._lock:
            preamble = self._preamble

        phase = chip_phase(self._correlation, self._spreading_factor)
        aligned = self._correlation[phase :: self._spreading_factor]
        if len(aligned) < self._burst_symbols:
            return False

        start, quality = self._find_frame_start(aligned, preamble)
        if start < 0 or quality < self._threshold:
            with self._lock:
                self._counts["no_sync"] += 1
            # Slide forward by the search margin only. Discarding a whole burst
            # here would throw away a burst that straddles two passes before it
            # could ever be assembled.
            self._correlation = self._correlation[self._search_margin_chips :]
            return False

        end = start + self._burst_symbols
        if end > len(aligned):
            return False

        symbols = aligned[start:end].astype(numpy.complex64)
        preamble_part = symbols[: len(preamble)] * preamble
        payload_part = symbols[len(preamble) :]

        if self._correct_frequency:
            offset_hz = estimate_frequency_offset(preamble_part, self._symbol_rate)
            if offset_hz:
                index = numpy.arange(len(payload_part), dtype=numpy.float64)
                payload_part = payload_part * numpy.exp(
                    -2j * numpy.pi * offset_hz * index / self._symbol_rate
                )

        soft_bits = differential_decode(payload_part).astype(numpy.float32)

        snr_db = estimate_snr_db(preamble_part)
        with self._lock:
            self._counts["burst"] += 1
            self._last_snr_db = snr_db

        self._publish(soft_bits, snr_db)

        consumed = (start + self._burst_symbols) * self._spreading_factor + phase
        self._correlation = self._correlation[consumed:]
        return True

    def work(self, input_items, output_items):
        correlation = input_items[0]
        self._correlation = numpy.concatenate((self._correlation, correlation))

        while self._process():
            pass

        # Hard cap so a link that never syncs cannot grow the buffer without
        # bound, while still holding more than one search window.
        limit = self._window_chips + self._burst_chips
        if len(self._correlation) > limit:
            self._correlation = self._correlation[-limit:]
        return len(correlation)


class dsss_despreader(gr.hier_block2):
    """Matched filter plus :class:`dsss_receiver`, wired together.

    The correlation runs inside ``filter.fft_filter_ccc`` -- C++, FFT-based, and
    the only component that touches every sample individually.
    """

    def __init__(self, code, preamble_bits, payload_bit_count, symbol_rate, threshold=0.25):
        gr.hier_block2.__init__(
            self,
            "dsss_despreader",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0),
        )
        self.message_port_register_hier_out("payload")

        self._matched_filter = gr_filter.fft_filter_ccc(1, matched_filter_taps(code).tolist())
        self._receiver = dsss_receiver(
            code, preamble_bits, payload_bit_count, symbol_rate, threshold=threshold
        )
        self.connect(self, self._matched_filter, self._receiver)
        self.msg_connect((self._receiver, "payload"), (self, "payload"))

    def set_code(self, code):
        """Retune the matched filter to a new spreading code.

        Only the filter needs the code: the receiver downstream consumes
        correlator output and cares about the code's *length* alone, which may
        not change. Retuning takes effect on the next samples the filter
        processes, which is a few milliseconds of buffering later -- that lag is
        why the hop controller switches codes mid-dwell rather than on the dwell
        boundary itself.
        """
        code = numpy.asarray(code, dtype=numpy.float32)
        if len(code) != self._receiver.spreading_factor:
            raise ValueError("spreading factor must not change while running")
        self._matched_filter.set_taps(matched_filter_taps(code).tolist())

    def set_preamble(self, preamble_bits):
        """Retune the frame-sync word. See :meth:`dsss_receiver.set_preamble`."""
        self._receiver.set_preamble(preamble_bits)

    @property
    def window_chips(self):
        return self._receiver.window_chips

    @property
    def counts(self):
        return self._receiver.counts

    @property
    def last_snr_db(self):
        return self._receiver.last_snr_db
