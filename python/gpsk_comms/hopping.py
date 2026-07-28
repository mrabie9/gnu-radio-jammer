"""Frequency hopping and band escape.

Hopping is what defeats a jammer that concentrates its power somewhere specific,
and it is the main source of low probability of intercept. Two tiers, because
the two useful hop rates are separated by orders of magnitude:

**Tier A -- digital, within the sampled window.** A complex NCO shifts the spread
signal between sub-slots inside the bandwidth already being sampled. Costs a
multiply, so it can change every burst. This is the tier that troubles a
reactive jammer, which must detect, retune and radiate before the burst it is
chasing has ended.

**Tier B -- LO retuning, across the 100 MHz allocation.** Moves the sampled
window itself. A B205-mini retune with relock takes on the order of a
millisecond, so this changes every few bursts, not every burst.

**Band escape (2.442 <-> 5.24 GHz)** is a third, much slower tier handled by
:class:`BandEscapeController`. With one RF chain per node it is genuinely an
escape rather than diversity: while a node sits on one band it is deaf to the
other, so both ends must agree on the schedule in advance rather than negotiate.

Everything here is pure scheduling logic with no UHD dependency, so it is
testable without hardware. :class:`TimedRetuner` is the only part that touches a
radio, and it is a thin adapter.
"""

import threading

from .security import hop_sequence

#: Centre frequencies of the two allocations, in Hz.
BAND_2G4 = 2_442_000_000
BAND_5G2 = 5_240_000_000
DEFAULT_BANDS = (BAND_2G4, BAND_5G2)

#: Usable width of each allocation, in Hz.
BAND_WIDTH = 100_000_000

#: Default LO step across an allocation. Twelve slots fit in 100 MHz.
DEFAULT_LO_CHANNELS = 12

#: Sub-slots addressed digitally inside one sampled window.
DEFAULT_SUBCHANNELS = 4


class HopPlan:
    """Keyed hop schedule over ``lo_channels * subchannels`` slots.

    A slot is a ``(lo_index, subchannel_index)`` pair. The visiting order within
    each epoch is a keyed permutation, so every usable slot is used exactly once
    per epoch -- bounding worst-case dwell on any one frequency and denying a
    jammer the chance to camp where the sequence lingers.
    """

    def __init__(
        self,
        hop_key,
        lo_channels=DEFAULT_LO_CHANNELS,
        subchannels=DEFAULT_SUBCHANNELS,
        band_width=BAND_WIDTH,
        sample_rate=8_000_000,
    ):
        self._hop_key = hop_key
        self._lo_channels = int(lo_channels)
        self._subchannels = int(subchannels)
        if self._lo_channels <= 0 or self._subchannels <= 0:
            raise ValueError("lo_channels and subchannels must be positive")
        self._band_width = float(band_width)
        self._sample_rate = float(sample_rate)
        self._slot_count = self._lo_channels * self._subchannels
        self._blacklist = frozenset()
        self._lock = threading.Lock()

    @property
    def slot_count(self):
        return self._slot_count

    @property
    def blacklist(self):
        with self._lock:
            return set(self._blacklist)

    def set_blacklist(self, slots):
        """Exclude slots from future epochs.

        Unused on a one-way link. It exists so that a reverse channel can be
        added later without a protocol revision: both ends stay in step for as
        long as they agree on the contents, which is precisely what a return
        path would carry.
        """
        with self._lock:
            self._blacklist = frozenset(int(slot) for slot in slots)

    def order(self, epoch):
        """Return the slot visiting order for ``epoch``."""
        with self._lock:
            blacklist = self._blacklist
        return hop_sequence(self._hop_key, epoch, self._slot_count, blacklist)

    def slot_at(self, epoch, position):
        """Return the slot index visited at ``position`` within ``epoch``."""
        order = self.order(epoch)
        return order[position % len(order)]

    def decompose(self, slot):
        """Split a slot index into ``(lo_index, subchannel_index)``."""
        slot = int(slot) % self._slot_count
        return divmod(slot, self._subchannels)

    def lo_frequency(self, band_centre, slot):
        """Absolute LO frequency for ``slot`` within the allocation."""
        lo_index, _ = self.decompose(slot)
        step = self._band_width / self._lo_channels
        # Centre the comb on the allocation rather than starting at its edge.
        offset = (lo_index - (self._lo_channels - 1) / 2.0) * step
        return float(band_centre + offset)

    def subchannel_offset(self, slot):
        """Baseband NCO offset in Hz for ``slot``, applied without retuning.

        Sub-slots are spaced across the sampled window and centred on zero, so
        the outermost never lands on the window edge where the anti-alias
        response rolls off.
        """
        _, sub_index = self.decompose(slot)
        step = self._sample_rate / (self._subchannels + 1)
        return float((sub_index - (self._subchannels - 1) / 2.0) * step)

    def tuning(self, band_centre, epoch, position):
        """Return ``(lo_frequency_hz, nco_offset_hz)`` for one dwell."""
        slot = self.slot_at(epoch, position)
        return self.lo_frequency(band_centre, slot), self.subchannel_offset(slot)


class BandEscapeController:
    """Decides when to abandon a band for the other one.

    The trigger is deliberately *energy present but nothing decodable*. Plain
    signal loss -- the transmitter switched off, the vehicle drove behind a
    building -- must not cause a band change, because changing bands during a
    real outage only adds an acquisition gap to an already broken link. That
    distinction is why the receiver's classifier reports ``jammed`` separately
    from ``no_signal``.

    ``dwells_before_escape`` consecutive jammed dwells are required, so a single
    unlucky burst never moves the link.
    """

    def __init__(self, bands=DEFAULT_BANDS, dwells_before_escape=4):
        self._bands = tuple(bands)
        if len(self._bands) < 2:
            raise ValueError("band escape needs at least two bands")
        self._required = int(dwells_before_escape)
        if self._required <= 0:
            raise ValueError("dwells_before_escape must be positive")
        self._index = 0
        self._consecutive = 0
        self._escapes = 0
        self._lock = threading.Lock()

    @property
    def band(self):
        with self._lock:
            return self._bands[self._index]

    @property
    def escapes(self):
        with self._lock:
            return self._escapes

    @property
    def consecutive_jammed(self):
        with self._lock:
            return self._consecutive

    def reset(self):
        with self._lock:
            self._consecutive = 0

    def observe(self, link_state):
        """Feed one dwell's classification. Returns the band to use next.

        Only ``jammed`` advances the escape counter. ``ok`` clears it;
        ``no_signal`` and ``degraded`` leave it untouched, so an outage neither
        triggers an escape nor cancels a jam assessment already in progress.
        """
        with self._lock:
            if link_state == "ok":
                self._consecutive = 0
            elif link_state == "jammed":
                self._consecutive += 1
                if self._consecutive >= self._required:
                    self._index = (self._index + 1) % len(self._bands)
                    self._consecutive = 0
                    self._escapes += 1
            return self._bands[self._index]


class TimedRetuner:
    """Schedules LO retunes on a UHD device at precise times.

    Timed commands let both ends change frequency at an agreed instant rather
    than whenever a Python call happens to execute, which is what keeps a
    one-way link in step. Kept deliberately thin: all scheduling decisions come
    from :class:`HopPlan`, so the hopping logic stays testable without a radio.

    ``device`` is a ``uhd.usrp_sink`` or ``uhd.usrp_source``. Passing ``None``
    makes this a no-op recorder, which is how the simulation harness uses it.
    """

    def __init__(self, device=None, settling_time=0.001):
        self._device = device
        self._settling_time = float(settling_time)
        self._history = []
        self._lock = threading.Lock()

    @property
    def history(self):
        with self._lock:
            return list(self._history)

    @property
    def settling_time(self):
        return self._settling_time

    def retune(self, frequency_hz, at_time=None):
        """Retune, optionally at a specific device time."""
        with self._lock:
            self._history.append((frequency_hz, at_time))
        if self._device is None:
            return
        if at_time is not None:
            from gnuradio import uhd  # imported lazily: simulation needs no UHD

            self._device.set_command_time(uhd.time_spec(float(at_time)))
        self._device.set_center_freq(float(frequency_hz), 0)
        if at_time is not None:
            self._device.clear_command_time()
