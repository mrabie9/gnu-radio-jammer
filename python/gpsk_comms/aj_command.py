"""Command transmitter and receiver, built as a ladder of feature levels.

The full stack is::

    command -> authenticated v2 frame -> convolutional code + interleaver
            -> DSSS spreading -> hop offset -> RF

and on the way back::

    RF -> derotate -> narrowband excision -> despread
       -> deinterleave + Viterbi -> MAC and replay check -> command

Every one of those stages is switchable, and :mod:`gpsk_comms.levels` arranges
them into levels 0 to 6 where each level is the previous one plus exactly one
stage. Pass ``level=`` to :class:`aj_command_tx` and :class:`aj_command_rx`;
both ends must be at the same level.

**Start at level 0 and work up.** Level 0 is a bare point-to-point BPSK link
with no key, no coding and no spreading, and if it does not carry traffic then
nothing above it can. The first level that fails identifies the layer at fault,
which is the whole reason the ladder exists: with the stack switched on all at
once, a fault anywhere in it presents identically as "no commands arrive".

The PMT contract is unchanged from the plain GMSK link at every level --
``{command, pressed}`` in, ``{command, sequence, session_id, link_state,
rx_time_ns}`` out -- so :mod:`gpsk_comms.keyboard_command_source` and
``keyboard_tx.py`` drive any of them without modification.

Hop synchronisation
-------------------
Both ends derive the current hop slot from their own clock as
``counter // dwell``. There is no reverse channel to negotiate over, so this
requires the two clocks to agree to well within one dwell -- with a 100 ms dwell,
ordinary NTP-level synchronisation is ample, but a free-running clock that has
never been set is not. Suspect time sync before suspecting the RF when level 4
passes and level 5 does not.

Two timing rules make hopping work, and both are enforced rather than assumed:

* **The transmitter emits exactly one burst per dwell, at the dwell boundary.**
  Its chip stream is anchored to the same wall clock the hop schedule is derived
  from, so a burst can never straddle a boundary and be spread with one code but
  despread with the next.
* **The receiver changes code mid-dwell, not on the boundary.** By then the
  dwell's burst has been buffered and searched, and the next one is still
  :data:`HOP_SWITCH_FRACTION` of a dwell away. Switching on the boundary itself
  would race the burst against the scheduler, GNU Radio's buffering and the
  receiver's own search window, all of which are worth milliseconds.
"""

import math
import threading
import time

import numpy
import pmt
from gnuradio import blocks, gr

from .dsss import PREAMBLE_SYMBOLS, build_burst, dsss_despreader, symbols_to_bits
from .excision import narrowband_excision
from .fec import decode_frame, encode_frame, encoded_length
from .hopping import BAND_2G4, BandEscapeController, HopPlan, TimedRetuner
from .levels import DEFAULT_LEVEL, profile as build_profile
from .protocol import (
    PAYLOAD_SIZE_V2,
    PacketAuthError,
    PacketError,
    PacketReplayError,
    ReplayWindow,
    COMMAND_TO_ID,
    decode_frame_v2,
    encode_frame_v2,
    monotonic_counter,
)
from .security import derive_keys, pn_code, public_keys, sync_word

#: Payload bits carried by one burst: the v2 frame, unpacked. The frame is the
#: same 12 bytes at every level, so the burst layout never changes as the ladder
#: is climbed and a level's failure cannot be an artefact of a different frame.
PAYLOAD_BITS = PAYLOAD_SIZE_V2 * 8

#: Interleaver depth. Chosen coprime with the code's constraint length so that
#: consecutive coded bits land far apart, which is what converts a jammer's
#: contiguous burst into the scattered errors a convolutional code handles well.
DEFAULT_INTERLEAVE_DEPTH = 17

#: Dwell on one hop slot, in counter ticks (microseconds).
DEFAULT_DWELL_US = 100_000

#: How far into a dwell the receiver switches to the *next* dwell's code. Late
#: enough that the current dwell's burst has been buffered and searched, early
#: enough that the next dwell's burst cannot arrive first.
HOP_SWITCH_FRACTION = 0.75

#: How far the transmitter's chip stream may drift behind real time, as a
#: fraction of one dwell, before it re-anchors to the wall clock.
#:
#: Only hopping cares. The hop slot is derived from the clock, so a stream that
#: has fallen behind is spreading bursts with a code the receiver has moved past,
#: and re-anchoring costs one dropped burst instead of every subsequent one.
#: Below level 5 there is no schedule to fall behind and drift is harmless, so
#: the transmitter does not re-anchor at all and a slow host merely transmits
#: more slowly than asked.
#:
#: Persistent re-anchoring means the flowgraph cannot keep up with the sample
#: rate. That is a host capacity problem, not an RF one; the counter is exposed
#: as ``aj_command_tx.reanchors`` so it can be told apart from one.
MAX_DRIFT_DWELLS = 0.25


def _pmt_text(value):
    if pmt.is_symbol(value):
        return pmt.symbol_to_string(value)
    raise ValueError("command value must be a PMT symbol")


def _dict_set(message, key, value):
    if isinstance(value, str):
        converted = pmt.intern(value)
    elif isinstance(value, bool):
        converted = pmt.from_bool(value)
    elif isinstance(value, int):
        converted = pmt.from_uint64(value) if value >= 0 else pmt.from_long(value)
    elif isinstance(value, float):
        converted = pmt.from_double(value)
    else:
        converted = pmt.to_pmt(value)
    return pmt.dict_add(message, pmt.intern(key), converted)


def bytes_to_bit_array(data):
    """Unpack bytes into an MSB-first uint8 array of 0/1."""
    return numpy.unpackbits(numpy.frombuffer(bytes(data), dtype=numpy.uint8))


def bit_array_to_bytes(bits):
    """Repack an MSB-first bit array into bytes."""
    return numpy.packbits(numpy.asarray(bits, dtype=numpy.uint8)).tobytes()


def dwell_index(counter, dwell=DEFAULT_DWELL_US):
    """Return the number of the dwell containing ``counter``.

    Both ends compute this independently from their own clocks; it is the only
    quantity the hop schedule depends on.
    """
    return int(counter) // int(dwell)


def hop_position(counter, dwell=DEFAULT_DWELL_US):
    """Return ``(epoch, position)`` for a counter value.

    An epoch is one full pass through the hop schedule; position indexes within
    it.
    """
    return divmod(dwell_index(counter, dwell), 1 << 16)


def coded_bit_count(link_profile, payload_bits=PAYLOAD_BITS, depth=DEFAULT_INTERLEAVE_DEPTH):
    """Bits on the air per burst, after coding and interleaving.

    Without FEC the payload travels uncoded and this is just ``payload_bits``.
    With it, the convolutional code doubles the count and adds a termination
    tail, and the interleaver pads up to a whole block.
    """
    if not link_profile.fec:
        return int(payload_bits)
    length = encoded_length(payload_bits)
    depth = int(depth)
    if depth > 1:
        length = -(-length // depth) * depth
    return length


def encode_payload_bits(link_profile, bits, depth=DEFAULT_INTERLEAVE_DEPTH):
    """Apply the coding stage, or pass the bits through when it is off."""
    if not link_profile.fec:
        return numpy.asarray(bits, dtype=numpy.uint8)
    return encode_frame(bits, depth)


def decode_payload_bits(link_profile, soft, payload_bits=PAYLOAD_BITS,
                        depth=DEFAULT_INTERLEAVE_DEPTH):
    """Invert :func:`encode_payload_bits`.

    Without FEC the soft values are hard-sliced directly, which is all an
    uncoded link can do: the sign convention is the one
    :func:`gpsk_comms.dsss.differential_decode` produces, positive for a zero.
    """
    if not link_profile.fec:
        return symbols_to_bits(numpy.asarray(soft)[:payload_bits])
    return decode_frame(soft, payload_bits, depth)


def spreading_code(keys, link_profile, slot, spreading_factor):
    """Return the chip sequence for one hop slot.

    With spreading off this is a constant, which makes each symbol a plain
    rectangular pulse ``spreading_factor`` samples long. The burst layout, the
    symbol rate and every timing constant are therefore identical whether
    spreading is on or off -- the signal is simply narrowband instead of spread,
    and the only thing the ``dsss`` rung changes is whether the chips carry a
    pseudorandom code. That is what makes a failure appearing at level 3
    attributable to spreading rather than to a reshuffled waveform.
    """
    if not link_profile.dsss:
        return numpy.ones(int(spreading_factor), dtype=numpy.float32)
    return pn_code(keys.pn, slot, spreading_factor).astype(numpy.float32)


def sync_bits(keys, link_profile, epoch, slot):
    """Return the frame sync word for one hop, as a bit array.

    Rotating it per hop denies a reactive jammer a trigger, but only matters
    once hopping is on; below that it is fixed, so an acquisition failure cannot
    be blamed on the two ends having disagreed about which sync word is current.
    """
    if not link_profile.hopping:
        epoch, slot = 0, 0
    return bytes_to_bit_array(sync_word(keys.sync, epoch, slot))


def link_keys(master_key, link_profile, allow_insecure_key=False):
    """Return the key set for ``link_profile``.

    Levels 0 and 1 need no key and are given the published one, so bringing a
    link up does not require key distribution to be working first. From level 2
    a real key is mandatory and an absent or all-zero one is refused.
    """
    if not link_profile.needs_key:
        return public_keys()
    if master_key is None:
        raise ValueError(
            f"level {link_profile.level} ({link_profile.name}) requires a master key; "
            "levels 0 and 1 do not, so use those to prove the radio first"
        )
    return derive_keys(master_key, allow_insecure=allow_insecure_key)


class _dwell_timer:
    """Calls ``action(dwell_index)`` once per dwell, at a fixed point within it.

    ``offset`` is the fraction of the dwell at which the call happens, and the
    index passed is the dwell that is about to *begin*. Everything that has to
    happen on the hop schedule -- retuning the despreader, stepping the LO --
    goes through here, so the schedule's timing is defined in one place and
    there is one place to look when the two ends disagree about it.

    A fresh thread is created per :meth:`start` rather than subclassing
    ``Thread``, so a flowgraph that is stopped and started again works. It is
    driven by the wall clock at every iteration instead of by accumulating
    sleeps, so a late wake-up costs one dwell rather than a permanent offset
    against the transmitter.
    """

    def __init__(self, dwell, offset, action, name="gpsk-dwell-timer"):
        self._dwell = int(dwell)
        self._offset = int(self._dwell * float(offset))
        self._action = action
        self._name = name
        self._stop = threading.Event()
        self._thread = None
        self._failures = 0

    @property
    def failures(self):
        return self._failures

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        while not self._stop.is_set():
            now = monotonic_counter()
            index = now // self._dwell
            fire_at = index * self._dwell + self._offset
            if now >= fire_at:
                index += 1
                fire_at = index * self._dwell + self._offset
            if self._stop.wait(max(0.0, (fire_at - now) / 1e6)):
                return
            try:
                self._action(index + 1)
            except Exception:  # pragma: no cover - defensive
                # A scheduling thread that dies takes the link down silently and
                # the counters blame the RF. Count it and carry on instead.
                self._failures += 1


class _aj_burst_source(gr.sync_block):
    """Emit command bursts as complex baseband, anchored to the wall clock.

    Retains the pacing and hold-to-send semantics of the original plain-GMSK
    ``_command_frame_source`` so behaviour visible to the operator is unchanged.

    The chip stream is anchored to :func:`monotonic_counter`: chip ``n`` of the
    stream is defined to leave the block at a known counter value. That is what
    lets the transmitter place a burst exactly on a dwell boundary, and it is the
    difference between hopping working and not -- a free-running stream drifts
    against the hop schedule and eventually spreads bursts with a code the
    receiver has already moved on from.
    """

    def __init__(
        self,
        keys,
        link_profile,
        chip_rate,
        repeat_rate,
        spreading_factor,
        hop_plan,
        dwell=DEFAULT_DWELL_US,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        hold_to_send=False,
    ):
        gr.sync_block.__init__(
            self, name="aj_burst_source", in_sig=None, out_sig=[numpy.complex64]
        )
        self._keys = keys
        self._profile = link_profile
        self._chip_rate = float(chip_rate)
        self._chip_rate_int = int(round(self._chip_rate))
        self._spreading_factor = int(spreading_factor)
        self._hop_plan = hop_plan
        self._dwell = int(dwell)
        self._interleave_depth = int(interleave_depth)
        self._max_drift_seconds = MAX_DRIFT_DWELLS * self._dwell / 1e6

        if repeat_rate <= 0:
            raise ValueError("repeat_rate must be positive")
        coded_bits = coded_bit_count(link_profile, PAYLOAD_BITS, self._interleave_depth)
        self._burst_chips = (PREAMBLE_SYMBOLS + coded_bits + 1) * self._spreading_factor

        if self._profile.hopping:
            # One burst per dwell, on the boundary. Any other rate would put a
            # burst somewhere inside the dwell where it could straddle the next
            # boundary, and the receiver would despread its two halves with two
            # different codes.
            self._interval_chips = self._dwell_chips(self._dwell)
            if self._burst_chips > self._interval_chips * HOP_SWITCH_FRACTION:
                raise ValueError(
                    f"a burst is {self._burst_chips} chips but only "
                    f"{int(self._interval_chips * HOP_SWITCH_FRACTION)} fit in the usable "
                    f"part of a {self._dwell} us dwell; raise dwell_us, raise sample_rate, "
                    "or lower spreading_factor"
                )
        else:
            # Round the repeat interval up to a whole number of symbols. If it is
            # not a multiple of the spreading factor, successive bursts start at
            # different chip phases; the receiver estimates one chip phase for its
            # whole correlator window, so whenever that window spans two bursts one
            # of them is sampled off-boundary and fails its integrity check. The
            # symptom is a high burst-detection rate with most frames failing
            # authentication even on a noiseless loopback.
            raw_interval = max(1, int(round(self._chip_rate / float(repeat_rate))))
            self._interval_chips = (
                -(-raw_interval // self._spreading_factor) * self._spreading_factor
            )
            if self._interval_chips < self._burst_chips:
                raise ValueError(
                    "repeat_rate is too high: each interval must fit one complete burst "
                    f"({self._burst_chips} chips at {self._spreading_factor} chips/symbol)"
                )

        self._command = "stop"
        self._hold_to_send = bool(hold_to_send)
        self._active = not self._hold_to_send
        self._sequence = 0
        self._lock = threading.Lock()

        self._pending = numpy.zeros(0, dtype=numpy.complex64)
        self._pace_origin = None
        self._origin_us = 0
        self._cursor = 0
        self._next_burst = 0
        self._chips_emitted = 0
        self._nco_phase = 0.0
        self._bursts = 0
        self._reanchors = 0
        self._timer = None

        self._diagnostics_port = pmt.intern("diagnostics")
        self.message_port_register_in(pmt.intern("command"))
        self.message_port_register_out(self._diagnostics_port)
        self.set_msg_handler(pmt.intern("command"), self._handle_command)
        # Bounded so pacing stays responsive; a whole dwell in one work call
        # would make the sleep granularity as coarse as the hop rate itself.
        self.set_max_noutput_items(min(self._interval_chips, 1 << 16))

    @property
    def command(self):
        with self._lock:
            return self._command

    @property
    def sequence(self):
        with self._lock:
            return self._sequence

    @property
    def bursts(self):
        with self._lock:
            return self._bursts

    @property
    def reanchors(self):
        """Times the stream fell behind real time and was re-anchored."""
        with self._lock:
            return self._reanchors

    @property
    def burst_chips(self):
        return self._burst_chips

    @property
    def interval_chips(self):
        return self._interval_chips

    def _dwell_chips(self, microseconds):
        return int(round(int(microseconds) * self._chip_rate / 1e6))

    def _counter_at(self, chip_index):
        """Wall-clock counter at which chip ``chip_index`` leaves the block."""
        return self._origin_us + (int(chip_index) * 1_000_000) // self._chip_rate_int

    def _chip_of_counter(self, counter):
        """First chip index at or after ``counter``."""
        delta = int(counter) - self._origin_us
        return -(-(delta * self._chip_rate_int) // 1_000_000)

    def _diagnostic(self, event, detail=None):
        message = _dict_set(pmt.make_dict(), "event", event)
        if detail is not None:
            message = _dict_set(message, "detail", str(detail))
        self.message_port_pub(self._diagnostics_port, message)

    def _anchor(self):
        """Pin the chip stream to the current wall clock and reset the schedule."""
        self._pace_origin = time.monotonic()
        self._origin_us = monotonic_counter()
        self._chips_emitted = 0
        self._cursor = 0
        self._pending = numpy.zeros(0, dtype=numpy.complex64)
        if self._profile.hopping:
            first = (self._origin_us // self._dwell + 1) * self._dwell
            self._next_burst = self._chip_of_counter(first)
        else:
            self._next_burst = 0

    def attach_timer(self, timer):
        """Bind a :class:`_dwell_timer` to this block's lifecycle.

        GNU Radio flattens hier blocks away before it starts anything, so
        ``start`` and ``stop`` on a hier block are never called. Anything that
        must run for as long as the flowgraph does has to hang off a leaf block,
        and this is the transmitter's.
        """
        self._timer = timer

    def start(self):
        self._anchor()
        if self._timer is not None:
            self._timer.start()
        return super().start()

    def stop(self):
        if self._timer is not None:
            self._timer.stop()
        return super().stop()

    def _handle_command(self, message):
        try:
            if not pmt.is_dict(message):
                raise ValueError("input must be a PMT dictionary")
            value = pmt.dict_ref(message, pmt.intern("command"), pmt.PMT_NIL)
            if pmt.eq(value, pmt.PMT_NIL):
                raise ValueError("dictionary has no command field")
            command = _pmt_text(value).lower()
            if command not in COMMAND_TO_ID:
                raise ValueError(f"unsupported command: {command!r}")
            pressed_value = pmt.dict_ref(message, pmt.intern("pressed"), pmt.PMT_T)
            if not pmt.is_bool(pressed_value):
                raise ValueError("pressed value must be a PMT boolean")
            pressed = pmt.to_bool(pressed_value)
        except (TypeError, ValueError) as error:
            self._diagnostic("invalid_command", error)
            return
        with self._lock:
            if self._hold_to_send and not pressed:
                self._active = False
                return
            if command != self._command or not self._active:
                self._command = command
                self._sequence = (self._sequence + 1) & 0xFFFF
            self._active = True

    def _slot(self, counter):
        if not self._profile.hopping:
            return 0
        epoch, position = hop_position(counter, self._dwell)
        return self._hop_plan.slot_at(epoch, position)

    def _apply_nco(self, chips, offset_hz):
        """Shift the burst to its sub-channel."""
        if not offset_hz:
            return chips
        index = numpy.arange(len(chips), dtype=numpy.float64)
        rotation = numpy.exp(
            2j * numpy.pi * offset_hz * index / self._chip_rate + 1j * self._nco_phase
        )
        self._nco_phase = float(
            (self._nco_phase + 2 * numpy.pi * offset_hz * len(chips) / self._chip_rate)
            % (2 * numpy.pi)
        )
        return (chips * rotation).astype(numpy.complex64)

    def _build_burst(self, chip_index):
        """Build the burst that starts at ``chip_index``, or silence if idle."""
        with self._lock:
            active = self._active
            command = self._command
        if not active:
            return numpy.zeros(self._burst_chips, dtype=numpy.complex64)

        counter = self._counter_at(chip_index)
        slot = self._slot(counter)
        epoch, _ = hop_position(counter, self._dwell)

        frame = encode_frame_v2(self._keys.mac, counter, command)
        coded = encode_payload_bits(
            self._profile, bytes_to_bit_array(frame), self._interleave_depth
        )
        code = spreading_code(self._keys, self._profile, slot, self._spreading_factor)
        preamble = sync_bits(self._keys, self._profile, epoch, slot)

        burst = build_burst(coded, code, preamble)
        if self._profile.hopping:
            burst = self._apply_nco(burst, self._hop_plan.subchannel_offset(slot))

        with self._lock:
            self._bursts += 1
        return burst

    def _advance_schedule(self):
        """Place the next burst: the next dwell boundary, or one interval on."""
        if not self._profile.hopping:
            self._next_burst += self._interval_chips
            return
        # Measured from the end of the burst just placed, so the next burst
        # always lands on a boundary strictly after it and one dwell can never
        # be asked to carry two.
        end = self._cursor + self._burst_chips
        boundary = (self._counter_at(end) // self._dwell + 1) * self._dwell
        self._next_burst = max(end, self._chip_of_counter(boundary))

    def _generate(self, count):
        """Produce at least ``count`` chips of the anchored stream."""
        chunks = []
        produced = 0
        while produced < count:
            if self._cursor < self._next_burst:
                gap = min(count - produced, self._next_burst - self._cursor)
                chunks.append(numpy.zeros(gap, dtype=numpy.complex64))
                self._cursor += gap
                produced += gap
                continue
            burst = self._build_burst(self._cursor)
            self._advance_schedule()
            chunks.append(burst)
            self._cursor += len(burst)
            produced += len(burst)
        return numpy.concatenate(chunks)

    def work(self, input_items, output_items):
        output = output_items[0]
        if self._pace_origin is None:
            self._anchor()

        deadline = self._pace_origin + self._chips_emitted / self._chip_rate
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif self._profile.hopping and -delay > self._max_drift_seconds:
            # The flowgraph could not keep up. Left alone the stream would sit
            # further and further behind the clock the hop schedule comes from,
            # and the link would fail in a way that looks like an RF problem.
            with self._lock:
                self._reanchors += 1
            self._anchor()
            self._diagnostic("clock_reanchor", f"{-delay:.3f}s behind real time")

        while len(self._pending) < len(output):
            self._pending = numpy.concatenate((self._pending, self._generate(len(output))))
        output[:] = self._pending[: len(output)]
        self._pending = self._pending[len(output) :]
        self._chips_emitted += len(output)
        return len(output)


class _aj_frame_decoder(gr.basic_block):
    """Decode despread soft bits into commands.

    Runs entirely on messages at burst rate, so the cost of Viterbi decoding and
    HMAC verification never touches the sample path.
    """

    def __init__(
        self,
        keys,
        link_profile,
        watchdog_timeout=1.0,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        max_advance=None,
        grace_period=0.0,
    ):
        gr.basic_block.__init__(self, name="aj_frame_decoder", in_sig=None, out_sig=None)
        self._keys = keys
        self._profile = link_profile
        self._interleave_depth = int(interleave_depth)
        self._expected_bits = coded_bit_count(
            link_profile, PAYLOAD_BITS, self._interleave_depth
        )
        self._watchdog_timeout = float(watchdog_timeout)
        if self._watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout must be positive")
        self._grace_period = float(grace_period)
        if self._grace_period < 0:
            raise ValueError("grace_period must not be negative")
        # Replay rejection is only meaningful once frames are authenticated:
        # without a secret key an attacker simply re-tags a replayed frame, and
        # enabling it below level 2 would add a failure mode that has nothing to
        # do with the layer under test.
        self._replay = None
        if self._profile.auth:
            self._replay = (
                ReplayWindow()
                if max_advance is None
                else ReplayWindow(max_advance=max_advance)
            )

        self._counts = {
            "valid": 0,
            "auth_failure": 0,
            "replay": 0,
            "malformed": 0,
            "timeout": 0,
            "duplicate": 0,
        }
        self._last_valid_time = None
        self._last_command = None
        self._last_valid_snr_db = float("-inf")
        self._timed_out = False
        self._state_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        self._session_id = int(monotonic_counter()) & 0xFFFF
        self._sequence = 0
        self._timer = None

        self._command_port = pmt.intern("command")
        self._diagnostics_port = pmt.intern("diagnostics")
        self.message_port_register_in(pmt.intern("payload"))
        self.message_port_register_out(self._command_port)
        self.message_port_register_out(self._diagnostics_port)
        self.set_msg_handler(pmt.intern("payload"), self._handle_payload)

    @property
    def counts(self):
        with self._state_lock:
            return dict(self._counts)

    def attach_timer(self, timer):
        """Bind the hop timer to this block's lifecycle.

        Hier blocks are flattened away before the flowgraph starts, so their
        ``start`` is never called; the receiver's long-running threads hang off
        this leaf block instead, alongside the watchdog that already does.
        """
        self._timer = timer

    def start(self):
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="gpsk-aj-watchdog", daemon=True
        )
        self._watchdog_thread.start()
        if self._timer is not None:
            self._timer.start()
        return True

    def stop(self):
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=max(0.1, self._watchdog_timeout))
            self._watchdog_thread = None
        if self._timer is not None:
            self._timer.stop()
        return True

    def _watchdog_loop(self):
        period = min(0.05, self._watchdog_timeout / 4.0)
        while not self._watchdog_stop.wait(period):
            self._check_watchdog()

    def _publish_command(self, command, link_state):
        message = pmt.make_dict()
        with self._state_lock:
            sequence = self._sequence
            session = self._session_id
        for key, value in (
            ("command", command),
            ("sequence", sequence),
            ("session_id", session),
            ("link_state", link_state),
            ("rx_time_ns", time.time_ns()),
        ):
            message = _dict_set(message, key, value)
        self.message_port_pub(self._command_port, message)

    def _diagnostic(self, event, **values):
        message = _dict_set(pmt.make_dict(), "event", event)
        for key, value in values.items():
            message = _dict_set(message, key, value)
        self.message_port_pub(self._diagnostics_port, message)

    def _count(self, name):
        with self._state_lock:
            self._counts[name] += 1
            return self._counts[name]

    @property
    def last_valid_snr_db(self):
        """SNR of the most recent *accepted* frame.

        Deliberately not the most recent detection: a correlator false alarm on
        noise carries a poor SNR and would otherwise be attributed to the link.
        """
        with self._state_lock:
            return self._last_valid_snr_db

    def _handle_payload(self, message):
        metadata = pmt.car(message)
        snr_db = float("-inf")
        if pmt.is_dict(metadata):
            value = pmt.dict_ref(metadata, pmt.intern("snr_db"), pmt.PMT_NIL)
            if not pmt.eq(value, pmt.PMT_NIL):
                snr_db = pmt.to_double(value)
        soft = numpy.array(pmt.f32vector_elements(pmt.cdr(message)), dtype=numpy.float32)
        if len(soft) < self._expected_bits:
            self._diagnostic("short_payload", count=self._count("malformed"))
            return

        bits = decode_payload_bits(
            self._profile, soft, PAYLOAD_BITS, self._interleave_depth
        )
        payload = bit_array_to_bytes(bits)
        try:
            counter, command = decode_frame_v2(self._keys.mac, payload)
        except PacketAuthError as error:
            # With authentication on this is what a forgery looks like, and it is
            # also what jamming looks like. With it off the key is public and the
            # tag is only an integrity check, so this means corruption alone.
            self._diagnostic(
                "auth_failure", detail=str(error), count=self._count("auth_failure")
            )
            return
        except PacketError as error:
            self._diagnostic("malformed", detail=str(error), count=self._count("malformed"))
            return

        if self._replay is not None:
            try:
                self._replay.check(counter)
            except PacketReplayError as error:
                self._diagnostic("replay", detail=str(error), count=self._count("replay"))
                return

        with self._state_lock:
            self._last_valid_time = time.monotonic()
            self._last_valid_snr_db = snr_db
            self._timed_out = False
            repeated = command == self._last_command
            self._last_command = command
            if not repeated:
                self._sequence = (self._sequence + 1) & 0xFFFF
            self._counts["valid"] += 1
            valid = self._counts["valid"]
            if repeated:
                self._counts["duplicate"] += 1

        self._diagnostic("valid_frame", count=valid)
        if not repeated:
            self._publish_command(command, "ok")

    def _check_watchdog(self):
        with self._state_lock:
            if self._last_valid_time is None or self._timed_out:
                return
            elapsed = time.monotonic() - self._last_valid_time
            if elapsed < self._watchdog_timeout + self._grace_period:
                return
            self._timed_out = True
            self._counts["timeout"] += 1
            count = self._counts["timeout"]
        self._publish_command("stop", "timeout")
        self._diagnostic("timeout", count=count)


class aj_command_tx(gr.hier_block2):
    """Command transmitter, at a selectable feature level.

    ``level`` selects how much of the stack is switched on; see
    :mod:`gpsk_comms.levels`. Individual features can be forced on or off on top
    of the level for bisecting a fault, but for normal use set the level alone
    and set it the same at both ends.

    ``master_key`` may be ``None`` at levels 0 and 1, which need no key.
    """

    def __init__(
        self,
        master_key=None,
        sample_rate=8_000_000,
        repeat_rate=10.0,
        spreading_factor=1023,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        dwell_us=DEFAULT_DWELL_US,
        level=DEFAULT_LEVEL,
        fec=None,
        auth=None,
        dsss=None,
        excision=None,
        hopping=None,
        band_escape=None,
        hopping_enabled=None,
        hold_to_send=False,
        allow_insecure_key=False,
        band=BAND_2G4,
        retuner=None,
    ):
        gr.hier_block2.__init__(
            self,
            "AJ Command TX",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )
        self.message_port_register_hier_in("command")
        self.message_port_register_hier_out("diagnostics")

        # hopping_enabled is the pre-ladder spelling and is still honoured.
        if hopping is None:
            hopping = hopping_enabled
        self._profile = build_profile(
            level,
            fec=fec,
            auth=auth,
            dsss=dsss,
            excision=excision,
            hopping=hopping,
            band_escape=band_escape,
        )
        self._keys = link_keys(master_key, self._profile, allow_insecure_key)
        self._dwell = int(dwell_us)
        self._hop_plan = HopPlan(self._keys.hop, sample_rate=sample_rate)
        self._source = _aj_burst_source(
            self._keys,
            self._profile,
            chip_rate=sample_rate,
            repeat_rate=repeat_rate,
            spreading_factor=spreading_factor,
            hop_plan=self._hop_plan,
            dwell=dwell_us,
            interleave_depth=interleave_depth,
            hold_to_send=hold_to_send,
        )
        self.msg_connect((self, "command"), (self._source, "command"))
        self.msg_connect((self._source, "diagnostics"), (self, "diagnostics"))
        self.connect(self._source, self)

        # Tier B: step the LO across the allocation on the same keyed schedule.
        # Derived from the clock and the key at both ends independently, exactly
        # as the sub-channel hop is, so it needs no coordination -- but it does
        # need a radio, and with retuner=None it records the schedule instead.
        self._band = int(band)
        self._retuner = retuner if retuner is not None else TimedRetuner()
        self._lo_timer = None
        if self._profile.band_escape:
            self._lo_timer = _dwell_timer(
                self._dwell, HOP_SWITCH_FRACTION, self._retune, name="gpsk-tx-lo"
            )
            self._source.attach_timer(self._lo_timer)
            self._retune(dwell_index(monotonic_counter(), self._dwell))

    def _retune(self, index):
        epoch, position = divmod(int(index), 1 << 16)
        slot = self._hop_plan.slot_at(epoch, position)
        self._retuner.retune(self._hop_plan.lo_frequency(self._band, slot))

    @property
    def profile(self):
        return self._profile

    @property
    def level(self):
        return self._profile.level

    @property
    def hop_plan(self):
        return self._hop_plan

    @property
    def retuner(self):
        return self._retuner

    @property
    def command(self):
        return self._source.command

    @property
    def bursts(self):
        return self._source.bursts

    @property
    def reanchors(self):
        return self._source.reanchors


class aj_command_rx(gr.hier_block2):
    """Command receiver, at a selectable feature level.

    ``level`` must match the transmitter's. See :class:`aj_command_tx`.

    ``grace_period`` extends the watchdog before ``stop`` is commanded. It must
    exceed the re-acquisition gap of a band escape or the fail-safe will fire
    during a legitimate band change. It defaults to zero -- immediate ``stop``,
    the safe choice -- because how long a moving vehicle may keep its last
    command is a safety decision for the operator, not a default worth guessing.
    """

    def __init__(
        self,
        master_key=None,
        sample_rate=8_000_000,
        spreading_factor=1023,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        watchdog_timeout=1.0,
        grace_period=0.0,
        dwell_us=DEFAULT_DWELL_US,
        level=DEFAULT_LEVEL,
        fec=None,
        auth=None,
        dsss=None,
        excision=None,
        hopping=None,
        band_escape=None,
        hopping_enabled=None,
        excision_enabled=None,
        detection_threshold=0.25,
        allow_insecure_key=False,
        band=BAND_2G4,
        retuner=None,
    ):
        gr.hier_block2.__init__(
            self,
            "AJ Command RX",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0),
        )
        self.message_port_register_hier_out("command")
        self.message_port_register_hier_out("diagnostics")

        if hopping is None:
            hopping = hopping_enabled
        if excision is None:
            excision = excision_enabled
        self._profile = build_profile(
            level,
            fec=fec,
            auth=auth,
            dsss=dsss,
            excision=excision,
            hopping=hopping,
            band_escape=band_escape,
        )
        self._keys = link_keys(master_key, self._profile, allow_insecure_key)
        self._sample_rate = float(sample_rate)
        self._spreading_factor = int(spreading_factor)
        self._dwell = int(dwell_us)
        self._hop_plan = HopPlan(self._keys.hop, sample_rate=sample_rate)
        self._escape = BandEscapeController()
        self._band = int(band)
        self._retuner = retuner if retuner is not None else TimedRetuner()

        index = dwell_index(monotonic_counter(), self._dwell)
        epoch, position = divmod(index, 1 << 16)
        slot = self._hop_plan.slot_at(epoch, position) if self._profile.hopping else 0
        code = spreading_code(self._keys, self._profile, slot, self._spreading_factor)
        preamble = sync_bits(self._keys, self._profile, epoch, slot)

        self._despreader = dsss_despreader(
            code,
            preamble,
            coded_bit_count(self._profile, PAYLOAD_BITS, interleave_depth),
            symbol_rate=sample_rate / spreading_factor,
            threshold=detection_threshold,
        )
        self._decoder = _aj_frame_decoder(
            self._keys,
            self._profile,
            watchdog_timeout=watchdog_timeout,
            interleave_depth=interleave_depth,
            grace_period=grace_period,
        )

        # The receive chain is assembled from whichever stages the level asks
        # for, so a disabled stage is genuinely absent rather than switched off
        # internally -- there is then no chance of a stage that is meant to be
        # off still perturbing the stream.
        chain = [self]
        self._rotator = None
        if self._profile.hopping:
            self._check_hop_timing()
            self._rotator = blocks.rotator_cc(0.0)
            chain.append(self._rotator)
        self._excision = None
        if self._profile.excision:
            self._excision = narrowband_excision()
            chain.append(self._excision)
        chain.append(self._despreader)
        self.connect(*chain)

        self.msg_connect((self._despreader, "payload"), (self._decoder, "payload"))
        self.msg_connect((self._decoder, "command"), (self, "command"))
        self.msg_connect((self._decoder, "diagnostics"), (self, "diagnostics"))

        self._tune_to(index)
        self._hop_timer = None
        if self._profile.hopping:
            self._hop_timer = _dwell_timer(
                self._dwell, HOP_SWITCH_FRACTION, self._on_dwell, name="gpsk-rx-hop"
            )
            self._decoder.attach_timer(self._hop_timer)

    def _check_hop_timing(self):
        """Refuse a configuration in which the code changes under a live burst.

        The receiver must finish searching for dwell N's burst before it retunes
        for dwell N+1. That needs a full search window to fit inside
        :data:`HOP_SWITCH_FRACTION` of a dwell. Raising here is far kinder than
        the alternative, which is a link that decodes nothing for reasons no
        counter explains.
        """
        window_seconds = self._despreader.window_chips / self._sample_rate
        usable = HOP_SWITCH_FRACTION * self._dwell / 1e6
        if window_seconds > usable:
            raise ValueError(
                f"hopping needs a search window ({window_seconds * 1e3:.1f} ms) shorter "
                f"than {HOP_SWITCH_FRACTION:g} of a dwell ({usable * 1e3:.1f} ms); "
                f"raise dwell_us above {math.ceil(window_seconds * 1e6 / HOP_SWITCH_FRACTION)}, "
                "raise sample_rate, or lower spreading_factor"
            )

    def _tune_to(self, index):
        """Install the spreading code, sync word and NCO offset for one dwell."""
        epoch, position = divmod(int(index), 1 << 16)
        slot = self._hop_plan.slot_at(epoch, position) if self._profile.hopping else 0
        self._despreader.set_code(
            spreading_code(self._keys, self._profile, slot, self._spreading_factor)
        )
        self._despreader.set_preamble(sync_bits(self._keys, self._profile, epoch, slot))
        if self._rotator is not None:
            # Undo the transmitter's sub-channel shift. Both ends compute the
            # offset from the same slot, so the residual is exactly zero and no
            # frequency acquisition is needed for the hop itself.
            offset = self._hop_plan.subchannel_offset(slot)
            self._rotator.set_phase_inc(-2.0 * math.pi * offset / self._sample_rate)
        if self._profile.band_escape:
            self._retuner.retune(self._hop_plan.lo_frequency(self._band, slot))

    def _on_dwell(self, index):
        if self._profile.band_escape:
            self.observe_dwell()
        self._tune_to(index)

    @property
    def profile(self):
        return self._profile

    @property
    def level(self):
        return self._profile.level

    @property
    def hop_plan(self):
        return self._hop_plan

    @property
    def retuner(self):
        return self._retuner

    @property
    def counts(self):
        return self._decoder.counts

    @property
    def despread_counts(self):
        return self._despreader.counts

    @property
    def excision_statistics(self):
        return None if self._excision is None else self._excision.statistics

    @property
    def snr_db(self):
        """SNR of the last accepted frame; -inf until one arrives."""
        return self._decoder.last_valid_snr_db

    @property
    def detection_snr_db(self):
        """SNR of the last correlator detection, accepted or not."""
        return self._despreader.last_snr_db

    @property
    def band(self):
        return self._escape.band

    @property
    def link_state(self):
        """Classify the link: ok, degraded, jammed, or no_signal.

        The distinction that matters is ``jammed`` -- energy present but nothing
        decodable -- versus ``no_signal``, where there is nothing there at all.
        Only the former should provoke a band escape; answering a genuine outage
        by adding an acquisition gap to it makes the outage worse.
        """
        counts = self._decoder.counts
        despread = self._despreader.counts
        snr = self._decoder.last_valid_snr_db

        if counts["valid"] and not counts["timeout"]:
            return "ok" if snr > 6.0 else "degraded"
        if despread["burst"] or despread["no_sync"]:
            # The correlator is seeing structure but nothing authenticates.
            if counts["auth_failure"] or despread["no_sync"]:
                return "jammed"
            return "degraded"
        return "no_signal"

    def observe_dwell(self):
        """Feed the current link state to the band-escape controller."""
        band = self._escape.observe(self.link_state)
        self._band = int(band)
        return band
