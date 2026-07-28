"""Anti-jam command transmitter and receiver.

Assembles the layers into a working link:

``command message -> authenticated v2 frame -> convolutional code + interleaver
-> DSSS spreading -> digital hop offset -> RF``

and on the way back:

``RF -> narrowband excision -> despread -> deinterleave + Viterbi -> MAC and
replay check -> command message``

The PMT contract is unchanged from the plain GMSK link -- ``{command, pressed}``
in, ``{command, sequence, session_id, link_state, rx_time_ns}`` out -- so
:mod:`gpsk_comms.keyboard_command_source` and ``keyboard_tx.py`` drive this
without modification.

Hop synchronisation
-------------------
Both ends derive the current hop slot from their own clock, as
``counter // dwell``. There is no reverse channel to negotiate over, so this
requires the two clocks to agree to well within one dwell -- with a 100 ms dwell,
ordinary NTP-level synchronisation is ample, but a free-running clock that has
never been set is not. When :meth:`aj_command_rx.hop_locked` reports False for an
extended period, suspect time sync before suspecting the RF.

Frequency-hopping the LO across the allocation additionally needs UHD timed
commands driven from :class:`gpsk_comms.hopping.TimedRetuner`; the flowgraph
owns the radio, so that is wired up by the application rather than here.
"""

import threading
import time

import numpy
import pmt
from gnuradio import gr

from .dsss import (
    PREAMBLE_SYMBOLS,
    build_burst,
    bits_to_symbols,
    dsss_despreader,
    symbols_to_bits,
)
from .excision import narrowband_excision
from .fec import decode_frame, encode_frame, encoded_length
from .hopping import BAND_2G4, BandEscapeController, HopPlan
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
from .security import derive_keys, pn_code, sync_word

#: Payload bits carried by one burst: the v2 frame, unpacked.
PAYLOAD_BITS = PAYLOAD_SIZE_V2 * 8

#: Interleaver depth. Chosen coprime with the code's constraint length so that
#: consecutive coded bits land far apart, which is what converts a jammer's
#: contiguous burst into the scattered errors a convolutional code handles well.
DEFAULT_INTERLEAVE_DEPTH = 17

#: Dwell on one hop slot, in counter ticks (microseconds).
DEFAULT_DWELL_US = 100_000


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


def hop_position(counter, dwell=DEFAULT_DWELL_US):
    """Return ``(epoch, position)`` for a counter value.

    An epoch is one full pass through the hop schedule; position indexes within
    it. Both ends compute this independently from their own clocks.
    """
    index = int(counter) // int(dwell)
    return divmod(index, 1 << 16)


class _aj_burst_source(gr.sync_block):
    """Emit spread, coded, authenticated command bursts as complex baseband.

    Retains the pacing and hold-to-send semantics of the original
    ``_command_frame_source`` so behaviour visible to the operator is unchanged.
    """

    def __init__(
        self,
        keys,
        chip_rate,
        repeat_rate,
        spreading_factor,
        hop_plan,
        dwell=DEFAULT_DWELL_US,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        hold_to_send=False,
        hopping_enabled=True,
    ):
        gr.sync_block.__init__(
            self, name="aj_burst_source", in_sig=None, out_sig=[numpy.complex64]
        )
        self._keys = keys
        self._chip_rate = float(chip_rate)
        self._spreading_factor = int(spreading_factor)
        self._hop_plan = hop_plan
        self._dwell = int(dwell)
        self._interleave_depth = int(interleave_depth)
        self._hopping_enabled = bool(hopping_enabled)

        if repeat_rate <= 0:
            raise ValueError("repeat_rate must be positive")
        # Round the repeat interval up to a whole number of symbols. If it is not
        # a multiple of the spreading factor, successive bursts start at
        # different chip phases; the receiver estimates one chip phase for its
        # whole correlator window, so whenever that window spans two bursts one
        # of them is sampled off-boundary and fails its MAC. The symptom is a
        # high burst-detection rate with most frames failing authentication even
        # on a noiseless loopback.
        raw_interval = max(1, int(round(self._chip_rate / float(repeat_rate))))
        self._interval_chips = (
            -(-raw_interval // self._spreading_factor) * self._spreading_factor
        )
        coded_bits = len(encode_frame(numpy.zeros(PAYLOAD_BITS, dtype=numpy.uint8),
                                      self._interleave_depth))
        self._burst_chips = (PREAMBLE_SYMBOLS + coded_bits + 1) * self._spreading_factor
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
        self._chips_emitted = 0
        self._nco_phase = 0.0
        self._bursts = 0

        self._diagnostics_port = pmt.intern("diagnostics")
        self.message_port_register_in(pmt.intern("command"))
        self.message_port_register_out(self._diagnostics_port)
        self.set_msg_handler(pmt.intern("command"), self._handle_command)
        self.set_max_noutput_items(self._interval_chips)

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

    def _diagnostic(self, event, detail=None):
        message = _dict_set(pmt.make_dict(), "event", event)
        if detail is not None:
            message = _dict_set(message, "detail", str(detail))
        self.message_port_pub(self._diagnostics_port, message)

    def start(self):
        self._pace_origin = time.monotonic()
        self._chips_emitted = 0
        return super().start()

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
        if not self._hopping_enabled:
            return 0
        epoch, position = hop_position(counter, self._dwell)
        return self._hop_plan.slot_at(epoch, position)

    def _apply_nco(self, chips, offset_hz):
        """Shift the burst to its sub-channel, keeping phase continuous."""
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

    def _next_interval(self):
        with self._lock:
            active = self._active
            command = self._command
        if not active:
            return numpy.zeros(self._interval_chips, dtype=numpy.complex64)

        counter = monotonic_counter()
        slot = self._slot(counter)
        frame = encode_frame_v2(self._keys.mac, counter, command)
        coded = encode_frame(bytes_to_bit_array(frame), self._interleave_depth)

        epoch, _ = hop_position(counter, self._dwell)
        preamble = bytes_to_bit_array(sync_word(self._keys.sync, epoch, slot))
        code = pn_code(self._keys.pn, slot, self._spreading_factor)
        burst = self._apply_nco(
            build_burst(coded, code.astype(numpy.float32), preamble),
            self._hop_plan.subchannel_offset(slot) if self._hopping_enabled else 0.0,
        )

        with self._lock:
            self._bursts += 1
        padding = numpy.zeros(self._interval_chips - len(burst), dtype=numpy.complex64)
        return numpy.concatenate((burst, padding))

    def work(self, input_items, output_items):
        output = output_items[0]
        if self._pace_origin is None:
            self._pace_origin = time.monotonic()
        deadline = self._pace_origin + self._chips_emitted / self._chip_rate
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        while len(self._pending) < len(output):
            self._pending = numpy.concatenate((self._pending, self._next_interval()))
        output[:] = self._pending[: len(output)]
        self._pending = self._pending[len(output) :]
        self._chips_emitted += len(output)
        return len(output)


class _aj_frame_decoder(gr.basic_block):
    """Decode despread soft bits into authenticated commands.

    Runs entirely on messages at burst rate, so the cost of Viterbi decoding and
    HMAC verification never touches the sample path.
    """

    def __init__(
        self,
        keys,
        watchdog_timeout=1.0,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        max_advance=None,
        grace_period=0.0,
    ):
        gr.basic_block.__init__(self, name="aj_frame_decoder", in_sig=None, out_sig=None)
        self._keys = keys
        self._interleave_depth = int(interleave_depth)
        self._watchdog_timeout = float(watchdog_timeout)
        if self._watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout must be positive")
        self._grace_period = float(grace_period)
        if self._grace_period < 0:
            raise ValueError("grace_period must not be negative")
        self._replay = (
            ReplayWindow() if max_advance is None else ReplayWindow(max_advance=max_advance)
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

    def start(self):
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="gpsk-aj-watchdog", daemon=True
        )
        self._watchdog_thread.start()
        return True

    def stop(self):
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=max(0.1, self._watchdog_timeout))
            self._watchdog_thread = None
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
        """SNR of the most recent *authenticated* frame.

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
        expected = encoded_length(PAYLOAD_BITS)
        if len(soft) < expected:
            self._diagnostic("short_payload", count=self._count("malformed"))
            return

        bits = decode_frame(soft, PAYLOAD_BITS, self._interleave_depth)
        payload = bit_array_to_bytes(bits)
        try:
            counter, command = decode_frame_v2(self._keys.mac, payload)
        except PacketAuthError as error:
            # Expected under jamming, and also exactly what a forgery looks like.
            self._diagnostic("auth_failure", detail=str(error), count=self._count("auth_failure"))
            return
        except PacketError as error:
            self._diagnostic("malformed", detail=str(error), count=self._count("malformed"))
            return

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
    """Anti-jam command transmitter."""

    def __init__(
        self,
        master_key,
        sample_rate=8_000_000,
        repeat_rate=10.0,
        spreading_factor=1023,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        dwell_us=DEFAULT_DWELL_US,
        hopping_enabled=True,
        hold_to_send=False,
        allow_insecure_key=False,
    ):
        gr.hier_block2.__init__(
            self,
            "AJ Command TX",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )
        self.message_port_register_hier_in("command")
        self.message_port_register_hier_out("diagnostics")

        self._keys = derive_keys(master_key, allow_insecure=allow_insecure_key)
        self._hop_plan = HopPlan(self._keys.hop, sample_rate=sample_rate)
        self._source = _aj_burst_source(
            self._keys,
            chip_rate=sample_rate,
            repeat_rate=repeat_rate,
            spreading_factor=spreading_factor,
            hop_plan=self._hop_plan,
            dwell=dwell_us,
            interleave_depth=interleave_depth,
            hold_to_send=hold_to_send,
            hopping_enabled=hopping_enabled,
        )
        self.msg_connect((self, "command"), (self._source, "command"))
        self.msg_connect((self._source, "diagnostics"), (self, "diagnostics"))
        self.connect(self._source, self)

    @property
    def hop_plan(self):
        return self._hop_plan

    @property
    def command(self):
        return self._source.command

    @property
    def bursts(self):
        return self._source.bursts


class aj_command_rx(gr.hier_block2):
    """Anti-jam command receiver.

    ``grace_period`` extends the watchdog before ``stop`` is commanded. It must
    exceed the re-acquisition gap of a band escape or the fail-safe will fire
    during a legitimate band change. It defaults to zero -- immediate ``stop``,
    the safe choice -- because how long a moving vehicle may keep its last
    command is a safety decision for the operator, not a default worth guessing.
    """

    def __init__(
        self,
        master_key,
        sample_rate=8_000_000,
        spreading_factor=1023,
        interleave_depth=DEFAULT_INTERLEAVE_DEPTH,
        watchdog_timeout=1.0,
        grace_period=0.0,
        dwell_us=DEFAULT_DWELL_US,
        hopping_enabled=True,
        excision_enabled=True,
        detection_threshold=0.25,
        allow_insecure_key=False,
    ):
        gr.hier_block2.__init__(
            self,
            "AJ Command RX",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0),
        )
        self.message_port_register_hier_out("command")
        self.message_port_register_hier_out("diagnostics")

        self._keys = derive_keys(master_key, allow_insecure=allow_insecure_key)
        self._hop_plan = HopPlan(self._keys.hop, sample_rate=sample_rate)
        self._dwell = int(dwell_us)
        self._hopping_enabled = bool(hopping_enabled)
        self._escape = BandEscapeController()

        slot = self._current_slot()
        epoch, _ = hop_position(monotonic_counter(), self._dwell)
        code = pn_code(self._keys.pn, slot, spreading_factor).astype(numpy.float32)
        preamble = bytes_to_bit_array(sync_word(self._keys.sync, epoch, slot))

        self._excision = narrowband_excision(enabled=excision_enabled)
        self._despreader = dsss_despreader(
            code,
            preamble,
            encoded_length(PAYLOAD_BITS),
            symbol_rate=sample_rate / spreading_factor,
            threshold=detection_threshold,
        )
        self._decoder = _aj_frame_decoder(
            self._keys,
            watchdog_timeout=watchdog_timeout,
            interleave_depth=interleave_depth,
            grace_period=grace_period,
        )

        self.connect(self, self._excision, self._despreader)
        self.msg_connect((self._despreader, "payload"), (self._decoder, "payload"))
        self.msg_connect((self._decoder, "command"), (self, "command"))
        self.msg_connect((self._decoder, "diagnostics"), (self, "diagnostics"))

    def _current_slot(self):
        if not self._hopping_enabled:
            return 0
        epoch, position = hop_position(monotonic_counter(), self._dwell)
        return self._hop_plan.slot_at(epoch, position)

    @property
    def counts(self):
        return self._decoder.counts

    @property
    def despread_counts(self):
        return self._despreader.counts

    @property
    def snr_db(self):
        """SNR of the last authenticated frame; -inf until one arrives."""
        return self._decoder.last_valid_snr_db

    @property
    def detection_snr_db(self):
        """SNR of the last correlator detection, authenticated or not."""
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
        return self._escape.observe(self.link_state)
