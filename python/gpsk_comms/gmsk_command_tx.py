"""GMSK command transmitter GNU Radio block."""

import secrets
import threading
import time

import numpy
import pmt
from gnuradio import digital, gr

from .protocol import DEFAULT_ACCESS_CODE, PAYLOAD_SIZE, encode_payload, normalise_access_code


def _pmt_text(value):
    if pmt.is_symbol(value):
        return pmt.symbol_to_string(value)
    raise ValueError("command value must be a PMT symbol")


class _command_frame_source(gr.sync_block):
    """Continuous packed-byte source containing repeated command frames."""

    def __init__(
        self,
        byte_rate,
        repeat_rate,
        access_code,
        session_id=None,
        command_cycle=None,
        cycle_period=0.5,
        hold_to_send=False,
    ):
        gr.sync_block.__init__(self, name="command_frame_source", in_sig=None, out_sig=[numpy.uint8])
        if byte_rate <= 0:
            raise ValueError("byte_rate must be positive")
        if repeat_rate <= 0:
            raise ValueError("repeat_rate must be positive")
        self._byte_rate = float(byte_rate)
        self._access_code = normalise_access_code(access_code)
        self._interval_bytes = max(1, int(round(float(byte_rate) / float(repeat_rate))))
        frame_size = len(self._access_code) + PAYLOAD_SIZE
        if self._interval_bytes < frame_size:
            raise ValueError(
                "repeat_rate is too high: each interval must fit one complete command frame"
            )
        self._session_id = secrets.randbits(16) if session_id is None else int(session_id) & 0xFFFF
        self._sequence = 0
        self._command_cycle = tuple(command_cycle or ())
        for command in self._command_cycle:
            encode_payload(self._session_id, self._sequence, command)
        self._cycle_period = float(cycle_period)
        if self._command_cycle and self._cycle_period <= 0:
            raise ValueError("cycle_period must be positive")
        self._cycle_period_bytes = max(
            self._interval_bytes,
            int(round(self._byte_rate * self._cycle_period)),
        )
        self._cycle_index = 0
        self._cycle_bytes_generated = 0
        self._next_cycle_byte = self._cycle_period_bytes
        self._command = self._command_cycle[0] if self._command_cycle else "stop"
        self._hold_to_send = bool(hold_to_send)
        if self._hold_to_send and self._command_cycle:
            raise ValueError("hold_to_send and command_cycle cannot both be enabled")
        self._active = not self._hold_to_send
        self._pending = bytearray()
        self._pace_origin = None
        self._bytes_emitted = 0
        self._lock = threading.Lock()
        self._diagnostics_port = pmt.intern("diagnostics")
        self.message_port_register_in(pmt.intern("command"))
        self.message_port_register_out(self._diagnostics_port)
        self.set_msg_handler(pmt.intern("command"), self._handle_command)
        self.set_max_noutput_items(self._interval_bytes)

    @property
    def session_id(self):
        return self._session_id

    @property
    def sequence(self):
        with self._lock:
            return self._sequence

    @property
    def command(self):
        with self._lock:
            return self._command

    def _diagnostic(self, event, detail=None):
        message = pmt.dict_add(pmt.make_dict(), pmt.intern("event"), pmt.intern(event))
        if detail is not None:
            message = pmt.dict_add(message, pmt.intern("detail"), pmt.intern(str(detail)))
        self.message_port_pub(self._diagnostics_port, message)

    def start(self):
        self._pace_origin = time.monotonic()
        self._bytes_emitted = 0
        return super().start()

    def _handle_command(self, message):
        try:
            if not pmt.is_dict(message):
                raise ValueError("input must be a PMT dictionary")
            value = pmt.dict_ref(message, pmt.intern("command"), pmt.PMT_NIL)
            if pmt.eq(value, pmt.PMT_NIL):
                raise ValueError("dictionary has no command field")
            command = _pmt_text(value).lower()
            encode_payload(self._session_id, self._sequence, command)
            pressed_value = pmt.dict_ref(message, pmt.intern("pressed"), pmt.PMT_T)
            if not pmt.is_bool(pressed_value):
                raise ValueError("pressed value must be a PMT boolean")
            pressed = pmt.to_bool(pressed_value)
        except (TypeError, ValueError) as error:
            self._diagnostic("invalid_command", error)
            return
        with self._lock:
            self._command_cycle = ()
            if self._hold_to_send and not pressed:
                self._active = False
                return
            if command != self._command or not self._active:
                self._command = command
                self._sequence = (self._sequence + 1) & 0xFFFF
            self._active = True

    def _next_interval(self):
        with self._lock:
            if (
                self._command_cycle
                and self._cycle_bytes_generated >= self._next_cycle_byte
            ):
                self._cycle_index = (self._cycle_index + 1) % len(self._command_cycle)
                self._command = self._command_cycle[self._cycle_index]
                self._sequence = (self._sequence + 1) & 0xFFFF
                self._next_cycle_byte += self._cycle_period_bytes
            self._cycle_bytes_generated += self._interval_bytes
            if not self._active:
                return bytes([0x55]) * self._interval_bytes
            payload = encode_payload(self._session_id, self._sequence, self._command)
        frame = self._access_code + payload
        return frame + bytes([0x55]) * (self._interval_bytes - len(frame))

    def work(self, input_items, output_items):
        output = output_items[0]
        if self._pace_origin is None:
            self._pace_origin = time.monotonic()
        deadline = self._pace_origin + self._bytes_emitted / self._byte_rate
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        while len(self._pending) < len(output):
            self._pending.extend(self._next_interval())
        output[:] = numpy.frombuffer(self._pending[: len(output)], dtype=numpy.uint8)
        del self._pending[: len(output)]
        self._bytes_emitted += len(output)
        return len(output)


class gmsk_command_tx(gr.hier_block2):
    """Message-controlled, continuously framed GMSK transmitter."""

    def __init__(
        self,
        sample_rate=1_000_000,
        samples_per_symbol=4,
        bt=0.35,
        repeat_rate=10.0,
        access_code=DEFAULT_ACCESS_CODE,
        session_id=None,
        command_cycle=None,
        cycle_period=0.5,
        hold_to_send=False,
    ):
        sample_rate = float(sample_rate)
        samples_per_symbol = int(samples_per_symbol)
        if samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be at least 2")
        symbol_rate = sample_rate / samples_per_symbol
        if symbol_rate <= 0:
            raise ValueError("sample_rate must be positive")

        gr.hier_block2.__init__(
            self,
            "GMSK Command TX",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )
        self.message_port_register_hier_in("command")
        self.message_port_register_hier_out("diagnostics")

        self._source = _command_frame_source(
            byte_rate=symbol_rate / 8.0,
            repeat_rate=float(repeat_rate),
            access_code=access_code,
            session_id=session_id,
            command_cycle=command_cycle,
            cycle_period=cycle_period,
            hold_to_send=hold_to_send,
        )
        source_buffer_bytes = max(512, self._source._interval_bytes * 2)
        self._source.set_max_output_buffer(0, source_buffer_bytes)
        self._modulator = digital.gmsk_mod(
            samples_per_symbol=samples_per_symbol,
            bt=float(bt),
            verbose=False,
            log=False,
            do_unpack=True,
        )
        # Bound queued waveform latency so a command change is observable well
        # inside the receiver watchdog interval, even with a hardware sink.
        max_buffer_samples = 4096
        self.set_max_output_buffer(0, max_buffer_samples)
        self._modulator.set_max_output_buffer(0, max_buffer_samples)
        self._modulator.unpack.set_max_output_buffer(0, 4096)
        self._modulator.nrz.set_max_output_buffer(0, 4096)
        self._modulator.gaussian_filter.set_max_output_buffer(0, 4096)
        self._modulator.fmmod.set_max_output_buffer(0, 4096)
        self.msg_connect((self, "command"), (self._source, "command"))
        self.msg_connect((self._source, "diagnostics"), (self, "diagnostics"))
        self.connect(self._source, self._modulator, self)
