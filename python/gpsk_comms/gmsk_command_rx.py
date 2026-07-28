"""GMSK command receiver GNU Radio block."""

import threading
import time

import numpy
import pmt
from gnuradio import digital, filter, gr

from .protocol import (
    DEFAULT_ACCESS_CODE,
    PAYLOAD_SIZE,
    PacketCRCError,
    PacketError,
    bytes_to_bits,
    decode_payload,
    normalise_access_code,
)


def _dict_set(message, key, value):
    if isinstance(value, str):
        converted = pmt.intern(value)
    elif isinstance(value, int):
        converted = pmt.from_uint64(value) if value >= 0 else pmt.from_long(value)
    else:
        converted = pmt.to_pmt(value)
    return pmt.dict_add(message, pmt.intern(key), converted)


class _command_frame_parser(gr.sync_block):
    """Scan an unpacked bitstream for fixed-size command frames."""

    def __init__(self, access_code, watchdog_timeout, access_code_threshold=2):
        gr.sync_block.__init__(self, name="command_frame_parser", in_sig=[numpy.uint8], out_sig=None)
        self._access_bits = bytes_to_bits(normalise_access_code(access_code))
        self._access_value = int.from_bytes(normalise_access_code(access_code), "big")
        self._access_mask = (1 << len(self._access_bits)) - 1
        self._access_code_threshold = int(access_code_threshold)
        if not 0 <= self._access_code_threshold <= 8:
            raise ValueError("access_code_threshold must be between 0 and 8")
        self._shift_register = 0
        self._payload_bits = []
        self._collecting = False
        self._watchdog_timeout = float(watchdog_timeout)
        if self._watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout must be positive")
        self._last_valid_time = None
        self._last_identity = None
        self._timed_out = False
        self._state_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        self._counts = {
            "valid": 0,
            "crc_failure": 0,
            "malformed": 0,
            "duplicate": 0,
            "timeout": 0,
        }
        self._command_port = pmt.intern("command")
        self._diagnostics_port = pmt.intern("diagnostics")
        self.message_port_register_out(self._command_port)
        self.message_port_register_out(self._diagnostics_port)

    @property
    def counts(self):
        with self._state_lock:
            return dict(self._counts)

    def start(self):
        started = super().start()
        if not started:
            return False
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="gpsk-command-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        return True

    def stop(self):
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=max(0.1, self._watchdog_timeout))
            self._watchdog_thread = None
        return super().stop()

    def _watchdog_loop(self):
        period = min(0.05, self._watchdog_timeout / 4.0)
        while not self._watchdog_stop.wait(period):
            self._check_watchdog()

    def _publish_diagnostic(self, event, **values):
        message = _dict_set(pmt.make_dict(), "event", event)
        for key, value in values.items():
            message = _dict_set(message, key, value)
        self.message_port_pub(self._diagnostics_port, message)

    def _publish_command(self, command, session_id, sequence, link_state):
        message = pmt.make_dict()
        for key, value in (
            ("command", command),
            ("sequence", sequence),
            ("session_id", session_id),
            ("link_state", link_state),
            ("rx_time_ns", time.time_ns()),
        ):
            message = _dict_set(message, key, value)
        self.message_port_pub(self._command_port, message)

    def _handle_payload(self):
        packed = bytearray()
        for offset in range(0, len(self._payload_bits), 8):
            value = 0
            for bit in self._payload_bits[offset : offset + 8]:
                value = (value << 1) | bit
            packed.append(value)
        try:
            session_id, sequence, command = decode_payload(packed)
        except PacketCRCError as error:
            with self._state_lock:
                self._counts["crc_failure"] += 1
                count = self._counts["crc_failure"]
            self._publish_diagnostic("crc_failure", detail=str(error), count=count)
            return
        except PacketError as error:
            with self._state_lock:
                self._counts["malformed"] += 1
                count = self._counts["malformed"]
            self._publish_diagnostic("malformed_packet", detail=str(error), count=count)
            return

        now = time.monotonic()
        identity = (session_id, sequence)
        with self._state_lock:
            self._last_valid_time = now
            self._timed_out = False
            if identity == self._last_identity:
                self._counts["duplicate"] += 1
                duplicate_count = self._counts["duplicate"]
                is_duplicate = True
            else:
                self._last_identity = identity
                self._counts["valid"] += 1
                valid_count = self._counts["valid"]
                is_duplicate = False
        if is_duplicate:
            self._publish_diagnostic("duplicate", count=duplicate_count)
            return
        self._publish_command(command, session_id, sequence, "ok")
        self._publish_diagnostic("valid_packet", count=valid_count)

    def _check_watchdog(self):
        with self._state_lock:
            if self._last_valid_time is None or self._timed_out:
                return
            if time.monotonic() - self._last_valid_time < self._watchdog_timeout:
                return
            self._timed_out = True
            self._counts["timeout"] += 1
            timeout_count = self._counts["timeout"]
            session_id, sequence = self._last_identity
        self._publish_command("stop", session_id, sequence, "timeout")
        self._publish_diagnostic("timeout", count=timeout_count)

    def work(self, input_items, output_items):
        for raw_bit in input_items[0]:
            bit = int(raw_bit) & 1
            if self._collecting:
                self._payload_bits.append(bit)
                if len(self._payload_bits) == PAYLOAD_SIZE * 8:
                    self._handle_payload()
                    self._payload_bits = []
                    self._collecting = False
                    self._shift_register = 0
                continue
            self._shift_register = ((self._shift_register << 1) | bit) & self._access_mask
            if (self._shift_register ^ self._access_value).bit_count() <= self._access_code_threshold:
                self._payload_bits = []
                self._collecting = True
        self._check_watchdog()
        return len(input_items[0])


class gmsk_command_rx(gr.hier_block2):
    """GMSK receiver with validation, duplicate suppression, and watchdog."""

    def __init__(
        self,
        sample_rate=1_000_000,
        samples_per_symbol=4,
        bt=0.35,
        access_code=DEFAULT_ACCESS_CODE,
        watchdog_timeout=0.5,
        access_code_threshold=2,
    ):
        del bt  # TX pulse shaping parameter retained for a symmetric GRC interface.
        sample_rate = float(sample_rate)
        samples_per_symbol = int(samples_per_symbol)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be at least 2")
        symbol_rate = sample_rate / samples_per_symbol

        gr.hier_block2.__init__(
            self,
            "GMSK Command RX",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0),
        )
        self.message_port_register_hier_out("command")
        self.message_port_register_hier_out("diagnostics")

        self._demodulator = digital.gmsk_demod(
            samples_per_symbol=samples_per_symbol,
            gain_mu=0.05,
            mu=0.5,
            omega_relative_limit=0.01,
            freq_error=0.0,
            verbose=False,
            log=False,
        )
        # The stock demodulator has no carrier-frequency recovery. RF carrier
        # offset appears as a DC term after quadrature demodulation and biases
        # its zero-threshold slicer, so remove only the slow component before
        # symbol timing recovery. The long form limits distortion of data runs.
        self._frequency_offset_filter = filter.dc_blocker_ff(256, True)
        self._demodulator.disconnect(
            self._demodulator.fmdemod,
            self._demodulator.clock_recovery,
        )
        self._demodulator.connect(
            self._demodulator.fmdemod,
            self._frequency_offset_filter,
            self._demodulator.clock_recovery,
        )
        self._channel_filter = filter.fir_filter_ccf(
            1,
            filter.firdes.low_pass(
                1.0,
                sample_rate,
                0.75 * symbol_rate,
                0.25 * symbol_rate,
            ),
        )
        self._parser = _command_frame_parser(
            access_code,
            watchdog_timeout,
            access_code_threshold=access_code_threshold,
        )
        self.connect(self, self._channel_filter, self._demodulator, self._parser)
        self.msg_connect((self._parser, "command"), (self, "command"))
        self.msg_connect((self._parser, "diagnostics"), (self, "diagnostics"))
