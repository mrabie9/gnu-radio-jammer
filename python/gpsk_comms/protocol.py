"""Wire format shared by the command transmitter and receiver.

Two frame versions coexist:

* **v1** -- the original CRC-32 frame. It is unauthenticated: the access code is
  a public constant and anyone can forge a command. Retained unchanged so the
  plain GMSK link and its tests keep working, but it must not be used on any
  link an adversary can reach.
* **v2** -- the authenticated frame used by the anti-jam link. A truncated
  HMAC replaces the CRC, providing error detection *and* forgery resistance,
  and a monotonic counter provides replay protection.
"""

import struct
import time
import zlib

from .security import MAC_SIZE, frame_mac, verify_mac

PROTOCOL_VERSION = 1
PROTOCOL_VERSION_V2 = 2
DEFAULT_ACCESS_CODE = "D391DA26"
ACCESS_CODE_BYTES = 4
BODY_FORMAT = ">BHHB"
BODY_SIZE = struct.calcsize(BODY_FORMAT)
PAYLOAD_SIZE = BODY_SIZE + 4

# v2: version 1B | counter 6B | command 1B | MAC 4B
BODY_FORMAT_V2 = ">B6sB"
BODY_SIZE_V2 = struct.calcsize(BODY_FORMAT_V2)
PAYLOAD_SIZE_V2 = BODY_SIZE_V2 + MAC_SIZE
COUNTER_BYTES = 6
COUNTER_MODULUS = 1 << (8 * COUNTER_BYTES)

#: Default freshness window for the replay guard, in counter ticks (1 tick =
#: 1 microsecond). A frame more than this far ahead of the last accepted one is
#: rejected, so a captured frame cannot be replayed later with the counter
#: fast-forwarded to lock the legitimate transmitter out.
DEFAULT_MAX_ADVANCE = 5_000_000

COMMAND_TO_ID = {
    "stop": 0,
    "forward": 1,
    "backward": 2,
    "left": 3,
    "right": 4,
}
ID_TO_COMMAND = {value: key for key, value in COMMAND_TO_ID.items()}


class PacketError(ValueError):
    """Base class for rejected command payloads."""


class PacketLengthError(PacketError):
    """Raised when a payload is not the fixed protocol length."""


class PacketCRCError(PacketError):
    """Raised when a payload CRC does not match its body."""


class PacketFormatError(PacketError):
    """Raised when a CRC-valid payload has unsupported field values."""


class PacketAuthError(PacketError):
    """Raised when a v2 payload fails its MAC check.

    Unlike a CRC failure this is not necessarily corruption -- it is what a
    forgery attempt looks like, so the receiver counts it separately.
    """


class PacketReplayError(PacketError):
    """Raised when an authentic v2 payload arrives with a stale counter."""


def normalise_access_code(value):
    """Return a four-byte access code from an 8-digit hexadecimal string."""
    if isinstance(value, bytes):
        raw = value
    else:
        text = str(value).strip().replace("0x", "").replace("_", "")
        if len(text) != ACCESS_CODE_BYTES * 2:
            raise ValueError("access_code must contain exactly 8 hexadecimal digits")
        try:
            raw = bytes.fromhex(text)
        except ValueError as error:
            raise ValueError("access_code must be hexadecimal") from error
    if len(raw) != ACCESS_CODE_BYTES:
        raise ValueError("access_code must contain exactly 4 bytes")
    return raw


def encode_payload(session_id, sequence, command):
    """Encode a command payload and append its big-endian CRC-32."""
    if command not in COMMAND_TO_ID:
        raise ValueError(f"unsupported command: {command!r}")
    body = struct.pack(
        BODY_FORMAT,
        PROTOCOL_VERSION,
        int(session_id) & 0xFFFF,
        int(sequence) & 0xFFFF,
        COMMAND_TO_ID[command],
    )
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_payload(payload):
    """Validate and decode a payload.

    Returns ``(session_id, sequence, command)``. Invalid data raises ValueError.
    """
    payload = bytes(payload)
    if len(payload) != PAYLOAD_SIZE:
        raise PacketLengthError(f"payload must be exactly {PAYLOAD_SIZE} bytes")
    body, crc_bytes = payload[:-4], payload[-4:]
    expected_crc = struct.unpack(">I", crc_bytes)[0]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise PacketCRCError("CRC mismatch")
    version, session_id, sequence, command_id = struct.unpack(BODY_FORMAT, body)
    if version != PROTOCOL_VERSION:
        raise PacketFormatError(f"unsupported protocol version: {version}")
    try:
        command = ID_TO_COMMAND[command_id]
    except KeyError as error:
        raise PacketFormatError(f"unsupported command id: {command_id}") from error
    return session_id, sequence, command


def bytes_to_bits(data):
    """Expand packed bytes into an MSB-first list of zero/one integers."""
    return [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)]


def monotonic_counter():
    """Return the current counter value in microseconds.

    Six bytes at microsecond resolution wrap after roughly nine years, which is
    far beyond any single deployment, so the counter is treated as strictly
    increasing and wraparound is not handled.
    """
    return time.time_ns() // 1000 % COUNTER_MODULUS


def encode_frame_v2(mac_key, counter, command):
    """Encode an authenticated command frame."""
    if command not in COMMAND_TO_ID:
        raise ValueError(f"unsupported command: {command!r}")
    counter = int(counter) % COUNTER_MODULUS
    body = struct.pack(
        BODY_FORMAT_V2,
        PROTOCOL_VERSION_V2,
        counter.to_bytes(COUNTER_BYTES, "big"),
        COMMAND_TO_ID[command],
    )
    return body + frame_mac(mac_key, body)


def decode_frame_v2(mac_key, payload):
    """Validate and decode an authenticated frame.

    Returns ``(counter, command)``. The MAC is checked before any field is
    interpreted, so malformed-field reporting cannot be used as an oracle by an
    attacker who does not hold the key.
    """
    payload = bytes(payload)
    if len(payload) != PAYLOAD_SIZE_V2:
        raise PacketLengthError(f"payload must be exactly {PAYLOAD_SIZE_V2} bytes")
    body, tag = payload[:-MAC_SIZE], payload[-MAC_SIZE:]
    if not verify_mac(mac_key, body, tag):
        raise PacketAuthError("MAC mismatch")
    version, counter_bytes, command_id = struct.unpack(BODY_FORMAT_V2, body)
    if version != PROTOCOL_VERSION_V2:
        raise PacketFormatError(f"unsupported protocol version: {version}")
    try:
        command = ID_TO_COMMAND[command_id]
    except KeyError as error:
        raise PacketFormatError(f"unsupported command id: {command_id}") from error
    return int.from_bytes(counter_bytes, "big"), command


class ReplayWindow:
    """Rejects stale and implausibly-advanced counters.

    Two rules, both needed:

    * The counter must strictly increase, which stops a captured frame from
      being replayed to re-issue a command the operator has since cancelled.
    * It must not jump more than ``max_advance`` ahead, which stops an attacker
      replaying one captured frame with a far-future counter to lock the
      legitimate transmitter out of the window for good.

    Without ``clock`` the very first frame after startup is accepted on trust,
    because there is nothing to compare it against -- an adversary who captured
    a frame earlier can replay it into a freshly booted receiver exactly once.
    Passing ``clock`` (any callable returning the counter's time base, normally
    :func:`monotonic_counter`) closes that hole by bounding every frame,
    including the first, against wall-clock time. It requires the two ends to
    agree on time to within ``max_advance``.
    """

    def __init__(self, max_advance=DEFAULT_MAX_ADVANCE, clock=None):
        self._max_advance = int(max_advance)
        if self._max_advance <= 0:
            raise ValueError("max_advance must be positive")
        self._clock = clock
        self._last = None

    @property
    def last(self):
        return self._last

    def reset(self):
        self._last = None

    def check(self, counter):
        """Raise :class:`PacketReplayError` unless ``counter`` is fresh."""
        counter = int(counter)
        if self._clock is not None:
            skew = counter - int(self._clock())
            if abs(skew) > self._max_advance:
                raise PacketReplayError(
                    f"counter {counter} is {skew} ticks from local time"
                )
        if self._last is not None:
            if counter <= self._last:
                raise PacketReplayError(
                    f"counter {counter} does not advance past {self._last}"
                )
            if counter - self._last > self._max_advance:
                raise PacketReplayError(
                    f"counter {counter} jumps more than {self._max_advance} past {self._last}"
                )
        self._last = counter
        return counter
